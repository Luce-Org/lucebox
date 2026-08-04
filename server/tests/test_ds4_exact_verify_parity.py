#!/usr/bin/env python3
"""Parser regressions for the DS4 reference-exact model gate."""

import importlib.util
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SERVER_DIR / "scripts" / "test_ds4_exact_verify_parity.py"
SPEC = importlib.util.spec_from_file_location("test_ds4_exact_verify_parity_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TokenTraceParserTest(unittest.TestCase):
    def test_parses_exact_trace(self):
        trace = MODULE.parse_token_trace(
            "[ds4-exact-verify-trace] reference_exact=1 speculation=1 n=3 ids=[8 13 21]\n",
            expected_reference_exact=True,
            expected_speculation=True,
        )
        self.assertEqual(trace.tokens, (8, 13, 21))

    def test_rejects_declared_count_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "declared 3 IDs but contained 2"):
            MODULE.parse_token_trace(
                "[ds4-exact-verify-trace] reference_exact=1 speculation=1 n=3 ids=[8 13]\n",
                expected_reference_exact=True,
                expected_speculation=True,
            )

    def test_rejects_empty_trace(self):
        with self.assertRaisesRegex(RuntimeError, "token trace is empty"):
            MODULE.parse_token_trace(
                "[ds4-exact-verify-trace] reference_exact=0 speculation=0 n=0 ids=[]\n",
                expected_reference_exact=False,
                expected_speculation=False,
            )

    def test_rejects_non_speculative_exact_run(self):
        with self.assertRaisesRegex(RuntimeError, "speculation mode mismatch"):
            MODULE.parse_token_trace(
                "[ds4-exact-verify-trace] reference_exact=1 speculation=0 n=2 ids=[3 5]\n",
                expected_reference_exact=True,
                expected_speculation=True,
            )

    def test_requires_positive_speculation_summary_and_exact_banner(self):
        summary = MODULE.require_speculation_work(
            "[ds4-spec] reference-exact verifier: sequential target replay "
            "with full rollback snapshots\n"
            "[ds4-spec] gen=9 steps=4 matched=5 offered=12 "
            "mean_accept=1.25/3.00 q_cap=4 full_snap=1\n"
        )
        self.assertEqual(summary.matched, 5)
        self.assertEqual(summary.offered, 12)
        self.assertTrue(summary.full_snapshot)

    def test_rejects_zero_speculation_steps(self):
        with self.assertRaisesRegex(RuntimeError, "did no work"):
            MODULE.require_speculation_work(
                "[ds4-spec] reference-exact verifier: sequential target replay "
                "with full rollback snapshots\n"
                "[ds4-spec] gen=0 steps=0 matched=0 offered=0 "
                "mean_accept=0/0 q_cap=4 full_snap=1\n"
            )

    def test_rejects_all_accepted_run_without_rollback(self):
        with self.assertRaisesRegex(RuntimeError, "did not exercise rejection rollback"):
            MODULE.require_speculation_work(
                "[ds4-spec] reference-exact verifier: sequential target replay "
                "with full rollback snapshots\n"
                "[ds4-spec] gen=12 steps=4 matched=12 offered=12 "
                "mean_accept=3.00/3.00 q_cap=4 full_snap=1\n"
            )

    def test_rejects_summary_without_exact_counters(self):
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            MODULE.require_speculation_work(
                "[ds4-spec] reference-exact verifier: sequential target replay "
                "with full rollback snapshots\n"
                "[ds4-spec] gen=9 steps=4 mean_accept=1.25/3.00 q_cap=4 full_snap=1\n"
            )

    def test_rejects_run_without_full_snapshots(self):
        with self.assertRaisesRegex(RuntimeError, "did not enable full rollback snapshots"):
            MODULE.require_speculation_work(
                "[ds4-spec] reference-exact verifier: sequential target replay "
                "with full rollback snapshots\n"
                "[ds4-spec] gen=9 steps=4 matched=5 offered=12 "
                "mean_accept=1.25/3.00 q_cap=4 full_snap=0\n"
            )

    def test_reports_positional_token_mismatch(self):
        self.assertEqual(
            MODULE.token_mismatch_message((3, 5, 8), (3, 7, 8)),
            "first token mismatch at 1: ar=(5, 8) exact=(7, 8)",
        )

    def test_reports_length_mismatch_after_common_prefix(self):
        self.assertEqual(
            MODULE.token_mismatch_message((3, 5), (3, 5, 8)),
            "token trace length mismatch after common prefix of 2: ar=2 exact=3",
        )


class EvidenceLifecycleTest(unittest.TestCase):
    def test_failed_attempt_does_not_occupy_final_evidence_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            attempt_dir = log_dir / "attempt-test"
            attempt_dir.mkdir()
            (attempt_dir / "ar.log").write_text("diagnostic", encoding="utf-8")

            failed_dir = MODULE.retain_failed_attempt(attempt_dir)

            self.assertEqual(failed_dir.name, "failed-test")
            self.assertEqual((failed_dir / "ar.log").read_text(encoding="utf-8"), "diagnostic")
            self.assertFalse((log_dir / "ar.log").exists())

    def test_promotes_complete_evidence_set(self):
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            attempt_dir = log_dir / "attempt-test"
            attempt_dir.mkdir()
            staged = [attempt_dir / name for name in ("ar.log", "exact.log", "manifest.json")]
            final = [log_dir / path.name for path in staged]
            for index, path in enumerate(staged):
                path.write_text(str(index), encoding="utf-8")

            MODULE.promote_evidence(staged, final)

            self.assertEqual([path.read_text(encoding="utf-8") for path in final], ["0", "1", "2"])

    def test_preexisting_final_rolls_back_only_new_links(self):
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            attempt_dir = log_dir / "attempt-test"
            attempt_dir.mkdir()
            staged = [attempt_dir / name for name in ("ar.log", "exact.log")]
            final = [log_dir / path.name for path in staged]
            for path in staged:
                path.write_text("diagnostic", encoding="utf-8")
            final[1].write_text("existing evidence", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                MODULE.promote_evidence(staged, final)

            self.assertTrue(all(path.exists() for path in staged))
            self.assertFalse(final[0].exists())
            self.assertEqual(final[1].read_text(encoding="utf-8"), "existing evidence")

    def test_denied_staged_cleanup_does_not_fail_completed_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            attempt_dir = log_dir / "attempt-test"
            attempt_dir.mkdir()
            staged = [attempt_dir / name for name in ("ar.log", "exact.log")]
            final = [log_dir / path.name for path in staged]
            for path in staged:
                path.write_text("evidence", encoding="utf-8")

            with mock.patch.object(MODULE.Path, "unlink", side_effect=PermissionError):
                MODULE.promote_evidence(staged, final)

            self.assertTrue(all(path.exists() for path in final))
            self.assertTrue(all(path.exists() for path in staged))


class CaseEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(
            draft=Path("draft.gguf"),
            spec_q=4,
            mmvq_max_ncols=4,
        )

    def test_arms_differ_only_by_reference_exact_activation(self):
        inherited = {
            "PATH": os.environ.get("PATH", ""),
            "DFLASH_DS4_FUSED_VERIFY": "1",
            "DFLASH_DS4_DRAFT_GPU": "7",
            "DFLASH_EXPERT_BUDGET_MB": "1234",
            "DFLASH_MOE_TP_BACKEND": "cuda",
            "DFLASH_MMQ_SUB_BATCH": "1",
            "GGML_BATCH_PEER_COPIES": "1",
            "LUCE_MMVQ_MAX_NCOLS": "99",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            ar = MODULE.case_environment(self.args, False)
            exact = MODULE.case_environment(self.args, True)

        intentional = {
            "DFLASH_DS4_DRAFT": "draft.gguf",
            "DFLASH_DS4_SPEC": "1",
            "DFLASH_DS4_SPEC_REFERENCE_EXACT": "1",
        }
        self.assertEqual({key: exact[key] for key in intentional}, intentional)
        self.assertEqual(
            {key: value for key, value in exact.items() if key not in intentional},
            ar,
        )

    def test_common_policy_is_fixed_and_inherited_policy_is_removed(self):
        inherited = {
            "DFLASH_DS4_ADAPTIVE_WIDTH": "1",
            "DFLASH_DS4_FUSED_VERIFY": "1",
            "DFLASH_EXPERT_BUDGET_MB": "1234",
            "DFLASH_MOE_TP_BACKEND": "cuda",
            "DFLASH_MMQ_SUB_BATCH": "1",
            "GGML_CUDA_BATCH_PEER_COPIES": "1",
            "LUCE_MMVQ_MAX_NCOLS": "99",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            ar = MODULE.case_environment(self.args, False)

        self.assertEqual(ar["DFLASH_DS4_ADAPTIVE_WIDTH"], "0")
        self.assertEqual(ar["DFLASH_DS4_SPEC_Q"], "4")
        self.assertEqual(ar["LUCE_MMVQ_MAX_NCOLS"], "4")
        for name in (
            "DFLASH_DS4_FUSED_VERIFY",
            "DFLASH_EXPERT_BUDGET_MB",
            "DFLASH_MOE_TP_BACKEND",
            "DFLASH_MMQ_SUB_BATCH",
            "GGML_CUDA_BATCH_PEER_COPIES",
        ):
            self.assertNotIn(name, ar)


if __name__ == "__main__":
    unittest.main()
