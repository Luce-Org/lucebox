from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "formal_plan.py"
REGISTRY = ROOT / "formal" / "contracts" / "registry.toml"

spec = importlib.util.spec_from_file_location("formal_plan", SCRIPT)
assert spec and spec.loader
formal_plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal_plan)


class FormalPlanTest(unittest.TestCase):
    def plan_fixture(self, name: str) -> dict:
        fixture = json.loads(
            (ROOT / "formal" / "contracts" / "fixtures" / name).read_text()
        )
        return formal_plan.make_plan(REGISTRY, ROOT, fixture["changed_paths"])

    def test_registry_is_valid(self) -> None:
        registry = formal_plan.load_registry(REGISTRY, ROOT)
        self.assertEqual(registry["schema_version"], 1)
        self.assertEqual(
            [target["id"] for target in registry["targets"]],
            [
                "prefix-cache-inline",
                "prefix-cache-abort-hole",
                "prefix-cache-full-lifecycle",
                "spec-commit-exactness",
                "kvflash-residency-map",
            ],
        )
        prefix_area = next(
            area
            for area in registry["critical_paths"]
            if area["id"] == "prefix-cache"
        )
        self.assertEqual(prefix_area["policy"], "advisory")
        self.assertEqual(prefix_area["include_roots"], ["server/src"])
        self.assertIn(
            "server/src/server/*eviction*.h",
            prefix_area["watch_paths"],
        )
        kvflash_target = next(
            target
            for target in registry["targets"]
            if target["id"] == "kvflash-residency-map"
        )
        self.assertEqual(kvflash_target["policy"], "advisory")
        self.assertEqual(
            kvflash_target["pr_defines"],
            ["LUCEBOX_FORMAL_BLOCKS=4"],
        )
        self.assertEqual(
            kvflash_target["nightly_defines"],
            ["LUCEBOX_FORMAL_BLOCKS=5"],
        )
        self.assertEqual(kvflash_target["timeout_seconds"], 180)

    def test_registry_and_legacy_manifest_pin_the_same_toolchain(self) -> None:
        registry = formal_plan.load_registry(REGISTRY, ROOT)
        manifest_path = registry["registry"]["compatibility_manifest"]
        manifest = tomllib.loads((ROOT / manifest_path).read_text())
        self.assertEqual(registry["toolchain"], manifest["toolchain"])

    def test_workflow_image_pins_match_registry_toolchain(self) -> None:
        toolchain = formal_plan.load_registry(REGISTRY, ROOT)["toolchain"]
        workflows = {
            ".github/workflows/formal.yml": {
                "VERIFIER_IMAGE": (toolchain["verifier_image"], 3),
                "REPAIR_IMAGE": (toolchain["repair_image"], 1),
            },
            ".github/workflows/formal-nightly.yml": {
                "VERIFIER_IMAGE": (toolchain["verifier_image"], 3),
                "REPAIR_IMAGE": (toolchain["repair_image"], 1),
            },
            ".github/workflows/formal-ai.yml": {
                "REPAIR_IMAGE": (toolchain["repair_image"], 4),
            },
        }
        assignment_pattern = re.compile(
            r"^\s*(VERIFIER_IMAGE|REPAIR_IMAGE):\s*(\S+)\s*$",
            re.MULTILINE,
        )
        for relative, expected in workflows.items():
            assignments = assignment_pattern.findall((ROOT / relative).read_text())
            self.assertEqual(
                {name for name, _ in assignments},
                set(expected),
                relative,
            )
            for name, (expected_value, expected_count) in expected.items():
                values = [value for actual_name, value in assignments if actual_name == name]
                self.assertEqual(len(values), expected_count, f"{relative}: {name}")
                self.assertEqual(
                    values,
                    [expected_value] * expected_count,
                    f"{relative}: {name}",
                )

    def test_prefix_cache_fixture_selects_approved_targets(self) -> None:
        plan = self.plan_fixture("prefix-cache-change.json")
        self.assertEqual(
            [target["id"] for target in plan["targets"]],
            [
                "prefix-cache-inline",
                "prefix-cache-abort-hole",
                "prefix-cache-full-lifecycle",
            ],
        )
        self.assertEqual(plan["coverage_gaps"], [])

    def test_spec_commit_fixture_selects_exactness_target(self) -> None:
        plan = self.plan_fixture("spec-commit-change.json")
        self.assertEqual(
            [target["id"] for target in plan["targets"]],
            ["spec-commit-exactness"],
        )
        self.assertEqual(plan["coverage_gaps"], [])

    def test_kvflash_fixture_selects_residency_target(self) -> None:
        plan = self.plan_fixture("kvflash-change.json")
        self.assertEqual(
            [target["id"] for target in plan["targets"]],
            ["kvflash-residency-map"],
        )
        self.assertEqual(plan["coverage_gaps"], [])

    def test_registry_execution_matches_legacy_capsules_during_dual_run(self) -> None:
        registry = formal_plan.load_registry(REGISTRY, ROOT)
        manifest = tomllib.loads((ROOT / "formal" / "manifest.toml").read_text())
        legacy = {capsule["id"]: capsule for capsule in manifest["capsules"]}
        for target in registry["targets"]:
            capsule = legacy[target["id"]]
            self.assertEqual(target["description"], capsule["description"])
            self.assertEqual(target["entry_function"], capsule["entry_function"])
            self.assertEqual(target["include_dirs"], capsule["include_dirs"])
            self.assertEqual(target["timeout_seconds"], capsule["timeout_seconds"])
            self.assertEqual(
                target["nightly_timeout_seconds"],
                capsule.get("nightly_timeout_seconds", capsule["timeout_seconds"]),
            )
            self.assertEqual(target["pr_defines"], capsule["defines"])
            self.assertEqual(target["nightly_defines"], capsule["nightly_defines"])
            self.assertEqual(target["pr_esbmc_args"], capsule["esbmc_args"])
            self.assertEqual(target["nightly_esbmc_args"], capsule["esbmc_args"])
            self.assertEqual(target["mutable_paths"], capsule["mutable_paths"])
            self.assertEqual(target.get("native_test"), capsule.get("native_test"))
            self.assertEqual(
                target.get("native_test_source"),
                capsule.get("native_test_source"),
            )

    def test_transitive_formal_template_includes_are_protected(self) -> None:
        registry = formal_plan.load_registry(REGISTRY, ROOT)
        expected_bodies = {
            "prefix-cache-full-lifecycle": (
                "formal/prefix_cache/full_lifecycle_harness_body.h"
            ),
            "spec-commit-exactness": (
                "formal/spec_commit/spec_commit_harness_body.h"
            ),
            "kvflash-residency-map": (
                "formal/kvflash/residency_map_harness_body.h"
            ),
        }
        for target in registry["targets"]:
            template = (ROOT / target["template"]).read_text(encoding="utf-8")
            resolved_formal_paths = set()
            for included in re.findall(r'#include\s+"([^"]+)"', template):
                candidates = [included]
                candidates.extend(
                    str(Path(include_dir) / included)
                    for include_dir in target["include_dirs"]
                )
                resolved_formal_paths.update(
                    path
                    for path in candidates
                    if path.startswith("formal/") and (ROOT / path).is_file()
                )

            if "formal" in target["include_dirs"]:
                self.assertTrue(resolved_formal_paths, target["id"])
            if target["id"] in expected_bodies:
                self.assertIn(
                    expected_bodies[target["id"]],
                    resolved_formal_paths,
                    target["id"],
                )
            for formal_path in resolved_formal_paths:
                self.assertIn(
                    formal_path,
                    target["contract_paths"],
                    target["id"],
                )
                self.assertIn(
                    formal_path,
                    target["trigger_paths"],
                    target["id"],
                )

    def test_uncovered_critical_path_is_advisory_gap(self) -> None:
        plan = self.plan_fixture("uncovered-streaming-change.json")
        self.assertEqual(plan["targets"], [])
        self.assertEqual(plan["coverage_gaps"][0]["id"], "streaming-lifecycle")
        self.assertEqual(plan["coverage_gaps"][0]["policy"], "advisory")

    def test_emit_copies_approved_template_and_records_hash(self) -> None:
        plan = self.plan_fixture("prefix-cache-change.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            formal_plan._emit_templates(plan, ROOT, output)
            emitted = plan["generated_harnesses"]
            self.assertEqual(len(emitted), 3)
            for item in emitted:
                self.assertTrue((output / item["path"]).is_file())
                self.assertEqual(len(item["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
