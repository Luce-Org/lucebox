from __future__ import annotations

import importlib.util
import json
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
            ["prefix-cache-inline", "prefix-cache-abort-hole"],
        )
        prefix_area = next(
            area
            for area in registry["critical_paths"]
            if area["id"] == "prefix-cache"
        )
        self.assertEqual(prefix_area["policy"], "advisory")
        self.assertEqual(prefix_area["include_roots"], ["server/src"])
        self.assertIn(
            "server/src/server/*eviction*",
            prefix_area["watch_paths"],
        )

    def test_registry_and_legacy_manifest_pin_the_same_toolchain(self) -> None:
        registry = formal_plan.load_registry(REGISTRY, ROOT)
        manifest = tomllib.loads((ROOT / "formal" / "manifest.toml").read_text())
        self.assertEqual(registry["toolchain"], manifest["toolchain"])

    def test_prefix_cache_fixture_selects_approved_targets(self) -> None:
        plan = self.plan_fixture("prefix-cache-change.json")
        self.assertEqual(
            [target["id"] for target in plan["targets"]],
            ["prefix-cache-inline", "prefix-cache-abort-hole"],
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

    def test_uncovered_critical_path_is_advisory_gap(self) -> None:
        plan = self.plan_fixture("uncovered-streaming-change.json")
        self.assertEqual(plan["targets"], [])
        self.assertEqual(plan["coverage_gaps"][0]["id"], "streaming-lifecycle")
        self.assertEqual(plan["coverage_gaps"][0]["policy"], "advisory")

    def test_watch_pattern_routes_new_eviction_file_without_claiming_coverage(self) -> None:
        plan = formal_plan.make_plan(
            REGISTRY,
            ROOT,
            ["server/src/server/cache_eviction_policy.h"],
        )
        self.assertEqual(plan["targets"], [])
        self.assertEqual(plan["coverage_gaps"][0]["id"], "prefix-cache")
        self.assertEqual(plan["coverage_gaps"][0]["policy"], "advisory")

    def test_emit_copies_approved_template_and_records_hash(self) -> None:
        plan = self.plan_fixture("prefix-cache-change.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            formal_plan._emit_templates(plan, ROOT, output)
            emitted = plan["generated_harnesses"]
            self.assertEqual(len(emitted), 2)
            for item in emitted:
                self.assertTrue((output / item["path"]).is_file())
                self.assertEqual(len(item["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
