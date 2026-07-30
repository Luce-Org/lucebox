#!/usr/bin/env python3
"""Plan approved minimal-core formal contracts for a Lucebox change.

This intentionally small, dependency-free tool is a Luce-side fixture
validator for the per-PR formal planner.  It does not invent contracts: it
selects only targets declared in a supplied registry, renders their approved
templates deterministically, and reports edits to critical paths that have no
approved target as advisory gaps.

CI integration and its base-revision blob loading are owned by the companion
planner.  This script is intentionally not the authoritative CI implementation.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
POLICIES = {"required", "advisory"}
TARGET_REQUIRED_FIELDS = {
    "id",
    "policy",
    "description",
    "source_paths",
    "trigger_paths",
    "symbol",
    "signature",
    "template",
    "entry_function",
    "include_dirs",
    "timeout_seconds",
    "pr_defines",
    "nightly_defines",
    "pr_esbmc_args",
    "nightly_esbmc_args",
    "mutable_paths",
    "contract_paths",
}


class RegistryError(ValueError):
    """A registry is not safe or complete enough for deterministic planning."""


def _repo_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{field} must be a non-empty repository path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise RegistryError(f"{field} must remain inside the repository: {value}")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RegistryError(f"{field} must be a list of strings")
    return list(value)


def _repo_paths(value: object, field: str) -> list[str]:
    return [_repo_path(item, field) for item in _string_list(value, field)]


def _template_sha256(root: Path, template: str) -> str:
    source = root / template
    if not source.is_file():
        raise RegistryError(f"template does not exist: {template}")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def load_registry(registry_path: Path, root: Path) -> dict[str, Any]:
    data = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError("unsupported registry schema_version")
    if not isinstance(data.get("registry"), dict):
        raise RegistryError("registry table is required")
    compatibility_manifest = data["registry"].get("compatibility_manifest")
    if compatibility_manifest is not None:
        data["registry"]["compatibility_manifest"] = _repo_path(
            compatibility_manifest,
            "registry.compatibility_manifest",
        )

    critical_paths = data.get("critical_paths")
    if not isinstance(critical_paths, list) or not critical_paths:
        raise RegistryError("at least one critical_paths table is required")
    critical_ids: set[str] = set()
    for area in critical_paths:
        if not isinstance(area, dict):
            raise RegistryError("critical_paths entries must be tables")
        area_id = area.get("id")
        if not isinstance(area_id, str) or not area_id:
            raise RegistryError("critical_paths.id must be a string")
        if area_id in critical_ids:
            raise RegistryError(f"duplicate critical path area: {area_id}")
        critical_ids.add(area_id)
        if not isinstance(area.get("description"), str):
            raise RegistryError(f"{area_id}: critical-path description is required")
        area["paths"] = _repo_paths(area.get("paths"), f"{area_id}.paths")
        area["watch_paths"] = _repo_paths(
            area.get("watch_paths", []),
            f"{area_id}.watch_paths",
        )
        area["include_roots"] = _repo_paths(
            area.get("include_roots", []),
            f"{area_id}.include_roots",
        )
        area_policy = area.get("policy", "advisory")
        if area_policy not in POLICIES:
            raise RegistryError(
                f"{area_id}: unsupported critical-path policy {area_policy!r}"
            )
        area["policy"] = area_policy
    if not isinstance(data.get("toolchain"), dict):
        raise RegistryError("toolchain table is required")
    if not isinstance(data["toolchain"].get("esbmc_version"), str):
        raise RegistryError("toolchain.esbmc_version must be a string")

    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RegistryError("at least one targets table is required")
    target_ids: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            raise RegistryError("targets entries must be tables")
        missing = TARGET_REQUIRED_FIELDS - target.keys()
        if missing:
            raise RegistryError(
                f"target missing required fields: {', '.join(sorted(missing))}"
            )
        target_id = target["id"]
        if not isinstance(target_id, str) or not target_id:
            raise RegistryError("target.id must be a non-empty string")
        if target_id in target_ids:
            raise RegistryError(f"duplicate target id: {target_id}")
        target_ids.add(target_id)
        if target["policy"] not in POLICIES:
            raise RegistryError(f"{target_id}: unsupported policy {target['policy']!r}")
        for field in ("description", "symbol", "signature", "entry_function"):
            if not isinstance(target[field], str) or not target[field]:
                raise RegistryError(f"{target_id}: {field} must be a non-empty string")
        if not isinstance(target["timeout_seconds"], int) or not 0 < target["timeout_seconds"] <= 3600:
            raise RegistryError(f"{target_id}: timeout_seconds must be between 1 and 3600")
        nightly_timeout = target.setdefault(
            "nightly_timeout_seconds", target["timeout_seconds"]
        )
        if (
            not isinstance(nightly_timeout, int)
            or nightly_timeout < target["timeout_seconds"]
            or nightly_timeout > 3600
        ):
            raise RegistryError(
                f"{target_id}: nightly_timeout_seconds must be between timeout_seconds and 3600"
            )
        for field in (
            "source_paths",
            "trigger_paths",
            "include_dirs",
            "mutable_paths",
            "contract_paths",
        ):
            target[field] = _repo_paths(target[field], f"{target_id}.{field}")
        native_test = target.get("native_test")
        native_source = target.get("native_test_source")
        if (native_test is None) != (native_source is None):
            raise RegistryError(
                f"{target_id}: native_test and native_test_source must be declared together"
            )
        if native_test is not None:
            if not isinstance(native_test, str) or not native_test:
                raise RegistryError(f"{target_id}: native_test must be a non-empty string")
            native_source = _repo_path(
                native_source,
                f"{target_id}.native_test_source",
            )
            if native_source not in target["contract_paths"]:
                raise RegistryError(
                    f"{target_id}: contract_paths must include native_test_source"
                )
        target["native_test"] = native_test
        target["native_test_source"] = native_source
        for field in (
            "pr_defines",
            "nightly_defines",
            "pr_esbmc_args",
            "nightly_esbmc_args",
        ):
            target[field] = _string_list(target[field], f"{target_id}.{field}")
        target["template"] = _repo_path(target["template"], f"{target_id}.template")
        variables = target.get("template_variables", {})
        if not isinstance(variables, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in variables.items()
        ):
            raise RegistryError(f"{target_id}: template_variables must be a string map")
        target["template_variables"] = variables
        target["template_sha256"] = _template_sha256(root, target["template"])
        if target["template"] not in target["contract_paths"]:
            raise RegistryError(f"{target_id}: contract_paths must include template")

    return data


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def selected_targets(registry: dict[str, Any], changed_paths: list[str]) -> list[dict[str, Any]]:
    return [
        target
        for target in registry["targets"]
        if any(_matches(path, target["trigger_paths"]) for path in changed_paths)
    ]


def coverage_gaps(registry: dict[str, Any], changed_paths: list[str]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    target_patterns = [
        pattern for target in registry["targets"] for pattern in target["trigger_paths"]
    ]
    for area in registry["critical_paths"]:
        uncovered = [
            path
            for path in changed_paths
            if _matches(path, area["paths"]) and not _matches(path, target_patterns)
        ]
        if uncovered:
            gaps.append(
                {
                    "id": area["id"],
                    "description": area["description"],
                    "policy": area["policy"],
                    "changed_paths": uncovered,
                }
            )
    return gaps


def make_plan(registry_path: Path, root: Path, changed_paths: list[str]) -> dict[str, Any]:
    registry = load_registry(registry_path, root)
    targets = selected_targets(registry, changed_paths)
    gaps = coverage_gaps(registry, changed_paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "registry": registry_path.relative_to(root).as_posix(),
        "changed_paths": changed_paths,
        "targets": [
            {
                key: target[key]
                for key in (
                    "id",
                    "policy",
                    "description",
                    "source_paths",
                    "symbol",
                    "signature",
                    "template",
                    "template_sha256",
                    "template_variables",
                    "entry_function",
                    "include_dirs",
                    "timeout_seconds",
                    "nightly_timeout_seconds",
                    "pr_defines",
                    "nightly_defines",
                    "pr_esbmc_args",
                    "nightly_esbmc_args",
                    "mutable_paths",
                    "contract_paths",
                    "native_test",
                    "native_test_source",
                )
            }
            for target in targets
        ],
        "coverage_gaps": gaps,
    }


def _changed_paths_from_git(root: Path, base_sha: str) -> list[str]:
    process = subprocess.run(
        ["git", "diff", "--name-only", f"{base_sha}...HEAD"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    if process.returncode:
        raise RegistryError(
            f"could not compute changed paths from {base_sha}: {process.stderr.strip()}"
        )
    return [line for line in process.stdout.splitlines() if line]


def _render_template(target: dict[str, Any], source: str) -> str:
    variables = {
        "ID": target["id"],
        "SYMBOL": target["symbol"],
        "SIGNATURE": target["signature"],
        **target["template_variables"],
    }
    for name, value in variables.items():
        source = source.replace("{{" + name + "}}", value)
    if "{{" in source or "}}" in source:
        raise RegistryError(f"{target['id']}: unresolved template token")
    return source


def _emit_templates(plan: dict[str, Any], root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, str]] = []
    for target in plan["targets"]:
        template = root / target["template"]
        destination = output / f"{target['id']}.cpp"
        destination.write_text(
            _render_template(target, template.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        generated.append(
            {
                "id": target["id"],
                "path": destination.name,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    plan["generated_harnesses"] = generated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--registry", type=Path, default=Path("formal/contracts/registry.toml")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    for command in ("plan", "emit"):
        subparser = subparsers.add_parser(command)
        changed = subparser.add_mutually_exclusive_group(required=True)
        changed.add_argument("--changed-path", action="append", default=[])
        changed.add_argument("--base-sha")
        subparser.add_argument("--out", type=Path, required=command == "emit")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    registry = args.registry if args.registry.is_absolute() else root / args.registry
    registry = registry.resolve()
    try:
        if args.command == "validate":
            load_registry(registry, root)
            print(f"valid registry: {registry.relative_to(root)}")
            return 0
        changed_paths = (
            _changed_paths_from_git(root, args.base_sha)
            if args.base_sha
            else args.changed_path
        )
        plan = make_plan(registry, root, changed_paths)
        if args.command == "emit":
            output = args.out if args.out.is_absolute() else root / args.out
            _emit_templates(plan, root, output.resolve())
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except (OSError, RegistryError, tomllib.TOMLDecodeError) as exc:
        print(f"formal-plan error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
