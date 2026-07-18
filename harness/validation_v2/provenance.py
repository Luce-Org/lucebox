"""Content-addressed experiment provenance collection and validation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .records import canonical_json, sha256_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.rstrip()


def repository_identity(root: Path) -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain"], root)
    if status:
        raise ValueError("publication provenance refuses a dirty git tree")
    submodules = _run(["git", "submodule", "status", "--recursive"], root)
    submodule_rows = submodules.splitlines() if submodules else []
    invalid_submodules = [row for row in submodule_rows if not row.startswith(" ")]
    if invalid_submodules:
        raise ValueError(
            "publication provenance requires initialized, pinned submodules: "
            + "; ".join(invalid_submodules)
        )
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], root),
        "tree": _run(["git", "rev-parse", "HEAD^{tree}"], root),
        "branch": _run(["git", "branch", "--show-current"], root),
        "submodules": [row.strip() for row in submodule_rows],
        "clean": True,
    }


_SECRET_KEY = re.compile(
    r"(?:api[-_]?key|access[-_]?token|auth[-_]?token|bearer[-_]?token|"
    r"(?:^|[-_])token(?:$|[-_=])|password|passwd|secret|authorization|credential)",
    re.I,
)
_IMMUTABLE_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def redact_command(command: list[str]) -> list[str]:
    """Redact common secret-bearing flags, assignments, and URL credentials."""

    redacted: list[str] = []
    hide_next = False
    for argument in command:
        if hide_next:
            redacted.append("[REDACTED]")
            hide_next = False
            continue
        parsed = urllib.parse.urlsplit(argument)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            hostname = parsed.hostname or ""
            if parsed.port is not None:
                hostname += f":{parsed.port}"
            redacted.append(
                urllib.parse.urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
            )
            continue
        if argument.startswith("-") and "=" in argument:
            name, _ = argument.split("=", 1)
            if _SECRET_KEY.search(name):
                redacted.append(f"{name}=[REDACTED]")
                continue
        if argument.startswith("-") and _SECRET_KEY.search(argument):
            redacted.append(argument)
            hide_next = True
            continue
        if "=" in argument and _SECRET_KEY.search(argument.split("=", 1)[0]):
            redacted.append(f"{argument.split('=', 1)[0]}=[REDACTED]")
            continue
        redacted.append(argument)
    return redacted


def redact_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact_mapping(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    return value


def build_manifest(
    *,
    root: Path,
    files: dict[str, Path],
    immutable_inputs: dict[str, str],
    build: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing manifest files: {missing}")
    mutable = [
        name for name, revision in immutable_inputs.items() if not _IMMUTABLE_SHA.fullmatch(revision)
    ]
    if mutable:
        raise ValueError(f"immutable inputs need commit-like revisions: {mutable}")
    manifest = {
        "schema": "lucebox.validation-v2.provenance/1",
        "repository": repository_identity(root),
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(files.items())
        },
        "immutable_inputs": dict(sorted(immutable_inputs.items())),
        "build": redact_mapping(build),
        "command": redact_command(command),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("manifest_hash", None)
    actual_hash = sha256_json(manifest)
    if expected_hash != actual_hash:
        raise ValueError("manifest content hash mismatch")
    for name, record in manifest["files"].items():
        file_path = Path(record["path"])
        if not file_path.is_file():
            raise ValueError(f"manifest input disappeared: {name}")
        if sha256_file(file_path) != record["sha256"]:
            raise ValueError(f"manifest input hash mismatch: {name}")
    manifest["manifest_hash"] = expected_hash
    return manifest


def redact_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Return only explicitly non-secret execution metadata."""

    source = os.environ if environment is None else environment
    allowed = ("CUDA_VISIBLE_DEVICES", "CMAKE_BUILD_TYPE", "TZ", "LANG")
    return {name: source[name] for name in allowed if name in source}
