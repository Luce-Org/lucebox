from __future__ import annotations

"""Shared backend/device placement plumbing for script-side test_dflash launchers."""

import os


def resolve_visible_devices(visible_devices: str | None,
                            fallback_device: int | None = None) -> str | None:
    if visible_devices:
        return visible_devices
    if fallback_device is None:
        return None
    return str(fallback_device)


def apply_backend_visible_devices(backend: str,
                                  *,
                                  visible_devices: str | None = None,
                                  fallback_device: int | None = None,
                                  base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    resolved = resolve_visible_devices(visible_devices, fallback_device)
    if resolved is None:
        return env
    if backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = resolved
        return env
    if backend == "hip":
        env["HIP_VISIBLE_DEVICES"] = resolved
        env["ROCR_VISIBLE_DEVICES"] = resolved
        return env
    raise ValueError(f"unknown backend: {backend!r}")
