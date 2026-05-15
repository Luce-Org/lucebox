from __future__ import annotations

"""Server-side placement resolution before daemon launch."""

from dataclasses import dataclass
from typing import MutableMapping

from .test_dflash_args import TestDflashLaunchArgs


@dataclass(frozen=True)
class ServerPlacement:
    env_updates: dict[str, str]
    daemon_args: list[str]
    prefix_cache_slots: int
    prefill_cache_slots: int
    target_gpu: int | None
    draft_gpu: int | None
    target_gpus: str | None
    target_layer_split: str | None
    draft_feature_mirror: bool
    peer_access: bool
    cache_slots_disabled: bool = False

    def apply_env(self, env: MutableMapping[str, str]) -> None:
        env.update(self.env_updates)

    def log_lines(self) -> list[str]:
        lines = [
            f"  placement = {'target-gpus' if self.target_gpus else 'single-target'}",
        ]
        if self.target_gpu is not None:
            lines.append(f"    target_gpu = {self.target_gpu}")
        if self.draft_gpu is not None:
            lines.append(f"    draft_gpu  = {self.draft_gpu}")
        if self.target_gpus:
            lines.append(f"    target_gpus = {self.target_gpus}")
            lines.append(f"    layer_split = {self.target_layer_split or '<default>'}")
        if self.draft_feature_mirror:
            lines.append("    draft_feature_mirror = on")
        if self.peer_access:
            lines.append("    peer_access = on")
        return lines


def resolve_server_placement(args) -> ServerPlacement:
    env_updates: dict[str, str] = {}
    if args.target_gpu is not None:
        env_updates["DFLASH_TARGET_GPU"] = str(args.target_gpu)
    if args.draft_gpu is not None:
        env_updates["DFLASH_DRAFT_GPU"] = str(args.draft_gpu)

    daemon_cfg = TestDflashLaunchArgs(
        draft_feature_mirror=args.draft_feature_mirror,
        peer_access=args.peer_access,
    )

    prefix_cache_slots = args.prefix_cache_slots
    prefill_cache_slots = args.prefill_cache_slots
    cache_slots_disabled = False

    if args.target_gpus:
        daemon_cfg = TestDflashLaunchArgs(
            draft_feature_mirror=args.draft_feature_mirror,
            peer_access=args.peer_access,
            target_gpus=args.target_gpus,
            target_layer_split=args.target_layer_split,
            target_split_load_draft=True,
            target_split_dflash=True,
        )
        cache_slots_disabled = prefix_cache_slots > 0 or prefill_cache_slots > 0
        if cache_slots_disabled:
            prefix_cache_slots = 0
            prefill_cache_slots = 0

    return ServerPlacement(
        env_updates=env_updates,
        daemon_args=daemon_cfg.to_cli_args(),
        prefix_cache_slots=prefix_cache_slots,
        prefill_cache_slots=prefill_cache_slots,
        target_gpu=args.target_gpu,
        draft_gpu=args.draft_gpu,
        target_gpus=args.target_gpus,
        target_layer_split=args.target_layer_split,
        draft_feature_mirror=args.draft_feature_mirror,
        peer_access=args.peer_access,
        cache_slots_disabled=cache_slots_disabled,
    )
