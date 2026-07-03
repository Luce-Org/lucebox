"""Daemon IPC for tool KV thin snapshots."""
from __future__ import annotations

import asyncio


async def snapshot_thin(
    *,
    daemon_stdin,
    await_reply,
    slot: int,
    kv_start: int,
    kv_end: int,
) -> bool:
    """Capture KV range ``[kv_start, kv_end)`` into a thin snapshot slot."""
    line = f"SNAPSHOT_THIN {slot} {kv_start} {kv_end}\n"
    daemon_stdin.write(line.encode("utf-8"))
    daemon_stdin.flush()
    try:
        reply = await await_reply("[snap] thin slot=", timeout=30.0)
    except (asyncio.TimeoutError, EOFError) as exc:
        print(f"[tool-split] SNAPSHOT_THIN slot={slot} failed: {exc}", flush=True)
        return False
    ok = reply.startswith(f"[snap] thin slot={slot} kv={kv_start},{kv_end}")
    if not ok:
        print(f"[tool-split] SNAPSHOT_THIN unexpected reply: {reply!r}", flush=True)
    return ok


async def commit_pending_tool_snap(
    *,
    orchestrator,
    daemon_stdin,
    await_reply,
    fingerprint: str,
    slot: int,
    kv_end: int,
) -> None:
    if kv_end <= 0:
        return
    ok = await snapshot_thin(
        daemon_stdin=daemon_stdin,
        await_reply=await_reply,
        slot=slot,
        kv_start=0,
        kv_end=kv_end,
    )
    if ok:
        orchestrator.tool_slots.confirm(fingerprint, slot)
        print(
            f"[tool-split] tool KV pinned slot={slot} len={kv_end} "
            f"fp={fingerprint[:12]}…",
            flush=True,
        )
