# Standalone component GPU-window review

**PASS** for the narrow `--component-only` mode at local/deployed harness SHA256 `152af330bb37f77f8e6f9c795ae46c12f787e8587c42fc210aeb671b205296d5`. Reviewed `/Users/marcelorm/workspace/lucebox/artifacts/ds4v-step2/hip-qualification.sh` and the matching `/tmp/ds4v-hip-qualification.sh` on soulf. No harness/probe/GPU execution, source edit, service action or converter action occurred.

The mode skips only the completed-text-proof prerequisite for an explicitly released standalone component window. It still requires the explicit release argument, fresh evidence, pinned runtime/binary/libraries/mmproj/fixtures and actual HIP device identity. Its summary records `component_only=true`, no text proof, and `standalone component; no chat/server acceptance`. The normal text-proof mode still validates its completed proof. Fixed comparisons, exit3 preservation, sequential hip0 probes and owned-child cleanup remain intact.

`idle_window()` fails closed on an active operator/nonzero MainPID, either TCP listener8016/8217, any KFD compute process, unavailable sysfs data, less than8GiB host available memory, missing/ambiguous discrete card, or less than8GiB free discrete VRAM. It executes once before evidence creation and again immediately before the first GPU probe, after provenance hashing. These are readiness snapshots within the parent's explicitly released window, not an interprocess GPU reservation.

CPU-only validation used the existing immutable reference venv with `python -I` under the exact clean HOME/PATH/locale/two-thread environment. Bash syntax and complete embedded Python AST parsing passed. Only AST-extracted `check` and `idle_window` function definitions were executed; no other harness statements or imports of model/GPU libraries ran.

Both independent idle-window invocations returned:

```text
ActiveState=inactive
MainPID=0
8016/8217 TCP listeners absent
KFD process directory empty
host_available_bytes=36079882240
discrete_free_vram_bytes=21430087680
```

The user-service query works in that clean environment with only `XDG_RUNTIME_DIR=/run/user/<uid>` added; no inherited DBus variable was needed. The `card[0-9]*/device` glob also visits DRM connector entries, but their nested `device` is a directory, so the `is_file()` predicate correctly excludes them. Exactly `/sys/class/drm/card0/device` has PCI ID `0x744c`; card1 is `0x1586`. The discrete card resolves to PCI `0000:c6:00.0`. No duplicate match or DBus issue was found.

This approves the prepared component-window gate under the parent's revised ordering. It claims no numerical/HIP result or full-chat acceptance, and does not authorize driving the operator service.
