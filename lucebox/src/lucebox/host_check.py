"""Readiness check: aggregate HostFacts (provided by lucebox.sh) with the
docker-daemon checks we can do from inside the container via the mounted
socket. Prints a status report and returns an aggregate severity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from rich.console import Console

from lucebox.types import HostFacts

Severity = Literal["ok", "warn", "fail"]
_SEVERITY_ORDER: dict[Severity, int] = {"ok": 0, "warn": 1, "fail": 2}


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    severity: Severity
    message: str
    hint: str | None = None


def run_checks(host: HostFacts) -> list[CheckResult]:
    return [
        _check_docker(host),
        _check_nvidia_driver(host),
        _check_ctk(host),
        _check_ram(host),
        _check_vram(host),
        _check_systemd(host),
    ]


def _check_docker(host: HostFacts) -> CheckResult:
    if not host.has_docker:
        return CheckResult(
            "docker",
            "fail",
            "docker daemon unreachable",
            "sudo systemctl start docker, or add your user to the 'docker' group",
        )
    return CheckResult("docker", "ok", f"daemon reachable ({host.docker_version})")


def _check_nvidia_driver(host: HostFacts) -> CheckResult:
    if host.gpu_vendor != "nvidia":
        if host.gpu_vendor == "amd":
            return CheckResult(
                "gpu",
                "fail",
                "AMD GPU detected — prebuilt images are NVIDIA-only",
                "Build dflash from source with HIP; see dflash/README.md",
            )
        return CheckResult("gpu", "fail", "no NVIDIA GPU detected")
    if not host.driver_version:
        return CheckResult(
            "driver",
            "warn",
            "nvidia-smi present but NVML query failed (likely driver/library mismatch)",
            "reboot, or reinstall the matching NVIDIA driver",
        )
    if host.driver_major < 525:
        return CheckResult(
            "driver",
            "fail",
            f"driver r{host.driver_major} too old (need r525+ for cuda12)",
            "upgrade the NVIDIA driver",
        )
    return CheckResult("driver", "ok", f"nvidia r{host.driver_major} ({host.driver_version})")


def _check_ctk(host: HostFacts) -> CheckResult:
    match host.ctk:
        case "runtime":
            return CheckResult("ctk", "ok", "NVIDIA Container Toolkit registered as docker runtime")
        case "cdi":
            return CheckResult("ctk", "ok", "NVIDIA Container Toolkit available via CDI")
        case "installed-unwired":
            return CheckResult(
                "ctk",
                "warn",
                "NVIDIA Container Toolkit installed but not wired into docker",
                "sudo nvidia-ctk runtime configure --runtime=docker && "
                "sudo systemctl restart docker",
            )
        case _:
            return CheckResult(
                "ctk",
                "fail",
                "NVIDIA Container Toolkit not installed",
                "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html",
            )


def _check_ram(host: HostFacts) -> CheckResult:
    if host.ram_gb == 0:
        return CheckResult("ram", "warn", "RAM unknown")
    if host.ram_gb < 16:
        return CheckResult("ram", "warn", f"{host.ram_gb} GB RAM — model load may swap")
    return CheckResult("ram", "ok", f"{host.ram_gb} GB RAM")


def _check_vram(host: HostFacts) -> CheckResult:
    if host.vram_gb == 0:
        return CheckResult("vram", "warn", "VRAM unknown")
    if host.vram_gb < 12:
        return CheckResult(
            "vram",
            "fail",
            f"VRAM {host.vram_gb} GB < 12 GB — 27B target won't fit",
            "use a smaller model preset or larger GPU",
        )
    if host.vram_gb < 22:
        return CheckResult(
            "vram",
            "warn",
            f"VRAM {host.vram_gb} GB — 27B fits but max_ctx will be capped near 32K",
        )
    return CheckResult("vram", "ok", f"VRAM {host.vram_gb} GB ({host.gpu_name})")


def _check_systemd(host: HostFacts) -> CheckResult:
    if not host.has_systemd:
        return CheckResult(
            "systemd",
            "warn",
            "user systemd not available",
            "WSL: enable systemd in /etc/wsl.conf; otherwise 'lucebox serve' "
            "still works in the foreground",
        )
    return CheckResult("systemd", "ok", "user systemd available")


def aggregate(results: list[CheckResult]) -> Severity:
    worst: Severity = "ok"
    for r in results:
        if _SEVERITY_ORDER[r.severity] > _SEVERITY_ORDER[worst]:
            worst = r.severity
    return worst


def render(console: Console, host: HostFacts, results: list[CheckResult]) -> Severity:
    """Print a status block, return the worst severity."""
    summary = f"[bold]Host:[/bold] {host.nproc} CPUs · {host.ram_gb} GB RAM"
    if host.gpu_vendor == "nvidia" and host.gpu_name:
        summary += f" · {host.gpu_name} · {host.vram_gb} GB VRAM" + (
            f" (sm_{host.gpu_sm})" if host.gpu_sm else ""
        )
    if host.is_wsl:
        summary += " · WSL2"
    console.print(summary)
    console.print()

    sev_style = {
        "ok": "[green]OK[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
    }
    for r in results:
        console.print(f"  {sev_style[r.severity]:<22} {r.name:<8} {r.message}")
        if r.hint:
            console.print(f"  {'':<22} {'':<8} [dim]{r.hint}[/dim]")

    render_host_facts(console)

    worst = aggregate(results)
    console.print()
    if worst == "ok":
        console.print("[green]All checks passed.[/green]")
    elif worst == "warn":
        console.print("[yellow]Checks passed with warnings.[/yellow]")
    else:
        console.print(
            "[red]Critical checks failed — fix the issues above before 'lucebox start'.[/red]"
        )
    return worst


def render_host_facts(console: Console) -> None:
    """Print a pretty 'Host facts' section sourced from LUCEBOX_HOST_*.

    Same data that ends up in /opt/lucebox-hub/HOST_INFO inside the
    container — printed here so the operator can sanity-check the
    rig classification BEFORE starting a long bench run, and so the
    CI exit-code gate (the pass/fail checks above) stays orthogonal
    to the informational host facts.

    Reads from the same LUCEBOX_HOST_* env the host wrapper exports
    (see lucebox.sh::probe_host). Quiet — emits the section header
    even when most facts are unset, since "no host facts probed at
    all" is itself a useful signal.
    """
    console.print()
    console.print("[bold]Host facts[/bold] (LUCEBOX_HOST_*, surfaced as /props.host)")
    facts = [
        ("os", os.environ.get("LUCEBOX_HOST_OS_PRETTY", "")),
        ("kernel", os.environ.get("LUCEBOX_HOST_KERNEL", "")),
        ("wsl_version", os.environ.get("LUCEBOX_HOST_WSL_VERSION", "")),
        ("docker", os.environ.get("LUCEBOX_HOST_DOCKER_VERSION", "")),
        ("nvidia_driver", os.environ.get("LUCEBOX_HOST_DRIVER_VERSION", "")),
        ("nvidia_ctk", os.environ.get("LUCEBOX_HOST_NVIDIA_CTK_VERSION", "")),
        ("cpu", os.environ.get("LUCEBOX_HOST_CPU_MODEL", "")),
        ("cuda_visible_devices", os.environ.get("LUCEBOX_HOST_CUDA_VISIBLE_DEVICES", "")),
    ]
    for key, value in facts:
        display = value if value else "[dim](unset)[/dim]"
        console.print(f"  {key:<22} {display}")

    # Multi-GPU table — one line per device. LUCEBOX_HOST_GPU_LIST_CSV
    # carries the verbatim nvidia-smi CSV the host wrapper probed.
    csv = os.environ.get("LUCEBOX_HOST_GPU_LIST_CSV", "")
    if csv:
        console.print("  gpus:")
        for line in csv.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [c.strip() for c in line.split(",")]
            if len(parts) >= 7:
                idx, _uuid, _pci, name, sm, mem, plimit = parts[:7]
                console.print(
                    f"    [{idx}] {name} (sm_{sm}, {mem}, {plimit})"
                )
            else:
                console.print(f"    {line}")
    else:
        console.print("  gpus                  [dim](none — nvidia-smi unavailable)[/dim]")
