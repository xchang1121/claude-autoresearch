"""Remote/local worker diagnostic classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    result: str
    suggest: str = ""


def _ssh_suggest(err: str) -> str:
    low = err.lower()
    if "could not resolve hostname" in low or "name or service not known" in low:
        return "check the ssh alias in ~/.ssh/config"
    if "timed out" in low or "no route to host" in low or "network is unreachable" in low:
        return "check VPN, route, and remote machine reachability"
    if "permission denied" in low or "publickey" in low:
        return "check ssh key and authorized_keys"
    if "host key verification failed" in low:
        return "refresh known_hosts for this host"
    return "run ssh <alias> manually for the raw connection error"


def classify(
    facts: dict,
    port: int,
    *,
    backend: Optional[str] = None,
    dsl: Optional[str] = None,
    for_start: bool = False,
) -> List[Finding]:
    ssh_err = facts.get("_SSH_ERROR")
    if ssh_err:
        return [Finding("fatal", "ssh", str(ssh_err)[:160], _ssh_suggest(str(ssh_err)))]

    backend_n = (backend or "ascend").strip().lower()
    dsl_n = (dsl or "").strip().lower()
    ascendish = backend_n in ("", "ascend")
    cudaish = backend_n == "cuda"
    cpuish = backend_n == "cpu"
    needs_triton = dsl_n.startswith("triton")
    findings: List[Finding] = []

    env_path = facts.get("ENV_PATH") or ""
    env_ok = facts.get("ENV_OK") or ""
    if not env_path:
        findings.append(Finding("info", "env_script", "not configured", "set remote_worker.hosts.<alias>.env_script if needed"))
    elif env_ok == "yes":
        findings.append(Finding("ok", "env_script", env_path))
    else:
        findings.append(Finding("fatal", "env_script", f"missing: {env_path}", "fix config.yaml env_script"))

    torch_npu = facts.get("TORCH_NPU") or ""
    if torch_npu == "ok":
        findings.append(Finding("ok", "torch_npu", "importable"))
    elif ascendish:
        findings.append(Finding("fatal", "torch_npu", torch_npu[:120] or "import failed", "source CANN/torch_npu env"))
    else:
        findings.append(Finding("info", "torch_npu", "not required", f"backend={backend_n}"))

    triton = facts.get("TRITON") or ""
    if triton == "ok":
        findings.append(Finding("ok", "triton", "importable"))
    else:
        sev = "fatal" if needs_triton else "warn"
        findings.append(Finding(sev, "triton", triton[:100] or "import failed", "required only for triton_* DSL"))

    npu_smi = facts.get("NPU_SMI") or ""
    if npu_smi == "ok":
        findings.append(Finding("ok", "npu-smi", "in PATH"))
    elif ascendish:
        findings.append(Finding("fatal", "npu-smi", "not in PATH", "source CANN set_env.sh"))
    else:
        findings.append(Finding("info", "npu-smi", "not required", f"backend={backend_n}"))

    nvidia_smi = facts.get("NVIDIA_SMI") or ""
    if nvidia_smi == "ok":
        findings.append(Finding("ok", "nvidia-smi", "in PATH"))
    elif cudaish:
        findings.append(Finding("fatal", "nvidia-smi", "not in PATH", "check CUDA driver/PATH/env_script"))
    else:
        findings.append(Finding("info", "nvidia-smi", "not required", f"backend={backend_n}"))

    arch = (facts.get("ARCH") or "").strip()
    cuda_arch = (facts.get("CUDA_ARCH") or "").strip()
    cpu_arch = (facts.get("CPU_ARCH") or "").strip()
    if arch:
        findings.append(Finding("ok", "npu arch", arch))
    elif ascendish:
        findings.append(Finding("warn", "npu arch", "not derived", "pass --arch explicitly if probe fails"))
    if cuda_arch:
        name = (facts.get("CUDA_NAME") or "").strip()
        findings.append(Finding("ok", "cuda arch", f"{cuda_arch} ({name})" if name else cuda_arch))
    elif cudaish:
        findings.append(Finding("warn", "cuda arch", "not derived", "pass --arch explicitly if probe fails"))
    if cpu_arch:
        findings.append(Finding("ok", "cpu arch", cpu_arch))
    elif cpuish:
        findings.append(Finding("warn", "cpu arch", "not derived", "pass --arch explicitly if probe fails"))

    npu_count = _to_int(facts.get("DEVICES"))
    if npu_count > 0:
        findings.append(Finding("ok", "npu devices", f"{npu_count} visible"))
    elif ascendish:
        findings.append(Finding("fatal", "npu devices", "0 visible", "check npu-smi info"))

    cuda_count = _to_int(facts.get("CUDA_DEVICES"))
    if cuda_count > 0:
        findings.append(Finding("ok", "cuda devices", f"{cuda_count} visible"))
    elif cudaish:
        findings.append(Finding("fatal", "cuda devices", "0 visible", "check nvidia-smi -L"))

    free_mb = _to_int(facts.get("DISK_FREE_MB"))
    if free_mb >= 500:
        findings.append(Finding("ok", "disk free", f"{free_mb} MB"))
    elif free_mb > 0:
        findings.append(Finding("fatal", "disk free", f"only {free_mb} MB", "clean remote /tmp or disk"))

    port_pid = (facts.get("PORT_PID") or "").strip()
    if not port_pid:
        findings.append(Finding("ok", f"remote :{port}", "free"))
    else:
        sev = "fatal" if for_start else "warn"
        findings.append(Finding(sev, f"remote :{port}", f"held by PID {port_pid}", "stop old daemon or choose another port"))

    return findings


def _to_int(value) -> int:
    try:
        return int(str(value or "0").strip())
    except ValueError:
        return 0


def has_fatal(findings: Iterable[Finding]) -> bool:
    return any(f.severity == "fatal" for f in findings)


def render_findings(findings: Iterable[Finding], log_tail: str = "") -> None:
    rows = list(findings)
    print("\nRemote diagnostics")
    print("------------------")
    for f in rows:
        line = f"[{f.severity.upper():<5}] {f.check:<14} {f.result}"
        if f.suggest:
            line += f"  -> {f.suggest}"
        print(line)
    if log_tail and log_tail.strip() and not log_tail.strip().startswith("(no log"):
        print("\nDaemon log tail:")
        print(log_tail)
