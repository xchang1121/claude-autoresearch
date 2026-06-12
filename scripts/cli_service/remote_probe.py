"""One-shot SSH probe for remote worker diagnostics."""
from __future__ import annotations

import shlex
import subprocess
from typing import Optional

from .remote_env import source_env_var_bash


_PROBE_BASH = r'''
env_script=__ENV_SCRIPT__
backend=__BACKEND__
probe_device=__PROBE_DEVICE__
port=__PORT__
log_file=__LOG_FILE__
repo_path=__REPO_PATH__
echo "ENV_PATH:$env_script"
echo "PROBE_BACKEND:$backend"
echo "PROBE_DEVICE:$probe_device"
if [ -n "$env_script" ]; then
  [ -f "$env_script" ] && echo "ENV_OK:yes" || echo "ENV_OK:no"
else
  echo "ENV_OK:"
fi
__ENV_SETUP__
if [ -n "$repo_path" ] && [ -d "$repo_path/scripts" ]; then
  export PYTHONPATH="$repo_path/scripts:${PYTHONPATH:-}"
fi
TORCH_NPU_OUT=$(python -c 'import torch_npu' 2>&1); TORCH_NPU_RC=$?
if [ $TORCH_NPU_RC -eq 0 ]; then echo "TORCH_NPU:ok"; else echo "TORCH_NPU:$(echo "$TORCH_NPU_OUT" | tail -1)"; fi
TRITON_OUT=$(python -c 'import triton' 2>&1); TRITON_RC=$?
if [ $TRITON_RC -eq 0 ]; then echo "TRITON:ok"; else echo "TRITON:$(echo "$TRITON_OUT" | tail -1)"; fi
if [ "$backend" = "cuda" ]; then
  echo "NPU_SMI:not_required"
  echo "NVIDIA_SMI:$(command -v nvidia-smi >/dev/null 2>&1 && echo ok || echo missing)"
  CUDA_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${probe_device:-0}" 2>/dev/null | head -1)
  echo "CUDA_NAME:$CUDA_NAME"
  echo "CUDA_ARCH:$(CUDA_NAME="$CUDA_NAME" python -c 'import os; from eval.arch_normalize import normalize_cuda_arch_name; print(normalize_cuda_arch_name(os.environ.get("CUDA_NAME", "")) or "")' 2>/dev/null)"
  echo "CUDA_DEVICES:$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
  echo "ARCH:"
elif [ "$backend" = "cpu" ]; then
  echo "NPU_SMI:not_required"
  echo "NVIDIA_SMI:not_required"
  echo "CPU_ARCH:$(python -c 'from eval.arch_normalize import normalize_cpu_arch_name; print(normalize_cpu_arch_name() or "")' 2>/dev/null)"
  echo "DEVICES:1"
  echo "ARCH:"
else
  echo "NPU_SMI:$(command -v npu-smi >/dev/null 2>&1 && echo ok || echo missing)"
  echo "NVIDIA_SMI:not_required"
  ASCEND_CHIP=$(npu-smi info 2>/dev/null | awk -v did="$probe_device" '/^\| +[0-9]+ +[0-9A-Z]/ { if (did == "" || $2 == did) { print $3; exit } }')
  echo "ARCH:$(ARCH_NAME="$ASCEND_CHIP" python -c 'import os; from eval.arch_normalize import normalize_ascend_arch_name; print(normalize_ascend_arch_name(os.environ.get("ARCH_NAME", "")) or "")' 2>/dev/null)"
  echo "DEVICES:$(npu-smi info 2>/dev/null | grep -cE '^\| +[0-9]+ +[0-9A-Z]')"
fi
if command -v lsof >/dev/null 2>&1; then
  echo "PORT_PID:$(lsof -ti :$port -sTCP:LISTEN 2>/dev/null | head -1)"
elif command -v ss >/dev/null 2>&1; then
  echo "PORT_PID:$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)"
else
  echo "PORT_PID:"
fi
echo "DISK_FREE_MB:$(df -kP /tmp / 2>/dev/null | awk 'NR>1 {print int($4/1024)}' | sort -n | head -1)"
echo "LOG_TAIL_BEGIN"
[ -f "$log_file" ] && tail -20 "$log_file" || echo "(no log: $log_file)"
'''


def _first_device_id(devices: Optional[list[int]]) -> Optional[int]:
    if not devices:
        return None
    return int(devices[0])


def probe_remote(
    ssh_alias: str,
    env_script: Optional[str],
    port: int,
    log_file: Optional[str] = None,
    repo_path: Optional[str] = None,
    devices: Optional[list[int]] = None,
    backend: Optional[str] = None,
) -> dict:
    probe_device = _first_device_id(devices)
    backend_n = (backend or "ascend").strip().lower()
    log = log_file or f"/tmp/akg_worker_{port}.log"
    bash = _PROBE_BASH
    replacements = {
        "__ENV_SCRIPT__": shlex.quote(env_script or ""),
        "__BACKEND__": shlex.quote(backend_n),
        "__PROBE_DEVICE__": shlex.quote("" if probe_device is None else str(probe_device)),
        "__PORT__": str(int(port)),
        "__LOG_FILE__": shlex.quote(log),
        "__REPO_PATH__": shlex.quote(repo_path or ""),
        "__ENV_SETUP__": source_env_var_bash("env_script"),
    }
    for key, value in replacements.items():
        bash = bash.replace(key, value)

    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_alias, f"bash -lc {shlex.quote(bash)}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=35,
        )
    except subprocess.TimeoutExpired:
        return {"_SSH_ERROR": "ssh probe timed out after 35s"}
    except Exception as exc:
        return {"_SSH_ERROR": str(exc)[:200]}
    if out.returncode != 0:
        return {"_SSH_ERROR": (out.stderr or out.stdout or f"ssh rc={out.returncode}")[:400]}

    facts: dict[str, str] = {}
    log_lines: list[str] = []
    in_log = False
    for line in out.stdout.splitlines():
        if line == "LOG_TAIL_BEGIN":
            in_log = True
            continue
        if in_log:
            log_lines.append(line)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            facts[key] = value.strip()
    facts["LOG_TAIL"] = "\n".join(log_lines)
    return facts
