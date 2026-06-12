"""Shell snippets used by remote worker dispatch."""
from __future__ import annotations

import shlex
from typing import Optional


_CONDA_HOOK_BASH = r'''
if command -v conda >/dev/null 2>&1; then
  __ar_conda_base="$(conda info --base 2>/dev/null || true)"
  if [ -n "$__ar_conda_base" ] && [ -f "$__ar_conda_base/etc/profile.d/conda.sh" ]; then
    . "$__ar_conda_base/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
  else
    eval "$(conda shell.bash hook 2>/dev/null)" >/dev/null 2>&1 || true
  fi
  unset __ar_conda_base
fi
'''.strip()


def source_env_script_bash(env_script: Optional[str]) -> str:
    parts = [_CONDA_HOOK_BASH]
    if env_script:
        parts.append(f"source {shlex.quote(env_script)}")
    return "\n".join(parts)


def source_env_var_bash(var_name: str) -> str:
    return "\n".join(
        [
            _CONDA_HOOK_BASH,
            f'if [ -n "${var_name}" ] && [ -f "${var_name}" ]; then',
            f'  source "${var_name}" >/dev/null 2>&1',
            "fi",
        ]
    )
