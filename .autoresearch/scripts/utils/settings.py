"""Shared config loader for framework defaults.

Every framework-level knob lives in `.autoresearch/config.yaml`:
default DSL fallback, profiler iteration counts, autotune behaviour,
precision-tolerance table, worker daemon defaults, batch-verify
timeouts, hallucinated-script aliases. This module reads that file
once per process and exposes small typed accessors — callers never
hand-build these tables inside Python modules.

Every getter has a hardcoded fallback (the value below mirrors the
documented default in config.yaml). A user can delete sections of
config.yaml and the framework still runs.
"""
from functools import lru_cache
import os
import re
from typing import Any, Dict, Optional, Tuple

import yaml

# __file__ now lives in scripts/utils/; climb two levels to reach .autoresearch/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTORESEARCH_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CONFIG_PATH = os.path.join(_AUTORESEARCH_DIR, "config.yaml")
_CODE_CHECKER_PATH = os.path.join(_AUTORESEARCH_DIR, "code_checker.yaml")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    return data


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Load config.yaml once. Missing file is a hard error — the framework
    ships with one and several modules depend on it."""
    return _load_yaml(_CONFIG_PATH)


@lru_cache(maxsize=1)
def _code_checker_raw() -> dict:
    """Load code_checker.yaml once. Missing file / key is a hard error:
    the checker has no fallback defaults, by design."""
    return _load_yaml(_CODE_CHECKER_PATH)


def default_dsl() -> str:
    return str(_raw().get("default_dsl", "triton_ascend"))


def worker_only_modules() -> frozenset:
    return frozenset(_raw().get("worker_only_modules", []))


def hallucinated_scripts() -> Dict[str, str]:
    return dict(_raw().get("hallucinated_scripts", {}))


# ---------------------------------------------------------------------
# profiler — user-facing latency measurement defaults
# ---------------------------------------------------------------------

_PROFILER_DEFAULTS = {"warmup_times": 10, "run_times": 100, "eval_timeout": 600}


def profiler_defaults() -> Dict[str, int]:
    """`warmup_times` / `run_times` / `eval_timeout` — used by `TaskConfig`
    when task.yaml leaves the corresponding field unset, and embedded in
    the generated eval-package tarball.
    """
    block = _raw().get("profiler") or {}
    return {k: int(block.get(k, v)) for k, v in _PROFILER_DEFAULTS.items()}


# ---------------------------------------------------------------------
# autotune — patched triton autotuner benchmark + disk cache
# ---------------------------------------------------------------------

_AUTOTUNE_DEFAULTS: Dict[str, Any] = {
    "benchmark_method": "sync_timer",
    "benchmark_warmup": 1,
    "benchmark_active": 3,
    "benchmark_clear_l2": False,
    "disk_cache_enabled": True,
}


def autotune_settings() -> Dict[str, Any]:
    """Knobs for `patches/triton_autotune_patch.py`. Keys:
      benchmark_method — `"sync_timer"` (default; AscendOpGenAgent-style,
        sub-second per trial) or `"profiler_npu"` (msprof-grade,
        seconds per trial — only worth it when sync_timer misorders
        close-race configs).
      benchmark_warmup / benchmark_active / benchmark_clear_l2 — passed
        into whichever bench method handles a trial. `clear_l2` is
        profiler_npu-only (sync_timer ignores it).
      disk_cache_enabled — when True, autotune winners persist to
        `~/.autoresearch_cache/autotune/<fn>_<src_hash>.json` after
        each `Autotuner.run`, and load back at next invocation so the
        bench loop is skipped entirely on subsequent rounds.
    """
    block = _raw().get("autotune") or {}
    out: Dict[str, Any] = {}
    for k, v in _AUTOTUNE_DEFAULTS.items():
        raw = block.get(k, v)
        if isinstance(v, bool):
            out[k] = bool(raw)
        elif isinstance(v, int):
            out[k] = int(raw)
        else:
            out[k] = raw
    return out


# ---------------------------------------------------------------------
# precision — per-dtype layered tolerance
# ---------------------------------------------------------------------

# Fallbacks mirror akg_agents torch adapter _get_tolerance; if config.yaml
# is missing or its `precision` block is incomplete, correctness.py still
# gates kernels the same way.
_PRECISION_FALLBACK: Dict[str, Tuple[float, float, float, float, float]] = {
    "torch.float32":  (1.22e-4, 1.0e-5, 1.22e-3, 1.0e-4, 0.001),
    "torch.float16":  (9.77e-4, 1.0e-3, 9.77e-3, 1.0e-2, 0.005),
    "torch.bfloat16": (7.81e-3, 1.0e-2, 7.81e-2, 1.0e-1, 0.010),
}

# config.yaml uses bare dtype names (`float32`, not `torch.float32`).
_DTYPE_NAME_MAP = {
    "float32":  "torch.float32",
    "float16":  "torch.float16",
    "bfloat16": "torch.bfloat16",
}


def precision_table() -> Dict[str, Tuple[float, float, float, float, float]]:
    """Per-dtype 5-tuple `(rtol, atol, outlier_rtol, outlier_atol, outlier_ratio)`.

    Keys are `str(torch.dtype)` (e.g. `"torch.float16"`) to match the
    in-process lookup in `utils/correctness.py:_tolerance_for`. Missing
    dtypes fall back to the fp32 row, mirroring the in-process default.
    """
    block = _raw().get("precision") or {}
    out: Dict[str, Tuple[float, float, float, float, float]] = dict(_PRECISION_FALLBACK)
    for name, spec in block.items():
        full = _DTYPE_NAME_MAP.get(str(name).lower(), str(name))
        if not isinstance(spec, dict):
            continue
        try:
            out[full] = (
                float(spec.get("rtol",          _PRECISION_FALLBACK.get(full, _PRECISION_FALLBACK["torch.float32"])[0])),
                float(spec.get("atol",          _PRECISION_FALLBACK.get(full, _PRECISION_FALLBACK["torch.float32"])[1])),
                float(spec.get("outlier_rtol",  _PRECISION_FALLBACK.get(full, _PRECISION_FALLBACK["torch.float32"])[2])),
                float(spec.get("outlier_atol", _PRECISION_FALLBACK.get(full, _PRECISION_FALLBACK["torch.float32"])[3])),
                float(spec.get("outlier_ratio", _PRECISION_FALLBACK.get(full, _PRECISION_FALLBACK["torch.float32"])[4])),
            )
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------
# worker — `ar_cli.py worker --start` defaults
# ---------------------------------------------------------------------

_WORKER_DEFAULTS: Dict[str, Any] = {"port": 9001, "host": "0.0.0.0"}


def worker_defaults() -> Dict[str, Any]:
    block = _raw().get("worker") or {}
    return {
        "port": int(block.get("port", _WORKER_DEFAULTS["port"])),
        "host": str(block.get("host", _WORKER_DEFAULTS["host"])),
    }


# ---------------------------------------------------------------------
# batch_verify — `batch/verify.py` subprocess timeouts
# ---------------------------------------------------------------------

_BATCH_VERIFY_DEFAULTS: Dict[str, int] = {"tier1_timeout": 30, "tier2_timeout": 600}


def batch_verify_timeouts() -> Dict[str, int]:
    block = _raw().get("batch_verify") or {}
    return {k: int(block.get(k, v)) for k, v in _BATCH_VERIFY_DEFAULTS.items()}


# Thin code_checker.yaml accessors. Missing keys → KeyError naturally.
# code_checker.py caches these into module-level constants at its import;
# callers should not invoke these repeatedly.

def code_checker_hard_ops() -> frozenset:
    return frozenset(_code_checker_raw()["hard_ops"])


def code_checker_soft_ops() -> frozenset:
    return frozenset(_code_checker_raw()["soft_ops"])


def code_checker_triton_decorators() -> frozenset:
    return frozenset(_code_checker_raw()["triton_decorators"])


def code_checker_torch_call_prefixes() -> frozenset:
    return frozenset(_code_checker_raw()["torch_call_prefixes"])


def code_checker_kernel_class_name() -> str:
    return _code_checker_raw()["kernel_class_name"]


def code_checker_kernel_forward_method() -> str:
    return _code_checker_raw()["kernel_forward_method"]


def code_checker_triton_module_name() -> str:
    return _code_checker_raw()["triton_module_name"]


def code_checker_dsl_compliance_prefix() -> str:
    return _code_checker_raw()["dsl_compliance_prefix"]


def code_checker_stray_text_re() -> "re.Pattern[str]":
    """Regex matching a run of `min_run` consecutive chars in any unicode_range."""
    cfg = _code_checker_raw()["stray_text"]
    cls = "".join(f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in cfg["unicode_ranges"])
    return re.compile(f"[{cls}]{{{cfg['min_run']},}}")


def code_checker_autotune_re() -> "re.Pattern[str]":
    """Regex matching `@<triton_module>.<decorator_attr>(`."""
    triton_mod = _code_checker_raw()["triton_module_name"]
    deco = _code_checker_raw()["autotune"]["decorator_attr"]
    return re.compile(rf"@{re.escape(triton_mod)}\.{re.escape(deco)}\s*\(", re.MULTILINE)


def code_checker_restore_value_re() -> "re.Pattern[str]":
    """Regex matching the required autotune kwarg assignment."""
    kwarg = _code_checker_raw()["autotune"]["required_kwarg"]
    return re.compile(rf"{re.escape(kwarg)}\s*=")
