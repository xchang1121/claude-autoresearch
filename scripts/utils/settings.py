"""Shared accessors for config.yaml.

config.yaml is the SINGLE SOURCE OF TRUTH for the framework reference
tables and tunable knobs below. This module reads it once per process and
exposes typed accessors. There are NO in-code defaults: a missing section
or key is a hard error, because config.yaml ships with every key present.
Retune by editing config.yaml — never by editing values here.
"""
from functools import lru_cache
import os
from typing import Dict

import yaml

# __file__ now lives in scripts/utils/; climb two levels to reach autoresearch/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTORESEARCH_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CONFIG_PATH = os.path.join(_AUTORESEARCH_DIR, "config.yaml")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    return data


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Load config.yaml once. Missing file is a hard error — the framework
    ships with one and every accessor below depends on it."""
    return _load_yaml(_CONFIG_PATH)


def _get(section: str, key: str):
    """Read config.yaml[section][key]. config.yaml is the single source of
    truth, so a missing section/key raises — there is no in-code default."""
    sect = _raw().get(section)
    if not isinstance(sect, dict) or key not in sect:
        raise KeyError(
            f"{_CONFIG_PATH}: missing required key '{section}.{key}'")
    return sect[key]


def hallucinated_scripts() -> Dict[str, str]:
    return dict(_raw().get("hallucinated_scripts", {}))


# --- task defaults -----------------------------------------------------
def default_max_rounds() -> int:
    """Default optimization-round budget when a task doesn't specify one.
    Single source for scaffold (new task.yaml) and loader (TaskConfig
    fallback) so the two cannot drift."""
    return _get("defaults", "max_rounds")


def default_eval_timeout() -> int:
    """Per-shape verify/profile budget (seconds) when a task omits it."""
    return _get("defaults", "eval_timeout")


def default_smoke_test_timeout() -> int:
    """quick_check smoke-test budget (seconds) when a task omits it."""
    return _get("defaults", "smoke_test_timeout")


def default_code_checker_enabled() -> bool:
    """Whether the triton-impl AST regression check runs by default."""
    return bool(_get("defaults", "code_checker_enabled"))


def default_metric() -> dict:
    """Primary-metric defaults (primary / lower_is_better /
    improvement_threshold). scaffold writes these into a new task.yaml;
    loader falls back to them when the task.yaml omits the metric block."""
    m = _get("defaults", "metric")
    if not isinstance(m, dict):
        raise ValueError(f"{_CONFIG_PATH}: 'defaults.metric' must be a mapping")
    return m


# --- eval timing measurement (read where the timing runs: on remote eval
#     that is the WORKER's config.yaml) ----------------------------------
def eval_warmup() -> int:
    return _get("eval", "warmup")


def eval_repeats() -> int:
    return _get("eval", "repeats")


# --- remote worker -----------------------------------------------------
def worker_port() -> int:
    """Worker TCP port. Single source for ar_cli (tunnel/status) and
    worker.server (bind) so the two cannot drift."""
    return _get("worker", "port")


def worker_ready_timeout() -> float:
    """Seconds ar_cli waits for a freshly started daemon to answer /status."""
    return float(_get("worker", "ready_timeout"))


def worker_ready_poll_interval() -> float:
    """Seconds between readiness polls while waiting for daemon startup."""
    return float(_get("worker", "ready_poll_interval"))


def worker_ready_probe_timeout() -> float:
    """Per-poll /status probe timeout during the readiness loop (short)."""
    return float(_get("worker", "ready_probe_timeout"))


def worker_status_timeout() -> float:
    """Seconds for a single /status reachability probe."""
    return float(_get("worker", "status_timeout"))


# --- batch pre-flight verification timeouts (seconds) ------------------
def batch_tier1_timeout() -> int:
    return _get("batch", "tier1_timeout")


def batch_tier2_timeout() -> int:
    return _get("batch", "tier2_timeout")


# --- batch driver knobs (overridable via run.py CLI flags) -------------
def batch_run_timeout_min() -> int:
    """Hard wall-clock cap per op in minutes (batch/run.py --timeout-min)."""
    return _get("batch", "run_timeout_min")


def batch_cooldown_sec() -> int:
    """Seconds to sleep between ops (batch/run.py --cooldown-sec)."""
    return _get("batch", "cooldown_sec")


# --- resume heartbeat freshness window (seconds) ----------------------
def heartbeat_fresh_seconds() -> int:
    return _get("resume", "heartbeat_fresh_seconds")


# --- speedup classification thresholds (x vs ref) ---------------------
def speedup_improved_above() -> float:
    return _get("metrics", "speedup_improved_above")


def speedup_regress_below() -> float:
    return _get("metrics", "speedup_regress_below")


def classify_speedup(v: float) -> str:
    """'improved' / 'on-par' / 'regress' per the configured thresholds.
    Single owner for the batch reporters (summarize, monitor)."""
    if v > speedup_improved_above():
        return "improved"
    if v < speedup_regress_below():
        return "regress"
    return "on-par"
