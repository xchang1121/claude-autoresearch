"""Convert raw eval transport responses into EvalResult.

This module owns metric semantics: sidecar interpretation, outcome
classification, per-shape aggregation, timing-method mismatch flags, and
verify failure detail. It has no transport, YAML, package, or Progress I/O.
"""
from __future__ import annotations

import math
from typing import Optional

from .metric_policy import EvalOutcome, EvalResult


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and 0 < v < float("inf")


def _avg_us(block: Optional[dict]) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    v = block.get("avg_time_us")
    return float(v) if _finite(v) else None


def _per_shape_us(block: Optional[dict]) -> Optional[list]:
    if not isinstance(block, dict):
        return None
    ps = block.get("per_shape")
    if not isinstance(ps, list) or not ps:
        return None
    return [(s.get("avg_time_us") if isinstance(s, dict) else None) for s in ps]


def _per_shape_methods(block: Optional[dict]) -> Optional[list]:
    """Per-shape `method` strings used to detect mixed timing semantics."""
    if not isinstance(block, dict):
        return None
    ps = block.get("per_shape")
    if not isinstance(ps, list) or not ps:
        return None
    return [(s.get("method") if isinstance(s, dict) else None) for s in ps]


def assemble_eval_result(resp: dict) -> EvalResult:
    """Convert a transport response into an EvalResult.

    `resp` shape (from both worker /run and utils.local_worker.local_eval):
        {"device_id": int, "returncode": int, "log": str,
         "eval_result": {"verify": {...}, "profile_gen": {...},
                          "profile_base": {...}, "ok": bool, "errors": [...]}}
    """
    log = resp.get("log", "")
    eval_result = resp.get("eval_result") or {}

    verify = eval_result.get("verify") or {}
    profile_gen = eval_result.get("profile_gen") or {}
    profile_base = eval_result.get("profile_base") or {}

    verify_ok = bool(verify.get("correctness"))
    error_source = verify.get("error_source")  # "ref" | "kernel" | None

    gen_time = _avg_us(profile_gen)
    base_time = _avg_us(profile_base)
    per_gen = _per_shape_us(profile_gen)
    per_base = _per_shape_us(profile_base)

    crashed_shapes = (
        [i for i, t in enumerate(per_gen) if not _finite(t)]
        if per_gen is not None else []
    )

    # Outcome - only two non-OK paths:
    #   error_source == "ref" -> broken --ref source file. INFRA_FAIL.
    #   anything else failing -> kernel responsibility. KERNEL_FAIL.
    # Pure transport failures set INFRA_FAIL before we reach this assembler.
    if error_source == "ref":
        outcome = EvalOutcome.INFRA_FAIL
    elif verify_ok and not crashed_shapes:
        outcome = EvalOutcome.OK
    else:
        outcome = EvalOutcome.KERNEL_FAIL

    metrics: dict = {}
    if _finite(gen_time):
        metrics["latency_us"] = gen_time
    if _finite(base_time):
        metrics["ref_latency_us"] = base_time
    if _finite(gen_time) and _finite(base_time):
        metrics["speedup_vs_ref"] = base_time / gen_time

    if per_gen is not None:
        metrics["num_cases"] = len(per_gen)
        metrics["per_shape_gen_us"] = per_gen
        gen_methods = _per_shape_methods(profile_gen)
        if gen_methods:
            metrics["per_shape_gen_method"] = gen_methods
            uniq_gen = sorted({m for m in gen_methods if m})
            if uniq_gen:
                metrics["timing_method_gen"] = (
                    uniq_gen[0] if len(uniq_gen) == 1 else "mixed")
        if crashed_shapes:
            metrics["profile_crashed_cases"] = crashed_shapes[:30]
            metrics["profile_crashed_count"] = len(crashed_shapes)
        if per_base is not None and len(per_base) == len(per_gen):
            metrics["per_shape_base_us"] = per_base
            base_methods = _per_shape_methods(profile_base)
            if base_methods:
                metrics["per_shape_base_method"] = base_methods
                uniq_base = sorted({m for m in base_methods if m})
                if uniq_base:
                    metrics["timing_method_base"] = (
                        uniq_base[0] if len(uniq_base) == 1 else "mixed")

            # "sticky" base is a reused profiler measurement, not a
            # different timing method.
            mg = metrics.get("timing_method_gen")
            mb = metrics.get("timing_method_base")
            if (mg and mb and mg != mb
                    and mg != "sticky" and mb != "sticky"):
                metrics["timing_method_mismatch"] = {"gen": mg, "base": mb}

            per_speedup = [
                (b / g) if (_finite(b) and _finite(g)) else None
                for b, g in zip(per_base, per_gen)
            ]
            metrics["per_shape_speedup"] = per_speedup
            bad = [i for i, s in enumerate(per_speedup) if not _finite(s)]
            if bad:
                metrics["per_shape_speedup_bad_cases"] = bad

            # Aggregation contract: latency = arithmetic mean; speedup =
            # geometric mean of per-shape ratios. Single shape collapses to
            # the same value as scalar base/gen.
            valid = [s for s in per_speedup if _finite(s)]
            if valid:
                metrics["speedup_vs_ref"] = math.exp(
                    sum(math.log(s) for s in valid) / len(valid))
                metrics["speedup_aggregation"] = "geomean"

        descs = [s.get("case_desc")
                 for s in (profile_gen.get("per_shape") or [])
                 if isinstance(s, dict)]
        if any(descs):
            metrics["per_shape_descs"] = descs

    # Verify-side failure detail lets DIAGNOSE pinpoint a failed shape
    # without scraping log text.
    if not verify_ok and verify:
        n_cases = verify.get("num_cases")
        if isinstance(n_cases, int) and n_cases >= 1:
            failed_idx = verify.get("failed_indices") or []
            if isinstance(failed_idx, list):
                metrics["correctness_failed_cases"] = failed_idx[:30]
                metrics["correctness_failed_count"] = len(failed_idx)
                metrics["correctness_total_cases"] = n_cases
            worst_idx = verify.get("worst_idx")
            if isinstance(worst_idx, int):
                metrics["correctness_worst_case"] = worst_idx
            worst_max = verify.get("worst_max_abs_diff")
            if isinstance(worst_max, (int, float)):
                metrics["correctness_worst_max_abs"] = worst_max

    if outcome == EvalOutcome.OK:
        error = None
    elif outcome == EvalOutcome.INFRA_FAIL:
        error = (f"reference.py failed: "
                 f"{verify.get('error') or '(no detail)'}")
    elif not eval_result:
        error = (f"kernel exited without producing verify result "
                 f"(rc={resp.get('returncode')})")
    elif not verify_ok:
        error = verify.get("error") or "kernel output != reference"
    else:
        error = (f"kernel crashed during profile on {len(crashed_shapes)} of "
                 f"{len(per_gen)} shapes")

    return EvalResult(
        outcome=outcome,
        metrics=metrics,
        error=error,
        raw_output=log[-4096:],
        error_source=error_source,
    )
