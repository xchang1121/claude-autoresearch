"""Layered tolerance output comparison — aligned with akg_agents torch adapter.

Single source of truth used by:
  - the generated eval script (`package_builder._gen_eval_script`) that
    runs inside the worker / local subprocess
  - (formerly) the batch-time pre-flight verify; now routed through
    `ar_cli.py verify --mode verify-only` to share this gate end-to-end

Semantics (ported from `akg_agents/op/verifier/adapters/framework/torch.py`):

  Per-dtype tolerance table (rtol, atol, outlier_rtol, outlier_atol, outlier_ratio).
  Outlier thresholds = 10× strict; outlier_ratio caps how many elements
  may exceed strict-but-stay-under-relaxed before the tensor fails.

    strict_tol  = atol         + rtol         * |ref|
    relaxed_tol = outlier_atol + outlier_rtol * |ref|

  Per tensor:
    - shape mismatch              → FAIL
    - NaN / Inf position mismatch → FAIL
    - Inf sign mismatch           → FAIL
    - hard_fail = count(|diff| > relaxed_tol);  any hard_fail > 0 → FAIL
    - outlier   = count(|diff| > strict_tol AND |diff| <= relaxed_tol)
                  fail when outlier > total * outlier_ratio
    - bool dtypes require exact equality (no tolerance applies)

  Plus MERE / MARE diagnostics:
    mere = mean(|diff| / (|ref| + atol))
    mare = max(|diff| / (|ref| + atol))

This is STRICTER than `torch.allclose(a, b, atol, rtol)` for the strict
band but allows a small fraction of "outlier" elements within a relaxed
10× band — matching CANN's MARE-threshold = 10× MERE-threshold convention
so claude-autoresearch classifies a kernel the same way akg_agents would.

Dependency-light (only torch) so this module can be bundled into the
eval-package tarball.
"""
from __future__ import annotations

from typing import Any, Optional


# Tolerance table keyed by torch dtype. Mirrors akg-hitl
# `akg_agents/op/verifier/adapters/framework/torch.py:_get_tolerance`.
# (rtol, atol, outlier_rtol, outlier_atol, outlier_ratio)
_TOLERANCE_BY_DTYPE_NAME = {
    "torch.float32":  (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001),
    "torch.float16":  (9.77e-4, 1e-3, 9.77e-3, 1e-2, 0.005),
    "torch.bfloat16": (7.81e-3, 1e-2, 7.81e-2, 1e-1, 0.010),
}
_DEFAULT_TOLERANCE = (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)


def _tolerance_for(dtype: Any) -> tuple[float, float, float, float, float]:
    """Look up `(rtol, atol, outlier_rtol, outlier_atol, outlier_ratio)`
    for a torch dtype. Falls back to fp32-grade tolerance for unknown
    dtypes (int / quantized / etc.) so callers always get a 5-tuple.
    """
    return _TOLERANCE_BY_DTYPE_NAME.get(str(dtype), _DEFAULT_TOLERANCE)


def compare_outputs(out_ref: list, out_new: list) -> dict:
    """Per-tensor layered-tolerance check.

    For each tensor pair:
      - tensors are detached, moved to CPU; the float comparisons cast
        to float32 for stable subtraction math
      - non-tensor outputs fail explicitly (a silent skip used to let
        scalar / numpy / mixed-structure outputs falsely classify as PASS)
      - NaN / Inf position mismatch → fail
      - shape mismatch → fail
      - hard_fail > 0  OR  outlier > total * outlier_ratio → fail
      - bool tensors require exact equality

    Returns:
      {"correctness": bool,
       "diagnostics": list[str],
       "max_abs_diff": float | None,
       "mere": float | None,    # mean |diff|/(|ref|+atol), across all tensors
       "mare": float | None,    # max  |diff|/(|ref|+atol)
       "hard_fail": int,
       "outlier": int,
       "outlier_cap": int}
    """
    import torch  # local import keeps module importable without torch

    diagnostics: list[str] = []
    all_close = True
    max_abs_overall: Optional[float] = None
    mere_acc = 0.0
    mere_count = 0
    mare_overall: Optional[float] = None
    hard_fail_total = 0
    outlier_total = 0
    cap_total = 0

    if len(out_ref) != len(out_new):
        return {
            "correctness": False,
            "diagnostics": [
                f"output count: ref={len(out_ref)} new={len(out_new)}"
            ],
            "max_abs_diff": None,
            "mere": None, "mare": None,
            "hard_fail": 0, "outlier": 0, "outlier_cap": 0,
        }

    for i, (r, n) in enumerate(zip(out_ref, out_new)):
        if not isinstance(r, torch.Tensor) or not isinstance(n, torch.Tensor):
            all_close = False
            diagnostics.append(
                f"out{i}: non-tensor output not supported "
                f"(ref={type(r).__name__}, new={type(n).__name__}); "
                f"both outputs must be torch.Tensor instances"
            )
            continue

        rf_dtype = r.dtype
        rf = r.detach().cpu()
        nf = n.detach().cpu()

        if rf.shape != nf.shape:
            all_close = False
            diagnostics.append(
                f"out{i} shape {tuple(rf.shape)} != kernel {tuple(nf.shape)}"
            )
            continue

        # NaN / Inf position must match. Position mismatch is unambiguous
        # kernel breakage even when subsequent finite-value comparison
        # would still pass (a NaN moved across cells produces wrong data).
        ref_nan = torch.isnan(rf)
        new_nan = torch.isnan(nf)
        if not torch.equal(ref_nan, new_nan):
            all_close = False
            diagnostics.append(
                f"out{i}: NaN position mismatch "
                f"(ref={int(ref_nan.sum())}, new={int(new_nan.sum())})"
            )
            continue

        ref_inf = torch.isinf(rf)
        new_inf = torch.isinf(nf)
        if not torch.equal(ref_inf, new_inf):
            all_close = False
            diagnostics.append(
                f"out{i}: Inf position mismatch "
                f"(ref={int(ref_inf.sum())}, new={int(new_inf.sum())})"
            )
            continue
        if ref_inf.any() and not torch.equal(
                torch.sign(rf[ref_inf]), torch.sign(nf[ref_inf])):
            all_close = False
            diagnostics.append(f"out{i}: Inf sign mismatch")
            continue

        # Bool tensors: no tolerance applies, demand exact equality.
        if rf.dtype == torch.bool:
            if not torch.equal(rf, nf):
                all_close = False
                diff_count = int((rf != nf).sum())
                diagnostics.append(
                    f"out{i}: bool mismatch ({diff_count}/{rf.numel()} cells)"
                )
            else:
                diagnostics.append(f"out{i}: OK (bool, n={rf.numel()})")
            continue

        finite_mask = torch.isfinite(rf) & torch.isfinite(nf)
        finite_count = int(finite_mask.sum())
        if finite_count == 0:
            diagnostics.append(
                f"out{i}: all values Inf/NaN, skipping precision check"
            )
            continue

        rf_finite = rf[finite_mask]
        nf_finite = nf[finite_mask]
        if nf_finite.dtype != rf_finite.dtype:
            nf_finite = nf_finite.to(rf_finite.dtype)

        rtol, atol, out_rtol, out_atol, out_ratio = _tolerance_for(rf_dtype)

        rf_f = rf_finite.float()
        nf_f = nf_finite.float()
        abs_diff = torch.abs(rf_f - nf_f)
        abs_ref = torch.abs(rf_f)
        strict_tol = atol + rtol * abs_ref
        relaxed_tol = out_atol + out_rtol * abs_ref

        max_abs = float(abs_diff.max().item())
        if max_abs_overall is None or max_abs > max_abs_overall:
            max_abs_overall = max_abs

        strict_pass = abs_diff <= strict_tol
        relaxed_pass = abs_diff <= relaxed_tol
        hard_fail = int((~relaxed_pass).sum().item())
        outlier = int(((~strict_pass) & relaxed_pass).sum().item())
        cap = int(rf_f.numel() * out_ratio)

        # MERE / MARE diagnostics (CANN convention: divide by |ref|+atol).
        denom = abs_ref + atol
        per_elem = abs_diff / denom
        mere_acc += float(per_elem.sum().item())
        mere_count += int(per_elem.numel())
        mare = float(per_elem.max().item())
        if mare_overall is None or mare > mare_overall:
            mare_overall = mare

        hard_fail_total += hard_fail
        outlier_total += outlier
        cap_total += cap

        if hard_fail > 0 or outlier > cap:
            all_close = False
            kind = "hard_fail" if hard_fail > 0 else "outlier-over-cap"
            diagnostics.append(
                f"out{i}: {kind} dtype={rf_dtype} total={rf_f.numel()} "
                f"hard_fail={hard_fail} outlier={outlier}/{cap} "
                f"max_abs={max_abs:.3e} mare={mare:.3e} "
                f"(rtol={rtol:.2e} atol={atol:.2e})"
            )
        else:
            diagnostics.append(
                f"out{i}: OK (dtype={rf_dtype}, total={rf_f.numel()}, "
                f"outlier={outlier}/{cap}, max_abs={max_abs:.3e})"
            )

    mere_overall = (mere_acc / mere_count) if mere_count > 0 else None
    return {
        "correctness": all_close,
        "diagnostics": diagnostics,
        "max_abs_diff": max_abs_overall,
        "mere": mere_overall,
        "mare": mare_overall,
        "hard_fail": hard_fail_total,
        "outlier": outlier_total,
        "outlier_cap": cap_total,
    }


def compare_outputs_per_case(out_ref_per_case: list,
                             out_new_per_case: list) -> dict:
    """Multi-shape layered-tolerance check; hard-gate on every case.

    Inputs are `List[List[Tensor]]` — one outer entry per shape case.
    Returns:
      `{"correctness", "per_case", "diagnostics" (flat with `[case i]`
       prefix), "max_abs_diff", "failed_indices", "worst_idx",
       "worst_max_abs_diff"}`.

    On multi-case failure appends one `CORRECTNESS_SUMMARY` line to the
    flat diagnostics for `failure_extractor` to pick up; DIAGNOSE
    consumes it directly.
    """
    if len(out_ref_per_case) != len(out_new_per_case):
        return {
            "correctness": False,
            "per_case": [],
            "diagnostics": [
                f"case count: ref={len(out_ref_per_case)} "
                f"new={len(out_new_per_case)}"
            ],
            "max_abs_diff": None,
            "failed_indices": [],
            "worst_idx": None,
            "worst_max_abs_diff": None,
        }

    per_case = []
    flat_diag: list[str] = []
    all_pass = True
    max_abs_overall: Optional[float] = None

    for i, (out_ref, out_new) in enumerate(zip(out_ref_per_case,
                                               out_new_per_case)):
        sub = compare_outputs(list(out_ref), list(out_new))
        per_case.append({
            "idx": i,
            "correctness": sub["correctness"],
            "diagnostics": sub["diagnostics"],
            "max_abs_diff": sub["max_abs_diff"],
            "mere": sub.get("mere"),
            "mare": sub.get("mare"),
            "hard_fail": sub.get("hard_fail", 0),
            "outlier": sub.get("outlier", 0),
            "outlier_cap": sub.get("outlier_cap", 0),
        })
        if not sub["correctness"]:
            all_pass = False
        for d in sub["diagnostics"]:
            flat_diag.append(f"[case {i}] {d}")
        m = sub["max_abs_diff"]
        if m is not None and (max_abs_overall is None or m > max_abs_overall):
            max_abs_overall = m

    failed_indices = [pc["idx"] for pc in per_case if not pc["correctness"]]
    worst_idx: Optional[int] = None
    worst_max: Optional[float] = None
    if not all_pass:
        candidates = [pc for pc in per_case
                      if not pc["correctness"]
                      and isinstance(pc.get("max_abs_diff"), (int, float))]
        if candidates:
            best = max(candidates, key=lambda x: x["max_abs_diff"])
            worst_idx = best["idx"]
            worst_max = best["max_abs_diff"]
        if len(per_case) > 1:
            flat_diag.append(
                f"[verify] CORRECTNESS_SUMMARY: failed={len(failed_indices)}/"
                f"{len(per_case)} failed_idx={failed_indices} "
                f"worst_case={worst_idx} max_abs="
                f"{(f'{worst_max:.3e}' if worst_max is not None else 'None')}"
            )

    return {
        "correctness": all_pass,
        "per_case": per_case,
        "diagnostics": flat_diag,
        "max_abs_diff": max_abs_overall,
        "failed_indices": failed_indices,
        "worst_idx": worst_idx,
        "worst_max_abs_diff": worst_max,
    }
