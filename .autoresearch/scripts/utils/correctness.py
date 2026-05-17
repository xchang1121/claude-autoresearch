"""Shared per-dtype atol/rtol output comparison.

Single source of truth used by:
  - the generated eval script (package_builder._gen_eval_script) that
    runs inside the worker / local subprocess
  - the batch-time pre-flight verify.py Tier 2 check

Tolerances are per-dtype (matches Ascend NPUKernelBench / kernel-verifier
skill conventions):

    float32 / float : rtol = 2^-13 ≈ 1.22e-4,  atol = 1e-5
    float16 / half  : rtol = 2^-10 ≈ 9.77e-4,  atol = 1e-3
    bfloat16        : rtol = 2^-7  ≈ 7.81e-3,  atol = 1e-2

The previous global atol=rtol=1e-2 was at bfloat16 level, which silently
let fp32 numerical bugs (orders of magnitude above the legitimate fp32
threshold) pass as "correct". Now each tensor's tolerance is keyed off
the REF tensor's original dtype before the fp32 promotion used for
stable subtraction.

`compare_outputs` is dependency-light (only torch) so it can be bundled
into the eval-package tarball without dragging the rest of
`.autoresearch/scripts/` along.
"""
from __future__ import annotations

from typing import Any, Optional


# Per-dtype tolerance table. Keys are the canonical torch dtype string
# repr (`str(t.dtype).split(".")[-1]`); add aliases as needed. Anything
# not in the table falls back to fp32 thresholds.
_TOL_BY_DTYPE: dict[str, tuple[float, float]] = {
    # name: (atol, rtol)
    "float32":  (1e-5, 2 ** -13),
    "float":    (1e-5, 2 ** -13),
    "float64":  (1e-6, 2 ** -13),
    "double":   (1e-6, 2 ** -13),
    "float16":  (1e-3, 2 ** -10),
    "half":     (1e-3, 2 ** -10),
    "bfloat16": (1e-2, 2 ** -7),
    # Integer / bool dtypes: exact match (atol=rtol=0). compare_outputs
    # promotes to fp32 for finite-difference math, so float-style
    # tolerance still applies; we keep the table conservative.
    "int8":     (0.0, 0.0),
    "int16":    (0.0, 0.0),
    "int32":    (0.0, 0.0),
    "int64":    (0.0, 0.0),
    "uint8":    (0.0, 0.0),
    "bool":     (0.0, 0.0),
}

_DEFAULT_TOL = _TOL_BY_DTYPE["float32"]


def tolerance_for_dtype(dtype) -> tuple[float, float]:
    """Return (atol, rtol) for a torch dtype or its name."""
    if hasattr(dtype, "__name__"):
        key = dtype.__name__
    else:
        key = str(dtype).split(".")[-1]
    return _TOL_BY_DTYPE.get(key.lower(), _DEFAULT_TOL)


def compare_outputs(out_ref: list, out_new: list) -> dict:
    """Per-dtype allclose comparison.

    For each (ref, new) tensor pair:
      - tolerance is derived from the REF tensor's dtype via
        `tolerance_for_dtype`
      - tensors are cast to fp32 BEFORE subtraction so the difference math
        doesn't itself underflow on half precision; the comparison
        threshold is still keyed off the original dtype
      - NaN positions on either side fail the case (no `equal_nan=True`)

    Returns:
      {"correctness": bool,
       "diagnostics": list[str],
       "max_abs_diff": float | None,
       "tolerances": [{"out": i, "dtype": str, "atol": float, "rtol": float}, ...]}
    """
    import torch

    diagnostics: list[str] = []
    tolerances: list[dict] = []
    all_close = True
    max_abs_overall: Optional[float] = None

    if len(out_ref) != len(out_new):
        return {
            "correctness": False,
            "diagnostics": [
                f"output count: ref={len(out_ref)} new={len(out_new)}"
            ],
            "max_abs_diff": None,
            "tolerances": [],
        }

    for i, (r, n) in enumerate(zip(out_ref, out_new)):
        # Only tensor pairs participate — autoresearch's eval never
        # materializes scalar-only outputs through this path.
        if not (isinstance(r, torch.Tensor) and isinstance(n, torch.Tensor)):
            continue

        ref_dtype = r.dtype
        atol, rtol = tolerance_for_dtype(ref_dtype)
        dtype_name = str(ref_dtype).split(".")[-1]
        tolerances.append({"out": i, "dtype": dtype_name,
                           "atol": atol, "rtol": rtol})

        rf = r.detach().cpu().float()
        nf = n.detach().cpu().float()

        if rf.shape != nf.shape:
            all_close = False
            diagnostics.append(
                f"out{i} ({dtype_name}) shape {tuple(rf.shape)} != "
                f"kernel {tuple(nf.shape)}"
            )
            continue

        abs_diff = (rf - nf).abs()
        max_abs = float(abs_diff.max().item())
        if max_abs_overall is None or max_abs > max_abs_overall:
            max_abs_overall = max_abs

        if not torch.allclose(rf, nf, atol=atol, rtol=rtol):
            all_close = False
            rel_denom = rf.abs().clamp_min(1e-12)
            max_rel = float((abs_diff / rel_denom).max().item())
            n_bad = int((abs_diff > (atol + rtol * rf.abs())).sum().item())
            n_tot = int(rf.numel())
            diagnostics.append(
                f"out{i} ({dtype_name}, atol={atol:.1e} rtol={rtol:.1e}): "
                f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
                f"bad_elems={n_bad}/{n_tot} ({100.0 * n_bad / n_tot:.2f}%)"
            )
        else:
            diagnostics.append(
                f"out{i} ({dtype_name}): OK (max_abs={max_abs:.3e})"
            )

    return {
        "correctness": all_close,
        "diagnostics": diagnostics,
        "max_abs_diff": max_abs_overall,
        "tolerances": tolerances,
    }


def compare_outputs_per_case(out_ref_per_case: list,
                             out_new_per_case: list) -> dict:
    """Multi-shape per-dtype allclose check.

    Inputs are List[List[Tensor]] — one outer entry per shape case. Returns
    {"correctness", "per_case", "diagnostics" (flat), "max_abs_diff",
     "failed_indices", "worst_idx", "worst_max_abs_diff"}.

    On multi-case failure, appends a CORRECTNESS_SUMMARY line to the flat
    diagnostics for failure_extractor to pick up.
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
            "tolerances": sub["tolerances"],
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
