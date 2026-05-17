"""Shared atol/rtol output comparison — matches akg/akg_agents semantics.

Single source of truth used by:
  - the generated eval script (package_builder._gen_eval_script) that
    runs inside the worker / local subprocess
  - the batch-time pre-flight verify.py Tier-2 check

Defaults and semantics come from akg/akg_agents
(`examples/kernel_related/bench_lite_common.py:_check_correctness`):

  DEFAULT_ATOL = DEFAULT_RTOL = 1e-2
  per-tensor check: max_abs_diff <= atol  AND  max_rel_diff <= rtol
                    where rel_diff = abs_diff / (|ref| + 1e-8)
  NaN / Inf in the solution → fail
  shape mismatch → fail

This is STRICTER than `torch.allclose(a, b, atol, rtol)`'s element-wise
`|a-b| <= atol + rtol*|b|` because the AND of maxima can't trade absolute
error against relative error within one element. Picking it deliberately
mirrors the benchmark suite's own correctness gate so claude-autoresearch
classifies a kernel the same way akg_agents would.

Dependency-light (only torch) so it can be bundled into the eval-package
tarball.
"""
from __future__ import annotations

from typing import Optional


# Single global tolerance, matching akg_agents.bench_lite_common.
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2

# Small epsilon prevents div-by-zero when ref values are exact 0. Same
# value akg_agents uses.
_REL_EPS = 1e-8


def compare_outputs(out_ref: list, out_new: list,
                    atol: float = DEFAULT_ATOL,
                    rtol: float = DEFAULT_RTOL) -> dict:
    """akg_agents-style AND-of-maxima check.

    For each tensor pair:
      - tensors are detached, moved to CPU, cast to fp32 for stable
        subtraction math
      - NaN or Inf anywhere in the solution → fail
      - shape mismatch → fail
      - max_abs_diff = max(|ref - sol|)
      - max_rel_diff = max(|ref - sol| / (|ref| + 1e-8))
      - PASS iff `max_abs_diff <= atol` AND `max_rel_diff <= rtol`

    Returns:
      {"correctness": bool,
       "diagnostics": list[str],
       "max_abs_diff": float | None,
       "atol": float, "rtol": float}
    """
    import torch  # local import keeps the module importable without torch

    diagnostics: list[str] = []
    all_close = True
    max_abs_overall: Optional[float] = None

    if len(out_ref) != len(out_new):
        return {
            "correctness": False,
            "diagnostics": [
                f"output count: ref={len(out_ref)} new={len(out_new)}"
            ],
            "max_abs_diff": None,
            "atol": atol, "rtol": rtol,
        }

    for i, (r, n) in enumerate(zip(out_ref, out_new)):
        # Only tensor pairs participate — autoresearch's eval never
        # materializes scalar-only outputs through this path.
        if not (isinstance(r, torch.Tensor) and isinstance(n, torch.Tensor)):
            continue

        rf = r.detach().cpu().float()
        nf = n.detach().cpu().float()

        if rf.shape != nf.shape:
            all_close = False
            diagnostics.append(
                f"out{i} shape {tuple(rf.shape)} != kernel {tuple(nf.shape)}"
            )
            continue

        if torch.isnan(nf).any() or torch.isinf(nf).any():
            all_close = False
            n_bad = int((torch.isnan(nf) | torch.isinf(nf)).sum().item())
            diagnostics.append(
                f"out{i}: kernel produced {n_bad} NaN/Inf value(s)"
            )
            continue

        abs_diff = (rf - nf).abs()
        rel_diff = abs_diff / (rf.abs() + _REL_EPS)
        max_abs = float(abs_diff.max().item())
        max_rel = float(rel_diff.max().item())

        if max_abs_overall is None or max_abs > max_abs_overall:
            max_abs_overall = max_abs

        # akg_agents AND-of-maxima: fail when EITHER threshold blown.
        if max_abs > atol or max_rel > rtol:
            all_close = False
            # bad_elems counts how many elements broke either threshold —
            # useful diagnostic the bare max numbers don't convey.
            bad_abs = abs_diff > atol
            bad_rel = rel_diff > rtol
            n_bad = int((bad_abs | bad_rel).sum().item())
            n_tot = int(rf.numel())
            diagnostics.append(
                f"out{i}: max_abs={max_abs:.3e} (atol={atol:.1e}) "
                f"max_rel={max_rel:.3e} (rtol={rtol:.1e}) "
                f"bad_elems={n_bad}/{n_tot} ({100.0 * n_bad / n_tot:.2f}%)"
            )
        else:
            diagnostics.append(
                f"out{i}: OK (max_abs={max_abs:.3e} max_rel={max_rel:.3e})"
            )

    return {
        "correctness": all_close,
        "diagnostics": diagnostics,
        "max_abs_diff": max_abs_overall,
        "atol": atol, "rtol": rtol,
    }


def compare_outputs_per_case(out_ref_per_case: list,
                             out_new_per_case: list,
                             atol: float = DEFAULT_ATOL,
                             rtol: float = DEFAULT_RTOL) -> dict:
    """Multi-shape AND-of-maxima check; hard-gate on every case.

    Inputs are List[List[Tensor]] — one outer entry per shape case.
    Returns {"correctness", "per_case", "diagnostics" (flat with
    `[case i]` prefix), "max_abs_diff", "failed_indices", "worst_idx",
    "worst_max_abs_diff", "atol", "rtol"}.

    On multi-case failure appends one CORRECTNESS_SUMMARY line to the
    flat diagnostics for failure_extractor to pick up; DIAGNOSE consumes
    it directly.
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
            "atol": atol, "rtol": rtol,
        }

    per_case = []
    flat_diag: list[str] = []
    all_pass = True
    max_abs_overall: Optional[float] = None

    for i, (out_ref, out_new) in enumerate(zip(out_ref_per_case,
                                               out_new_per_case)):
        sub = compare_outputs(list(out_ref), list(out_new), atol, rtol)
        per_case.append({
            "idx": i,
            "correctness": sub["correctness"],
            "diagnostics": sub["diagnostics"],
            "max_abs_diff": sub["max_abs_diff"],
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
        "atol": atol, "rtol": rtol,
    }
