#!/usr/bin/env python3
"""Thin CLI wrapper around scripts/eval/KernelVerifier.

Three eval phases — verify / profile_gen / profile_base — all execute
inside KernelVerifier-spawned subprocesses (LocalWorker manages the
DevicePool acquire/release). This CLI's only job is to:

  1. Read reference.py + kernel.py as source strings.
  2. Build LocalWorker(DevicePool([device_id])) + KernelVerifier.
  3. `await verifier.run({"coder_code": kernel_src})` for verify.
  4. `await verifier.run_profile(...)` for profile.
  5. Re-shape KernelVerifier outputs into autoresearch's
     `.eval_result.json` sidecar.

per-case granularity:
  - verify: the rendered verify script (kernel_verify_template_refactored.j2)
    runs a three-stage per-case loop (ref / impl / compare each in its own
    try/except) and drops a structured `verify_result.json` next to the
    script. `KernelVerifier.run()` reads it onto
    `verifier.last_verify_sidecar` — autoresearch's `verify` block is a
    direct passthrough of that dict (idx, case_desc, error_source,
    failure_kind, failed_indices, diagnostics).
  - profile: `KernelVerifier.run_profile()` returns the canonical
    per-shape dict (per_shape_gen_us / per_shape_base_us / case_descs +
    gen_method / base_method). We just lay those into per_shape rows
    without re-reading the underlying JSON.

Sidecar shape — schema preserved verbatim for eval_client:
  {
    ok: bool, errors: list,
    verify: {correctness, error_source, ref_source, num_cases,
             failed_indices,
             per_case[{idx, case_desc, correctness, error,
                       error_source, failure_kind?}],
             diagnostics, worst_idx, worst_max_abs_diff},
    profile_base | profile_gen: {
      avg_time_us, execution_time_us, execution_time_ms,
      warmup_times, run_times, num_cases,
      per_shape[{idx, case_desc, avg_time_us, method}]
    }
  }

Standalone reproducer:
    python scripts/engine/eval_kernel.py \\
        --task-dir <task_dir> --op-name <op> \\
        --kernel-file kernel --ref-file reference \\
        --device-id 0 --backend ascend --dsl triton_ascend \\
        --arch ascend910b4 \\
        --warmup 10 --repeats 100 --phases verify,profile_gen,profile_base
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import traceback
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Profile result re-shaping (canonical KernelVerifier output -> per_shape)
# ---------------------------------------------------------------------------

def _shape_profile_to_sidecar(per_case_us: list,
                              case_descs: list,
                              method: Optional[str],
                              warmup: Any,
                              runs: Any) -> Optional[dict]:
    """Re-shape ``KernelVerifier.run_profile()`` outputs into autoresearch's
    per_shape + aggregate dict.

    Inputs come straight from the canonical profile result:

      - ``per_case_us``: `[float, ...]` from ``per_shape_gen_us`` /
        ``per_shape_base_us`` (always populated; length 1 for static).
      - ``case_descs``: `[str, ...]` from ``case_descs`` (verify sidecar's
        `inputs[N]: shape=... dtype=...` strings).
      - ``method``: timer name (e.g. ``msprof`` / ``loop_timer``).

    Returns ``None`` when ``per_case_us`` is empty (caller treats as
    "phase not run / failed"). ``caseN`` labels back-fill any
    desc/timing length mismatch — symptom of a verify-vs-profile shape
    disagreement, not load-bearing for the per-shape array shape."""
    if not per_case_us:
        return None
    per_shape: list = []
    for i, raw in enumerate(per_case_us):
        try:
            us = float(raw)
            if not math.isfinite(us) or us <= 0:
                us = float("inf")
        except (TypeError, ValueError):
            us = float("inf")
        desc = case_descs[i] if i < len(case_descs) else f"case{i}"
        per_shape.append({
            "idx": i,
            "case_desc": desc,
            "avg_time_us": us,
            "method": method if math.isfinite(us) else None,
        })
    finites = [r["avg_time_us"] for r in per_shape
               if math.isfinite(r["avg_time_us"])]
    agg_us = sum(finites) / len(finites) if finites else float("inf")
    return {
        "avg_time_us": agg_us,
        "execution_time_us": agg_us if math.isfinite(agg_us) else None,
        "execution_time_ms": (agg_us / 1000.0) if math.isfinite(agg_us) else None,
        "warmup_times": warmup,
        "run_times": runs,
        "num_cases": len(per_shape),
        "per_shape": per_shape,
    }


def _verify_sidecar_to_autoresearch(sidecar: Optional[dict],
                                    log: str,
                                    passed: bool) -> dict:
    """Pass KernelVerifier's verify_result.json straight through —
    the template now emits exactly the autoresearch shape. Fall back
    to a minimal one-case dict when the sidecar is missing (template
    crash before the json.dump landed)."""
    if isinstance(sidecar, dict) and sidecar.get("num_cases") is not None:
        return sidecar
    # Sidecar missing — synthesize a one-case stand-in from (passed, log).
    log_short = (log or "").strip()
    if passed:
        return {
            "correctness": True, "error_source": None,
            "ref_source": "computed", "num_cases": 1,
            "failed_indices": [],
            "per_case": [{
                "idx": 0, "case_desc": "aggregate",
                "correctness": True, "error": None,
                "error_source": None, "failure_kind": None,
            }],
            "diagnostics": [], "worst_idx": None,
            "worst_max_abs_diff": None,
        }
    return {
        "correctness": False, "error_source": "kernel",
        "ref_source": "computed", "num_cases": 1,
        "failed_indices": [0],
        "per_case": [{
            "idx": 0, "case_desc": "aggregate",
            "correctness": False, "error": log_short[:500],
            "error_source": "kernel", "failure_kind": "kernel_crash",
        }],
        "diagnostics": [log_short[:2000]] if log_short else [],
        "worst_idx": None, "worst_max_abs_diff": None,
    }


def _wrap_phase_error(phase: str, e: BaseException) -> dict:
    return {
        "phase": phase,
        "type": type(e).__name__,
        "msg": str(e),
        "trace": traceback.format_exc(),
    }


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------

async def _run_eval(args: argparse.Namespace) -> int:
    from utils.json_io import sanitize_floats
    from eval.kernel_verifier import KernelVerifier
    from eval.worker.device_pool import DevicePool
    from eval.worker.local_worker import LocalWorker

    task_dir = os.path.abspath(args.task_dir)
    ref_path = os.path.join(task_dir, args.ref_file + ".py")
    kernel_path = os.path.join(task_dir, args.kernel_file + ".py")
    out_path = args.output or os.path.join(task_dir, ".eval_result.json")

    requested = {p.strip() for p in args.phases.split(",") if p.strip()}
    do_verify = "verify" in requested
    do_gen = "profile_gen" in requested
    do_base = "profile_base" in requested

    result: dict = {
        "verify": None,
        "profile_base": None,
        "profile_gen": None,
        "ok": True,
        "errors": [],
    }

    def _write(rc: int) -> int:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sanitize_floats(result), f, default=str)
        print(f"[eval_kernel] result -> {out_path}", file=sys.stderr)
        return rc

    # --- Read source files ---
    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_src = f.read()
    except Exception as e:
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("read_ref", e))
        result["verify"] = {
            "correctness": False, "error_source": "ref",
            "ref_source": "computed",
            "error": f"read reference failed: {type(e).__name__}: {e}",
            "num_cases": 0, "per_case": [], "diagnostics": [],
            "failed_indices": [], "worst_idx": None,
            "worst_max_abs_diff": None,
        }
        return _write(1)

    try:
        with open(kernel_path, "r", encoding="utf-8") as f:
            kernel_src = f.read()
    except Exception as e:
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("read_kernel", e))
        result["verify"] = _verify_sidecar_to_autoresearch(
            None, f"read kernel failed: {type(e).__name__}: {e}", False)
        return _write(1)

    catlass_meta: dict = {"task_dir": task_dir}
    if args.catlass_root:
        catlass_meta["catlass_root"] = args.catlass_root
    if args.catlass_op_dir:
        catlass_meta["catlass_op_dir"] = args.catlass_op_dir
    task_info = {"coder_code": kernel_src, **catlass_meta}

    # --- Build worker + verifier ---
    log_dir = os.path.join(task_dir, ".eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    try:
        from task_config.loader import load_task_config
        from task_config.task_files import read_declared_files
        cfg = load_task_config(task_dir)
        framework_aux_files = read_declared_files(
            task_dir,
            getattr(cfg, "data_files", None) or [],
            field_name="data_files",
        )
    except Exception as e:
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("read_data_files", e))
        result["verify"] = {
            "correctness": False, "error_source": "ref",
            "ref_source": "computed",
            "error": f"read data_files failed: {type(e).__name__}: {e}",
            "num_cases": 0, "per_case": [], "diagnostics": [],
            "failed_indices": [], "worst_idx": None,
            "worst_max_abs_diff": None,
        }
        return _write(1)

    config = {
        "log_dir": log_dir,
        "verify_timeout": args.verify_timeout,
        "framework_aux_files": framework_aux_files,
        "framework_module_name": args.ref_file,
        "framework_filename": f"{args.ref_file}.py",
        **catlass_meta,
    }
    try:
        device_pool = DevicePool([args.device_id])
        worker = LocalWorker(device_pool=device_pool, backend=args.backend)
        verifier = KernelVerifier(
            op_name=args.op_name,
            framework_code=ref_src,
            task_id=args.task_id,
            framework=args.framework,
            dsl=args.dsl,
            backend=args.backend,
            arch=args.arch,
            config=config,
            worker=worker,
        )
    except Exception as e:
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("verifier_init", e))
        return _write(1)

    # --- verify (KernelVerifier subprocess + verify_result.json passthrough) ---
    verify_case_descs: list = []
    if do_verify:
        try:
            passed, vlog = await verifier.run(
                task_info, current_step=args.current_step)
            sidecar = getattr(verifier, "last_verify_sidecar", None)
            result["verify"] = _verify_sidecar_to_autoresearch(
                sidecar, str(vlog), bool(passed))
            for entry in result["verify"].get("per_case", []) or []:
                verify_case_descs.append(entry.get("case_desc", "") or "")
        except Exception as e:
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("verify", e))
            result["verify"] = _verify_sidecar_to_autoresearch(
                None, f"{type(e).__name__}: {e}", False)

    # --- profile (KernelVerifier subprocess) ---
    # KernelVerifier.run_profile returns canonical per-shape arrays
    # directly (per_shape_gen_us / per_shape_base_us / case_descs +
    # gen_method / base_method). We just lay those into per_shape rows
    # — no JSON re-reading needed.
    if do_gen or do_base:
        try:
            prof_res = await verifier.run_profile(
                task_info,
                current_step=args.current_step,
                profile_settings={
                    "warmup_times": args.warmup,
                    "run_times": args.repeats,
                    "skip_base_profile": not do_base,
                    "enable_roofline": False,
                },
            )
            # Prefer profile's case_descs (sourced from the verify sidecar
            # inside KernelVerifier); fall back to verify_case_descs we
            # captured above when verify ran in this same invocation.
            case_descs = (list(prof_res.get("case_descs") or [])
                          or verify_case_descs)
            if do_gen:
                result["profile_gen"] = _shape_profile_to_sidecar(
                    list(prof_res.get("per_shape_gen_us") or []),
                    case_descs,
                    prof_res.get("gen_method"),
                    args.warmup, args.repeats,
                )
            if do_base:
                result["profile_base"] = _shape_profile_to_sidecar(
                    list(prof_res.get("per_shape_base_us") or []),
                    case_descs,
                    prof_res.get("base_method"),
                    args.warmup, args.repeats,
                )
        except Exception as e:
            result["ok"] = False
            phase = "profile_gen+base" if do_base else "profile_gen"
            result["errors"].append(_wrap_phase_error(phase, e))

    return _write(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from utils.settings import target_backend, target_framework, target_dsl
        default_backend = target_backend()
        default_framework = target_framework()
        default_dsl = target_dsl()
    except Exception:
        default_backend = "ascend"
        default_framework = "torch"
        default_dsl = "triton_ascend"

    ap = argparse.ArgumentParser(
        description="autoresearch eval orchestrator — thin wrapper around "
                    "scripts/eval/KernelVerifier (verify + profile)")
    ap.add_argument("--task-dir", required=True,
                    help="task directory containing reference.py + kernel.py")
    ap.add_argument("--op-name", required=True,
                    help="operator name (used for KernelVerifier internal "
                         "dir naming + the impl file basename)")
    ap.add_argument("--kernel-file", required=True,
                    help="kernel module name without .py (convention: kernel)")
    ap.add_argument("--ref-file", required=True,
                    help="reference module name without .py (convention: reference)")
    ap.add_argument("--device-id", type=int, default=0)
    from eval.worker.interface import (
        DEFAULT_EVAL_TIMEOUT_S as _DEFAULT_EVAL_TIMEOUT_S,
        DEFAULT_WARMUP_TIMES as _DEFAULT_WARMUP_TIMES,
        DEFAULT_RUN_TIMES as _DEFAULT_RUN_TIMES,
    )
    ap.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP_TIMES,
                    help="profile warmup iterations")
    ap.add_argument("--repeats", type=int, default=_DEFAULT_RUN_TIMES,
                    help="profile measured iterations")
    ap.add_argument("--phases", default="verify,profile_gen,profile_base",
                    help="comma-separated subset of {verify, profile_gen, "
                         "profile_base}; KernelVerifier always emits both "
                         "base + gen profile JSONs, this flag picks which "
                         "we copy into the sidecar")
    ap.add_argument("--output", default=None,
                    help="JSON sidecar path (default: <task_dir>/.eval_result.json)")
    ap.add_argument("--framework", default=default_framework,
                    choices=["torch", "mindspore", "numpy"])
    ap.add_argument("--dsl", default=default_dsl,
                    help="DSL the kernel is written in (triton_ascend, "
                         "triton_cuda, swft, ascendc, ...; see "
                         "scripts/eval/adapters/factory.py)")
    ap.add_argument("--backend", default=default_backend,
                    choices=["ascend", "cuda", "cpu"])
    ap.add_argument("--arch", default="ascend910b4",
                    help="hardware arch (ascend910b1..b4, ascend310p3, "
                         "a100, v100, h20, l20, rtx3090, ...)")
    ap.add_argument("--catlass-root", default=None,
                    help="CATLASS repo root for ascendc_catlass tasks")
    ap.add_argument("--catlass-op-dir", default=None,
                    help="task-local CATLASS project directory "
                         "(default: catlass_op)")
    ap.add_argument("--task-id", default="0",
                    help="task identifier — used in verify_dir naming so "
                         "concurrent eval runs on the same task_dir don't "
                         "clobber each other")
    ap.add_argument("--current-step", type=int, default=0,
                    help="round number for verify_dir naming "
                         "(Iteration<task_id>_Step<NN>_verify)")
    ap.add_argument("--verify-timeout", type=int, default=_DEFAULT_EVAL_TIMEOUT_S,
                    help="verify subprocess timeout in seconds")
    args = ap.parse_args()

    valid = {"verify", "profile_gen", "profile_base"}
    bad = {p.strip() for p in args.phases.split(",") if p.strip()} - valid
    if bad:
        print(f"unknown phase(s): {sorted(bad)}; valid: {sorted(valid)}",
              file=sys.stderr)
        sys.exit(2)

    sys.exit(asyncio.run(_run_eval(args)))


if __name__ == "__main__":
    main()
