"""Eval script generation + tar.gz package assembly.

The generated `eval_<op>.py` is the contract between this client and the
worker (or local subprocess runner). Both transports unpack the same
tarball and run the same single script, which does verify + profile_gen
+ profile_base in ONE Python process and writes `eval_result.json` as a
sidecar.

Why one script (not three):

  - Triton JIT cache and autotune state are in-process. The previous
    3-subprocess layout meant profile_gen's autotune re-explored the
    config space from scratch — warmup budget burned on exploration
    instead of measurement, contaminating per-shape timing.
  - CANN's tiling-struct warnings to stdout could pollute the
    stdout-tail JSON parsed by `_last_json_line` and silently corrupt
    verify classification. Sidecar JSON eliminates that hazard.
  - Three cold `torch_npu` inits per round added ~30s of fixed
    overhead on Ascend.

This file owns:
  - DSL adapter resolution (`_get_dsl_adapter`, `_detect_device_type`).
  - The single eval-script template (`_gen_eval_script`).
  - Tarball assembly (`_build_package`).
"""
import io
import os
import sys
import tarfile
import textwrap
from typing import Optional

from .loader import TaskConfig

EVAL_SIDECAR = "eval_result.json"


# ---------------------------------------------------------------------------
# DSL / device-type resolution
# ---------------------------------------------------------------------------

def _detect_device_type(config: TaskConfig) -> str:
    """torch.device prefix ('npu' / 'cuda' / 'cpu'). Derived from DSL via
    hw_detect (DSL → backend → device_type)."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from utils.hw_detect import device_type_for_dsl
    try:
        return device_type_for_dsl(config.dsl or "")
    except Exception:
        return "cpu"


def _get_dsl_adapter(dsl: Optional[str]):
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from ar_vendored.op.verifier.adapters.factory import get_dsl_adapter
    return get_dsl_adapter(dsl or "triton_ascend")


def _gen_benchmark_body(adapter, config: TaskConfig,
                        warmup: int, repeats: int) -> tuple[str, str]:
    """Kernel benchmark body. Goes through the DSL adapter so DSL-specific
    setup (triton autotune trigger, tilelang compile, ...) gets included,
    then the adapter calls profiler_npu under the hood for NPU backends.

    `clear_l2_cache=False` is forced so kernel timing is directly
    comparable to the ref measurement below (which can't use the
    triton_ascend L2-clear kernel — it corrupts aclnnArange state).
    Both measurements now reflect warm-cache steady-state behavior, the
    same as akg-hitl's default.

    framework_model="impl_model" — adapter templates reference
    {framework_model} verbatim; tilelang_npuir / swft would otherwise
    emit `return None(*inputs)` or a call to an undefined name.

    case_idx="{case_idx}" — adapters that bake the index into a filename
    or output directory (triton_ascend's autotune_info_case_{case_idx}.json,
    pypto's prof_generation_output_case{case_idx}) embed `{case_idx}`
    inside an f-string LITERAL in their template. The literal string
    "{case_idx}" makes the outer f-string emit `{case_idx}` verbatim
    into the inner f-string slot — resolved at runtime against the
    `case_idx` parameter our `_bench_*` functions receive from the
    `_run_profile` per-case loop.
    """
    import inspect
    kwargs: dict = dict(
        impl_func_name="TargetModel", inputs="inputs",
        warmup=warmup, runs=repeats,
        backend=config.backend or "", op_name=config.name,
        case_idx="{case_idx}", device_id=0,
        framework_model="impl_model",
    )
    # Only the triton_ascend / tilelang_npuir adapters take clear_l2_cache.
    # Other DSL adapters reject unknown kwargs — feed it conditionally.
    sig = inspect.signature(adapter.benchmark_impl)
    if "clear_l2_cache" in sig.parameters:
        kwargs["clear_l2_cache"] = False
    raw = adapter.benchmark_impl(**kwargs)
    if raw and raw.strip():
        return (textwrap.indent(textwrap.dedent(raw), "    "),
                f"adapter ({type(adapter).__name__})")
    # Adapter has no benchmark_impl — fall back to direct profiler_npu /
    # do_bench depending on backend. Same body as _base_benchmark_body.
    return _base_benchmark_body(warmup, repeats)


def _base_benchmark_body(warmup: int, repeats: int) -> tuple[str, str]:
    """Reference benchmark body. Bypasses the DSL adapter and goes
    directly to `profiler_npu` (for NPU) or `triton.testing.do_bench`
    (other backends) — semantically matched to akg-hitl's
    `profiler_npu_core` so ref and kernel timings can be divided to
    produce an honest speedup.

    No L2-cache-clear: triton_ascend's clear kernel corrupts the next
    aclnnArange dispatch, and clearing for ref alone would bias the
    ratio. Kernel measurement is also called with clear_l2_cache=False
    above for symmetry.
    """
    body = textwrap.indent(textwrap.dedent(f"""\
        if device_type == "npu":
            from ar_vendored.op.verifier.profiler import profiler_npu
            def _ref_bench():
                with torch.no_grad():
                    impl_model(*inputs)
            execution_time_us = profiler_npu(
                _ref_bench,
                warmup={warmup}, active={repeats},
                prof_dir_name=f"prof_base_case_{{case_idx}}",
                keep_res=False, suppress_warnings=True,
                clear_l2_cache=False, dsl="other",
            )
            execution_time_ms = execution_time_us / 1000
            method = "profiler_npu_base"
        else:
            import triton.testing
            def _ref_bench():
                with torch.no_grad():
                    impl_model(*inputs)
            execution_time_ms = triton.testing.do_bench(
                _ref_bench, warmup={warmup}, rep={repeats},
                return_mode="min")
            execution_time_us = execution_time_ms * 1000
            method = "do_bench_base"
    """), "    ")
    return body, "profiler_npu_base / do_bench"


def _gen_eval_script(config: TaskConfig) -> str:
    """Generate `eval_<op>.py` — verify + profile_gen + profile_base, with
    a single `eval_result.json` sidecar.

    Iteration counts come from `config.warmup_times` and `config.run_times`
    (task.yaml `eval.warmup_times` / `eval.run_times`), defaulting to
    10/100. ref and kernel use the same counts so their timings are
    directly comparable.

    Phase layout (all caught independently; partial results still land in
    the sidecar):

      Phase A — ref-side setup           error_source="ref"   (always)
      Phase B — kernel-side import       error_source="kernel"
      Phase C — verify (Model vs ModelNew, per-case)
                first-failing case decides error_source
      Phase D — profile_gen (ModelNew under adapter benchmark_impl)
      Phase E — profile_base (Model under profiler_npu, matching akg-hitl)

    `AR_EVAL_PHASE` env selects which subset runs:
      "ref_only"     — Phase A + E only (no kernel import; immune to
                       kernel SIGKILL / device hang in a sibling pass)
      "kernel_only"  — Phase A + B + C + D only (no profile_base)
      "all" (default for ad-hoc reproducer) — A + B + C + D + E

    Production callers spawn this script TWICE per round (ref_only then
    kernel_only) so a kernel UB overflow / device fault in pass 2 can't
    erase ref data that pass 1 already wrote to its sidecar.

    Profile D uses the warm JIT/autotune state populated by verify — no
    autotune exploration during warmup, so the measured shape timing
    reflects the actual best config rather than mid-exploration cost.
    """
    warmup = config.warmup_times
    repeats = config.run_times
    device = _detect_device_type(config)
    kernel_file = config.editable_files[0].replace(".py", "")
    ref_file = config.ref_file.replace(".py", "")

    adapter = _get_dsl_adapter(config.dsl)
    dsl_imports = adapter.get_import_statements(config.framework or "torch")
    dsl_setup = (adapter.get_special_setup_code()
                 if hasattr(adapter, "get_special_setup_code") else "")

    bench_gen_body, bench_gen_label = _gen_benchmark_body(
        adapter, config, warmup=warmup, repeats=repeats)
    bench_base_body, bench_base_label = _base_benchmark_body(
        warmup=warmup, repeats=repeats)

    return f'''\
#!/usr/bin/env python3
"""Auto-generated single-process eval (dsl={config.dsl}, backend={config.backend}).

Phases run in one Python interpreter so verify warms the JIT / autotune
cache that profile_gen reuses. Result is a sidecar `eval_result.json` —
NOT stdout JSON (which CANN warnings could corrupt).

profile_gen benchmark = {bench_gen_label}
profile_base benchmark = {bench_base_label}
"""
import os, sys, json, math, time, traceback

# Bundled at tarball root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device_type = "{device}"
device_id = int(os.environ.get("DEVICE_ID", "0"))

if device_type == "npu":
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))
    import torch
    import torch_npu
    device = torch.device("npu:0")
elif device_type == "cuda":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device_id))
    import torch
    device = torch.device("cuda:0")
else:
    import torch
    device = torch.device("cpu")

# DSL-specific imports (triton autotune patches, tilelang patches, etc.)
{dsl_imports}
{dsl_setup}

SIDECAR_PATH = os.environ.get(
    "AR_EVAL_SIDECAR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "{EVAL_SIDECAR}"),
)
# Phase selector — production callers spawn TWO subprocesses per round
# (ref_only then kernel_only). "all" is the legacy single-process
# behavior, kept for ad-hoc reproducer use.
PHASE = os.environ.get("AR_EVAL_PHASE", "all")
if PHASE not in ("all", "ref_only", "kernel_only"):
    print(f"[eval] WARN: unknown AR_EVAL_PHASE={{PHASE!r}}, defaulting to 'all'",
          file=sys.stderr)
    PHASE = "all"
DO_KERNEL_PHASES = PHASE in ("all", "kernel_only")
DO_REF_PHASE = PHASE in ("all", "ref_only")

result = {{
    "verify": None,
    "profile_gen": None,
    "profile_base": None,
    "ok": True,
    "errors": [],
}}


def _empty_cache():
    if device_type == "npu":
        torch.npu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()


def _to_cpu_list(out):
    if isinstance(out, torch.Tensor):
        return [out.detach().cpu()]
    if isinstance(out, (list, tuple)):
        return [o.detach().cpu() if hasattr(o, "detach") else o for o in out]
    return [out]


def _write_and_exit(rc):
    try:
        with open(SIDECAR_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, default=str)
        print(f"[eval] sidecar -> {{SIDECAR_PATH}}", file=sys.stderr)
    except Exception as e:
        print(f"[eval] failed to write sidecar: {{e}}", file=sys.stderr)
    sys.exit(rc)


# === Phase A: ref-side setup ============================================
# Failures here all classify as error_source="ref". Caller (scaffold /
# pipeline) routes ref-side failures back to the user instead of
# dragging the agent into a futile kernel-rewrite loop.
try:
    import {ref_file} as _ref_mod
    from {ref_file} import Model, get_init_inputs
    from input_groups import (
        resolve as _resolve_groups,
        describe_case as _describe_case,
    )
    init_inputs = get_init_inputs()
    cases_cpu = _resolve_groups(_ref_mod)
    num_cases = len(cases_cpu)
    if num_cases == 0:
        raise RuntimeError("reference module returned 0 input cases")
except Exception as e:
    traceback.print_exc()
    result["ok"] = False
    result["verify"] = {{
        "correctness": False,
        "error_source": "ref",
        "error": f"reference setup failed: {{type(e).__name__}}: {{e}}",
        "num_cases": 0,
        "per_case": [], "diagnostics": [], "failed_indices": [],
        "worst_idx": None, "worst_max_abs_diff": None,
    }}
    _write_and_exit(2)


# === Phase B: kernel-side import ========================================
# Skipped in ref_only mode (ref pass doesn't need ModelNew). When the
# kernel pass hits an import error we still let Phase E try to measure
# ref — that's the whole point of the split, ref shouldn't be held
# hostage by a broken kernel.
ModelNew = None
kernel_imported = False
if DO_KERNEL_PHASES:
    try:
        from {kernel_file} import ModelNew
        kernel_imported = True
    except Exception as e:
        traceback.print_exc()
        result["ok"] = False
        result["verify"] = {{
            "correctness": False,
            "error_source": "kernel",
            "error": (f"import failed: cannot import name 'ModelNew' from "
                      f"'{kernel_file}' ({{type(e).__name__}}: {{e}})"),
            "num_cases": num_cases,
            "per_case": [], "diagnostics": [], "failed_indices": [],
            "worst_idx": None, "worst_max_abs_diff": None,
        }}


# === Phase C: verify (warms JIT cache for profile_gen) ==================
# Per-case loop with ref-side vs kernel-side blame. The first failure
# decides error_source — ref-side wins over kernel-side because a broken
# ref invalidates the entire eval regardless of kernel correctness.
# Whole block skipped when kernel_imported is False (ref_only mode or
# Phase B import failed). `correctness` is bundled at tarball root and
# is needed regardless of mode by other tooling — keep the import
# unconditional.
from correctness import (
    compare_outputs_per_case, DEFAULT_ATOL, DEFAULT_RTOL,
)

if kernel_imported:
    out_ref_per_case = []
    out_new_per_case = []
    verify_error_source = None
    verify_error_msg = None

    try:
        model_ref = Model(*init_inputs).to(device).eval()
        with torch.no_grad():
            for case in cases_cpu:
                ref_inputs_dev = [x.to(device) if hasattr(x, "to") else x
                                  for x in case]
                out_ref_per_case.append(_to_cpu_list(model_ref(*ref_inputs_dev)))
                del ref_inputs_dev
        del model_ref
        _empty_cache()
    except Exception as e:
        traceback.print_exc()
        verify_error_source = "ref"
        verify_error_msg = (f"reference forward failed on device: "
                            f"{{type(e).__name__}}: {{e}}")

    if verify_error_source is None:
        try:
            model_new = ModelNew(*init_inputs).to(device).eval()
            with torch.no_grad():
                for case in cases_cpu:
                    inputs_dev = [x.to(device) if hasattr(x, "to") else x
                                  for x in case]
                    out_new_per_case.append(_to_cpu_list(model_new(*inputs_dev)))
                    del inputs_dev
            # Keep model_new around briefly — describe_case may use its
            # metadata. Delete after we're done with the compare block.
        except Exception as e:
            traceback.print_exc()
            verify_error_source = "kernel"
            verify_error_msg = (f"kernel forward failed: "
                                f"{{type(e).__name__}}: {{e}}")
            model_new = None

    if verify_error_source is not None:
        result["verify"] = {{
            "correctness": False,
            "error_source": verify_error_source,
            "error": verify_error_msg,
            "num_cases": num_cases,
            "per_case": [], "diagnostics": [], "failed_indices": [],
            "worst_idx": None, "worst_max_abs_diff": None,
        }}
    else:
        try:
            cmp_result = compare_outputs_per_case(
                out_ref_per_case, out_new_per_case,
                DEFAULT_ATOL, DEFAULT_RTOL)

            for d in cmp_result["diagnostics"]:
                print(d, file=sys.stderr)

            # FAILED_SHAPES line for failure_extractor.multi_shape_failed_shapes
            # — gives DIAGNOSE the offending case desc directly.
            if not cmp_result["correctness"] and num_cases > 1:
                failed = cmp_result.get("failed_indices") or []
                if failed:
                    shape_strs = []
                    for i in failed[:10]:
                        if 0 <= i < num_cases:
                            shape_strs.append(
                                f"case {{i}}={{_describe_case(cases_cpu[i], model_new)}}")
                    if shape_strs:
                        suffix = " ..." if len(failed) > 10 else ""
                        print("[verify] FAILED_SHAPES: " +
                              "; ".join(shape_strs) + suffix, file=sys.stderr)

            result["verify"] = {{
                "correctness": cmp_result["correctness"],
                "error_source": None if cmp_result["correctness"] else "kernel",
                "error": (None if cmp_result["correctness"]
                          else "kernel output != reference"),
                "num_cases": num_cases,
                "per_case": cmp_result.get("per_case", []),
                "diagnostics": cmp_result["diagnostics"],
                "failed_indices": cmp_result.get("failed_indices", []),
                "worst_idx": cmp_result.get("worst_idx"),
                "worst_max_abs_diff": cmp_result.get("worst_max_abs_diff"),
            }}
        except Exception as e:
            traceback.print_exc()
            result["verify"] = {{
                "correctness": False,
                "error_source": "kernel",
                "error": f"comparison failed: {{type(e).__name__}}: {{e}}",
                "num_cases": num_cases,
                "per_case": [], "diagnostics": [], "failed_indices": [],
                "worst_idx": None, "worst_max_abs_diff": None,
            }}

    # Free verify-side tensors before profile pass — HBM doesn't fit both.
    try:
        del model_new
    except NameError:
        pass
    out_ref_per_case.clear()
    out_new_per_case.clear()
    _empty_cache()


# === Benchmark helpers ===================================================
# kernel (gen) goes through the DSL adapter (autotune setup + profiler_npu);
# ref (base) goes through profiler_npu directly with dsl="other" so
# triton_ascend's L2-clear kernel — which corrupts the next aclnnArange
# in the ref Model — never runs. Both have clear_l2_cache=False for
# symmetric warm-cache measurement, matching akg-hitl's default.
#
# `case_idx` is the per-shape index from the enclosing for-loop. Adapter
# templates that bake the index into output dirs / filenames (pypto's
# prof_generation_output_case{{case_idx}}, triton_ascend's
# autotune_info_case_{{case_idx}}.json) reference this local name through
# an f-string embedded in the body — see package_builder docstring.

def _bench_gen(impl_model, inputs, case_idx):
    execution_time_us = None
    execution_time_ms = None
    method = None
{bench_gen_body}
    if execution_time_us is None and execution_time_ms is not None:
        execution_time_us = execution_time_ms * 1000
    return execution_time_us, method


def _bench_base(impl_model, inputs, case_idx):
    execution_time_us = None
    execution_time_ms = None
    method = None
{bench_base_body}
    if execution_time_us is None and execution_time_ms is not None:
        execution_time_us = execution_time_ms * 1000
    return execution_time_us, method


def _run_profile(target_cls, bench_fn, mode_label):
    per_shape = []
    for idx, case in enumerate(cases_cpu):
        impl_model = None
        try:
            impl_model = target_cls(*init_inputs)
            if hasattr(impl_model, "to"):
                impl_model = impl_model.to(device)
            if hasattr(impl_model, "eval"):
                impl_model.eval()
            inputs = [x.to(device) if hasattr(x, "to") else x for x in case]
            avg_us, method = bench_fn(impl_model, inputs, idx)
            if (avg_us is None or avg_us <= 0
                    or avg_us == float("inf")):
                # adapter ran but returned a non-finite / non-positive
                # value. Mark crashed; don't silently fall back to a
                # different timing method (e.g. CPU wall-clock) — that
                # would silently mix semantics across rounds and
                # produce a dishonest speedup_vs_ref.
                print(f"[profile {{mode_label}}] case {{idx}} adapter "
                      f"returned invalid avg_us={{avg_us!r}}; marking "
                      f"crashed",
                      file=sys.stderr)
                avg_us, method = float("inf"), "crashed"
        except Exception as e:
            print(f"[profile {{mode_label}}] case {{idx}} setup/timing "
                  f"failed: {{e}}", file=sys.stderr)
            traceback.print_exc()
            avg_us, method = float("inf"), "crashed"
        per_shape.append({{
            "idx": idx,
            "case_desc": _describe_case(case, impl_model),
            "avg_time_us": avg_us,
            "method": method,
        }})
        del impl_model
        _empty_cache()

    finite = [s["avg_time_us"] for s in per_shape
              if isinstance(s["avg_time_us"], (int, float))
              and math.isfinite(s["avg_time_us"])]
    avg_us = (sum(finite) / len(finite)) if finite else float("inf")
    execution_time_ms = (avg_us / 1000) if math.isfinite(avg_us) else None
    return {{
        "avg_time_us": avg_us,
        "execution_time_us": avg_us,
        "execution_time_ms": execution_time_ms,
        "warmup_times": {warmup},
        "run_times": {repeats},
        "num_cases": num_cases,
        "per_shape": per_shape,
    }}


# === Phase D: profile_gen (warm cache from verify) ======================
# Reads verify_block populated by Phase C. In ref_only mode the verify
# block is None — verify_ok defaults to False and we skip Phase D
# regardless.
verify_block = result.get("verify") or {{}}
verify_ok = bool(verify_block.get("correctness"))
verify_error_source = verify_block.get("error_source")

if kernel_imported and verify_error_source != "ref":
    # ref-side failure short-circuits Phase D in the same subprocess
    # (profile_gen needs ref forward as part of adapter setup for some
    # DSLs). Phase E is independent and may still try its own ref
    # measurement — that's the whole point of the split when this script
    # is run as the kernel-only pass.
    try:
        result["profile_gen"] = _run_profile(ModelNew, _bench_gen, "gen")
    except Exception as e:
        traceback.print_exc()
        result["ok"] = False
        result["errors"].append({{
            "phase": "profile_gen",
            "type": type(e).__name__, "msg": str(e),
        }})


# === Phase E: profile_base (PyTorch reference) ==========================
# Skipped in kernel_only mode (DO_REF_PHASE=False); the ref subprocess
# from the two-pass runner owns ref measurement. Sticky baseline reuse
# is handled by the runner (utils.eval_runner.synth_sticky_ref_payload),
# which builds the ref payload externally so this script never has to
# materialise anything when ref doesn't need re-measuring.
if DO_REF_PHASE:
    try:
        result["profile_base"] = _run_profile(Model, _bench_base, "base")
    except Exception as e:
        traceback.print_exc()
        result["ok"] = False
        result["errors"].append({{
            "phase": "profile_base",
            "type": type(e).__name__, "msg": str(e),
        }})

_write_and_exit(0 if verify_ok else 1)
'''


# ---------------------------------------------------------------------------
# Tarball assembly
# ---------------------------------------------------------------------------

def _exclude_pycache(tarinfo: tarfile.TarInfo):
    base = os.path.basename(tarinfo.name)
    if base == "__pycache__" or base.endswith(".pyc") or base.startswith("."):
        return None
    return tarinfo


def _build_package(task_dir: str, config: TaskConfig) -> bytes:
    """Build a tar.gz package containing:
      - kernel.py / reference.py / other .py support files from task_dir
      - eval_<op>.py (single-process orchestrator)
      - correctness.py + input_groups.py (lib modules at tarball root)
      - ar_vendored/ (DSL adapters + profiler_npu + patches)

    Device id is NOT baked in — the worker / local runner exports
    DEVICE_ID at run time and the generated script picks it up.
    """
    op_name = config.name
    buf = io.BytesIO()

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fname in config.editable_files:
            fpath = os.path.join(task_dir, fname)
            if os.path.exists(fpath):
                tar.add(fpath, arcname=fname)

        ref_path = os.path.join(task_dir, config.ref_file)
        if os.path.exists(ref_path):
            tar.add(ref_path, arcname=config.ref_file)

        for f in os.listdir(task_dir):
            if (f.endswith(".py")
                    and f not in config.editable_files
                    and f != config.ref_file
                    and not f.startswith(".")):
                fpath = os.path.join(task_dir, f)
                if os.path.isfile(fpath):
                    tar.add(fpath, arcname=f)

        def _add_script(name: str, content: str):
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        utils_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils",
        )
        for lib_name in ("correctness.py", "input_groups.py"):
            lib_src = os.path.join(utils_dir, lib_name)
            if os.path.isfile(lib_src):
                tar.add(lib_src, arcname=lib_name)

        _add_script(f"eval_{op_name}.py", _gen_eval_script(config))

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vendored_root = os.path.join(script_dir, "ar_vendored")
        if os.path.isdir(vendored_root):
            tar.add(vendored_root, arcname="ar_vendored",
                    filter=_exclude_pycache)

    return buf.getvalue()
