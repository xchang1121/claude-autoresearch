"""Verify/profile script generation + tar.gz package assembly.

The generated scripts are the contract between this client and the
remote worker (or local subprocess runner). Both transports unpack the
same tarball and run the same auto-generated `verify_<op>.py` /
`profile_<op>_<mode>.py`. This file owns:

  - DSL-adapter resolution (`_get_dsl_adapter`, `_detect_device_type`).
  - The verify-script template (`_gen_verify_script`).
  - The profile-script template (`_gen_profile_script`).
  - Tarball assembly (`_build_package`) + the pycache/dotfile filter.

What's NOT here:
  - HTTP transport / device pool / `run_*_eval` — those live in
    eval_client; this module only produces bytes for them to ship.
  - Metric comparison / EvalResult — those live in metric_policy; the
    generated scripts emit JSON that eval_client parses, not us.
"""
import io
import os
import sys
import tarfile
from typing import Optional

from .loader import TaskConfig


# ---------------------------------------------------------------------------
# DSL / device-type resolution (kept here because only the script generators
# need them — eval_client never touches DSL adapters directly).
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
    """Return the vendored DSL adapter for `dsl`. Raises if unknown.

    Cached-once import — the factory touches all DSL adapter modules on first
    call, which pulls in pandas/numpy/etc. Keep the import local to callers
    that need it, not at module scope.
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from ar_vendored.op.verifier.adapters.factory import get_dsl_adapter
    return get_dsl_adapter(dsl or "triton_ascend")


# ---------------------------------------------------------------------------
# Verify script template
# ---------------------------------------------------------------------------

def _gen_verify_script(config: TaskConfig, device_id: int = 0) -> str:
    """Generate verify_{op_name}.py for the Worker Service.

    Reference outputs are re-computed on the device every round. We used
    to torch.save them to /tmp/ar_cache/<op>_<sha>/reference.pt and
    torch.load on subsequent rounds; on multi-shape ops that .pt is
    List[List[Tensor]] of length num_cases * outputs-per-case and the
    weights_only=False reload rebuilds the entire pickle graph, which
    on NPU workers with slow /tmp stalled for minutes — looking like
    the worker had frozen. Re-computing the ref every round is cheaper
    end-to-end (one forward * num_cases) and the triton JIT cache for
    ModelNew is the actual amortisation source across rounds.

    DSL adapter (ar_vendored.op.verifier.adapters) supplies DSL-specific
    imports (triton autotune patches, tilelang compile patches, etc.) via
    `get_import_statements`. The verify body itself is uniform across DSLs:
    instantiate ModelNew, run forward per case, allclose vs ref.
    """
    device = _detect_device_type(config)
    kernel_file = config.editable_files[0].replace(".py", "")
    ref_file = config.ref_file.replace(".py", "")
    # Tolerance is locked to correctness.DEFAULT_ATOL / DEFAULT_RTOL —
    # the single source of truth for the comparison. TaskConfig no longer
    # carries atol/rtol; yaml/CLI no longer override.
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from utils.correctness import DEFAULT_ATOL as atol, DEFAULT_RTOL as rtol

    adapter = _get_dsl_adapter(config.dsl)
    dsl_imports = adapter.get_import_statements(config.framework or "torch")
    dsl_setup = adapter.get_special_setup_code() if hasattr(adapter, "get_special_setup_code") else ""

    return f'''\
#!/usr/bin/env python3
"""Auto-generated verify script (dsl={config.dsl}, backend={config.backend}).

Reference outputs are re-computed on the device every round - no .pt
cache, no torch.load. The previous /tmp/ar_cache/<op>_<sha>/reference.pt
path stalled multi-shape workers on the weights_only=False reload.
"""
import os, sys, json, traceback

# ar_vendored is bundled at the tarball root (same dir as this script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device_type = "{device}"
device_id = int(os.environ.get("DEVICE_ID", {device_id}))

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

# DSL-specific imports (triton / tilelang patches, etc.)
{dsl_imports}
{dsl_setup}

ATOL = {atol!r}
RTOL = {rtol!r}

try:
    from {kernel_file} import ModelNew
except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"correctness": False,
                      "error": f"import failed: cannot import name 'ModelNew' from '{kernel_file}' ({{e}})"}}))
    sys.exit(1)

def _to_cpu_list(out):
    if isinstance(out, torch.Tensor):
        return [out.detach().cpu()]
    if isinstance(out, (list, tuple)):
        return [o.detach().cpu() if hasattr(o, "detach") else o for o in out]
    return [out]

try:
    # input_groups.resolve duck-types between get_input_groups() (multi-shape)
    # and get_inputs() (legacy single-shape) and always returns List[List].
    # Single-shape ops collapse to a 1-element list.
    import {ref_file} as _ref_mod
    from {ref_file} import Model, get_init_inputs
    from input_groups import resolve as _resolve_groups, describe_case as _describe_case
    init_inputs = get_init_inputs()
    cases_cpu = _resolve_groups(_ref_mod)
    num_cases = len(cases_cpu)
    if num_cases == 0:
        raise RuntimeError("reference module returned 0 input cases")

    # --- Compute reference outputs per case ---
    # Previously cached to /tmp/ar_cache/<op>_<sha>/reference.pt via
    # torch.save and reloaded with torch.load(weights_only=False) on
    # subsequent rounds. On multi-shape ops the .pt is List[List[Tensor]]
    # of length num_cases * outputs-per-case, and the pickle graph rebuild
    # on NPU workers stalled for minutes (looked like a hang). Re-computing
    # every round is cheaper than the load + the triton JIT cache amortises
    # ModelNew across rounds anyway.
    model_ref = Model(*init_inputs).to(device).eval()
    out_ref_per_case = []
    with torch.no_grad():
        for case in cases_cpu:
            ref_inputs_dev = [x.to(device) if hasattr(x, "to") else x
                              for x in case]
            out_ref_raw = model_ref(*ref_inputs_dev)
            out_ref_per_case.append(_to_cpu_list(out_ref_raw))
            del ref_inputs_dev, out_ref_raw
    ref_source = "computed-worker"
    # Free the ref model before ModelNew allocates — BatchNorm-scale
    # tensors on HBM don't fit both at once.
    del model_ref
    if device_type == "npu":
        torch.npu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()

    # --- Run kernel on device for every case ---
    model_new = ModelNew(*init_inputs).to(device).eval()
    out_new_per_case = []
    with torch.no_grad():
        for case in cases_cpu:
            inputs_dev = [x.to(device) if hasattr(x, "to") else x for x in case]
            out_new_raw = model_new(*inputs_dev)
            out_new_per_case.append(_to_cpu_list(out_new_raw))
            del inputs_dev, out_new_raw

    # --- Compare (delegated to shared correctness module) ---
    # `correctness.py` is bundled into the tarball at root by _build_package;
    # both this generated script and the batch verifier call into the same
    # compare_outputs_per_case so single- and multi-shape semantics can't
    # drift.
    from correctness import compare_outputs_per_case
    cmp_result = compare_outputs_per_case(
        out_ref_per_case, out_new_per_case, ATOL, RTOL)

    for d in cmp_result["diagnostics"]:
        print(d, file=sys.stderr)

    # Surface FAILED_SHAPES line (failure_extractor.multi_shape_failed_shapes
    # parses this) so DIAGNOSE can name the offending case(s) directly.
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
                print("[verify] FAILED_SHAPES: " + "; ".join(shape_strs) + suffix,
                      file=sys.stderr)

    print(json.dumps({{
        "correctness": cmp_result["correctness"],
        "ref_source": ref_source,
        "atol": cmp_result["atol"], "rtol": cmp_result["rtol"],
        "diagnostics": cmp_result["diagnostics"],
        "num_cases": num_cases,
        "per_case": cmp_result.get("per_case", []),
        "failed_indices": cmp_result.get("failed_indices", []),
        "worst_idx": cmp_result.get("worst_idx"),
        "worst_max_abs_diff": cmp_result.get("worst_max_abs_diff"),
    }}))
    sys.exit(0 if cmp_result["correctness"] else 1)

except Exception as e:
    traceback.print_exc()
    print(json.dumps({{"correctness": False, "error": str(e)}}))
    sys.exit(1)
'''


# ---------------------------------------------------------------------------
# Profile script template
# ---------------------------------------------------------------------------

def _gen_profile_script(config: TaskConfig, device_id: int = 0,
                        mode: str = "generation",
                        warmup: int = 10, repeats: int = 100) -> str:
    """Generate profile_{op_name}_{mode}.py, adapter-driven.

    Structure:
      1. Outer skeleton (device setup, model instantiation) — uniform.
      2. Adapter-supplied `get_import_statements` + `get_special_setup_code`
         — DSL-specific imports and one-time patches (triton autotune,
         tilelang compile).
      3. Adapter-supplied `benchmark_impl` — the timing block. For
         triton_ascend this wraps `profiler_npu` (torch_npu.profiler); for
         triton_cuda / tilelang_cuda it's `triton.testing.do_bench`; for
         ascendc / cuda_c it's empty (those DSLs rely on msprof/nsys, routed
         at local_worker.py, not here).
      4. Fallback timing block — used when adapter's benchmark_impl is
         empty or crashes at runtime.

    mode='base' profiles Model (reference); mode='generation' profiles
    ModelNew (kernel).
    """
    import textwrap

    device = _detect_device_type(config)
    kernel_file = config.editable_files[0].replace(".py", "")
    ref_file = config.ref_file.replace(".py", "")

    if mode == "base":
        target_import = (f"from {ref_file} import Model as TargetModel\n"
                         f"from {ref_file} import get_init_inputs\n"
                         f"import {ref_file} as _ref_mod")
    else:
        target_import = (f"from {kernel_file} import ModelNew as TargetModel\n"
                         f"from {ref_file} import get_init_inputs\n"
                         f"import {ref_file} as _ref_mod")

    adapter = _get_dsl_adapter(config.dsl)
    dsl_imports = adapter.get_import_statements(config.framework or "torch")
    dsl_setup = adapter.get_special_setup_code() if hasattr(adapter, "get_special_setup_code") else ""

    # Adapter's benchmark_impl returns a code string indented 8-space for
    # upstream's kernel_verifier (which calls it inside a `for case` loop).
    # Dedent to column 0, then re-indent at 4-space for our function body.
    #
    # mode='base' force-routes through the adapter's `if backend=="ascend"`
    # else-branch (triton.testing.do_bench) by passing backend="". The
    # if-branch uses profiler_npu with dsl="triton_ascend" + a triton L2-
    # cache-clear kernel; running that against the PyTorch Model corrupts
    # NPU state and crashes the next aclnnArange with aivec error. The
    # else-branch (do_bench) works for both Triton kernels and PyTorch.
    # mode='generation' keeps the original backend so kernel timing uses
    # profiler_npu (more accurate, filters L2-cache-clear ops).
    benchmark_backend = "" if mode == "base" else (config.backend or "")
    raw = adapter.benchmark_impl(
        impl_func_name="TargetModel", inputs="inputs",
        warmup=warmup, runs=repeats,
        backend=benchmark_backend, op_name=config.name,
        case_idx=0, device_id=device_id,
    )
    if raw and raw.strip():
        benchmark_body = textwrap.indent(textwrap.dedent(raw), "    ")
        benchmark_source = f"adapter ({type(adapter).__name__})"
    else:
        # Adapter had no benchmark (ascendc / cuda_c): do_bench fallback so
        # the local subprocess still produces a timing. Real msprof/nsys
        # goes through local_worker.py when backend matches.
        benchmark_body = textwrap.indent(textwrap.dedent(f"""\
            import triton.testing
            def _bench():
                with torch.no_grad():
                    return impl_model(*inputs)
            execution_time_ms = triton.testing.do_bench(
                _bench, warmup={warmup}, rep={repeats}, return_mode="min")
            execution_time_us = execution_time_ms * 1000
            method = "triton_do_bench (adapter has no benchmark_impl)"
        """), "    ")
        benchmark_source = "fallback-do_bench"

    return f'''\
#!/usr/bin/env python3
"""Auto-generated {mode} profile script (dsl={config.dsl}, benchmark={benchmark_source}).

Multi-shape aware: iterates input_groups.resolve(ref_module) and times each
case independently. Single-shape refs collapse to a 1-element loop. The
top-level avg_time_us is the arithmetic mean of finite per-case timings;
crashed cases are recorded as inf in `per_shape` but don't poison the
aggregate.
"""
import os, sys, json, math, time, traceback

# ar_vendored is bundled at tarball root (same dir as this script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device_type = "{device}"
device_id = int(os.environ.get("DEVICE_ID", {device_id}))

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

# --- DSL-specific imports + patches (from adapter.get_import_statements) ---
{dsl_imports}
{dsl_setup}

{target_import}
from input_groups import resolve as _resolve_groups, describe_case as _describe_case

init_inputs = get_init_inputs()
cases_cpu = _resolve_groups(_ref_mod)
num_cases = len(cases_cpu)
if num_cases == 0:
    raise RuntimeError("reference module returned 0 input cases")

def _run_adapter_benchmark(impl_model, inputs):
    # Variables the adapter code assigns:
    #   execution_time_us / execution_time_ms / method
    execution_time_us = None
    execution_time_ms = None
    method = None
{benchmark_body}
    if execution_time_us is None and execution_time_ms is not None:
        execution_time_us = execution_time_ms * 1000
    if execution_time_ms is None and execution_time_us is not None:
        execution_time_ms = execution_time_us / 1000
    return execution_time_us, execution_time_ms, method


def _bench_one_case(impl_model, inputs):
    """Per-case timing: try adapter benchmark first, fall back to cpu timer
    on any exception so a single misbehaving case doesn't abort the whole
    profile pass."""
    try:
        avg_us, ms, method = _run_adapter_benchmark(impl_model, inputs)
        if avg_us is None or avg_us <= 0 or avg_us == float("inf"):
            raise RuntimeError(f"adapter benchmark returned invalid avg_us={{avg_us!r}}")
        return avg_us, ms, method
    except Exception as e:
        print(f"[profile {mode}] adapter benchmark failed: {{e}}; falling back to cpu timer",
              file=sys.stderr)
        traceback.print_exc()
        for _ in range({warmup}):
            with torch.no_grad():
                impl_model(*inputs)
        if device_type == "npu":
            torch.npu.synchronize()
        elif device_type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range({repeats}):
            if device_type == "npu":
                torch.npu.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                impl_model(*inputs)
            if device_type == "npu":
                torch.npu.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)
        avg_us = sum(times) / len(times)
        return avg_us, avg_us / 1000, "cpu_timer_fallback"


per_shape = []
for idx, case in enumerate(cases_cpu):
    impl_model = TargetModel(*init_inputs)
    if hasattr(impl_model, "to"):
        impl_model = impl_model.to(device)
    if hasattr(impl_model, "eval"):
        impl_model.eval()
    inputs = [x.to(device) if hasattr(x, "to") else x for x in case]
    try:
        avg_us, _ms, method = _bench_one_case(impl_model, inputs)
    except Exception as e:
        print(f"[profile {mode}] case {{idx}} timing failed: {{e}}",
              file=sys.stderr)
        traceback.print_exc()
        avg_us, method = float("inf"), "crashed"
    per_shape.append({{
        "idx": idx,
        "case_desc": _describe_case(case, impl_model),
        "avg_time_us": avg_us,
        "method": method,
    }})
    del impl_model, inputs
    if device_type == "npu":
        torch.npu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()

# Aggregate over finite per-case timings only - crashed cases are inf so
# the simple mean would otherwise be infected.
_finite = [s["avg_time_us"] for s in per_shape
           if isinstance(s["avg_time_us"], (int, float))
           and math.isfinite(s["avg_time_us"])]
avg_us = (sum(_finite) / len(_finite)) if _finite else float("inf")
execution_time_ms = (avg_us / 1000) if math.isfinite(avg_us) else None

result_data = {{
    "avg_time_us": avg_us,
    "execution_time_us": avg_us,
    "execution_time_ms": execution_time_ms,
    "warmup_times": {warmup},
    "run_times": {repeats},
    "num_cases": num_cases,
    "per_shape": per_shape,
}}
result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "{mode}_profile_result.json")
with open(result_file, "w") as f:
    json.dump(result_data, f, indent=2)
print(f"PROFILE_RESULT: {{avg_us}}")
'''


# ---------------------------------------------------------------------------
# Tarball assembly
# ---------------------------------------------------------------------------

# /tmp/ar_cache/<op>_<sha>/reference.pt used to live here. Removed when
# the multi-shape .pt reload was found to stall NPU workers — refs are
# now recomputed every round inside verify_<op>.py.


def _exclude_pycache(tarinfo: tarfile.TarInfo):
    """tarfile.add filter: skip __pycache__ / *.pyc / editor temp files."""
    base = os.path.basename(tarinfo.name)
    if base == "__pycache__" or base.endswith(".pyc") or base.startswith("."):
        return None
    return tarinfo


def _build_package(task_dir: str, config: TaskConfig, device_id: int = 0) -> bytes:
    """Build a tar.gz package with worker-compatible scripts.

    Generates and includes:
      - verify_{op_name}.py     (correctness check; recomputes ref per round)
      - profile_{op_name}_base.py (reference timing)
      - profile_{op_name}_generation.py (kernel timing)
      - kernel.py, reference.py, and any support .py files

    Reference outputs are never shipped in the tarball and never cached on
    the worker — verify_<op>.py recomputes Model on every round.
    """
    op_name = config.name
    buf = io.BytesIO()

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add editable files
        for fname in config.editable_files:
            fpath = os.path.join(task_dir, fname)
            if os.path.exists(fpath):
                tar.add(fpath, arcname=fname)

        # Add reference file (always set; default is "reference.py")
        ref_path = os.path.join(task_dir, config.ref_file)
        if os.path.exists(ref_path):
            tar.add(ref_path, arcname=config.ref_file)

        # Add any other .py files in task_dir root (support files)
        for f in os.listdir(task_dir):
            if (f.endswith(".py")
                    and f not in config.editable_files
                    and f != config.ref_file
                    and not f.startswith(".")):
                fpath = os.path.join(task_dir, f)
                if os.path.isfile(fpath):
                    tar.add(fpath, arcname=f)

        # Generate and add worker scripts
        def _add_script(name: str, content: str):
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        # Shared lib modules — imported by the generated verify / profile
        # scripts from the tarball root. Both worker subprocesses resolve
        # them from the same sys.path entry the generated scripts insert.
        # Source lives in scripts/utils/ post-restructure; bundled at the
        # tarball root so the worker-side `from input_groups import ...`
        # (no `utils.` prefix) keeps resolving.
        #   correctness.py  — compare_outputs[_per_case]
        #   input_groups.py — resolve() / describe_case() / num_cases()
        utils_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils",
        )
        for lib_name in ("correctness.py", "input_groups.py"):
            lib_src = os.path.join(utils_dir, lib_name)
            if os.path.isfile(lib_src):
                tar.add(lib_src, arcname=lib_name)

        _add_script(f"verify_{op_name}.py",
                     _gen_verify_script(config, device_id))
        _add_script(f"profile_{op_name}_base.py",
                     _gen_profile_script(config, device_id, mode="base"))
        _add_script(f"profile_{op_name}_generation.py",
                     _gen_profile_script(config, device_id, mode="generation"))

        # Bundle the vendored adapter/profiler tree at tarball root. Generated
        # verify/profile scripts prepend sys.path with their own dir, so
        # `import ar_vendored` resolves without any PYTHONPATH setup on the
        # worker side. ~150 KB compressed — acceptable overhead per eval.
        # task_config package lives one level inside scripts/, so go up two.
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vendored_root = os.path.join(script_dir, "ar_vendored")
        if os.path.isdir(vendored_root):
            tar.add(vendored_root, arcname="ar_vendored",
                    filter=_exclude_pycache)

    return buf.getvalue()
