"""Eval dispatcher + transports.

Public entry point: `run_eval(task_dir, config, device_id=None,
worker_urls=None) -> EvalResult`. It picks one of three paths:

  1. Explicit worker URLs (from CLI or task.yaml.worker.urls) → remote.
  2. `local_worker.detect_local_backend(config.backend)` reports the
     runtime is available → local subprocess.
  3. Otherwise → EvalResult with a clear "no execution backend" error.

Both transports unpack the same tar.gz from package_builder and converge
on `_assemble_eval_result`, so downstream sees identical EvalResult shapes.

What lives here:
  - Worker URL discovery (`_normalize_worker_url`, `_worker_status`,
    `_select_worker`).
  - HTTP client (`_multipart_post`, `_worker_acquire_device`,
    `_worker_release_device`, `_worker_verify`, `_worker_profile`).
  - Result assembly (`_assemble_eval_result`).
  - The three eval entry points (`run_remote_eval`, `run_local_eval`,
    `run_eval`).

What's NOT here:
  - Tarball assembly / DSL adapters / verify-script templates — those
    live in package_builder.
  - EvalResult / improvement / constraints — those live in metric_policy.
  - YAML parsing — those live in loader.
"""
import json
import math
import os
import sys
import uuid
from typing import Optional
from urllib.request import Request, urlopen

from .loader import TaskConfig
from .metric_policy import EvalOutcome, EvalResult
from .package_builder import _build_package

# Subprocess JSON-tail parser is the shared util — used to be duplicated
# here as `_last_json_line` and in phase_machine.state_store as
# `parse_last_json_line`.
_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from utils.json_io import parse_last_json_line as _last_json_line  # noqa: E402


# ---------------------------------------------------------------------------
# Per-shape eval_timeout scaling + case-count probe
# ---------------------------------------------------------------------------

def _count_ref_cases(task_dir: str, config: TaskConfig) -> int:
    """Probe the ref module locally and count input cases.

    Mirrors what the generated verify script does: import ref + run
    input_groups.resolve, which duck-types between get_input_groups
    (multi-shape, NPUKernelBench) and get_inputs (single-shape collapsed
    to N=1). Used purely to scale eval_timeout - any failure falls back
    to 1 (single-shape semantics, no scaling).

    Cost: O(materialise all input tensors) once per eval call. Bounded
    by case set size; ~100ms-1s for typical multi-shape benches and
    swamped by the verify/profile call that follows.
    """
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    ref_path = os.path.join(task_dir, config.ref_file)
    if not os.path.isfile(ref_path):
        return 1
    ref_dir = os.path.dirname(ref_path) or "."
    sys_path_added = ref_dir not in sys.path
    if sys_path_added:
        sys.path.insert(0, ref_dir)
    try:
        import importlib.util
        from utils.input_groups import resolve as _resolve  # type: ignore
        spec = importlib.util.spec_from_file_location(
            f"_count_ref_{config.name}", ref_path)
        if spec is None or spec.loader is None:
            return 1
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        n = len(_resolve(mod))
        return max(n, 1)
    except Exception as e:
        print(f"[eval_client] WARN: case-count probe failed ({type(e).__name__}: "
              f"{e}); eval_timeout will not scale per shape.", file=sys.stderr)
        return 1
    finally:
        if sys_path_added:
            try:
                sys.path.remove(ref_dir)
            except ValueError:
                pass


def _effective_timeout(config: TaskConfig, num_cases: int) -> int:
    """Per-shape semantics: total = config.eval_timeout * num_cases.

    config.eval_timeout is documented as the budget for ONE shape; scaling
    keeps single-shape behaviour identical (num_cases=1) while keeping a
    multi-shape verify from being killed at the first JIT compile.
    """
    return int(config.eval_timeout) * max(int(num_cases), 1)


# ---------------------------------------------------------------------------
# Worker URL discovery
# ---------------------------------------------------------------------------

def _normalize_worker_url(url: str) -> str:
    """Ensure URL has scheme. '127.0.0.1:9111' → 'http://127.0.0.1:9111'."""
    url = url.strip()
    if not url.startswith("http"):
        url = f"http://{url}"
    return url.rstrip("/")


def _worker_status(worker_url: str, timeout: float = 5.0) -> Optional[dict]:
    """GET /api/v1/status. Returns parsed JSON or None on failure."""
    url = f"{_normalize_worker_url(worker_url)}/api/v1/status"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _select_worker(worker_urls: list) -> Optional[str]:
    """Pick the first reachable worker. Simple round-robin fallback."""
    for url in worker_urls:
        url = _normalize_worker_url(url)
        status = _worker_status(url)
        if status is not None:
            return url
    return None


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def _multipart_post(url: str, fields: dict, files: dict, timeout: float) -> dict:
    """POST multipart/form-data using only stdlib.

    Args:
        url: Target URL
        fields: {name: value} for text fields
        files: {name: (filename, data_bytes, content_type)} for file fields
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response dict.
    """
    boundary = f"----AutoResearch{uuid.uuid4().hex}"
    body_parts = []

    for name, value in fields.items():
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        )

    for name, (filename, data, content_type) in files.items():
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        )
        body_parts.append(header)
        body_parts.append(data)
        body_parts.append(b"\r\n" if isinstance(data, bytes) else "\r\n")

    body_parts.append(f"--{boundary}--\r\n")

    # Assemble body as bytes
    body = b""
    for part in body_parts:
        if isinstance(part, str):
            body += part.encode("utf-8")
        else:
            body += part

    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _worker_acquire_device(worker_url: str, task_id: str, timeout: float = 30.0) -> Optional[int]:
    """POST /api/v1/acquire_device → device_id or None."""
    url = f"{worker_url}/api/v1/acquire_device"
    try:
        resp = _multipart_post(url, {"task_id": task_id}, {}, timeout)
        return resp.get("device_id")
    except Exception as e:
        print(f"[worker] acquire_device failed: {e}", file=sys.stderr)
        return None


def _worker_release_device(worker_url: str, task_id: str, device_id: int, timeout: float = 10.0):
    """POST /api/v1/release_device."""
    url = f"{worker_url}/api/v1/release_device"
    try:
        _multipart_post(url, {"task_id": task_id, "device_id": str(device_id)}, {}, timeout)
    except Exception as e:
        print(f"[worker] release_device failed: {e}", file=sys.stderr)


def _worker_verify(worker_url: str, package: bytes, task_id: str,
                   op_name: str, timeout: float) -> dict:
    """POST /api/v1/verify with tar.gz package. Returns parsed JSON."""
    url = f"{worker_url}/api/v1/verify"
    fields = {
        "task_id": task_id,
        "op_name": op_name,
        "timeout": str(int(timeout)),
    }
    files = {
        "package": ("package.tar.gz", package, "application/gzip"),
    }
    return _multipart_post(url, fields, files, timeout=timeout + 30)


def _worker_profile(worker_url: str, package: bytes, task_id: str,
                    op_name: str, timeout: float,
                    profile_settings: Optional[dict] = None) -> dict:
    """POST /api/v1/profile with tar.gz package. Returns parsed JSON."""
    url = f"{worker_url}/api/v1/profile"
    fields = {
        "task_id": task_id,
        "op_name": op_name,
    }
    if profile_settings:
        fields["profile_settings"] = json.dumps(profile_settings)
    files = {
        "package": ("package.tar.gz", package, "application/gzip"),
    }
    return _multipart_post(url, fields, files, timeout=timeout + 30)


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _finite(v) -> bool:
    return isinstance(v, (int, float)) and 0 < v < float("inf")


def _resolve_profile(resp: dict, key: str, artifact_name: str):
    """Return (top_level_time, parsed_artifact). Falls back to artifact
    `avg_time_us` when the transport didn't surface the field at top-level."""
    t = resp.get(key)
    artifacts = resp.get("artifacts") or {}
    art = None
    if artifact_name in artifacts:
        try:
            art = json.loads(artifacts[artifact_name])
        except (json.JSONDecodeError, TypeError):
            art = None
    if t is None and art is not None:
        t = art.get("avg_time_us")
    return t, art


def _per_shape_floats(art: Optional[dict]) -> Optional[list]:
    """List of `avg_time_us` values from a profile artifact, or None."""
    if not art:
        return None
    ps = art.get("per_shape")
    if not isinstance(ps, list) or not ps:
        return None
    return [(s.get("avg_time_us") if isinstance(s, dict) else None) for s in ps]


def _assemble_eval_result(verify_resp: dict, profile_resp: dict) -> EvalResult:
    """Combine verify + profile responses into an EvalResult.

    Shared by `run_remote_eval` (HTTP transport) and `run_local_eval`
    (subprocess transport). Both transports return the same dict shape:

        verify_resp:  {"success": bool, "log": str, "artifacts": {...}}
        profile_resp: {"gen_time": float|None, "base_time": float|None,
                       "log": str, "artifacts": {...}}

    so this function is the single place that decides correctness, picks
    metrics, and computes speedup. Keeping it transport-agnostic means
    fixing a parsing bug in one place fixes it for both.

    Multi-shape: when the profile artifact carries `per_shape` (each
    generated profile script always emits it now; single-shape ops produce
    a length-1 array), per-shape timings flow into metrics so DIAGNOSE /
    report can surface which shape regressed. Correctness is verify_ok AND
    no per-shape profile timing is non-finite.
    """
    verify_log = verify_resp.get("log", "")
    verify_ok = bool(verify_resp.get("success", False))
    verify_json = _last_json_line(verify_log) or {}
    # error_source is set by the verify script template (Phase 1/2 -> "ref",
    # Phase 3/4 -> "kernel"). Empty when verify succeeded.
    error_source = verify_json.get("error_source") if not verify_ok else None

    gen_time, gen_art = _resolve_profile(profile_resp, "gen_time",
                                         "generation_profile_result.json")
    base_time, base_art = _resolve_profile(profile_resp, "base_time",
                                           "base_profile_result.json")
    gen_ok = _finite(gen_time)
    base_ok = _finite(base_time)

    per_gen = _per_shape_floats(gen_art)
    per_base = _per_shape_floats(base_art)

    # `latency_us` aggregate is the mean of finite per-shape timings - so
    # gen_ok being True does NOT imply every shape finished. The strict
    # crashed-shape list is what gates correctness.
    crashed_shapes = (
        [i for i, t in enumerate(per_gen) if not _finite(t)]
        if per_gen is not None else []
    )

    # Outcome — see EvalOutcome docstring for definitions.
    # error_source="ref" supersedes the per_gen/verify decision: a broken
    # reference invalidates the whole eval regardless of whether the
    # profile happened to produce data.
    if error_source == "ref":
        outcome = EvalOutcome.REF_FAIL
    elif per_gen is None and not verify_ok:
        outcome = EvalOutcome.FRAMEWORK_ERROR
    elif not verify_ok:
        outcome = EvalOutcome.KERNEL_VERIFY_FAIL
    elif crashed_shapes:
        outcome = EvalOutcome.KERNEL_PROFILE_CRASH
    else:
        outcome = EvalOutcome.OK
    correctness = outcome == EvalOutcome.OK

    metrics: dict = {}

    # --- timing + speedup -------------------------------------------------
    # ref_latency_us and latency_us are INDEPENDENT measurements. A broken
    # kernel round (gen_time=None) must NOT drop the ref baseline — base
    # profile runs reference.py only, has nothing to do with kernel.py.
    # The previous `if gen_ok and base_ok: metrics["ref_latency_us"]=...`
    # path silently swallowed a valid ref reading whenever the kernel
    # crashed, leaving baseline_init to fall back to seed_fallback even
    # though we successfully measured the ref. Surface each independently;
    # gate only the speedup RATIO on both being valid.
    if gen_ok:
        metrics["latency_us"] = gen_time
    else:
        print(f"[eval] WARNING: no valid gen_time (got {gen_time!r}) — "
              f"kernel profile likely failed", file=sys.stderr)
    if base_ok:
        metrics["ref_latency_us"] = base_time
    else:
        print(f"[eval] WARNING: no valid base_time (got {base_time!r}) — "
              f"speedup vs reference unavailable", file=sys.stderr)
    if gen_ok and base_ok:
        metrics["speedup_vs_ref"] = base_time / gen_time
    elif profile_resp.get("speedup"):
        metrics["speedup_vs_ref"] = profile_resp["speedup"]

    # --- per-shape detail -------------------------------------------------
    # Single-shape ops collapse to N=1 under the same schema (`per_shape` of
    # length 1), so downstream readers see uniform keys regardless of shape
    # count and don't need single-vs-multi branches.
    if per_gen is not None:
        metrics["num_cases"] = len(per_gen)
        metrics["per_shape_gen_us"] = per_gen
        if crashed_shapes:
            metrics["profile_crashed_cases"] = crashed_shapes[:30]
            metrics["profile_crashed_count"] = len(crashed_shapes)
        if per_base is not None and len(per_base) == len(per_gen):
            metrics["per_shape_base_us"] = per_base
            per_speedup = [
                (b / g) if (_finite(b) and _finite(g)) else None
                for b, g in zip(per_base, per_gen)
            ]
            metrics["per_shape_speedup"] = per_speedup

            # Aggregate speedup: geomean of valid (>0, finite) per-shape
            # ratios. NaN / inf / non-positive shapes drop out of the
            # geomean but their indices are recorded. For N=1 the geomean
            # equals the plain ratio set above; the override is harmless
            # and keeps `speedup_aggregation` populated uniformly.
            valid_sp = [s for s in per_speedup if _finite(s)]
            if valid_sp:
                metrics["speedup_vs_ref"] = math.exp(
                    sum(math.log(s) for s in valid_sp) / len(valid_sp))
                metrics["speedup_aggregation"] = "geomean"
            bad_sp = [i for i, s in enumerate(per_speedup) if not _finite(s)]
            if bad_sp:
                metrics["per_shape_speedup_bad_cases"] = bad_sp
        descs = [s.get("case_desc")
                 for s in ((gen_art or {}).get("per_shape") or [])
                 if isinstance(s, dict)]
        if any(descs):
            metrics["per_shape_descs"] = descs

    # --- pass-through scalars from profile_resp ---------------------------
    _PROFILE_RESP_RESERVED = {"success", "log", "gen_time", "base_time",
                              "speedup", "artifacts", "task_id", "returncode"}
    for k, v in profile_resp.items():
        if k not in _PROFILE_RESP_RESERVED and isinstance(v, (int, float)):
            metrics[k] = v

    # --- verify failure detail (only on verify-side failure) --------------
    # The verify-script template emits failed_indices / worst_idx /
    # worst_max_abs_diff. Surfacing them lets DIAGNOSE / EDIT pinpoint
    # which shape the kernel is mis-handling without scraping stderr.
    if not verify_ok and verify_json:
        n_cases = verify_json.get("num_cases")
        if isinstance(n_cases, int) and n_cases >= 1:
            failed_idx = verify_json.get("failed_indices") or []
            if isinstance(failed_idx, list):
                metrics["correctness_failed_cases"] = failed_idx[:30]
                metrics["correctness_failed_count"] = len(failed_idx)
                metrics["correctness_total_cases"] = n_cases
            worst_idx = verify_json.get("worst_idx")
            if isinstance(worst_idx, int):
                metrics["correctness_worst_case"] = worst_idx
            worst_max = verify_json.get("worst_max_abs_diff")
            if isinstance(worst_max, (int, float)):
                metrics["correctness_worst_max_abs"] = worst_max

    if outcome == EvalOutcome.OK:
        error = None
    elif outcome == EvalOutcome.REF_FAIL:
        error = (f"reference.py failed: {verify_json.get('error') or '(no detail)'}")
    elif outcome == EvalOutcome.KERNEL_PROFILE_CRASH:
        error = (f"kernel crashed during profile on {len(crashed_shapes)} of "
                 f"{len(per_gen)} shapes")
    else:
        error = {
            EvalOutcome.FRAMEWORK_ERROR:
                "eval framework produced no per-shape data (timeout / crash / OOM)",
            EvalOutcome.KERNEL_VERIFY_FAIL: "kernel output != reference",
        }[outcome]

    profile_log = profile_resp.get("log", "")
    return EvalResult(
        outcome=outcome,
        metrics=metrics,
        error=error,
        raw_output=(verify_log + "\n" + profile_log)[-4096:],
        error_source=error_source,
    )


# ---------------------------------------------------------------------------
# Remote eval (HTTP transport)
# ---------------------------------------------------------------------------

def run_remote_eval(task_dir: str, config: TaskConfig,
                    worker_urls: Optional[list] = None) -> EvalResult:
    """Run eval via remote Worker Service.

    Flow:
      1. Select a reachable worker
      2. Acquire a device slot (the device id is baked into the package's
         generated scripts; the vendored worker doesn't override DEVICE_ID
         env at exec time, so the slot must be resolved BEFORE building).
      3. Build tar.gz package with the acquired device id baked in.
      4. POST /api/v1/verify → correctness check
      5. POST /api/v1/profile → latency metrics (always run, even on
         verify failure — the ref baseline is still useful)
      6. Release device, return EvalResult

    Compatible with the Worker Service API from ar_vendored.worker.server.
    """
    urls = worker_urls or config.worker_urls
    if not urls:
        return EvalResult(outcome=EvalOutcome.FRAMEWORK_ERROR, error="no worker_urls configured")

    urls = [_normalize_worker_url(u) for u in urls]

    # Select reachable worker
    worker_url = _select_worker(urls)
    if worker_url is None:
        return EvalResult(
            outcome=EvalOutcome.FRAMEWORK_ERROR,
            error=f"no reachable worker from: {urls}",
        )

    task_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
    # Per-shape eval_timeout scaling: probe the ref once and scale before
    # building the package so both transports pass the wall-clock budget
    # to the worker / subprocess that needs it.
    num_cases = _count_ref_cases(task_dir, config)
    eff_timeout = _effective_timeout(config, num_cases)
    if num_cases > 1:
        print(f"[remote_eval] multi-shape: num_cases={num_cases}, "
              f"timeout={config.eval_timeout}s/shape x {num_cases} cases = "
              f"{eff_timeout}s", file=sys.stderr)
    print(f"[remote_eval] Using worker: {worker_url}", file=sys.stderr)

    # Acquire device BEFORE building the package. The vendored worker
    # (`ar_vendored/core/worker/local_worker.py`) by contract trusts the
    # device_id baked into the generated verify/profile scripts and does
    # NOT override DEVICE_ID env at execution time. If we built the
    # package first and acquired the device second, the package would
    # bake in `_build_package`'s default (device_id=0) and the entire
    # multi-device device_pool would be ineffective — every task would
    # land on NPU 0 regardless of which slot was acquired.
    acquired_id = _worker_acquire_device(worker_url, task_id)
    if acquired_id is None:
        print("[remote_eval] WARNING: acquire_device returned None; falling "
              "back to device 0 baked into the package. The release call "
              "in the finally block will be skipped — no device to release.",
              file=sys.stderr)
        device_id = 0
    else:
        device_id = acquired_id

    # Build package — device_id is now the acquired slot (or 0 fallback),
    # baked into the generated verify/profile scripts so the worker runs
    # them on the right card.
    try:
        package = _build_package(task_dir, config, device_id=device_id)
    except Exception as e:
        if acquired_id is not None:
            _worker_release_device(worker_url, task_id, acquired_id)
        return EvalResult(outcome=EvalOutcome.FRAMEWORK_ERROR, error=f"failed to build package: {e}")

    try:
        # Step 1: Verify (correctness check)
        print(f"[remote_eval] Running verify...", file=sys.stderr)
        try:
            verify_resp = _worker_verify(
                worker_url, package, task_id, config.name, eff_timeout,
            )
        except Exception as e:
            return EvalResult(outcome=EvalOutcome.FRAMEWORK_ERROR, error=f"verify request failed: {e}")

        # Step 2: Profile — ALWAYS run it, even if verify failed. The profile
        # endpoint runs both profile_base.py (PyTorch reference, uses
        # reference.py only) and profile_generation.py (the seed/kernel,
        # needs kernel.py correct). A broken kernel still lets us measure the
        # ref baseline, which is the user-facing anchor for speedup.
        #
        # profile_settings is REQUIRED for the worker to route correctly:
        #   - triton_ascend / triton_cuda / pypto: script-direct path
        #     (worker just runs the scripts and reads back the
        #     *_profile_result.json each script wrote itself).
        #   - ascend / cuda backends (no recognised dsl): msprof / nsys
        #     path (worker wraps the scripts in a profiler and analyses
        #     op_summary).
        # Without dsl set, the worker falls back to the msprof path even
        # for triton_ascend kernels — and msprof's op-count analyser can't
        # match the script's own profiler_npu warmup/run counts, so
        # base_time silently comes back as None.
        # warmup_times/run_times match _gen_profile_script defaults so
        # the msprof analyser path (when used) has the right expectation.
        profile_settings = {
            "dsl": config.dsl or "",
            "backend": config.backend or "",
            "warmup_times": 10,
            "run_times": 100,
        }
        print(f"[remote_eval] Running profile...", file=sys.stderr)
        try:
            profile_resp = _worker_profile(
                worker_url, package, task_id, config.name, eff_timeout,
                profile_settings=profile_settings,
            )
        except Exception as e:
            # verify succeeded but profile transport failed — kernel ran but
            # we have no per-shape data, framework-side issue.
            return EvalResult(
                outcome=EvalOutcome.FRAMEWORK_ERROR,
                metrics={},
                error=f"verify={verify_resp.get('success', False)}; "
                      f"profile request failed: {e}",
                raw_output=verify_resp.get("log", "")[-2048:],
            )

        return _assemble_eval_result(verify_resp, profile_resp)

    finally:
        # Release only what we actually acquired. The fallback path sets
        # device_id=0 without calling acquire — releasing that would
        # decrement the pool's count for a slot we never reserved.
        if acquired_id is not None:
            _worker_release_device(worker_url, task_id, acquired_id)


# ---------------------------------------------------------------------------
# Local eval (subprocess transport, same generated scripts as remote)
# ---------------------------------------------------------------------------

def run_local_eval(task_dir: str, config: TaskConfig,
                   device_id: Optional[int] = None) -> EvalResult:
    """Run eval entirely in local subprocesses.

    Builds the same tar.gz package the remote worker would receive, then runs
    the auto-generated `verify_<op>.py` and `profile_<op>_*.py` scripts via
    `local_worker.local_verify` / `local_worker.local_profile`. Both
    transports converge on `_assemble_eval_result` so downstream code can't
    tell them apart.

    The pre-refactor implementation ran `config.eval_script` as a
    user-supplied entry point. That field was never set by scaffold and is
    no longer consulted; it stays in TaskConfig only for yaml back-compat.
    """
    # local_worker lives in scripts/utils/.
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.local_worker import local_verify, local_profile

    if device_id is not None:
        dev = int(device_id)
    elif config.devices:
        dev = int(config.devices[0])
    else:
        # No explicit device on the call AND task.yaml has no `devices`
        # field — fall back to NPU 0. We emit a loud warning instead of
        # raising because legitimate callers (notebooks, ad-hoc reruns)
        # do hit this path, but a SILENT fallback to 0 is what once let
        # `--devices 6` get rewritten to 0 and OOM on a busy NPU. The
        # warning surfaces the implicit choice so the user can spot it
        # before sinking minutes into a wrong-card eval.
        dev = 0
        print(
            "[local_eval] WARNING: no device specified (no device_id arg, "
            "no `devices` field in task.yaml). Defaulting to NPU 0. If "
            "another card is intended, pass --device-id N or set "
            "`devices: [N]` in task.yaml.",
            file=sys.stderr,
        )
    try:
        package = _build_package(task_dir, config, device_id=dev)
    except Exception as e:
        return EvalResult(outcome=EvalOutcome.FRAMEWORK_ERROR, error=f"failed to build package: {e}")

    # Per-shape eval_timeout scaling: probe the ref once and scale the
    # subprocess budget so multi-shape verify isn't killed mid-loop.
    num_cases = _count_ref_cases(task_dir, config)
    eff_timeout = _effective_timeout(config, num_cases)
    if num_cases > 1:
        print(f"[local_eval] multi-shape: num_cases={num_cases}, "
              f"timeout={config.eval_timeout}s/shape x {num_cases} cases = "
              f"{eff_timeout}s", file=sys.stderr)

    print(f"[local_eval] Running verify...", file=sys.stderr)
    verify_resp = local_verify(package, config.name, eff_timeout, dev)
    print(f"[local_eval] Running profile (dsl={config.dsl}, backend={config.backend})...",
          file=sys.stderr)
    profile_resp = local_profile(
        package, config.name, eff_timeout, dev,
        dsl=config.dsl, backend=config.backend,
    )
    return _assemble_eval_result(verify_resp, profile_resp)


# ---------------------------------------------------------------------------
# Unified eval entry point
# ---------------------------------------------------------------------------

def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None,
             worker_urls: Optional[list] = None) -> EvalResult:
    """Three-way routing:

      1. Explicit worker URLs (CLI or task.yaml) → remote.
      2. Else, if `local_worker.detect_local_backend(config.backend)`
         reports the runtime is available → local subprocess.
      3. Else → EvalResult with a clear "no execution backend" error so the
         user knows to either pass --worker-url or install the matching
         runtime (torch / torch_npu / CUDA driver).

    The local and remote branches share the same package and the same
    result-assembly function (`_assemble_eval_result`), so downstream code
    sees identical EvalResult shapes regardless of transport.
    """
    urls = worker_urls or config.worker_urls
    if urls:
        return run_remote_eval(task_dir, config, worker_urls=urls)

    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.local_worker import detect_local_backend
    backend_key = (config.backend or "cpu").lower()
    ok, why = detect_local_backend(backend_key)
    if ok:
        print(f"[eval] local backend ok ({backend_key}): {why}", file=sys.stderr)
        return run_local_eval(task_dir, config, device_id=device_id)

    return EvalResult(
        outcome=EvalOutcome.FRAMEWORK_ERROR,
        error=(
            f"no execution backend available for backend={backend_key!r}: "
            f"{why}. Either pass --worker-url to use a remote worker, or "
            f"install the matching runtime locally (torch + torch_npu for "
            f"ascend, torch + CUDA for cuda, torch alone for cpu)."
        ),
    )
