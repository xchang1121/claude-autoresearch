"""Eval client — two transports, one EvalResult assembler.

Public entry point: `run_eval(task_dir, config, device_id=None,
worker_urls=None) -> EvalResult`. Routing:

  1. `worker_urls` (CLI or task.yaml) non-empty → ship package to HTTP
     worker (`/api/v1/run`).
  2. Else `device_id` / `config.devices[0]` → direct local subprocess
     (`utils.local_worker.local_eval`).
  3. Else → EvalResult with FRAMEWORK_ERROR explaining what's missing.

Both transports return the same {verify, profile, device_id} dict so
`_assemble_eval_result` is transport-agnostic.
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
    (multi-shape) and get_inputs (single-shape collapsed to N=1). Any
    failure falls back to 1 (single-shape semantics, no scaling).
    """
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
        return max(len(_resolve(mod)), 1)
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
    """config.eval_timeout is the budget per shape; scale by case count."""
    return int(config.eval_timeout) * max(int(num_cases), 1)


# ---------------------------------------------------------------------------
# Worker URL discovery + HTTP transport
# ---------------------------------------------------------------------------

def _normalize_worker_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = f"http://{url}"
    return url.rstrip("/")


def _worker_status(worker_url: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        req = Request(f"{worker_url}/api/v1/status", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _select_worker(urls: list[str]) -> Optional[str]:
    for url in urls:
        if _worker_status(url) is not None:
            return url
    return None


def _multipart_post(url: str, fields: dict, files: dict, timeout: float) -> dict:
    """POST multipart/form-data using only stdlib. Returns parsed JSON."""
    boundary = f"----AutoResearch{uuid.uuid4().hex}"
    body = b""
    for name, value in fields.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                 f"{value}\r\n").encode("utf-8")
    for name, (filename, data, content_type) in files.items():
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{name}"; '
                 f'filename="{filename}"\r\n'
                 f"Content-Type: {content_type}\r\n\r\n").encode("utf-8")
        body += data if isinstance(data, bytes) else data.encode("utf-8")
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")

    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _worker_run(worker_url: str, package: bytes, task_id: str,
                op_name: str, timeout: float) -> dict:
    """POST /api/v1/run. Returns the worker's combined verify+profile dict."""
    return _multipart_post(
        f"{worker_url}/api/v1/run",
        fields={"task_id": task_id, "op_name": op_name,
                "timeout": str(int(timeout))},
        files={"package": ("package.tar.gz", package, "application/gzip")},
        timeout=timeout + 30,
    )


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _finite(v) -> bool:
    return isinstance(v, (int, float)) and 0 < v < float("inf")


def _parse_profile_artifact(artifacts: dict, key: str) -> Optional[dict]:
    raw = artifacts.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _per_shape_us(art: Optional[dict]) -> Optional[list]:
    if not art:
        return None
    ps = art.get("per_shape")
    if not isinstance(ps, list) or not ps:
        return None
    return [(s.get("avg_time_us") if isinstance(s, dict) else None) for s in ps]


def _assemble_eval_result(verify_resp: dict, profile_resp: dict) -> EvalResult:
    """Combine verify + profile responses into an EvalResult."""
    verify_log = verify_resp.get("log", "")
    verify_ok = bool(verify_resp.get("success", False))
    verify_json = _last_json_line(verify_log) or {}
    error_source = verify_json.get("error_source") if not verify_ok else None

    artifacts = profile_resp.get("artifacts") or {}
    gen_time = profile_resp.get("gen_time")
    base_time = profile_resp.get("base_time")
    gen_art = _parse_profile_artifact(artifacts, "generation_profile_result.json")
    base_art = _parse_profile_artifact(artifacts, "base_profile_result.json")

    per_gen = _per_shape_us(gen_art)
    per_base = _per_shape_us(base_art)

    gen_ok = _finite(gen_time)
    base_ok = _finite(base_time)
    crashed_shapes = (
        [i for i, t in enumerate(per_gen) if not _finite(t)]
        if per_gen is not None else []
    )

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

    metrics: dict = {}
    if gen_ok:
        metrics["latency_us"] = gen_time
    if base_ok:
        metrics["ref_latency_us"] = base_time
    if gen_ok and base_ok:
        metrics["speedup_vs_ref"] = base_time / gen_time

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
            # Geomean over valid per-shape ratios; N=1 collapses to the
            # plain ratio set above (override is harmless and keeps the
            # `speedup_aggregation` field populated uniformly).
            valid = [s for s in per_speedup if _finite(s)]
            if valid:
                metrics["speedup_vs_ref"] = math.exp(
                    sum(math.log(s) for s in valid) / len(valid))
                metrics["speedup_aggregation"] = "geomean"
            bad = [i for i, s in enumerate(per_speedup) if not _finite(s)]
            if bad:
                metrics["per_shape_speedup_bad_cases"] = bad
        descs = [s.get("case_desc")
                 for s in ((gen_art or {}).get("per_shape") or [])
                 if isinstance(s, dict)]
        if any(descs):
            metrics["per_shape_descs"] = descs

    # Verify-side failure detail (failed cases, worst idx) — surfaces which
    # shape the kernel mishandled so DIAGNOSE can pinpoint it.
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
        error = f"reference.py failed: {verify_json.get('error') or '(no detail)'}"
    elif outcome == EvalOutcome.KERNEL_PROFILE_CRASH:
        error = (f"kernel crashed during profile on {len(crashed_shapes)} of "
                 f"{len(per_gen)} shapes")
    elif outcome == EvalOutcome.FRAMEWORK_ERROR:
        error = "eval framework produced no per-shape data (timeout / crash / OOM)"
    else:
        error = "kernel output != reference"

    profile_log = profile_resp.get("log", "")
    return EvalResult(
        outcome=outcome,
        metrics=metrics,
        error=error,
        raw_output=(verify_log + "\n" + profile_log)[-4096:],
        error_source=error_source,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_remote_eval(task_dir: str, config: TaskConfig,
                    worker_urls: list) -> EvalResult:
    """Ship the built package to one of the configured worker URLs."""
    urls = [_normalize_worker_url(u) for u in worker_urls]
    worker_url = _select_worker(urls)
    if worker_url is None:
        return EvalResult(
            outcome=EvalOutcome.FRAMEWORK_ERROR,
            error=f"no reachable worker from: {urls}",
        )

    task_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
    num_cases = _count_ref_cases(task_dir, config)
    eff_timeout = _effective_timeout(config, num_cases)
    if num_cases > 1:
        print(f"[remote_eval] multi-shape: num_cases={num_cases}, "
              f"timeout={config.eval_timeout}s/shape x {num_cases} = "
              f"{eff_timeout}s", file=sys.stderr)
    print(f"[remote_eval] worker={worker_url} task={task_id}", file=sys.stderr)

    try:
        package = _build_package(task_dir, config)
    except Exception as e:
        return EvalResult(outcome=EvalOutcome.FRAMEWORK_ERROR,
                          error=f"failed to build package: {e}")

    try:
        resp = _worker_run(worker_url, package, task_id, config.name, eff_timeout)
    except Exception as e:
        return EvalResult(outcome=EvalOutcome.FRAMEWORK_ERROR,
                          error=f"worker /run failed: {e}")

    return _assemble_eval_result(resp.get("verify", {}), resp.get("profile", {}))


def run_local_eval(task_dir: str, config: TaskConfig,
                   device_id: Optional[int] = None) -> EvalResult:
    """Run verify + profile in a local subprocess (no HTTP)."""
    if device_id is not None:
        dev = int(device_id)
    elif config.devices:
        dev = int(config.devices[0])
    else:
        dev = 0
        print("[local_eval] WARNING: no device specified — defaulting to 0",
              file=sys.stderr)

    try:
        package = _build_package(task_dir, config)
    except Exception as e:
        return EvalResult(outcome=EvalOutcome.FRAMEWORK_ERROR,
                          error=f"failed to build package: {e}")

    num_cases = _count_ref_cases(task_dir, config)
    eff_timeout = _effective_timeout(config, num_cases)
    if num_cases > 1:
        print(f"[local_eval] multi-shape: num_cases={num_cases}, "
              f"timeout={config.eval_timeout}s/shape x {num_cases} = "
              f"{eff_timeout}s", file=sys.stderr)
    print(f"[local_eval] device={dev}", file=sys.stderr)

    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.local_worker import local_eval as _local_eval

    resp = _local_eval(package, config.name, eff_timeout, dev)
    return _assemble_eval_result(resp.get("verify", {}), resp.get("profile", {}))


def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None,
             worker_urls: Optional[list] = None) -> EvalResult:
    """Pick a transport and assemble the EvalResult.

      1. CLI `worker_urls` or `config.worker_urls` non-empty → remote.
      2. Else, with a device id available (arg or config) → local subprocess.
      3. Else → FRAMEWORK_ERROR explaining the missing input.
    """
    urls = worker_urls or config.worker_urls
    if urls:
        return run_remote_eval(task_dir, config, worker_urls=urls)

    if device_id is not None or config.devices:
        return run_local_eval(task_dir, config, device_id=device_id)

    return EvalResult(
        outcome=EvalOutcome.FRAMEWORK_ERROR,
        error=("no execution transport: pass --worker-url (HTTP worker) "
               "or --devices N / `devices: [N]` in task.yaml (local "
               "subprocess)."),
    )
