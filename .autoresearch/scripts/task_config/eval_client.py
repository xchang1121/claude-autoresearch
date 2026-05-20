"""Eval client — two transports, one EvalResult assembler.

Public entry point: `run_eval(task_dir, config, device_id=None,
worker_urls=None) -> EvalResult`. Routing:

  1. `worker_urls` (CLI or task.yaml) non-empty → ship package to HTTP
     worker (`/api/v1/run`).
  2. Else `device_id` / `config.devices[0]` → direct local subprocess
     (`utils.local_worker.local_eval`).
  3. Else → EvalResult with INFRA_FAIL explaining what's missing.

Both transports run the SAME generated `eval_<op>.py` script in one
Python process and return the same dict:

    {"device_id", "returncode", "log", "eval_result"}

where `eval_result` is the sidecar JSON written by the script. We never
parse stdout-tail JSON — CANN's tiling warnings could (and did) corrupt
it.
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


# ---------------------------------------------------------------------------
# Per-shape eval_timeout scaling
# ---------------------------------------------------------------------------

def _count_ref_cases(task_dir: str, config: TaskConfig) -> int:
    """Probe the ref module locally and count input cases."""
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
        print(f"[eval_client] WARN: case-count probe failed "
              f"({type(e).__name__}: {e})", file=sys.stderr)
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


def _override_base_from_progress(task_dir: str) -> Optional[float]:
    """Sticky baseline: read the recorded baseline_metric so the eval
    script can skip profile_base. Only honoured when the prior
    baseline_init recorded baseline_source='ref'.
    """
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    try:
        from phase_machine import load_progress  # type: ignore
        progress = load_progress(task_dir) or {}
    except Exception:
        return None
    if progress.get("baseline_source") != "ref":
        return None
    v = progress.get("baseline_metric")
    if isinstance(v, (int, float)) and 0 < v < float("inf"):
        return float(v)
    return None


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
    """Prefer the reachable worker with the most free device slots.

    `free` is reported by /api/v1/status. Picking by max-free spreads
    parallel agents across workers instead of piling them onto the
    first one in the list (which would queue while peers sit idle).
    Tie-break by the URL's position in the caller-supplied list so
    deterministic behaviour for single-worker setups is unchanged.
    """
    best: Optional[tuple[int, int, str]] = None  # (-free, index, url)
    for idx, url in enumerate(urls):
        status = _worker_status(url)
        if status is None:
            continue
        free = status.get("free", 0)
        if not isinstance(free, int):
            free = 0
        candidate = (-free, idx, url)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best is not None else None


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
                op_name: str, timeout: float,
                override_base_us: Optional[float] = None) -> dict:
    """POST /api/v1/run. Returns {device_id, returncode, log, eval_result}."""
    fields: dict = {
        "task_id": task_id, "op_name": op_name,
        "timeout": str(int(timeout)),
    }
    if override_base_us is not None and override_base_us > 0:
        fields["override_base_us"] = f"{override_base_us:.6f}"
    return _multipart_post(
        f"{worker_url}/api/v1/run",
        fields=fields,
        files={"package": ("package.tar.gz", package, "application/gzip")},
        timeout=timeout + 30,
    )


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

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


def _assemble_eval_result(resp: dict) -> EvalResult:
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

    # Outcome — only two non-OK paths:
    #   error_source == "ref" or sidecar missing  → INFRA_FAIL (operator only).
    #   anything else failing                      → KERNEL_FAIL (PLAN-recoverable).
    # Pure transport failures (worker unreachable / no NPU) set INFRA_FAIL
    # in run_eval before we ever call _assemble.
    if error_source == "ref":
        outcome = EvalOutcome.INFRA_FAIL
    elif not eval_result:
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
            valid = [s for s in per_speedup if _finite(s)]
            if valid:
                metrics["speedup_vs_ref"] = math.exp(
                    sum(math.log(s) for s in valid) / len(valid))
                metrics["speedup_aggregation"] = "geomean"
            bad = [i for i, s in enumerate(per_speedup) if not _finite(s)]
            if bad:
                metrics["per_shape_speedup_bad_cases"] = bad
        descs = [s.get("case_desc")
                 for s in (profile_gen.get("per_shape") or [])
                 if isinstance(s, dict)]
        if any(descs):
            metrics["per_shape_descs"] = descs

    # Verify-side failure detail — surfaces which shape the kernel
    # mishandled so DIAGNOSE can pinpoint without scraping log text.
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
        if error_source == "ref":
            error = (f"reference.py failed: "
                     f"{verify.get('error') or '(no detail)'}")
        elif not eval_result:
            error = (f"eval script crashed before writing sidecar "
                     f"(rc={resp.get('returncode')})")
        else:
            error = "eval framework produced no per-shape data"
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


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_remote_eval(task_dir: str, config: TaskConfig,
                    worker_urls: list) -> EvalResult:
    """Ship the built package to one of the configured worker URLs."""
    urls = [_normalize_worker_url(u) for u in worker_urls]
    worker_url = _select_worker(urls)
    if worker_url is None:
        return EvalResult(
            outcome=EvalOutcome.INFRA_FAIL,
            error=f"no reachable worker from: {urls}",
        )

    task_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
    num_cases = _count_ref_cases(task_dir, config)
    eff_timeout = _effective_timeout(config, num_cases)
    if num_cases > 1:
        print(f"[remote_eval] multi-shape: num_cases={num_cases}, "
              f"timeout={config.eval_timeout}s/shape x {num_cases} = "
              f"{eff_timeout}s", file=sys.stderr)
    print(f"[remote_eval] worker={worker_url} task={task_id}",
          file=sys.stderr)

    try:
        package = _build_package(task_dir, config)
    except Exception as e:
        return EvalResult(outcome=EvalOutcome.INFRA_FAIL,
                          error=f"failed to build package: {e}")

    override = _override_base_from_progress(task_dir)
    if override is not None:
        print(f"[remote_eval] sticky baseline override={override:.2f} us",
              file=sys.stderr)

    try:
        resp = _worker_run(worker_url, package, task_id, config.name,
                           eff_timeout, override_base_us=override)
    except Exception as e:
        return EvalResult(outcome=EvalOutcome.INFRA_FAIL,
                          error=f"worker /run failed: {e}")

    return _assemble_eval_result(resp)


def run_local_eval(task_dir: str, config: TaskConfig,
                   device_id: Optional[int] = None) -> EvalResult:
    """Run the single eval_<op>.py in a local subprocess."""
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
        return EvalResult(outcome=EvalOutcome.INFRA_FAIL,
                          error=f"failed to build package: {e}")

    num_cases = _count_ref_cases(task_dir, config)
    eff_timeout = _effective_timeout(config, num_cases)
    if num_cases > 1:
        print(f"[local_eval] multi-shape: num_cases={num_cases}, "
              f"timeout={config.eval_timeout}s/shape x {num_cases} = "
              f"{eff_timeout}s", file=sys.stderr)
    print(f"[local_eval] device={dev}", file=sys.stderr)

    override = _override_base_from_progress(task_dir)
    if override is not None:
        print(f"[local_eval] sticky baseline override={override:.2f} us",
              file=sys.stderr)

    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from utils.local_worker import local_eval as _local_eval
    resp = _local_eval(package, config.name, eff_timeout, dev,
                       override_base_us=override)
    return _assemble_eval_result(resp)


def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None,
             worker_urls: Optional[list] = None) -> EvalResult:
    """Pick a transport and assemble the EvalResult.

      1. CLI `worker_urls` or `config.worker_urls` non-empty → remote.
      2. Else, with a device id available (arg or config) → local subprocess.
      3. Else → INFRA_FAIL explaining the missing input.
    """
    urls = worker_urls or config.worker_urls
    if urls:
        return run_remote_eval(task_dir, config, worker_urls=urls)

    if device_id is not None or config.devices:
        return run_local_eval(task_dir, config, device_id=device_id)

    return EvalResult(
        outcome=EvalOutcome.INFRA_FAIL,
        error=("no execution transport: pass --worker-url (HTTP worker) "
               "or --devices N / `devices: [N]` in task.yaml (local "
               "subprocess)."),
    )
