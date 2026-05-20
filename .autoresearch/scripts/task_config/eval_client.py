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
import sys
from typing import Optional

from .eval_assemble import assemble_eval_result as _assemble_eval_result
from .eval_request import (
    build_eval_request,
    count_ref_cases as _count_ref_cases,
    effective_timeout as _effective_timeout,
    override_base_from_progress as _override_base_from_progress,
)
from .eval_transport import (
    normalize_worker_url as _normalize_worker_url,
    run_local_transport,
    run_remote_transport,
    select_worker as _select_worker,
)
from .loader import TaskConfig
from .metric_policy import EvalOutcome, EvalResult
from .package_builder import _build_package

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _log_request(prefix: str, request) -> None:
    if request.num_cases > 1:
        print(f"[{prefix}] multi-shape: num_cases={request.num_cases}, "
              f"timeout={request.config.eval_timeout}s/shape x "
              f"{request.num_cases} = {request.timeout}s", file=sys.stderr)
    note = request.sticky_note()
    if note:
        print(f"[{prefix}] {note}", file=sys.stderr)


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

    request = build_eval_request(task_dir, config)
    _log_request("remote_eval", request)
    print(f"[remote_eval] worker={worker_url} task={request.task_id}",
          file=sys.stderr)

    try:
        package = _build_package(task_dir, config)
    except Exception as e:
        return EvalResult(outcome=EvalOutcome.INFRA_FAIL,
                          error=f"failed to build package: {e}")

    try:
        resp = run_remote_transport(worker_url, request, package)
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

    request = build_eval_request(task_dir, config)
    _log_request("local_eval", request)
    print(f"[local_eval] device={dev}", file=sys.stderr)

    resp = run_local_transport(request, package, dev)
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
