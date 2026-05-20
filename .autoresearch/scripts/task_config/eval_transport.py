"""Raw eval transports.

Transports take an EvalRequest plus a package and return the canonical raw
response shape:

    {"device_id": int, "returncode": int, "log": str, "eval_result": dict|None}

They do not interpret metrics or mutate Progress.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Optional
from urllib.request import Request, urlopen

from .eval_request import EvalRequest

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


def normalize_worker_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = f"http://{url}"
    return url.rstrip("/")


def worker_status(worker_url: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        req = Request(f"{worker_url}/api/v1/status", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def select_worker(urls: list[str]) -> Optional[str]:
    """Prefer the reachable worker with the most free device slots."""
    best: Optional[tuple[int, int, str]] = None  # (-free, index, url)
    for idx, url in enumerate(urls):
        status = worker_status(url)
        if status is None:
            continue
        free = status.get("free", 0)
        if not isinstance(free, int):
            free = 0
        candidate = (-free, idx, url)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best is not None else None


def multipart_post(url: str, fields: dict, files: dict,
                   timeout: float) -> dict:
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


def run_remote_transport(worker_url: str, request: EvalRequest,
                         package: bytes) -> dict:
    """POST /api/v1/run."""
    fields: dict = {
        "task_id": request.task_id,
        "op_name": request.config.name,
        "timeout": str(int(request.timeout)),
    }
    if request.override_base_us is not None and request.override_base_us > 0:
        fields["override_base_us"] = f"{request.override_base_us:.6f}"
    if request.override_base_per_shape_us:
        fields["override_base_per_shape_us"] = json.dumps(
            [float(v) for v in request.override_base_per_shape_us])
    return multipart_post(
        f"{worker_url}/api/v1/run",
        fields=fields,
        files={"package": ("package.tar.gz", package, "application/gzip")},
        timeout=request.timeout + 30,
    )


def run_local_transport(request: EvalRequest, package: bytes,
                        device_id: int) -> dict:
    from utils.local_worker import local_eval as _local_eval  # type: ignore
    return _local_eval(
        package,
        request.config.name,
        request.timeout,
        device_id,
        override_base_us=request.override_base_us,
        override_base_per_shape_us=request.override_base_per_shape_us,
    )
