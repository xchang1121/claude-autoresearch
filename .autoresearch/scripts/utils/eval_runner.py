"""Shared eval-runner helpers.

Both transports (worker HTTP server and local subprocess) extract the
same tar.gz produced by `package_builder._build_package`, export the
same env, write the same sidecar JSON, and read it back the same way.
They differ only in how they spawn the python subprocess (asyncio vs
plain subprocess). This module owns the parts they share so the two
sides can't drift on path-safety, env vars, or sidecar shape.

Public surface:

    safe_extract(package_bytes, dst)
        Reject absolute paths and `..` components; raise on traversal.
    env_for(device_id, override_base_us=None)
        Build the env dict the generated eval_<op>.py expects.
    read_sidecar(workdir)
        Parse <workdir>/eval_result.json or return None on miss.
    build_response(device_id, returncode, log, eval_result)
        Assemble the `{device_id, returncode, log, eval_result}` dict
        both transports return.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tarfile
from typing import Optional

logger = logging.getLogger(__name__)

EVAL_SIDECAR = "eval_result.json"


def safe_extract(package_bytes: bytes, dst: str) -> None:
    """Extract a tar.gz into `dst`, rejecting members that would escape it.

    Defends against path-traversal in malicious tarballs (`..` in names,
    absolute paths, symlinks pointing outside dst). Python 3.12+ has
    `tar.extractall(filter='data')` for the same purpose; this is the
    portable equivalent.
    """
    dst = os.path.abspath(dst)
    with tarfile.open(fileobj=io.BytesIO(package_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if name.startswith("/") or name.startswith(os.sep):
                raise RuntimeError(
                    f"tar member has absolute path: {name!r}")
            parts = name.replace("\\", "/").split("/")
            if any(p == ".." for p in parts):
                raise RuntimeError(
                    f"tar member contains parent reference: {name!r}")
            target = os.path.abspath(os.path.join(dst, name))
            if not (target == dst or target.startswith(dst + os.sep)):
                raise RuntimeError(
                    f"tar member would extract outside dst: {name!r}")
            # Symlinks / hardlinks: reject anything that doesn't resolve
            # under dst. linkname is a path relative to the link itself,
            # so we cannot pre-validate it from os.path.join(dst, name)
            # alone — easiest is to forbid links entirely. The package
            # builder doesn't emit any.
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"tar member is a link (not allowed): {name!r}")
        tar.extractall(dst)


def env_for(device_id: int,
            override_base_us: Optional[float] = None,
            override_base_per_shape_us: Optional[list] = None) -> dict:
    """Env vars the generated eval_<op>.py reads.

    DEVICE_ID selects which NPU/GPU to bind.
    AR_OVERRIDE_BASE_TIME_US, when set, tells the script to skip
    profile_base and reuse the caller-supplied aggregate baseline.
    AR_OVERRIDE_BASE_PER_SHAPE_US, when set (JSON list of floats),
    additionally populates `profile_base.per_shape` so speedup_vs_ref
    keeps computing as a geomean of per-shape ratios — sticky rounds
    aggregate the same way the SEED round did.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DEVICE_ID"] = str(device_id)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Windows libiomp5 double-load workaround (no-op on Linux).
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if override_base_us is not None and override_base_us > 0:
        env["AR_OVERRIDE_BASE_TIME_US"] = f"{override_base_us:.6f}"
    if (isinstance(override_base_per_shape_us, list)
            and override_base_per_shape_us):
        env["AR_OVERRIDE_BASE_PER_SHAPE_US"] = json.dumps(
            [float(v) for v in override_base_per_shape_us])
    return env


def read_sidecar(workdir: str) -> Optional[dict]:
    """Read <workdir>/eval_result.json. Returns None if the script
    crashed before writing it or the file is unparseable."""
    path = os.path.join(workdir, EVAL_SIDECAR)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("eval_runner: cannot parse %s: %s", path, e)
        return None


def build_response(device_id: int, returncode: int, log: str,
                   eval_result: Optional[dict]) -> dict:
    """Both transports return the same dict shape; centralise it here so
    `eval_client._assemble_eval_result` is transport-agnostic."""
    return {
        "device_id": device_id,
        "returncode": returncode,
        "log": log,
        "eval_result": eval_result,
    }
