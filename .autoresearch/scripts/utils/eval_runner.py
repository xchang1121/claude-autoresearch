"""Shared eval-runner helpers.

Both transports (worker HTTP server and local subprocess) extract the
same tar.gz produced by `package_builder._build_package`, export the
same env, write per-phase sidecar JSON, and merge them the same way.
They differ only in how they spawn the python subprocess (asyncio vs
plain subprocess). This module owns the parts they share so the two
sides can't drift on path-safety, env vars, or sidecar shape.

Two-pass execution (post fix for kernel-induced subprocess death):
each round runs the generated eval_<op>.py TWICE — once with
AR_EVAL_PHASE=ref_only (writes profile_base only) and once with
AR_EVAL_PHASE=kernel_only (writes verify + profile_gen). Per-phase
sidecars are merged into eval_result.json at the end. A kernel
SIGKILL / device hang in pass 2 cannot erase ref data that pass 1
already wrote. When override_base_us is supplied (sticky baseline),
pass 1 is skipped — ref doesn't need re-measuring this round.

Public surface:

    safe_extract(package_bytes, dst)
        Reject absolute paths and `..` components; raise on traversal.
    env_for(device_id, override_base_us=None, ..., phase=None,
            sidecar_path=None)
        Build the env dict the generated eval_<op>.py expects.
        `phase` ∈ {"ref_only", "kernel_only", None}; None ⇒ legacy
        "all" behavior. `sidecar_path` overrides AR_EVAL_SIDECAR so
        the two passes can write to distinct files.
    read_sidecar(workdir, name="eval_result.json")
        Parse <workdir>/<name> or return None on miss.
    merge_sidecars(ref, kernel)
        Combine per-phase sidecars into one canonical eval_result dict.
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
            override_base_per_shape_us: Optional[list] = None,
            phase: Optional[str] = None,
            sidecar_path: Optional[str] = None) -> dict:
    """Env vars the generated eval_<op>.py reads.

    DEVICE_ID selects which NPU/GPU to bind.
    AR_OVERRIDE_BASE_TIME_US, when set, tells the script to skip
    profile_base and reuse the caller-supplied aggregate baseline.
    AR_OVERRIDE_BASE_PER_SHAPE_US, when set (JSON list of floats),
    additionally populates `profile_base.per_shape` so speedup_vs_ref
    keeps computing as a geomean of per-shape ratios — sticky rounds
    aggregate the same way the SEED round did.
    AR_EVAL_PHASE ∈ {"ref_only", "kernel_only"} restricts which phases
    of the generated script run; omit / None for the legacy all-phases
    behavior (kept for ad-hoc reproducer use).
    AR_EVAL_SIDECAR overrides the canonical eval_result.json path so
    the two-pass runner can keep ref and kernel sidecars distinct.
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
    if phase in ("ref_only", "kernel_only"):
        env["AR_EVAL_PHASE"] = phase
    if sidecar_path:
        env["AR_EVAL_SIDECAR"] = sidecar_path
    return env


def read_sidecar(workdir: str, name: str = EVAL_SIDECAR) -> Optional[dict]:
    """Read <workdir>/<name>. Returns None if the script
    crashed before writing it or the file is unparseable."""
    path = os.path.join(workdir, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("eval_runner: cannot parse %s: %s", path, e)
        return None


def merge_sidecars(ref: Optional[dict],
                   kernel: Optional[dict]) -> Optional[dict]:
    """Combine per-phase sidecars into the canonical eval_result shape.

    `ref` (from AR_EVAL_PHASE=ref_only) contributes `profile_base`.
    `kernel` (from AR_EVAL_PHASE=kernel_only) contributes `verify`
    and `profile_gen`. `errors` is union; `ok` is AND of both sides.
    Missing sidecars (subprocess SIGKILL'd before write) are tolerated
    — downstream `_assemble_eval_result` treats None blocks as failure.
    """
    if ref is None and kernel is None:
        return None
    ref = ref or {}
    kernel = kernel or {}
    # verify: prefer kernel's (it's where verify actually runs); fall
    # back to ref's only if kernel didn't produce one (e.g. ref's
    # Phase A failed and wrote verify with error_source=ref).
    verify = kernel.get("verify") or ref.get("verify")
    return {
        "verify": verify,
        "profile_gen": kernel.get("profile_gen"),
        "profile_base": ref.get("profile_base"),
        "ok": bool(ref.get("ok", True)) and bool(kernel.get("ok", True)),
        "errors": (ref.get("errors") or []) + (kernel.get("errors") or []),
    }


def write_merged_sidecar(workdir: str, merged: Optional[dict]) -> None:
    """Persist the merged eval_result.json so ad-hoc inspectors / tests
    looking at the canonical path still see one combined artifact."""
    if merged is None:
        return
    path = os.path.join(workdir, EVAL_SIDECAR)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, default=str)
    except Exception as e:
        logger.warning("eval_runner: cannot write merged sidecar %s: %s",
                       path, e)


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
