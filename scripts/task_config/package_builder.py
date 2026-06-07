"""Build a tar.gz package for the remote worker.

The worker has its own claude-autoresearch checkout, so eval_kernel.py /
skill modules / utils ride along on the worker side — the package only
needs the per-task artifacts:

  - task.yaml                  (always)
  - ref_file                   (config.ref_file, default reference.py)
  - editable_files[*]          (kernel.py and any aux files/directories in
                                config.editable_files)
  - data_files[*]              (sibling files the ref reads at runtime —
                                NPUKernelBench-style `<op>.json` shape
                                lists, sglang-style `ref.pt` output
                                caches, auxiliary `.py` imports.
                                Declared in task.yaml `data_files:`.

The data_files field is REQUIRED for any ref that reads sibling
files at runtime — there's no reliable static way to detect such
deps (open() / torch.load() paths can be dynamic), so we ask the
task author to spell them out.
"""
from __future__ import annotations

import io
import os
import tarfile
from typing import Iterable

from .loader import TaskConfig, REF_FILE_DEFAULT


def _resolve_declared_path(task_dir: str, name: str) -> tuple[str, str]:
    """Return (absolute path, normalized archive name) for a task-local path."""
    if not name:
        raise ValueError("package path is empty")
    if (os.path.isabs(name)
            or (len(name) >= 2 and name[1] == ":")
            or name.startswith(("/", "\\"))):
        raise ValueError(f"package path {name!r} must be relative")

    normalized = os.path.normpath(name)
    parts = [p for p in normalized.replace("\\", "/").split("/")
             if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"package path {name!r} escapes task_dir")
    arcname = "/".join(parts)

    base = os.path.realpath(task_dir)
    src = os.path.abspath(os.path.join(task_dir, normalized))
    src_real = os.path.realpath(src)
    if not (src_real == base or src_real.startswith(base + os.sep)):
        raise ValueError(f"package path {name!r} escapes task_dir")
    return src, arcname


def _add_regular_file(tar: tarfile.TarFile, src: str, arcname: str,
                      seen: set) -> None:
    if arcname in seen:
        return
    if os.path.islink(src) or not os.path.isfile(src):
        raise ValueError(f"package entry {arcname!r} is not a regular file")
    tar.add(src, arcname=arcname, recursive=False)
    seen.add(arcname)


def _add_path(tar: tarfile.TarFile, task_dir: str, name: str,
              seen: set) -> None:
    """Add a declared task-local file or directory into the archive."""
    src, arcname = _resolve_declared_path(task_dir, name)
    if os.path.isfile(src):
        _add_regular_file(tar, src, arcname, seen)
        return
    if not os.path.isdir(src):
        raise ValueError(
            f"package path {name!r} not found in task_dir "
            f"({src!r}) -- check task.yaml paths.")
    if os.path.islink(src):
        raise ValueError(f"package directory {name!r} is a symlink")

    base = os.path.realpath(task_dir)
    for dirpath, dirnames, filenames in os.walk(src):
        for dirname in list(dirnames):
            full_dir = os.path.join(dirpath, dirname)
            if os.path.islink(full_dir):
                rel = os.path.relpath(full_dir, task_dir).replace(os.sep, "/")
                raise ValueError(f"package directory {rel!r} is a symlink")
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            raw = os.path.join(dirpath, filename)
            if os.path.islink(raw) or not os.path.isfile(raw):
                rel = os.path.relpath(raw, task_dir).replace(os.sep, "/")
                raise ValueError(
                    f"package entry {rel!r} is not a regular file")
            full = os.path.realpath(raw)
            if not (full == base or full.startswith(base + os.sep)):
                raise ValueError(
                    f"package path {filename!r} escapes task_dir")
            rel = os.path.relpath(raw, task_dir).replace(os.sep, "/")
            _add_regular_file(tar, raw, rel, seen)


def _add_file(tar: tarfile.TarFile, task_dir: str, name: str,
              seen: set) -> None:
    """Add task_dir/name into the archive at top-level `name`.

    Fails fast (ValueError) instead of silently skipping: every file we
    pack is declared in task.yaml (task.yaml / ref / editable / data_files)
    and must exist, else the client would ship an incomplete package that
    surfaces as a confusing ref/kernel failure on the worker.
    """
    if name in seen:
        return
    src = os.path.join(task_dir, name)
    if not os.path.isfile(src):
        raise ValueError(
            f"package file {name!r} not found in task_dir "
            f"({src!r}) — check task.yaml paths.")
    tar.add(src, arcname=name)
    seen.add(name)


def build_package(task_dir: str, config: TaskConfig,
                  extra_files: Iterable[str] = ()) -> bytes:
    """Pack task.yaml + ref + editable + data_files + extras into tar.gz.

    `extra_files` is an ad-hoc escape hatch for callers that know about
    additional sibling files outside task.yaml (rare). Normal usage is
    to list everything in task.yaml `data_files:` instead.
    """
    seen: set = set()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        _add_path(tar, task_dir, "task.yaml", seen)
        _add_path(tar, task_dir, config.ref_file or REF_FILE_DEFAULT, seen)
        for ef in config.editable_files or []:
            _add_path(tar, task_dir, ef, seen)
        for df in config.data_files or []:
            _add_path(tar, task_dir, df, seen)
        for ex in extra_files:
            if ex:
                _add_path(tar, task_dir, ex, seen)
    return buf.getvalue()
