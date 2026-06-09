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
from .task_files import iter_declared_files


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
    for src, arcname in iter_declared_files(
            task_dir, [name], field_name="package path"):
        _add_regular_file(tar, src, arcname, seen)


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
