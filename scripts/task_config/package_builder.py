"""Build a tar.gz package for the remote worker.

The worker has its own AscendOpGenAgent checkout, so eval_kernel.py /
skill modules / utils ride along on the worker side — the package only
needs the per-task artifacts:

  - task.yaml                  (always)
  - ref_file                   (config.ref_file, default reference.py)
  - editable_files[*]          (kernel.py and any aux files in
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
        _add_file(tar, task_dir, "task.yaml", seen)
        _add_file(tar, task_dir, config.ref_file or REF_FILE_DEFAULT, seen)
        for ef in config.editable_files or []:
            _add_file(tar, task_dir, ef, seen)
        for df in config.data_files or []:
            _add_file(tar, task_dir, df, seen)
        for ex in extra_files:
            if ex:
                _add_file(tar, task_dir, ex, seen)
    return buf.getvalue()
