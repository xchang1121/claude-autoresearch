"""Task-local file path utilities.

This module owns the safety and normalization rules for paths declared in
task.yaml (``ref_file``, ``editable_files``, ``data_files``). Consumers that
read, package, or re-materialize task-local files should go through here
instead of each spelling its own path containment checks.
"""
from __future__ import annotations

import os
from typing import Iterable, Iterator


def normalize_task_relative_path(name: str, field_name: str = "task path") -> str:
    """Return a slash-normalized task-relative path, or raise ValueError.

    Rejects absolute paths, Windows drive-letter forms, and any ``..``
    segment. The returned value uses ``/`` separators so it can double as a
    tar archive name and as a stable key in aux-file dictionaries.
    """
    if not name:
        raise ValueError(f"{field_name} is empty")
    raw = str(name)
    if (os.path.isabs(raw)
            or (len(raw) >= 2 and raw[1] == ":")
            or raw.startswith(("/", "\\"))):
        raise ValueError(f"{field_name} {raw!r} must be task-relative")

    normalized = os.path.normpath(raw)
    parts = [p for p in normalized.replace("\\", "/").split("/")
             if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"{field_name} {raw!r} escapes task_dir")
    return "/".join(parts)


def is_task_relative_path(name: str) -> bool:
    """Return True iff ``name`` is a safe task-relative path."""
    try:
        normalize_task_relative_path(str(name))
        return True
    except ValueError:
        return False


def resolve_task_path(task_dir: str, name: str,
                      field_name: str = "task path") -> tuple[str, str]:
    """Return ``(absolute_source_path, normalized_relative_path)``.

    The realpath containment check catches symlink tricks and platform
    separator mismatches after lexical normalization.
    """
    arcname = normalize_task_relative_path(name, field_name)
    base = os.path.realpath(task_dir)
    src = os.path.abspath(os.path.join(task_dir, os.path.normpath(arcname)))
    src_real = os.path.realpath(src)
    if not (src_real == base or src_real.startswith(base + os.sep)):
        raise ValueError(f"{field_name} {name!r} escapes task_dir")
    return src, arcname


def iter_declared_files(task_dir: str, names: Iterable[str],
                        field_name: str = "task path",
                        allow_dirs: bool = True
                        ) -> Iterator[tuple[str, str]]:
    """Yield regular files declared by ``names`` as ``(src, relname)``.

    Directory entries are recursively expanded in deterministic order when
    ``allow_dirs`` is true. Symlinks are rejected; task packages and verifier
    aux bundles should contain concrete regular files only.
    """
    base = os.path.realpath(task_dir)
    seen: set[str] = set()

    for name in names:
        src, arcname = resolve_task_path(task_dir, name, field_name)
        if os.path.isfile(src):
            if os.path.islink(src):
                raise ValueError(f"{field_name} {arcname!r} is a symlink")
            if arcname not in seen:
                seen.add(arcname)
                yield src, arcname
            continue

        if not allow_dirs or not os.path.isdir(src):
            raise ValueError(
                f"{field_name} {name!r} not found in task_dir ({src!r})")
        if os.path.islink(src):
            raise ValueError(f"{field_name} directory {arcname!r} is a symlink")

        for dirpath, dirnames, filenames in os.walk(src):
            for dirname in list(dirnames):
                full_dir = os.path.join(dirpath, dirname)
                if os.path.islink(full_dir):
                    rel = os.path.relpath(full_dir, task_dir).replace(
                        os.sep, "/")
                    raise ValueError(
                        f"{field_name} directory {rel!r} is a symlink")
            dirnames.sort()
            filenames.sort()
            for filename in filenames:
                raw = os.path.join(dirpath, filename)
                if os.path.islink(raw) or not os.path.isfile(raw):
                    rel = os.path.relpath(raw, task_dir).replace(os.sep, "/")
                    raise ValueError(
                        f"{field_name} entry {rel!r} is not a regular file")
                full = os.path.realpath(raw)
                if not (full == base or full.startswith(base + os.sep)):
                    raise ValueError(
                        f"{field_name} entry {filename!r} escapes task_dir")
                rel = os.path.relpath(raw, task_dir).replace(os.sep, "/")
                if rel not in seen:
                    seen.add(rel)
                    yield raw, rel


def read_declared_files(task_dir: str, names: Iterable[str],
                        field_name: str = "task path",
                        allow_dirs: bool = True) -> dict[str, bytes]:
    """Read declared task-local files as ``{normalized_relname: bytes}``."""
    files: dict[str, bytes] = {}
    for src, rel in iter_declared_files(
            task_dir, names, field_name=field_name, allow_dirs=allow_dirs):
        with open(src, "rb") as f:
            files[rel] = f.read()
    return files
