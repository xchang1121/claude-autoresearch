"""Manifest loading + progress JSON I/O for the batch runner.

Workspace convention:
    <batch_dir>/
        manifest.yaml | manifest.json    # user-authored
        batch_progress.json              # written here
        batch.log                        # written here
        <ref_dir>/<op_name>_ref.py
        <kernel_dir>/<op_name>_kernel.py

YAML support is optional (requires pyyaml). JSON works with stdlib only.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "batch_progress.json"
LOG_FILENAME = "batch.log"
VALID_MODES = ("ref-kernel",)
VALID_STATUSES = ("pending", "running", "done", "error", "skip")


class ManifestError(Exception):
    pass


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ManifestError(
            f"{path.name} is YAML but pyyaml is not installed. "
            f"Either `pip install pyyaml` or rename to manifest.json (JSON format)."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_manifest(batch_dir: Path) -> Path:
    """Return path to manifest.yaml or manifest.json, preferring YAML."""
    yaml_path = batch_dir / "manifest.yaml"
    if yaml_path.exists():
        return yaml_path
    json_path = batch_dir / "manifest.json"
    if json_path.exists():
        return json_path
    raise ManifestError(
        f"no manifest.yaml or manifest.json in {batch_dir}"
    )


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.suffix == ".yaml" or manifest_path.suffix == ".yml":
        data = _load_yaml(manifest_path)
    elif manifest_path.suffix == ".json":
        data = _load_json(manifest_path)
    else:
        raise ManifestError(f"unknown manifest extension: {manifest_path}")
    if not isinstance(data, dict):
        raise ManifestError(f"manifest root must be a mapping, got {type(data).__name__}")
    return data


def resolve_cases(batch_dir: Path, manifest: dict, mode: str) -> list[dict]:
    """Apply the <op_name>_{ref,kernel}.py naming convention and return resolved
    case dicts. Pre-flight check that every referenced file exists.

    Returns a list of dicts with keys: op_name, ref (abs path), kernel
    (abs path).
    """
    if mode not in VALID_MODES:
        raise ManifestError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    ops = manifest.get("ops")
    if not ops or not isinstance(ops, list):
        raise ManifestError("manifest.ops must be a non-empty list")

    ref_dir_raw = manifest.get("ref_dir")
    if not ref_dir_raw:
        raise ManifestError("manifest.ref_dir is required")
    ref_dir = (batch_dir / ref_dir_raw).resolve()
    if not ref_dir.is_dir():
        raise ManifestError(f"ref_dir not found: {ref_dir}")

    kernel_dir_raw = manifest.get("kernel_dir")
    if not kernel_dir_raw:
        raise ManifestError("kernel_dir is required")
    kernel_dir = (batch_dir / kernel_dir_raw).resolve()
    if not kernel_dir.is_dir():
        raise ManifestError(f"kernel_dir not found: {kernel_dir}")

    cases: list[dict] = []
    seen: set[str] = set()
    for entry in ops:
        if not isinstance(entry, str):
            raise ManifestError(
                f"manifest.ops entries must be strings (op names); got {entry!r}"
            )
        op_name = entry.strip()
        if not op_name:
            raise ManifestError("empty op_name in manifest.ops")
        if op_name in seen:
            raise ManifestError(f"duplicate op_name: {op_name}")
        seen.add(op_name)

        ref_path = ref_dir / f"{op_name}_ref.py"
        if not ref_path.is_file():
            raise ManifestError(f"{ref_path.relative_to(batch_dir)} not found")

        kernel_path = kernel_dir / f"{op_name}_kernel.py"
        if not kernel_path.is_file():
            raise ManifestError(
                f"{kernel_path.relative_to(batch_dir)} not found"
            )

        cases.append({
            "op_name": op_name,
            "ref": str(ref_path),
            "kernel": str(kernel_path),
        })

    return cases


def load_progress(batch_dir: Path) -> dict:
    path = batch_dir / PROGRESS_FILENAME
    if not path.exists():
        return {"batch_dir": str(batch_dir.resolve()), "cases": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(f"corrupt progress file at {path}: {e}")


def save_progress(batch_dir: Path, progress: dict) -> None:
    path = batch_dir / PROGRESS_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def merge_cases(progress: dict, resolved_cases: list[dict],
                mode: str, dsl: str) -> tuple[dict, list[str]]:
    """Merge freshly-resolved cases into the progress dict.

    The manifest is the source of truth: ops that exist in the old progress
    file but no longer in `resolved_cases` are dropped (so a user filtering
    or deleting ops via discover.py / manual manifest edits actually shrinks
    the queue, matching the docs' "ops list fully replaced" promise).

    New cases are inserted as pending; surviving cases keep their status but
    their ref/kernel paths refresh.

    Returns (progress, dropped_op_names).
    """
    progress["mode"] = mode
    progress["dsl"] = dsl
    old_cases = progress.get("cases", {})
    resolved_ops = {c["op_name"] for c in resolved_cases}
    dropped = sorted(op for op in old_cases if op not in resolved_ops)

    new_cases: dict = {}
    for c in resolved_cases:
        op = c["op_name"]
        existing = old_cases.get(op)
        if existing is None:
            new_cases[op] = {
                "op_name": op,
                "ref": c["ref"],
                "kernel": c["kernel"],
                "status": "pending",
                "task_dir": None,
                "started_at": None,
                "finished_at": None,
                "final_phase": None,
                "rc": None,
                "result": {
                    "baseline_metric": None,
                    "best_metric": None,
                    "rounds": None,
                    "consecutive_failures": None,
                },
                "note": "",
            }
        else:
            existing["ref"] = c["ref"]
            existing["kernel"] = c["kernel"]
            new_cases[op] = existing
    progress["cases"] = new_cases
    return progress, dropped


def update_case(batch_dir: Path, op_name: str, **fields: Any) -> None:
    """Atomic update of one case's fields. Reloads progress file on each call
    so concurrent edits (e.g. by hand) aren't clobbered."""
    progress = load_progress(batch_dir)
    case = progress.get("cases", {}).get(op_name)
    if case is None:
        raise ManifestError(f"unknown op_name: {op_name}")
    if "status" in fields and fields["status"] not in VALID_STATUSES:
        raise ManifestError(f"invalid status: {fields['status']}")
    case.update(fields)
    save_progress(batch_dir, progress)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_task_state(task_dir: Path) -> dict:
    """Pull the result block from <task_dir>/.ar_state/progress.json. Returns
    a dict with whichever fields could be read."""
    out: dict = {
        "baseline_metric": None,
        "best_metric": None,
        "rounds": None,
        "consecutive_failures": None,
    }
    for name in ("progress.json", ".progress.json"):
        pf = task_dir / ".ar_state" / name
        if not pf.exists():
            continue
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out["baseline_metric"] = data.get("baseline_metric")
        out["best_metric"] = data.get("best_metric")
        out["rounds"] = data.get("eval_rounds")
        out["consecutive_failures"] = data.get("consecutive_failures")
        break
    return out


def read_phase(task_dir: Path) -> str:
    pf = task_dir / ".ar_state" / ".phase"
    if pf.exists():
        try:
            return pf.read_text(encoding="utf-8").strip() or "UNKNOWN"
        except OSError:
            pass
    return "UNKNOWN"


def repo_root() -> Path:
    """The claude-autoresearch repo root, derived from this file's location.

    Layout: <repo>/.autoresearch/scripts/batch/manifest.py
    """
    return Path(__file__).resolve().parent.parent.parent.parent


_SCAFFOLD_RESULT_STATUSES = frozenset({"ok", "error"})


def parse_scaffold_result_line(line: str) -> Path | None:
    """Extract a task_dir from a single line of claude's stdout when it
    contains a `scaffold.py` result JSON — success OR failure.

    scaffold prints one of:
      {"task_dir": "<abs>", "status": "ok"}                       (baseline OK)
      {"task_dir": "<abs>", "status": "error", ...}               (kernel-side
                                                                   FAIL — task
                                                                   activates,
                                                                   PLAN takes
                                                                   over)
      {"task_dir": "<abs>", "status": "error", ...}               (REF_FAIL /
                                                                   FRAMEWORK_
                                                                   ERROR —
                                                                   stuck at
                                                                   BASELINE)
      {"status": "error", "error": "..."}                          (early
                                                                   pre-scaffold
                                                                   error — no
                                                                   task_dir
                                                                   yet)

    All three task_dir-carrying shapes name the dir scaffold created
    for THIS run. The case's final outcome (done / error) is decided
    later from `.phase` and `proc.returncode`, not from this status
    field, so binding on `status="error"` doesn't misreport completion
    — it just records WHICH dir this run produced.

    This is "process-identity-level" binding: claude's stdout is owned
    by THIS subprocess, so a JSON line we read here came from a
    `scaffold.py` invocation made by THIS run. Concurrent same-op
    batches that both land in `pick_new_task_dir`'s snapshot diff are
    disambiguated correctly via this parser, including the kernel-fail
    branch that the snapshot-diff fallback would have raced on by
    mtime.

    Returns None when the line isn't a scaffold result JSON shape, or
    `task_dir` is missing / not a string / points at a path that no
    longer exists on disk."""
    s = line.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    if d.get("status") not in _SCAFFOLD_RESULT_STATUSES:
        return None
    td = d.get("task_dir")
    if not isinstance(td, str):
        return None
    p = Path(td)
    return p if p.is_dir() else None


def snapshot_task_dirs() -> set[Path]:
    """Snapshot the current set of `ar_tasks/<dir>` entries.

    `run_one` calls this immediately before launching claude. After the
    subprocess exits we diff the post-snapshot against this set to find
    exactly the directories created during this run — robust against
    concurrent batches or manual sessions hitting the same op name.
    Previously we did `glob('<op>_*')` filtered by mtime, which could
    grab a sibling batch's task_dir that happened to be touched in the
    same window.
    """
    tasks_root = repo_root() / "ar_tasks"
    if not tasks_root.is_dir():
        return set()
    return {d for d in tasks_root.iterdir() if d.is_dir()}


def pick_new_task_dir(pre_snapshot: set[Path], op_name: str) -> Path | None:
    """Return the task_dir created since `pre_snapshot` matching
    `<op_name>_*`. Scaffold names task dirs `<op>_<ts>_<rand>` so a
    fresh entry under that pattern is this run's task. If multiple new
    dirs match (e.g. claude was retried mid-run), pick the most recent.
    Returns None when nothing new yet — caller can poll until claude
    has actually scaffolded the dir.
    """
    tasks_root = repo_root() / "ar_tasks"
    if not tasks_root.is_dir():
        return None
    try:
        current = {d for d in tasks_root.iterdir() if d.is_dir()}
    except OSError:
        return None
    new = current - pre_snapshot
    matches = [d for d in new if d.name.startswith(f"{op_name}_")]
    if not matches:
        return None
    matches.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return matches[0]


def find_running_case_task_dir(batch_dir: Path) -> Path | None:
    """Return the task_dir of the case currently `status="running"` in
    THIS batch's progress.json. Used by monitor / `monitor --dashboard`
    to scope "active" strictly to the batch we were pointed at — the
    repo-wide active-task pointer could belong to a sibling batch or a
    manual session sharing `ar_tasks/`. Falls back to None while the
    case is still scaffolding (task_dir not yet populated)."""
    progress = load_progress(batch_dir)
    if not progress:
        return None
    cases = progress.get("cases", {}) or {}
    running = [v for v in cases.values()
               if isinstance(v, dict) and v.get("status") == "running"]
    if not running:
        return None
    # Most-recently-started running case wins if the batch races.
    running.sort(key=lambda v: v.get("started_at", ""), reverse=True)
    td = running[0].get("task_dir")
    if not td:
        return None
    p = Path(td)
    return p if p.is_dir() else None
