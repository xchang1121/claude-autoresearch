"""Shared git helpers for autoresearch lifecycle code.

`scaffold.py` (seed commit) and `workflow.round.record_round` (per-round
KEEP commit) both stage files inside a task's git repo. Historically each
path open-coded its own `subprocess.run(["git", ...])` sequence with
subtly different error-handling — scaffold used `check=True` (crash loud
on any git failure) while the earlier per-round commit helper swallowed
every error as a stderr WARNING and let phase advance anyway. The latter
caused a class of bugs where the seed kernel never made it into HEAD,
baseline ran fine on the in-tree code, phase walked forward, and the
problem only surfaced two phases later as a misleading "uncommitted
changes from previous round" block in the EDIT-phase git gate.

This module is the single canonical implementation of "stage these files
and commit". Both scaffold and the round writer now route through it.
The contract:

    ok, info = commit_in_task(task_dir, paths, message)
        ok   == True  → either created a commit (info=short hash) or there
                        was nothing to commit (info="noop"); both safe to
                        proceed.
        ok   == False → commit really failed (info = human-readable cause).
                        Caller decides whether to abort, hold phase, or
                        propagate.

Defensive `git config user.name/email` runs every call so we don't depend
on `.git/config` having been set by an earlier path. Repeated config sets
are idempotent.
"""
import os
import subprocess
import sys
from typing import Optional

_GIT_USER_NAME = "autoresearch"
_GIT_USER_EMAIL = "auto@research"


def _run(cmd: list, cwd: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def ensure_git_identity(task_dir: str) -> None:
    """Set local user.name/email if missing.

    Idempotent: writing identity that's already there is a no-op. Doing
    this in every commit path closes the "user.name=undefined on a fresh
    box" failure mode that hit seed commits on fresh CI workers.
    """
    _run(["git", "config", "user.name", _GIT_USER_NAME], cwd=task_dir)
    _run(["git", "config", "user.email", _GIT_USER_EMAIL], cwd=task_dir)


def commit_in_task(task_dir: str, paths, message: str) -> tuple:
    """Stage `paths` under `task_dir` and create a commit.

    `paths` are task-dir-relative ("kernel.py", "reference.py", ...) or the
    literal "." for an "add everything" first commit. Missing files are
    skipped silently — caller decides whether that's an error.

    Returns (True, "<short hash>") on success, (True, "noop") if there was
    nothing staged worth committing, (False, "<reason>") on any other
    failure.
    """
    try:
        # Identity bootstrap inside the try so a missing/invalid task_dir
        # surfaces as (False, "...") instead of leaking NotADirectoryError /
        # FileNotFoundError out of the function. Callers expect a tuple.
        ensure_git_identity(task_dir)

        for p in paths:
            if p != "." and not os.path.exists(os.path.join(task_dir, p)):
                continue
            r = _run(["git", "add", "--", p], cwd=task_dir)
            if r.returncode != 0:
                return False, f"git add {p!r} failed: {(r.stderr or r.stdout).strip()[-300:]}"

        r = _run(["git", "commit", "-m", message], cwd=task_dir)
        if r.returncode != 0:
            blob = (r.stdout or "") + (r.stderr or "")
            # "nothing to commit"            — clean tree, no diff staged
            # "no changes added to commit"   — modifications exist but
            #                                   none were `git add`-ed
            # "nothing added to commit but untracked files present"
            #                                — nothing staged AND there
            #                                   are unrelated untracked
            #                                   paths (e.g. .ar_state).
            # All three mean "nothing committed but not an error" —
            # treat as noop so KEEP doesn't get demoted to FAIL just
            # because some sidecar file is untracked.
            if ("nothing to commit" in blob
                    or "no changes added" in blob
                    or "nothing added to commit" in blob):
                return True, "noop"
            return False, blob.strip()[-400:] or "git commit returned non-zero with no output"

        h = _run(["git", "rev-parse", "--short", "HEAD"], cwd=task_dir)
        return True, h.stdout.strip() if h.returncode == 0 else "ok"

    except subprocess.TimeoutExpired:
        return False, "git operation timed out (>10s) — check for index lock or fs contention"
    except Exception as e:
        return False, f"unexpected error: {e}"


def is_working_tree_clean(task_dir: str) -> bool:
    """True iff `git status --porcelain` reports no changes in `task_dir`.

    Used by resume.py and the post-Bash hook to decide whether the
    `.edit_started` marker is stale (clean tree → nothing to resume →
    marker is leftover). Both call sites previously open-coded the same
    `git status --porcelain` subprocess; centralizing here keeps the
    "what does clean mean" decision in one place.

    On any git failure (no repo, timeout, exception) returns False
    — we'd rather leave a marker around than incorrectly declare clean.
    """
    try:
        r = _run(["git", "status", "--porcelain"], cwd=task_dir, timeout=5)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode != 0:
        return False
    return not r.stdout.strip()


def current_head_short(task_dir: str) -> Optional[str]:
    """Return the short hash of HEAD inside `task_dir`, or None if the
    rev-parse call fails / there is no HEAD. Used by round.py when a
    KEEP commit was a no-op (working tree already matched HEAD) — the
    kernel we just evaluated is exactly what HEAD points at, so
    best_commit must track HEAD instead of getting set to None.
    """
    try:
        r = _run(["git", "rev-parse", "--short", "HEAD"], cwd=task_dir)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    h = r.stdout.strip()
    return h or None


def auto_rollback(task_dir: str):
    """Revert editable_files to HEAD via `git checkout HEAD --`.

    Used by keep_or_discard (DISCARD/FAIL paths) and by the pipeline's
    quick-check failure branch. Reads `task.yaml.editable_files` to know
    which files to revert. Silent on git failures: rollback is a recovery
    path, swallowing here is appropriate (caller has already decided to
    abandon the round).
    """
    try:
        # __file__ is scripts/utils/git_utils.py — climb two to reach scripts/.
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from task_config import load_task_config
        config = load_task_config(task_dir)
        if config is None:
            return
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, capture_output=True, text=True,
        ).stdout.strip()
        for f in config.editable_files:
            fpath = os.path.relpath(os.path.join(task_dir, f), repo_root)
            subprocess.run(["git", "checkout", "HEAD", "--", fpath],
                           cwd=repo_root, capture_output=True)
    except Exception as e:
        print(f"[AR] Rollback failed: {e}", file=sys.stderr)
