"""Round-0 (SEED) eval recorder.

Lifted from `_baseline_init.py` so the body is reusable from other
entry points (notebook re-runs, future scaffold one-shots) without
re-invoking the CLI shell. The shell `_baseline_init.py` stays as a
thin wrapper so existing `python _baseline_init.py <td> <eval_json>`
call sites keep working.

Exit codes: see `_EXIT_FOR` below. Phase transition is owned by
PhaseController.on_baseline_settled (dispatches on `baseline_outcome`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    Progress, append_history, load_progress, save_progress,
)
from task_config import EvalOutcome, load_task_config  # noqa: E402

from .transition import PhaseController


# Outcome → exit code. on_baseline_settled reads `baseline_outcome` from
# progress and dispatches phase independently — exit codes are kept for
# scaffold.py's "rc != 0 → surface error" check. REF_FAIL gets its own
# exit code (5) so scaffold can distinguish "reference broken; user must
# fix --ref source" from "kernel broken; PLAN will rewrite".
_EXIT_FOR = {
    EvalOutcome.OK: 0,
    EvalOutcome.REF_FAIL: 5,
    EvalOutcome.KERNEL_VERIFY_FAIL: 3,
    EvalOutcome.KERNEL_PROFILE_CRASH: 3,
    EvalOutcome.FRAMEWORK_ERROR: 4,
}


def _read_outcome(eval_data: dict) -> EvalOutcome:
    """Resolve outcome from eval_data, with legacy fallback when the
    `outcome` field isn't set (older wire format / external test producer
    pre-refactor)."""
    s = eval_data.get("outcome")
    if s is None:
        s = (EvalOutcome.OK.value if eval_data.get("correctness")
             else EvalOutcome.KERNEL_VERIFY_FAIL.value)
    try:
        return EvalOutcome(s)
    except ValueError:
        return EvalOutcome.FRAMEWORK_ERROR


def _valid(v) -> bool:
    return isinstance(v, (int, float)) and 0 < v < float("inf")


def _git_short_head(task_dir: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=task_dir, capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_baseline_init(task_dir: str, eval_json: str) -> int:
    """Library entry point. CLI shell `_baseline_init.py` calls this with
    sys.argv[2]. Returns process exit code. Side effects (progress,
    history, phase) are durable on disk before this returns."""
    config = load_task_config(task_dir)
    if config is None:
        print("[baseline] ERROR: task.yaml not found", file=sys.stderr)
        return 1

    eval_data = json.loads(eval_json)
    outcome = _read_outcome(eval_data)
    correctness = outcome == EvalOutcome.OK
    error_source = eval_data.get("error_source")  # "ref" | "kernel" | None
    metrics = eval_data.get("metrics", {})
    seed_val = metrics.get(config.primary_metric) if _valid(
        metrics.get(config.primary_metric)) else None
    ref_val = metrics.get("ref_latency_us") if _valid(
        metrics.get("ref_latency_us")) else None

    # Sticky baseline: pin first ref capture so SEED retries don't drift
    # the speedup anchor. Earlier versions overwrote baseline_metric on
    # every re-run, which made the "1.27x vs ref" message compare round-N
    # kernels against a different anchor than round-0 kernels.
    existing = load_progress(task_dir) or Progress()
    existing_baseline = existing.baseline_metric
    existing_source = existing.baseline_source
    existing_baseline_commit = existing.baseline_commit

    if _valid(existing_baseline) and existing_source == "ref":
        baseline_val = existing_baseline
        baseline_source = "ref"
        print(f"[baseline] sticky baseline = {existing_baseline} "
              f"(kept from first eval; this round's ref={ref_val} ignored)",
              file=sys.stderr)
    elif ref_val is not None:
        baseline_val = ref_val
        baseline_source = "ref"
        existing_baseline_commit = None
        print(f"[baseline] baseline = ref_latency_us = {ref_val} "
              f"(PyTorch reference)", file=sys.stderr)
    else:
        baseline_val = seed_val
        baseline_source = "seed_fallback"
        existing_baseline_commit = None
        print("[baseline] WARNING: ref_latency_us missing - baseline falls "
              "back to seed metric", file=sys.stderr)

    # initial_best stays None when seed didn't profile.
    # keep_or_discard's `if best is None: KEEP` then accepts the first
    # real kernel timing without comparing against a fake anchor. The
    # earlier fallback `seed_val if seed_val is not None else ref_val`
    # compared kernels against the PyTorch ref instead of seed, producing
    # the "fake 1.00x" baseline bug.
    initial_best = seed_val
    baseline_commit = _git_short_head(task_dir)
    baseline_commit_recorded = existing_baseline_commit or baseline_commit

    # Multi-shape: eval_client populates `num_cases` + `per_shape_descs` from
    # the profile artifact. Store them on the Progress dataclass so PLAN's
    # multi-shape note + DIAGNOSE's failed-shape block can look them up
    # without re-parsing history.jsonl. Single-shape ops leave both absent.
    n_cases = metrics.get("num_cases")
    descs = metrics.get("per_shape_descs")
    save_progress(task_dir, Progress(
        task=config.name,
        eval_rounds=0,
        max_rounds=config.max_rounds,
        best_metric=initial_best,
        best_commit=(baseline_commit if seed_val is not None
                     else "seed_profile_failed"),
        baseline_commit=baseline_commit_recorded,
        baseline_metric=baseline_val,
        baseline_source=baseline_source,
        baseline_outcome=outcome.value,
        baseline_correctness=correctness,
        baseline_error_source=error_source,
        seed_metric=seed_val,
        consecutive_failures=0,
        plan_version=0,
        status="no_plan",
        num_cases=(int(n_cases) if isinstance(n_cases, int)
                   and n_cases >= 1 else None),
        per_shape_descs=(
            [str(d) for d in descs if d]
            if isinstance(descs, list) and descs else None
        ),
    ), stamp=True)

    # Round 0 logs the SEED kernel's initial eval. `metrics.latency_us` is the
    # seed's timing; `metrics.ref_latency_us` (if present) is the PyTorch
    # baseline used as the speedup anchor.
    append_history(task_dir, {
        "round": 0,
        "description": "seed kernel initial eval",
        "decision": "SEED",
        "metrics": metrics,
        "outcome": outcome.value,
        "correctness": correctness,
        "commit": baseline_commit,
    })

    if outcome == EvalOutcome.REF_FAIL:
        # Reference is broken — the whole task is invalid. Scaffold reads
        # this exit code (5) and surfaces "fix --ref source" to the user
        # without activating the task. Don't advance phase here either.
        print(f"[baseline] REF_FAIL: {eval_data.get('error') or '(no detail)'}",
              file=sys.stderr)
        print(f"[baseline] reference.py is broken — fix the source file "
              f"passed via --ref and re-run /autoresearch.", file=sys.stderr)
        return _EXIT_FOR[outcome]

    if outcome != EvalOutcome.OK:
        print(f"[baseline] {outcome.value}: {eval_data.get('error') or '(no detail)'}",
              file=sys.stderr)
        return _EXIT_FOR[outcome]

    if seed_val is None:
        # Degenerate: outcome=OK but no primary metric (rare).
        print(f"[baseline] ERROR: outcome=OK but no valid "
              f"{config.primary_metric}; treating as kernel-no-timing.",
              file=sys.stderr)
        return 2

    PhaseController(task_dir).on_baseline_init_success()
    print(f"[baseline] Initialized: task={config.name}, "
          f"seed_{config.primary_metric}={seed_val}, "
          f"baseline({baseline_source})={baseline_val}, "
          f"commit={baseline_commit_recorded}", file=sys.stderr)
    print("[baseline] Phase -> PLAN", file=sys.stderr)
    return 0
