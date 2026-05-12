"""Round-N (post-EDIT) eval recorder.

Lifted from `keep_or_discard.py` so pipeline.py can call it as a library
(no subprocess + stdout JSON round-trip). The shell `keep_or_discard.py`
stays as a thin wrapper.

`record_round(task_dir, eval_data, description, plan_item) -> dict`
returns the same shape the CLI used to print as JSON: decision,
best_metric, eval_rounds, max_rounds, consecutive_failures.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    Progress, append_history, auto_rollback, load_progress, save_progress,
)
from task_config import (  # noqa: E402
    EvalResult, check_constraints, is_improvement, load_task_config,
)
from git_utils import commit_in_task  # noqa: E402


def record_round(task_dir: str, eval_data: dict,
                 description: str = "optimization round",
                 plan_item: Optional[str] = None) -> dict:
    """Single library entry point for one round of EDIT settlement.

    Decision flow: correctness gate -> constraint gate -> primary-metric
    presence -> improvement check. Mirrors keep_or_discard.py byte-for-
    byte; only the shell-vs-library wrapping differs."""
    config = load_task_config(task_dir)
    if config is None:
        return {"decision": "ERROR", "error": "task.yaml not found"}

    progress = load_progress(task_dir) or Progress()
    eval_result = EvalResult(
        correctness=eval_data.get("correctness", False),
        metrics=eval_data.get("metrics", {}),
        error=eval_data.get("error"),
    )

    round_num = progress.eval_rounds + 1
    decision = "DISCARD"
    commit_hash: Optional[str] = None
    new_failures = progress.consecutive_failures
    new_best_metric = progress.best_metric
    new_best_commit = progress.best_commit

    # `consecutive_failures` counts ONLY real failures (FAIL = kernel broken).
    # DISCARD (correct but not faster) is not a failure — it's a signal to
    # REPLAN with different ideas, which happens naturally when all plan
    # items settle. DIAGNOSE is reserved for broken kernels.
    if not eval_result.correctness:
        decision = "FAIL"
        new_failures = progress.consecutive_failures + 1
        print("[keep_or_discard] FAIL: correctness check failed",
              file=sys.stderr)
    else:
        violations = (check_constraints(eval_result, config.constraints)
                      if config.constraints else [])
        if violations:
            decision = "FAIL"
            new_failures = progress.consecutive_failures + 1
            print(f"[keep_or_discard] FAIL: constraint violations: "
                  f"{violations}", file=sys.stderr)
        else:
            cur = eval_result.metrics.get(config.primary_metric)
            best = progress.best_metric
            # NaN/missing-metric guard: correctness=True but the primary
            # metric came back missing, non-numeric, or NaN. Earlier we
            # took the `best is None: KEEP` branch and wrote that poisoned
            # value to progress.json, where every downstream is_improvement
            # then compared against it.
            if (not isinstance(cur, (int, float))
                    or cur != cur):  # NaN guard
                decision = "FAIL"
                new_failures = progress.consecutive_failures + 1
                print(f"[keep_or_discard] FAIL: correctness=PASS but primary "
                      f"metric '{config.primary_metric}' missing from "
                      f"{sorted(eval_result.metrics)}", file=sys.stderr)
            elif best is None:
                decision = "KEEP"
            else:
                best_er = EvalResult(correctness=True,
                                     metrics={config.primary_metric: best})
                if is_improvement(
                    eval_result, best_er,
                    metric=config.primary_metric,
                    lower_is_better=config.lower_is_better,
                    threshold=config.improvement_threshold,
                ):
                    decision = "KEEP"
                else:
                    decision = "DISCARD"

    if decision == "KEEP":
        metric_val = eval_result.metrics.get(config.primary_metric)
        metric_str = f"{config.primary_metric}={metric_val}"
        ok, info = commit_in_task(
            task_dir, config.editable_files,
            f"autoresearch: {description} | {metric_str}",
        )
        if not ok:
            # Couldn't preserve kernel state. Earlier we still wrote
            # best_metric=<this round's value> and best_commit=None,
            # which left progress.json pointing at a kernel that no
            # commit captured — rollback / resume / report all became
            # unreliable. Demote to FAIL: roll the working tree back,
            # bump consecutive_failures, leave best_* untouched.
            print(f"[keep_or_discard] git commit failed: {info}; demoting "
                  f"KEEP -> FAIL (kernel state not preserved)",
                  file=sys.stderr)
            decision = "FAIL"
            new_failures = progress.consecutive_failures + 1
            auto_rollback(task_dir)
        else:
            commit_hash = info if info != "noop" else None
            new_best_metric = metric_val
            new_best_commit = commit_hash
            new_failures = 0
            print(f"[keep_or_discard] KEEP: {metric_str} "
                  f"(commit: {commit_hash})", file=sys.stderr)
    else:
        auto_rollback(task_dir)
        print(f"[keep_or_discard] {decision}: rolled back editable files",
              file=sys.stderr)

    progress = progress.apply(
        eval_rounds=round_num,
        consecutive_failures=new_failures,
        best_metric=new_best_metric,
        best_commit=new_best_commit,
    )
    save_progress(task_dir, progress)

    hist: dict[str, Any] = {
        "round": round_num,
        "plan_item": plan_item,
        "description": description,
        "decision": decision,
        "metrics": eval_result.metrics,
        "correctness": eval_result.correctness,
        "error": eval_result.error,
        "commit": commit_hash,
    }
    # FAIL rows carry the structured failure context so DIAGNOSE has
    # something concrete to reason about. KEEP/DISCARD passed correctness
    # so their signals dict is empty noise — skip to keep history.jsonl
    # slim. raw_output_tail is truncated to 1500 chars; the actionable
    # detail is in `failure_signals`, the tail is just there for FAILs
    # whose pattern wasn't matched.
    if decision == "FAIL":
        sig = eval_data.get("failure_signals")
        if isinstance(sig, dict) and (sig.get("primary")
                                      or sig.get("python_error")
                                      or sig.get("signals")):
            hist["failure_signals"] = sig
        tail = (eval_data.get("raw_output_tail") or "").strip()
        if tail:
            hist["raw_output_tail"] = tail[-1500:]
    append_history(task_dir, hist)

    return {
        "decision": decision,
        "best_metric": progress.best_metric,
        "eval_rounds": round_num,
        "max_rounds": progress.max_rounds or config.max_rounds,
        "consecutive_failures": progress.consecutive_failures,
    }
