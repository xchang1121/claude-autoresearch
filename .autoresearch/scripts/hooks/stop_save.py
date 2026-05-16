#!/usr/bin/env python3
"""Stop hook: allow Stop at FINISH or when the task is structurally stuck.

Default behaviour blocks Stop in every non-FINISH phase so the agent
can't bail out of the optimisation loop. Two outcomes break that rule —
the task is unrecoverable from inside the agent loop:

  - baseline_outcome == "ref_fail": the source --ref file is broken,
    only the user can fix it (guard_edit also blocks the agent from
    editing reference.py / workspace/<op>_ref.py).
  - baseline_outcome == "framework_error": eval framework crashed (worker
    disconnect, OOM-killed, timeout) — needs operator intervention, not
    a kernel rewrite.

Both leave phase pinned at BASELINE (on_baseline_settled refuses to
advance). Without this carve-out the agent would loop forever:
guard_edit blocks any attempt to "fix" ref / external files; baseline.py
just reproduces the same failure; Stop is the only sensible exit. The
emitted status text tells the user exactly what to fix and how to resume.
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, emit_status
from phase_machine import (
    BASELINE, FINISH, get_guidance, get_task_dir,
    load_progress, read_phase, update_progress,
)
from task_config.metric_policy import STUCK_BASELINE_OUTCOMES


def _block_stop_with_reason(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def _is_stuck(phase: str, progress) -> bool:
    """True iff the task can't be advanced from inside the agent loop.

    Triggers when baseline produced an outcome the agent can't fix
    (ref-side breakage or eval framework crash) and phase is still at
    BASELINE — i.e. on_baseline_settled refused to advance to PLAN and
    no kernel-side recovery path exists. PLAN/EDIT etc. are not "stuck"
    even if eval is failing; max_rounds will route them to FINISH.
    """
    if phase != BASELINE:
        return False
    outcome = progress.get("baseline_outcome")
    return outcome in STUCK_BASELINE_OUTCOMES


def main():
    stop_reason = read_hook_input().get("stop_reason", "unknown")

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)

    progress = load_progress(task_dir)
    if progress is None:
        sys.exit(0)

    phase = read_phase(task_dir)
    stuck = _is_stuck(phase, progress)
    if phase != FINISH and not stuck:
        _block_stop_with_reason(
            f"[AR] Cannot Stop at phase={phase}. Continue the loop:\n\n"
            f"{get_guidance(task_dir)}"
        )

    update_progress(
        task_dir,
        last_stop_reason=stop_reason,
        last_stop_time=datetime.now(timezone.utc).isoformat(),
    )

    if stuck:
        outcome = progress.get("baseline_outcome")
        err_src = progress.get("baseline_error_source") or outcome
        if outcome == "ref_fail":
            emit_status(
                f"\n[AR] Task aborted at BASELINE: reference is broken "
                f"(error_source={err_src})."
            )
            emit_status(
                f"[AR] The agent cannot fix this from EDIT — reference.py "
                f"is treated as ground truth and is not editable. Fix the "
                f"SOURCE file you passed via --ref and re-run /autoresearch "
                f"from scratch (do NOT --resume this task; the task_dir "
                f"only exists for inspection)."
            )
        else:  # framework_error
            emit_status(
                f"\n[AR] Task aborted at BASELINE: eval framework crashed."
            )
            emit_status(
                f"[AR] No per-shape data was produced — check worker "
                f"availability, eval.timeout, OOM, or device contention. "
                f"After fixing, /autoresearch --resume {task_dir} to retry "
                f"baseline."
            )
        return

    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 0)
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")

    improv = ""
    if best is not None and baseline is not None and baseline != 0:
        pct = (baseline - best) / abs(baseline) * 100
        improv = f" ({pct:+.1f}%)"

    emit_status(f"\n[AR] Session stopped at FINISH: {stop_reason}")
    emit_status(f"[AR] {rounds}/{max_rounds} rounds | Best: {best}{improv}")
    emit_status(f"[AR] Resume: /autoresearch --resume {task_dir}")


if __name__ == "__main__":
    main()
