#!/usr/bin/env python3
"""
Stop hook: blocks Stop in every phase except FINISH.

The autoresearch loop is designed to run to completion (FINISH phase, or
max_rounds exhausted). Any earlier Stop abandons useful work — the seed
hasn't been baselined, the plan hasn't been edited through, the
DIAGNOSE artifact hasn't been turned into a plan, etc. Phase-specific
block messages tell Claude exactly what action to take instead of
stopping. FINISH is the only legal Stop point.

A no-task session (no AR_TASK_DIR / no progress.json) Stops normally —
the gate only applies once a task is active.
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status
from phase_machine import (
    INIT, GENERATE_REF, GENERATE_KERNEL, BASELINE, PLAN, EDIT,
    DIAGNOSE, REPLAN, FINISH,
    DIAGNOSE_ATTEMPTS_CAP, diagnose_state, get_task_dir,
    load_progress, read_phase, update_progress,
    DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
)


def _block_stop_with_reason(reason: str) -> None:
    """Tell Claude Code to refuse the stop and re-prompt the agent. Wire
    format follows Stop hook decision schema (`{decision: "block",
    reason: ...}`)."""
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


# Per-phase block messages. DIAGNOSE has its own dynamic branch below
# (different sub-states); every other non-FINISH phase gets a static
# "do X instead of stopping" message. Keys mirror phase_machine
# constants; the dict's KeyError on an unknown phase is intentional —
# we want a loud failure rather than a silent allow-Stop on a phase we
# forgot to wire.
_PHASE_BLOCK_MESSAGE = {
    INIT: (
        "[AR] Cannot Stop: phase=INIT — no task is active yet. "
        "Run `export AR_TASK_DIR=\"<task_dir>\"` to activate, then "
        "proceed with the phase the activation hook prints."
    ),
    GENERATE_REF: (
        "[AR] Cannot Stop: phase=GENERATE_REF — reference.py is not "
        "yet a runnable seed. Write reference.py at <task_dir>/"
        "reference.py with class Model + get_init_inputs() + one of "
        "get_inputs()/get_input_groups(); the post-Edit hook will "
        "advance phase."
    ),
    GENERATE_KERNEL: (
        "[AR] Cannot Stop: phase=GENERATE_KERNEL — kernel.py is not "
        "yet a runnable seed. Write the kernel into the editable "
        "file(s) declared by task.yaml; the post-Edit hook advances "
        "phase to BASELINE on validation pass."
    ),
    BASELINE: (
        "[AR] Cannot Stop: phase=BASELINE — seed hasn't been profiled. "
        "Run `python .autoresearch/scripts/baseline.py \"$AR_TASK_DIR\" "
        "[--worker-url ...]`; on correctness + valid seed_metric the "
        "hook advances phase to PLAN."
    ),
    PLAN: (
        "[AR] Cannot Stop: phase=PLAN — no plan written yet. Read "
        "task.yaml + reference.py + skills/, then Write "
        ".ar_state/plan_items.xml and run "
        "`python .autoresearch/scripts/create_plan.py \"$AR_TASK_DIR\"`. "
        "Hook advances phase to EDIT on validation pass."
    ),
    EDIT: (
        "[AR] Cannot Stop: phase=EDIT — the current ACTIVE plan item "
        "hasn't been settled. Make the edit, then run "
        "`python .autoresearch/scripts/pipeline.py \"$AR_TASK_DIR\"`. "
        "Loop until phase moves to REPLAN / DIAGNOSE / FINISH."
    ),
    REPLAN: (
        "[AR] Cannot Stop: phase=REPLAN — every plan item has settled "
        "but the budget isn't exhausted. Read history.jsonl, then "
        "Write a fresh plan_items.xml and run create_plan.py to keep "
        "iterating. Stop is only legal at FINISH."
    ),
}


def main():
    stop_reason = read_hook_input().get("stop_reason", "unknown")

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)

    progress = load_progress(task_dir)
    if progress is None:
        sys.exit(0)

    phase = read_phase(task_dir)

    # FINISH is the only legal Stop point. Everything else gets a
    # phase-specific "what to do next" message.
    if phase == DIAGNOSE:
        state = diagnose_state(task_dir, progress=progress)
        if state.action == DIAGNOSE_READY:
            _block_stop_with_reason(
                f"[AR] Cannot Stop: DIAGNOSE artifact is ready, but the "
                f"phase still needs a new plan. Write plan_items.xml from "
                f"diagnose_v{state.plan_version}.md, then run "
                f"create_plan.py."
            )
        elif state.action == DIAGNOSE_MANUAL_FALLBACK:
            _block_stop_with_reason(
                f"[AR] Cannot Stop: phase=DIAGNOSE requires a new plan. "
                f"Subagent attempts exhausted "
                f"({state.attempts}/{DIAGNOSE_ATTEMPTS_CAP}); switch to "
                f"manual planning — Write plan_items.xml directly using "
                f"history.jsonl + plan.md, then run create_plan.py."
            )
        else:
            _block_stop_with_reason(
                f"[AR] Cannot Stop: phase=DIAGNOSE requires a new plan. "
                f"Re-issue Task with subagent_type='ar-diagnosis'; only "
                f"create_plan.py advancing the phase out of DIAGNOSE "
                f"makes Stop legal. Attempts so far: "
                f"{state.attempts}/{DIAGNOSE_ATTEMPTS_CAP}."
            )

    if phase != FINISH:
        msg = _PHASE_BLOCK_MESSAGE.get(phase)
        if msg is None:
            # Defensive: unknown phase. Refuse Stop loudly so the missing
            # rule shows up in logs instead of silently allowing exit.
            msg = (f"[AR] Cannot Stop: phase={phase!r} has no Stop rule "
                   f"wired in hook_stop_save.py. This is a hook bug — "
                   f"fix the _PHASE_BLOCK_MESSAGE table. Stop is only "
                   f"legal at FINISH.")
        _block_stop_with_reason(msg)

    # phase == FINISH — allow Stop, stamp the reason + print summary.
    update_progress(
        task_dir,
        last_stop_reason=stop_reason,
        last_stop_time=datetime.now(timezone.utc).isoformat(),
    )

    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 0)
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")

    improv = ""
    if best is not None and baseline is not None and baseline != 0:
        pct = (baseline - best) / abs(baseline) * 100
        improv = f" ({pct:+.1f}%)"

    emit_status(f"\n[AR] Session stopped at FINISH: {stop_reason}")
    emit_status(f"[AR] Progress: {rounds}/{max_rounds} rounds | Best: {best}{improv}")
    emit_status(f"[AR] Resume with: /autoresearch --resume {task_dir}")


if __name__ == "__main__":
    main()
