#!/usr/bin/env python3
"""Stop hook: block Stop unless phase == FINISH."""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, emit_status
from phase_machine import (
    FINISH, get_guidance, get_task_dir,
    load_progress, read_phase, update_progress,
)


def _block_stop_with_reason(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def main():
    stop_reason = read_hook_input().get("stop_reason", "unknown")

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)

    progress = load_progress(task_dir)
    if progress is None:
        sys.exit(0)

    phase = read_phase(task_dir)
    if phase != FINISH:
        _block_stop_with_reason(
            f"[AR] Cannot Stop at phase={phase}. Continue the loop:\n\n"
            f"{get_guidance(task_dir)}"
        )

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
    emit_status(f"[AR] {rounds}/{max_rounds} rounds | Best: {best}{improv}")
    emit_status(f"[AR] Resume: /autoresearch --resume {task_dir}")


if __name__ == "__main__":
    main()
