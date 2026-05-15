"""PhaseController — single owner of .ar_state/.phase writes.

Before this module, write_phase was called from many sites across
hook_post_bash, hook_post_edit, pipeline.py, and _baseline_init.
Each site embedded its own decision logic (read progress, inspect
files, branch on subagent state, ...). When phase rules drifted in one
place they didn't drift in the others, which is how baseline-vs-retry
behaviour got tangled.

PhaseController takes EVENTS as input (what just happened) and is the
only thing that decides the target phase + writes it. Callers no longer
do `write_phase(...)` directly; they do `PhaseController(td).on_xxx()`.
The set of events here is exhaustive for the current call sites, by
design: a new event has to land here, not in the caller.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    BASELINE, EDIT, FINISH, PLAN,
    compute_next_phase, compute_resume_phase, load_progress, read_phase,
    write_phase,
)


class PhaseController:
    def __init__(self, task_dir: str):
        self.task_dir = task_dir

    # ---- Activation -----------------------------------------------------
    def on_activation_resume(self) -> str:
        phase = compute_resume_phase(self.task_dir)
        return self._write(phase)

    def on_activation_ready(self) -> str:
        return self._write(BASELINE)

    # ---- Baseline -------------------------------------------------------
    def on_baseline_settled(self) -> str:
        """Single owner of the post-baseline phase transition. Called
        from workflow.run_baseline_init at the end of its body, so both
        the Bash-hook flow and any direct library caller (notebook
        re-runs, tests) go through the same decision rule.

        Advance phase based on baseline outcome:
          - ok / kernel_* → PLAN (seed PASS goes to optimize; seed FAIL
            goes to plan-and-rewrite)
          - framework_error → leave phase untouched (no per-shape data,
            agent should retry baseline)
          - ref_fail → leave phase untouched (reference is broken; the
            agent cannot fix it from EDIT, user must fix --ref source)
        Legacy progress (no outcome) maps via the old
        (seed_metric, baseline_correctness) rule."""
        progress = load_progress(self.task_dir)
        if progress is None:
            return read_phase(self.task_dir)
        outcome = getattr(progress, "baseline_outcome", None) or (
            "ok" if (progress.baseline_correctness
                     and progress.seed_metric is not None)
            else "kernel_verify_fail"
        )
        if outcome in ("framework_error", "ref_fail"):
            return read_phase(self.task_dir)
        return self._write(PLAN)

    # ---- Plan -----------------------------------------------------------
    def on_plan_validated(self) -> str:
        return self._write(EDIT)

    # ---- Round (post settle.py) ----------------------------------------
    def on_round_settled(self) -> str:
        """End of one EDIT round. Delegates to compute_next_phase, which
        reads eval_rounds / consecutive_failures / pending plan items
        from progress + plan.md."""
        next_phase = compute_next_phase(self.task_dir)
        return self._write(next_phase)

    # ---- internal -------------------------------------------------------
    def _write(self, phase: str) -> str:
        write_phase(self.task_dir, phase)
        return phase

    # ---- Convenience constants ----------------------------------------
    # Re-exported so callers don't need a second `from phase_machine
    # import FINISH` next to their PhaseController call.
    FINISH = FINISH
