"""workflow/ — orchestration layer between hooks and state_store.

Owns the rules that turn "what just happened" into "next phase" and
record_round / baseline_init bodies that pipeline.py used to spawn as
subprocesses. The CLI shells (_baseline_init.py, keep_or_discard.py)
stay as thin wrappers around the library entry points here.
"""
from .transition import PhaseController
from .planning import PlanStore
from .baseline import run_baseline_init
from .round import record_round

__all__ = ["PhaseController", "PlanStore", "run_baseline_init", "record_round"]
