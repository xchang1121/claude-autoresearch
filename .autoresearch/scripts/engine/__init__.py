"""engine/ — orchestration scripts the LLM and hooks invoke via subprocess.

Holds the blessed-script set: pipeline.py (main post-edit driver) plus the
single-purpose CLIs it spawns (quick_check, eval_wrapper, keep_or_discard,
settle), the BASELINE-phase entry (baseline.py + _baseline_init.py), the
PLAN-phase entry (create_plan.py), and the /autoresearch arg dispatcher
(parse_args.py).

These are CLIs, not a library — they are exec'd via Bash, not imported.
The package marker exists so cross-package imports (e.g. phase_machine
referencing engine.quick_check.check_editable_files) resolve cleanly.
"""
