"""utils/ contains stateless library modules imported by engine,
hooks, phase_machine, workflow, task_config, and batch.

No CLI entry points live here. Static kernel checks are implemented by
utils.code_checker.CodeChecker and invoked through engine.quick_check.
Nothing in this package mutates task state.

Import style invariant: modules inside utils/ should import siblings via
relative imports (``from .settings import ...``) unless they are deliberately
loaded as top-level scripts.
"""
