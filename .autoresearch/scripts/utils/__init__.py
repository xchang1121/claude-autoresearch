"""utils/ — stateless library modules imported by engine/, hooks/,
phase_machine/, workflow/, task_config/, and batch/.

No CLI entry points live here; nothing in this package mutates state.
Splitting them out makes the dependency direction obvious: utils sits
at the bottom of the stack and never imports from any sibling package.
"""
