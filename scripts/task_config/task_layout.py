"""AR task directory layout — single source of truth for the conventions
shared across the scaffolder, verifier, and eval transport.

Lives under task_config/ alongside the TaskConfig dataclass so the
"what's in a task_dir" contract has one neighbourhood. Other modules
(scaffold, eval_client, eval/kernel_verifier) import from here instead
of hardcoding filenames.

================================================================================
1. Per-task layout (under ``ar_tasks/<op>_<ts>_<hex6>/``)
================================================================================

  task_dir/
      reference.py          ← REF_FILE_DEFAULT, PyTorch Model + get_inputs
                              (single file, DSL-agnostic)
      <primary_editable>    ← editable_files[0], adapter-driven
                              (kernel.py for triton / tilelang / pypto /
                               catlass wrapper / ascendc meta-Python;
                               <op>_kernel.cpp for hypothetical pure-C++
                               DSLs, etc.)
      <project_subtree>/    ← multi-file DSLs only (e.g. catlass_op/),
                              listed in editable_files[1:] from
                              adapter.kernel_project_files
      task.yaml             ← TaskConfig (ref_file, editable_files, ...)
      program.md / SKILL.md ← agent instructions
      .git/                 ← per-task baseline + per-KEEP commit history
      .ar_state/            ← phase machine, plan, history, report

================================================================================
2. Per-batch layout (under ``<batch_dir>/`` for batch.run.py)
================================================================================

  Single-file DSLs:
      <batch_dir>/
          manifest.yaml | manifest.json
          batch_progress.json
          batch.log
          <ref_dir>/<op_name>_ref.py
          <kernel_dir>/<op_name>_kernel.py

  Multi-file DSLs (e.g. ascendc_catlass — adapter sets
  ``kernel_arg_is_directory = True`` + ``kernel_project_dir_name = "catlass_op"``):
      <batch_dir>/
          manifest.yaml | manifest.json
          <ref_dir>/<op_name>_ref.py
          <kernel_dir>/<op_name>/
              <primary_editable>        # adapter.primary_editable_template
              <kernel_project_dir>/     # adapter.kernel_project_dir_name
                  kernel/, include/, src/, CMakeLists.txt, ...

  Resolution of which case fires is owned by
  ``batch/manifest.py::resolve_kernel_paths_for_op`` — the per-DSL rule
  has a single owner; ``batch/discover.py`` delegates to it.

================================================================================
3. DSL-adapter knobs that drive the layout
================================================================================

The base :class:`DSLAdapter` (``eval/adapters/dsl/base.py``) exposes the
structural metadata; this module documents what each knob means for the
layout:

  primary_editable_template (str)
      Format-string filename for ``editable_files[0]`` — the file the
      LLM mainly works on. ``{op_name}`` slot supported but typically
      unused (single literal "kernel.py" works for every Python-style
      DSL). Pure C++ DSLs override to e.g. ``"{op_name}_kernel.cpp"``.

  kernel_arg_is_directory (bool)
      False (default) → ``--kernel`` is a Python file. The wrapper is
      the kernel; ``editable_files = [primary_editable]``.
      True → ``--kernel`` is a directory containing a sibling Python
      wrapper + a per-DSL project subtree. ``editable_files =
      [primary_editable] + kernel_project_files``.

  kernel_project_dir_name (Optional[str])
      Subdirectory name (relative to per-op root) holding the project
      subtree when ``kernel_arg_is_directory=True``. catlass uses
      ``"catlass_op"``.

  kernel_project_files (list)
      Path entries (files or directories) that belong to the DSL's
      kernel project besides the Python wrapper — sources, headers,
      build scripts. Listed in ``editable_files[1:]``. Single-file DSLs
      leave empty.

  static_check_via_python_ast (bool)
      True iff ``editable_files[0]`` is Python source CodeChecker should
      ``ast.parse``. False for ascendc (meta-Python that exec's into
      string vars but doesn't define ``ModelNew``) and any pure-C++
      adapter. NOT the same as "primary editable is .py" — ascendc's
      primary is .py but isn't ast-checkable.

  needs_binary_io (bool)
      True iff the DSL uses file-based tensor I/O (swft).

================================================================================
4. Consumer rules (how to read editable_files / ref_file)
================================================================================

Any consumer reading from a task_dir should follow these rules instead
of hardcoding ``"reference.py"`` / ``"kernel.py"``:

  Want the reference module path → ``task_dir / config.ref_file``
      ``config.ref_file`` defaults to ``REF_FILE_DEFAULT``.

  Want the LLM-edited primary file → ``task_dir / config.editable_files[0]``
      What the wrapper is named is DSL-driven. Don't assume Python; for
      "is this a parseable Python module?" check
      ``adapter.static_check_via_python_ast``.

  Want the full kernel project for handoff → iterate ``config.editable_files``
      Each entry may be a file or a directory; copy / tar / pass as
      ``--kernel`` accordingly.
"""


# Reference file written by every AR scaffolder. Per-task overridable via
# ``TaskConfig.ref_file`` in task.yaml; this constant is the on-disk
# default for the scaffolder, the verifier's
# ``_materialize_framework_bundle`` target, and any reader that needs to
# find the framework Model before task.yaml is loaded.
REF_FILE_DEFAULT = "reference.py"


def primary_editable_filename(adapter, op_name: str) -> str:
    """Resolve the DSL adapter's ``primary_editable_template`` into a
    concrete filename.

    Thin wrapper so consumers don't repeat the ``.format(op_name=...)``
    pattern (and so a future template that takes more slots can be
    threaded through one place). Returns whatever string the adapter
    declared; caller should not assume Python (.py).
    """
    return adapter.primary_editable_template.format(op_name=op_name)


def task_editable_files(adapter, op_name: str) -> list:
    """Build the ``task.yaml: editable_files`` list for a given DSL.

    Returns ``[primary] + adapter.kernel_project_files`` —
    position [0] is the LLM's primary edit target (per
    :func:`primary_editable_filename`); positions [1:] are project
    subtree entries declared by the adapter (catlass's ``catlass_op/``
    etc.). Single-file DSLs (triton / tilelang / pypto / ascendc) end up
    with a one-element list.
    """
    return [primary_editable_filename(adapter, op_name)] + list(
        adapter.kernel_project_files)


def pick_primary_editable(editable_files, default: str = "kernel.py") -> str:
    """Return ``editable_files[0]`` with a defensive fallback for the
    degraded case where ``editable_files`` failed to load.

    ``loader.py`` refuses an empty list at load time, so on the happy
    path this is just ``editable_files[0]``. The fallback only fires
    when ``editable_files`` is falsy (None / empty), which means the
    task is already broken; the placeholder ``"kernel.py"`` makes the
    eventual "file not found" error name something concrete instead of
    failing on IndexError.
    """
    if not editable_files:
        return default
    return editable_files[0]
