# CATLASS Migration Progress

This file is the migration checkpoint. Update it before moving to a new
phase and after each commit/smoke test so context compaction cannot erase
the current state.

## Ground Rules

- Treat this file as the source of truth for migration state.
- Keep commits small and run smoke tests on `npu` after each code commit.
- Do not rely on conversation context to remember what has landed.
- Avoid complicated file transfer/compression flows; prefer git-native or
  direct small-file sync.

## Completed

1. Target triple config
   - Local commit: `ec2839e Add configurable target triple`
   - NPU commit: `e6b6a57 Add configurable target triple`
   - Smoke: passed on `npu` as part of Step 2/3 follow-up.

2. DSL adapter extension hooks
   - Local commit: `09e1f40 Add DSL adapter extension hooks`
   - NPU commit: `6bf9cf9 Add DSL adapter extension hooks`
   - Smoke: `adapter protocol smoke ok` on `npu`.

3. AscendC CATLASS adapter
   - Local commit: `f11b708 Add AscendC CATLASS adapter`
   - NPU commit: `2f25816 Add AscendC CATLASS adapter`
   - Smoke: `catlass adapter npu smoke ok` on `npu`.

## In Progress

4. Multi-file DSL task/scaffold/package flow
   - Goal: allow `ascendc_catlass` tasks to pass `catlass_op/` as the
     kernel handoff while keeping single-file DSL behavior unchanged.
   - Current focus:
     - `scripts/scaffold.py`
     - `scripts/task_config/loader.py`
     - `scripts/task_config/package_builder.py`
     - `scripts/utils/eval_runner.py`
     - `scripts/worker/server.py`
     - `scripts/batch/*`
   - Split plan:
     - 4a: scaffold + TaskConfig loading for CATLASS task directories.
     - 4b: packaging / worker / batch path resolution for multi-file DSLs.

### Step 4a: Scaffold + TaskConfig

- Local commit: `8450532 wip commit`
- NPU commit: `5a3ac7e Support CATLASS task scaffolding`
- Scope:
  - `scripts/scaffold.py` uses the selected DSL adapter for `--kernel`
    path interpretation and project-tree materialization.
  - `scripts/task_config/loader.py` parses `catlass.root` and
    `catlass.op_dir` / `catlass.catlass_op_dir`.
- Local checks:
  - `python -m compileall scripts\scaffold.py scripts\task_config\loader.py`
    passed on 2026-06-07.
  - `git diff --check -- scripts/scaffold.py scripts/task_config/loader.py MIGRATION_PROGRESS.md`
    passed on 2026-06-07.
  - Local no-hardware behavior smoke passed on 2026-06-07:
    `catlass scaffold local smoke ok`.
- NPU sync/smoke:
  - `python -m compileall scripts/scaffold.py scripts/task_config/loader.py`
    passed on `npu` on 2026-06-07.
  - `catlass scaffold npu smoke ok` passed on `npu` on 2026-06-07.

### Step 4b: Package / Worker / Batch Multi-File Flow

- Status: local implementation complete; NPU sync/smoke pending.
- Scope:
  - `scripts/task_config/package_builder.py`: package task-local
    directory entries safely when editable/data/extra files name a
    project subtree.
  - `scripts/batch/manifest.py`: resolve `kernel` as a directory for
    multi-file DSLs and add `kernel_module` for importable Python wrapper.
  - `scripts/batch/verify.py`: use `kernel_module` for tier-1/tier-2
    checks while batch run still passes `kernel`.
  - `scripts/utils/eval_runner.py` and `scripts/engine/eval_kernel.py`:
    pass `task_dir`, `catlass_root`, and `catlass_op_dir` through to
    the DSL adapter.
- Local checks:
  - `python -m compileall` passed for all touched 4b Python files on
    2026-06-07.
  - `git diff --check` passed for all touched 4b files on 2026-06-07.
  - Local no-hardware behavior smoke passed on 2026-06-07:
    `catlass 4b local smoke ok`.
- NPU sync/smoke:
  - Pending.

## Pending

5. Step 4b packaging / worker / batch path resolution for multi-file DSLs.
7. Multi-DSL CodeChecker/static checks.
8. End-to-end local + NPU verification and final migration notes.

## Latest Known Repo State

- Local `claude-autoresearch` head: `8450532 wip commit`
- NPU `/home/yyz/cxy/claude-autoresearch` head: `5a3ac7e`
- NPU tracked files were clean after Step 3; only runtime dirs were untracked:
  `.autoresearch/`, `.session_tasks/`, `.task_dir_pointers/`, `extra-info/`.
