# CATLASS Migration Progress

This file is the migration checkpoint. Update it before moving to a new
phase and after each commit/smoke test so context compaction cannot erase
the current state.

## Ground Rules

- Treat this file as the source of truth for migration state.
- Keep commits small and run smoke tests on `npu` after each code commit.
- Do not rely on conversation context to remember what has landed.
- Sync changed files to `npu` with `scp` before remote smoke tests.

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

## Runtime Flow Completed

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

- Local commit: `2f34319 Support multi-file DSL task packaging`
- NPU commit: `4dac92a Support multi-file DSL task packaging`
- Status: complete.
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
  - `python -m compileall` passed for all touched 4b Python files on
    `npu` on 2026-06-07.
  - `catlass 4b npu smoke ok` passed on `npu` on 2026-06-07.

## Static Checks and Final Runtime Verification

5. Multi-DSL CodeChecker/static checks.
   - Local commit: `eb537d5 Add DSL-aware static CodeChecker`
   - NPU commit: `c5a0d37 Add DSL-aware static CodeChecker`
   - Status: complete.
   - Scope:
     - `scripts/utils/code_checker.py`: local `CodeChecker` compatibility
       wrapper with Triton delegation and CATLASS `torch.ops.catlass.*`
       checks.
     - `scripts/engine/quick_check.py`: use `CodeChecker` instead of
       direct `validate_triton_impl`.
     - `scripts/batch/verify.py`: use `CodeChecker` for tier-1 kernel
       static checks.
     - User-facing static-check text updated away from Triton-only wording.
   - Local checks:
     - `python -m compileall` / `py_compile` passed for touched Step 5
       files on 2026-06-07.
     - `git diff --check` passed for touched Step 5 files on 2026-06-07.
     - Local behavior smoke passed on 2026-06-07:
       `code checker local smoke ok`.
   - NPU sync/smoke:
     - Files synced via `scp`.
     - `python -m compileall` passed for touched Step 5 files on `npu`
       on 2026-06-07.
     - `code checker npu smoke ok` passed on `npu` on 2026-06-07.
6. End-to-end local + NPU verification and final migration notes.
   - Status: complete.
   - Local final checks:
     - `python -m compileall` passed for the migrated runtime files on
       2026-06-07.
     - Legacy package/repo import/path scan over `scripts --glob "*.py"`
       found no runtime imports; remaining hits were comments/docstrings.
   - NPU final checks:
     - `catlass final npu integration smoke ok` passed on `npu` on
       2026-06-07. This smoke exercised CATLASS scaffold, CMake patching,
       directory packaging, quick_check, batch manifest resolution, and
       batch tier-1 validation.

## Overall Status

- CATLASS runtime migration is complete through scaffold, packaging/worker/batch
  path handling, and DSL-aware static checks.
- Standalone docs/skills cleanup and CATLASS skill sync are complete.
- Per-code-commit NPU smoke tests passed for Steps 2, 3, 4a, 4b, and 5;
  the Step 7 cleanup smoke also passed on `npu`.

## Completed Cleanup

7. Standalone docs/skills cleanup.
   - Status: complete.
   - Scope:
     - Remove stale legacy package/repo references from user-facing docs,
       templates, comments, and logs.
     - Sync local CATLASS skills to `npu` with `scp`.
     - Re-run cheap local checks and NPU smoke/static checks after sync.
   - Local checks:
     - `python -m compileall` passed for touched Step 7 Python files on
       2026-06-07.
     - `git diff --check` passed on 2026-06-07.
     - Full repo legacy package/repo reference scan returned no matches on
       2026-06-07.
   - NPU sync/checks completed:
     - Changed docs/code files were synced to `/home/yyz/cxy/claude-autoresearch`
       via `scp`.
     - `skills/ascendc-catlass/` was synced via `scp`; the four CATLASS
       skill files are present on `npu`.
     - `python -m compileall` passed on the touched Step 7 Python files on
       `npu` on 2026-06-07.
     - Legacy package/repo reference scans passed on tracked files and
       `skills/ascendc-catlass/` on `npu` on 2026-06-07.
     - `scripts/ar_cli.py` on `npu` was updated only for the stale
       checkout comment after reverting an overly broad scp attempt.
   - NPU smoke:
     - `.step7_catlass_smoke.py` was created, fixed for the current
       `CodeChecker.check` 3-tuple return and YAML dedent, synced to `npu`
       via `scp`, and executed successfully on 2026-06-07:
       `catlass step7 npu smoke ok`.
     - Temporary smoke harness was removed after the run.

8. ar_cli worker/status simplification.
   - Status: complete.
   - Scope:
     - Keep the public CLI surface focused on `worker`.
     - Fold local/remote diagnostics into the `worker --status` failure
       path instead of exposing a separate diagnostic command.
     - Keep `repo_path` as the single remote checkout pointer; remote
       invocation is `python scripts/ar_cli.py worker ...`.
     - Configure `npu.repo_path` for the remote claude-autoresearch
       checkout.
     - Update `AUTORESEARCH.md` to describe the simpler worker flow.
   - Local checks:
     - `python -m py_compile scripts/ar_cli.py` passed on 2026-06-07.
     - `python scripts/ar_cli.py --help` shows only the `worker` subcommand.
     - `python scripts/ar_cli.py worker --help` documents status diagnostics.
     - Local `worker --status --backend cpu --dsl ascendc_catlass` on an
       unused port returned unreachable diagnostics as expected.
     - Remote `worker --remote-host npu --status --backend ascend
       --dsl ascendc_catlass` on an unused port returned remote diagnostics
       as expected.
   - NPU sync/checks:
     - Synced `scripts/ar_cli.py`, `config.yaml`, `AUTORESEARCH.md`, and
       `MIGRATION_PROGRESS.md` via `scp`.
     - Removed an accidental untracked root-level `ar_cli.py` copy after
       verifying it was not tracked.
     - `git diff --check -- scripts/ar_cli.py config.yaml AUTORESEARCH.md
       MIGRATION_PROGRESS.md` passed on NPU.
     - `python -m py_compile scripts/ar_cli.py` passed on NPU.
     - `python scripts/ar_cli.py --help` and `python scripts/ar_cli.py
       worker --help` passed on NPU and expose only `worker`.
     - NPU local `worker --status --backend cpu --dsl ascendc_catlass`
       on an unused port returned unreachable diagnostics as expected.
     - NPU `config.yaml` contains only `repo_path` and `env_script` for
       the `npu` remote worker host.

9. Remote python override removal.
   - Status: complete.
   - Scope:
     - Remove the `remote_worker.hosts.<alias>.python` override from the
       standalone config contract.
     - Always invoke `python` after `env_script` is sourced, so the
       environment controls the interpreter.
     - Apply the same remote worker command rule to the AKG-side dispatch.
     - Keep worker startup behavior aligned across both repos: `worker
       --start` prints one logo locally, while `--status` / `--stop` and
       recursive remote starts stay quiet.
   - Local checks:
     - `python -m py_compile scripts/ar_cli.py` passed.
     - `git diff --check -- scripts/ar_cli.py config.yaml
       MIGRATION_PROGRESS.md` passed.
     - No removed remote-python override references remain in the
       standalone repo.
     - AKG-side `misc.py` / `remote_dispatch.py` py_compile and diff check
       passed locally.
   - NPU sync/checks:
     - Synced standalone `scripts/ar_cli.py`, `config.yaml`, and
       `MIGRATION_PROGRESS.md`.
     - Synced AKG `misc.py` and `remote_dispatch.py`; removed an accidental
       untracked AKG `service/misc.py` copy after verifying it was not tracked.
     - Standalone and AKG py_compile/help smoke passed on NPU.
     - Standalone and AKG `worker --status` on an unused port returned
       diagnostics without logo output.
     - NPU scans confirmed both repos no longer reference the removed
       `python` remote-worker override.

10. Documentation and YAML contract alignment.
   - Status: complete.
   - Scope:
     - Align standalone YAML comments and docs with the env-owned remote
       interpreter contract: `env_script` prepares PATH, then remote dispatch
       invokes plain `python`.
     - Align the sibling workspace docs and YAML comments with the same
       remote worker contract and remove stale `python` field examples.
     - Keep wording focused on usage and behavior, with neutral technical
       phrasing in the migration-facing docs.
   - Local checks:
     - Standalone `git diff --check -- scripts/ar_cli.py config.yaml
       CLAUDE.md AUTORESEARCH.md MIGRATION_PROGRESS.md` passed.
     - Standalone `python -m py_compile scripts/ar_cli.py` passed.
     - Sibling workspace `git diff --check -- workspace_autoresearch/
       config.yaml workspace_autoresearch/AGENTS.md
       workspace_autoresearch/AUTORESEARCH.md` passed.
     - Local scans found no removed remote-python override references or
       migration-facing colloquial terms targeted by this cleanup.
   - NPU sync/checks:
     - Synced standalone docs/YAML/progress and `scripts/ar_cli.py` via
       exact `scp` destinations.
     - Synced sibling workspace docs/YAML via `scp`.
     - NPU `git diff --check` passed for all synced files after CRLF
       normalization.
     - NPU standalone `python -m py_compile scripts/ar_cli.py` passed.
     - NPU sibling worker-related py_compile passed for `misc.py` and
       `remote_dispatch.py`.
     - NPU grep scans found no removed remote-python override references or
       migration-facing colloquial terms targeted by this cleanup.

## Latest Known Repo State

- This file is committed as part of the Step 7 cleanup changes; use
  `git log -1 --oneline` locally and on `npu` for the final cleanup commit IDs.
- NPU runtime dirs are expected to remain untracked:
  `.autoresearch/`, `.session_tasks/`, `.task_dir_pointers/`, `extra-info/`.
