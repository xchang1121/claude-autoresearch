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

8. ar_cli UX and diagnostics follow-up.
   - Status: complete.
   - Scope:
     - Compare `ar_cli` against `akg_cli` worker UX/diagnostics behavior.
     - Add human-facing logo/list/doctor output while keeping
       `worker --status` machine-readable.
     - Add remote preflight checks before spawning a remote worker:
       ssh/env_script, torch_npu, triton policy, npu-smi, arch/device
       visibility, disk space, and remote port ownership.
     - Sync via `scp` and smoke on `npu`.
   - Local checks:
     - `python -m py_compile scripts/ar_cli.py` passed on 2026-06-07.
     - `python scripts/ar_cli.py list` passed and prints the new logo.
     - `python scripts/ar_cli.py doctor --help` passed.
     - `python scripts/ar_cli.py doctor --remote-host npu --backend ascend --dsl ascendc_catlass --port 65534`
       passed; all remote diagnostics were OK.
   - NPU sync/checks:
     - `scripts/ar_cli.py` and `MIGRATION_PROGRESS.md` synced via `scp`.
     - `python -m py_compile scripts/ar_cli.py` passed on `npu`.
     - `python scripts/ar_cli.py list` passed on `npu`.
     - `python scripts/ar_cli.py doctor --backend ascend --dsl ascendc_catlass --port 65534`
       passed on `npu`.
   - Notes:
     - A full remote `worker --start` smoke on port 65534 exceeded the
       command timeout; follow-up checks found no remote worker/listener
       or worker log, and the local 65534 ssh tunnel was cleaned up.

9. npu remote akg host configuration.
   - Status: complete.
   - Scope:
     - Configure local `ar_cli` host `npu` to target the remote akg project
       at `/home/yyz/cxy/akg/akg_agents` with `/home/yyz/env.sh`.
     - Add `remote_cli: akg_cli` support so `ar_cli` can dispatch to the
       upstream akg project layout instead of requiring `scripts/ar_cli.py`.
   - Local checks:
     - `python -m py_compile scripts/ar_cli.py` passed on 2026-06-07.
     - `python scripts/ar_cli.py doctor --remote-host npu --backend ascend --dsl ascendc_catlass --port 65534`
       passed; all remote diagnostics were OK.
     - `_build_remote_ar_cli_cmd(...)` constructed
       `akg_cli worker --stop --port 65534` with `AKG_CLI_QUIET=1`.
   - NPU sync/checks:
     - Synced `config.yaml`, `MIGRATION_PROGRESS.md`, and `scripts/ar_cli.py`
       to `/home/yyz/cxy/claude-autoresearch`.
     - `git diff --check -- scripts/ar_cli.py config.yaml MIGRATION_PROGRESS.md`
       and `python -m py_compile scripts/ar_cli.py` passed on NPU.
     - Remote config contains `npu` -> `/home/yyz/cxy/akg/akg_agents`
       with `/home/yyz/env.sh` and `remote_cli: akg_cli`.
   - Adjustment:
     - Removed the extra `akg_npu` host alias and made `npu` itself point
       at the akg project, as requested.

10. AUTORESEARCH.md remote CLI usage docs.
   - Status: complete.
   - Scope:
     - Update `AUTORESEARCH.md` B-section remote worker setup to document
       `list`, `doctor`, `worker --remote-host`, `--dsl`, and `remote_cli`.
     - Add a generic AKG project entry example using placeholders and
       `remote_cli: akg_cli`.
     - Refresh the ar_cli reference sections so `list` / `doctor` / `worker`
       match the current CLI surface.
   - Local checks:
     - `git diff --check -- AUTORESEARCH.md` passed on 2026-06-07.
     - Generic-example and discouraged-wording scan passed for
       `AUTORESEARCH.md`.

## Latest Known Repo State

- This file is committed as part of the Step 7 cleanup changes; use
  `git log -1 --oneline` locally and on `npu` for the final cleanup commit IDs.
- NPU runtime dirs are expected to remain untracked:
  `.autoresearch/`, `.session_tasks/`, `.task_dir_pointers/`, `extra-info/`.
