# Claude AutoResearch

An iterative optimization framework powered by Claude Code: plan → edit →
eval → keep/discard, looping against a measurable metric. Standalone, no
external deps beyond Python + PyYAML.

## Quick Start

```bash
# Drop sources into workspace/<op_name>_ref.py and workspace/<op_name>_kernel.py,
# then start a task. --devices is required.
/autoresearch --ref workspace/<op_name>_ref.py --kernel workspace/<op_name>_kernel.py \
              --op-name <op_name> --devices 0

# Resume later
/autoresearch --resume

# Monitor in a separate terminal
python scripts/dashboard.py <task_dir> --watch
```

Full operational details in [.claude/commands/autoresearch.md](.claude/commands/autoresearch.md).

### Remote eval (optional)

If the local machine has no NPU, eval can run on a remote Ascend box:

```bash
# Start a worker daemon on the remote (SSH-launched + auto ssh -L tunnel).
# `my-npu` is an entry under remote_worker.hosts in config.yaml.
python scripts/ar_cli.py worker --remote-host my-npu --start \
    --backend ascend --arch ascend910b3 --devices 0 --port 9111

# Point /autoresearch (or baseline.py / pipeline.py) at the tunneled port.
/autoresearch --ref ... --kernel ... --devices 0 --worker-url 127.0.0.1:9111
```

`ar_cli worker --remote-host my-npu --stop --port 9111` tears down both
the remote daemon and the local tunnel.

## Skills Library

Root: `../skills/` — DSL-partitioned skill docs. Top-level dirs are
DSLs (`triton-ascend/`, `triton-cuda/`, `pypto/`, `cpp/`, `cuda-c/`,
`tilelang-cuda/`); under each DSL the relevant subdirs are
`fundamentals/`, `guides/`, `cases/`, `examples/`, and
`evolved-improvement/` — each leaf carries a `SKILL.md`.

During PLAN, Glob `../skills/<dsl>/**/SKILL.md` for the current DSL
and Read 1-3 most relevant; cite filename in plan rationales.

## Invariants (hook-driven flow)

1. **`.ar_state/plan.md` is the source of truth.** Only `create_plan.py`
   and `pipeline.py`'s inlined settle step write it (both via
   `workflow.PlanStore`). Never hand-edit. TodoWrite is a UI mirror,
   not a substitute.
2. **Plan IDs are globally monotonic.** `p1, p2, ...` from
   `state.next_pid`. Never reuse, never skip.
3. **Every `pN` either settles (KEEP / DISCARD / FAIL in `history.jsonl`)
   or is silently dropped at a REPLAN/DIAGNOSE boundary** — pid counter
   still advances, no synthetic DISCARD row written.
4. **Phase transitions are owned by `workflow.PhaseController`.** Never
   write `state.json` manually. The hook (`hooks/post_bash.py`)
   triggers the controller after activation and after `create_plan.py`
   validates; the engine scripts (`workflow.run_baseline_init` inside
   `engine/baseline.py`, `_post_settle` inside `engine/pipeline.py`)
   trigger it after baseline / round settlement. Either way every
   write goes through `PhaseController.on_*`. Listen to the
   `[AR Phase: ...]` messages on stderr; don't poke `phase_machine`
   directly (it's a library, not a CLI — `hooks/guard_bash.py` rejects
   direct invocation).
5. **Editable files are scoped by `task.yaml.editable_files`.** Editing
   anything else is rejected by `hooks/guard_edit.py`.
6. **After a session break, resume with `/autoresearch --resume`.** Do
   not patch state files to recover.
7. **`create_plan.py` rejects mean the plan has a real problem**
   (diversity, repeated failure keywords, short rationale). Read stderr
   and rewrite — don't retry the same XML payload.
8. **TodoWrite sync is mandatory.** When a hook emits `additionalContext`
   with a TodoWrite payload, call TodoWrite with it verbatim next turn.
9. **AR scripts run as direct top-level Bash invocations only.**
   To *invoke* a blessed CLI the command must be a single foreground
   call: `python scripts/engine/<name>.py <task_dir>
   [args...]` (pipeline, baseline, create_plan, parse_args). The
   top-level lifecycle scripts use the flat path:
   `python scripts/<name>.py` (scaffold, resume,
   dashboard). Env-var prefixes, Python flags, and FD redirection
   (`> log 2>&1`) are fine. Wrappers (`nohup`, `bash -lc`, `sh -c`,
   subshells, `$(...)`), chains (`&&`, `||`, `;`, `|`), and
   backgrounding (`&`) are unsupported and rejected by
   `hooks/guard_bash.py`. Run multiple AR scripts as separate Bash
   tool calls.

   *Reading* AR scripts (e.g. `cat scripts/engine/pipeline.py`,
   `git diff -- scripts/engine/baseline.py`) is allowed because the
   classifier sees those heads as read-only and the args don't execute.
   The Read tool is still preferred — it's the idiomatic way to inspect
   file contents in Claude Code.
10. **DIAGNOSE phase ends with a new plan.** Two paths to that end:
   - **Preferred (subagent route).** Call `Task(subagent_type='ar-diagnosis')`;
     the subagent's prompt asks it to Write a structured artifact at
     `<task_dir>/.ar_state/diagnose_v<plan_version>.md` containing three
     sections (`Root cause` / `Fix directions` / `What to avoid`),
     useful citations of recent FAIL rounds by `R<n>`, and the marker
     line `[AR DIAGNOSE COMPLETE marker_v<plan_version>]`. The host
     gates on file presence, marker, and section names; then write
     `plan_items.xml` and run `create_plan.py`.
   - **Fallback (manual planning).** After 5 failed Task attempts on the
     same `plan_version`, the artifact gate is relaxed: write
     `plan_items.xml` yourself using `history.jsonl` + `plan.md`, then
     run `create_plan.py`. Further Task calls are blocked at this point.

   While the artifact is invalid AND attempts < cap, Bash is locked to
   read-only / lifecycle ops (no AR scripts beyond `create_plan.py`,
   which is itself gated on artifact validity).

   Provenance note: hook payloads do NOT distinguish main agent from
   subagent, so the host validates the artifact's CONTENT only — not who
   wrote it. The subagent path is preferred because the prompt and
   read-only-by-default tool isolation produce a more reliable diagnosis,
   not because the host can prove the subagent wrote the file.

11. **Stop is only legal at phase FINISH.** `hooks/stop_save.py` blocks
    early Stop in every other phase; the block message embeds
    `get_guidance(task_dir)` so the agent sees the next action.
    `max_rounds` + auto-DIAGNOSE-on-3-fails are the budget. If stuck,
    go through DIAGNOSE (#10), not Stop.

## Dependencies

- Python >= 3.10
- PyYAML (`pip install pyyaml`)
- Claude Code CLI
