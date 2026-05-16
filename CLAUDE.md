# Claude AutoResearch

An iterative optimization framework powered by Claude Code: plan → edit →
eval → keep/discard, looping against a measurable metric. Standalone, no
external deps beyond Python + PyYAML.

## Quick Start

```bash
# Drop sources into workspace/<op_name>_ref.py and workspace/<op_name>_kernel.py,
# then start a task. --dsl required; pick --devices N XOR --worker-url.
/autoresearch --ref workspace/<op_name>_ref.py --kernel workspace/<op_name>_kernel.py \
              --op-name <op_name> --dsl triton_cuda --devices 0

# Resume later
/autoresearch --resume

# Monitor in a separate terminal
python .autoresearch/scripts/dashboard.py <task_dir> --watch
```

Hook-driven — every Bash / Edit emits a fresh `[AR Phase: ...]` line on
stderr with the next legal action. **Read those messages**; don't poke
state files by hand. Full operational details (slash-command flags, the
DIAGNOSE artifact contract, the canonical-form Bash policy, etc.) live
in [.claude/commands/autoresearch.md](.claude/commands/autoresearch.md);
the hooks enforce them — they're documentation, not a checklist you
have to memorize.

For unattended long runs: `/loop /autoresearch --resume`.

## Invariants

1. `.ar_state/plan.md`, `progress.json`, `history.jsonl`, `.phase` are
   owned by AR scripts. Never hand-edit. TodoWrite is a UI mirror of
   plan.md, not a substitute.
2. Edits are scoped to `task.yaml.editable_files`. Anything else is
   rejected by `hooks/guard_edit.py`.
3. After a session break, recover with `/autoresearch --resume` — do
   not patch state files.

## Dependencies

- Python >= 3.10
- PyYAML (`pip install pyyaml`)
- Claude Code CLI
