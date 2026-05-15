"""Shared JSON helpers for autoresearch subprocess I/O.

Every AR script that talks to a child process follows the same wire
protocol: arbitrary log lines on stdout/stderr, with the final non-empty
stdout line being a single-line JSON object carrying the result. This
module owns the parser. Previously `phase_machine.state_store` and
`task_config.eval_client` each carried their own copy (under different
names — `parse_last_json_line` vs `_last_json_line`); two
implementations of the same protocol is exactly the kind of quiet drift
this codebase tries to avoid, so both now route through here.
"""
from __future__ import annotations

import json
from typing import Optional


def parse_last_json_line(text: str) -> Optional[dict]:
    """Scan `text` from the bottom up and return the last standalone JSON
    object (a line that starts with '{' and ends with '}'), or None if no
    line matches. Lines that look JSON-shaped but fail to parse are
    skipped and the search continues upward — non-JSON warning lines
    printed after the result line don't cause a false negative.
    """
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
