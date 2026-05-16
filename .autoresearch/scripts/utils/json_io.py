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
import os
from typing import List, Optional


def _read_whole_file(path: str) -> str:
    """Read `path` to EOF without short-reads.

    Python's `open().read()` and "for line in f" iterate via a buffered
    read that has been observed to truncate large history.jsonl files
    (multi-shape runs trivially pass 256 KB after ~25 rounds with 60
    cases). Looping `os.read` until EOF avoids that — silently dropping
    the tail of history made the dashboard look frozen on recent rounds.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        os.close(fd)


def load_jsonl(path: str) -> List[dict]:
    """Load every JSON object from a JSONL file. Missing file → []. Blank
    lines are skipped; malformed lines are dropped silently (the file
    is treated as an audit log where partial readability matters more
    than strict validation).

    Uses `_read_whole_file` to avoid the short-read truncation issue
    that affected the plain-`open` history loader in report.py.
    """
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    for line in _read_whole_file(path).split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


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
