#!/usr/bin/env python3
"""Thin CLI wrapper around workflow.baseline.run_baseline_init.

Body lives in workflow/baseline.py so notebook / batch / scaffold call
sites can invoke it in-process. This script preserves the legacy
`python _baseline_init.py <task_dir> <eval_json>` shell contract.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from workflow import run_baseline_init


def main():
    if len(sys.argv) < 3:
        print("Usage: python _baseline_init.py <task_dir> <eval_json>",
              file=sys.stderr)
        sys.exit(1)
    sys.exit(run_baseline_init(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
