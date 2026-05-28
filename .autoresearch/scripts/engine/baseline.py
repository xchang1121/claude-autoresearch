#!/usr/bin/env python3
"""Run baseline eval and initialize .ar_state.

Calls `task_config.run_eval` in-process — same Python tree as
pipeline.py and the rest of the engine, so no subprocess / sentinel /
rc-decoding ceremony between this script and the eval pipeline. The
eval pipeline itself still spawns `eval_<op>.py` (via local_worker)
or POSTs to a remote worker; that's where the crash isolation lives.

Usage:
    python .autoresearch/scripts/engine/baseline.py <task_dir>
        [--device-id N] [--worker-url URL[,URL,...]]
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPTS_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from task_config import load_task_config, run_eval
from utils.failure_extractor import extract_failure_signals, format_for_stdout
from workflow import run_baseline_init


def _eval_result_to_dict(result) -> dict:
    """Match the dict shape `run_baseline_init` (and the historical
    `ar_cli.py verify` sentinel JSON) consume. EvalResult is a dataclass,
    not a dict; flatten its public fields and attach failure_signals /
    raw_output_tail only on failure to match the runtime contract."""
    eval_data = {
        "outcome": result.outcome.value,
        "correctness": result.correctness,
        "metrics": result.metrics or {},
        "error": result.error,
        "error_source": result.error_source,
    }
    if not result.correctness or result.error:
        eval_data["failure_signals"] = extract_failure_signals(
            result.raw_output).to_dict()
        eval_data["raw_output_tail"] = (result.raw_output or "")[-4000:]
    return eval_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--worker-url", default=None,
                        help="Worker URL(s), comma-separated. Overrides "
                             "task.yaml worker.urls.")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    config = load_task_config(task_dir)
    if config is None:
        print("[baseline] ERROR: task.yaml not found in task_dir",
              file=sys.stderr)
        sys.exit(1)

    worker_urls = None
    if args.worker_url:
        worker_urls = [u.strip() for u in args.worker_url.split(",") if u.strip()]

    print("[baseline] Running baseline eval...", flush=True)
    try:
        result = run_eval(task_dir, config,
                          device_id=args.device_id,
                          worker_urls=worker_urls)
    except Exception as e:
        # run_eval is expected to convert its own internal failures to
        # EvalResult(INFRA_FAIL, ...); reaching this branch means it
        # raised instead — log and exit 4 (the INFRA_FAIL exit code
        # workflow.baseline._EXIT_FOR would have used).
        print(f"[baseline] run_eval raised {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(4)

    eval_data = _eval_result_to_dict(result)

    # Pretty-print structured failure signals (UB overflow, aivec trap,
    # OOM, correctness mismatch, ...) — mirrors pipeline.py so the
    # seed-failure → PLAN flow surfaces the same actionable summary the
    # EDIT loop does.
    if not eval_data.get("correctness", False) or eval_data.get("error"):
        if eval_data.get("error"):
            print(f"[baseline] Error: {eval_data['error']}", flush=True)
        pretty = format_for_stdout(eval_data.get("failure_signals") or {})
        if pretty:
            print(pretty, flush=True)
        elif eval_data.get("raw_output_tail"):
            print("[baseline] Worker log tail (no structured signals matched):",
                  flush=True)
            print(eval_data["raw_output_tail"], flush=True)

    # workflow.run_baseline_init owns progress.json / history.jsonl /
    # phase writes; its int return becomes this script's exit code.
    sys.exit(run_baseline_init(task_dir, eval_data))


if __name__ == "__main__":
    main()
