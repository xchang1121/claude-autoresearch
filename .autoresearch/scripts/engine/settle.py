#!/usr/bin/env python3
"""
Mechanical plan.md settlement — no LLM needed.

After keep_or_discard.py runs, this script:
1. Reads the decision (KEEP/DISCARD/FAIL) from keep_or_discard output
2. Updates plan.md: mark active item [x] with result, advance (ACTIVE)
3. Returns the next phase

Usage:
    python settle.py <task_dir> <decision_json>

Output (stdout, last line):
    {"next_phase": "EDIT", "settled_item": "p1", "decision": "KEEP", "metric": 1294.8}

All plan.md mutation goes through workflow.PlanStore so the parse / render
formats can't drift across files. Phase advancement goes through
workflow.PhaseController so the phase rule lives in one place.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow import PhaseController, PlanStore


def main():
    if len(sys.argv) != 3:
        print(json.dumps({
            "error": "invalid arguments",
            "usage": "python settle.py <task_dir> <decision_json>",
            "received_args": sys.argv[1:],
        }))
        sys.exit(1)

    task_dir = sys.argv[1]
    decision_json = sys.argv[2]

    try:
        decision_data = json.loads(decision_json)
    except json.JSONDecodeError as exc:
        print(json.dumps({
            "error": "invalid decision_json",
            "details": str(exc),
        }))
        sys.exit(1)
    decision = decision_data.get("decision", "FAIL")
    best_metric = decision_data.get("best_metric")
    # For KEEP, best_metric is this round's value. For DISCARD we have no metric.
    metric_val = best_metric if decision == "KEEP" else None

    store = PlanStore(task_dir)
    if not store.exists():
        print(json.dumps({"error": "plan.md not found"}))
        sys.exit(1)

    try:
        settled_item_id, _settled_desc = store.settle_active(decision, metric_val)
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    next_phase = PhaseController(task_dir).on_round_settled()

    print(json.dumps({
        "settled_item": settled_item_id,
        "decision": decision,
        "metric": metric_val,
        "next_phase": next_phase,
    }))


if __name__ == "__main__":
    main()
