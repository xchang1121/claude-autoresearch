#!/usr/bin/env python3
"""
Task directory scaffolder for Claude Code autoresearch.

Zero external dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - reference.py (correctness baseline; AST-checked via utils.ref_ast.
    validate_ref before scaffold copies it. Runtime correctness is
    validated by --run-baseline whose verify script tags error_source.)
  - kernel.py (editable seed; written from the user's --kernel file)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

Usage:
    # NOTE: --devices values below are placeholders; pass the actual free
    # device id at invocation time.

    # Local eval (arch auto-derived via npu-smi):
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --dsl triton_ascend --devices <DEV>

    # Remote worker (arch fetched from /api/v1/status):
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --dsl triton_ascend --worker-url 127.0.0.1:9111

    # Custom output directory:
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --dsl triton_cuda --devices <DEV> --output-dir /tmp/tasks

Output (last line of stdout):
    {"task_dir": "/absolute/path/to/task_dir", "status": "ok"}
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

import yaml


# ---------------------------------------------------------------------------
# Reference validation — delegated to the standalone library module so
# phase_machine.validators can call the same rule without importing this
# CLI script. The local re-export keeps callers that imported
# `scaffold.validate_ref` working.
# ---------------------------------------------------------------------------
from utils.ref_ast import validate_ref  # noqa: E402, F401  (re-export)


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def scaffold_task_dir(
    *,
    ref_code: str,
    kernel_code: str,
    op_name: str,
    desc: str = "",
    dsl: str = "",
    framework: str = "torch",
    backend: str = "",
    arch: str = "",
    devices: list | None = None,
    worker_urls: list | None = None,
    max_rounds: int = 20,
    eval_timeout: int = 120,
    output_dir: str | None = None,
    editable_filename: str = "kernel.py",
    code_checker_enabled: bool = True,
) -> str:
    """Create task directory with all files. Returns absolute path."""
    # Determine base directory
    if output_dir:
        base_dir = output_dir
    else:
        base_dir = os.path.join(os.getcwd(), "ar_tasks")

    dir_name = f"{op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, dir_name)
    os.makedirs(task_dir)

    # Write reference.py and the seed kernel.py from the user's files.
    _write(task_dir, "reference.py", ref_code)
    _write(task_dir, editable_filename, kernel_code)

    # Generate task.yaml
    task_yaml = {
        "name": op_name,
        "description": desc or f"Optimize {op_name}",
        "dsl": dsl or None,
        "framework": framework or None,
        "backend": backend or None,
        "arch": arch or None,
        "editable_files": [editable_filename],
        "eval": {
            "timeout": eval_timeout,
        },
        "metric": {
            "primary": "latency_us",
            "lower_is_better": True,
            # atol / rtol used to live here too; they're hardcoded in
            # correctness.DEFAULT_ATOL / DEFAULT_RTOL now (single source
            # of truth). Loader silently ignores any stale fields a user
            # might still have in their task.yaml.
        },
        "agent": {
            "ref_file": "reference.py",
            "max_rounds": max_rounds,
        },
    }
    if devices:
        task_yaml["devices"] = list(devices)

    # Only emit the code_checker block when disabled — default-true tasks
    # stay clean. quick_check.py and phase_machine.validate_kernel honor
    # this field.
    if not code_checker_enabled:
        task_yaml["code_checker"] = {"enabled": False}

    # Add worker config if provided
    if worker_urls:
        task_yaml["worker"] = {"urls": worker_urls}

    yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True)
    _write(task_dir, "task.yaml", yaml_content)

    # Create .ar_state directory
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    # Git init + baseline commit
    _git_init(task_dir)

    return os.path.abspath(task_dir)


def _write(task_dir: str, rel_path: str, content: str):
    full_path = os.path.join(task_dir, rel_path)
    parent = os.path.dirname(full_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def _git_init(task_dir: str):
    """Initialize git repo and create baseline commit.

    The actual commit goes through git_utils.commit_in_task — same code
    path hooks use for round commits, so reliability is consistent.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.git_utils import commit_in_task

    subprocess.run(["git", "init"], cwd=task_dir, capture_output=True, check=True)
    ok, info = commit_in_task(task_dir, ["."], "scaffold: baseline")
    if not ok:
        raise RuntimeError(f"scaffold baseline commit failed: {info}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    """Construct scaffold's argparse, with no side effects.

    Extracted out of main() so parse_args.py can reuse the exact same flag
    spec without duplicating it. Single source of truth for which flags
    /autoresearch accepts and how they're typed/defaulted.
    """
    parser = argparse.ArgumentParser(
        description="Scaffold a task directory for Claude Code autoresearch",
    )
    parser.add_argument("--ref", required=True,
                        help="Path to reference.py (Model/get_inputs format)")
    parser.add_argument("--kernel", required=True,
                        help="Path to seed kernel file")
    parser.add_argument("--op-name", default=None,
                        help="Operator name (required)")
    # DSL = primary pivot. backend is a pure function of DSL; arch is
    # derived from hardware (local: npu-smi on --devices; remote: worker
    # /api/v1/status). Neither needs to be user-facing.
    # Pull the canonical DSL list from hw_detect at construction time so
    # the help string can't drift from _DSL_BACKEND.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.hw_detect import list_supported_dsls
    parser.add_argument("--dsl", default=None,
                        help=f"DSL name (one of: {', '.join(list_supported_dsls())}). "
                             f"Defaults to config.yaml:default_dsl.")
    parser.add_argument("--framework", default="torch",
                        choices=["torch", "mindspore", "numpy"],
                        help="Framework for the reference/kernel code "
                             "(default: torch).")
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device IDs for local eval "
                             "(e.g. '5' or '0,1,2,3'). Mutually exclusive "
                             "with --worker-url.")
    parser.add_argument("--worker-url", default=None,
                        help="Remote worker URL(s), comma-separated. "
                             "Mutually exclusive with --devices.")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")
    parser.add_argument("--no-code-checker", action="store_true",
                        help=("Disable the static CodeChecker pipeline "
                              "(syntax / imports / DSL / autotune compliance) "
                              "for this task. Useful when the DSL rules are "
                              "too strict for the chosen kernel style. Writes "
                              "`code_checker: {enabled: false}` into "
                              "task.yaml; flip the field to re-enable later."))
    # --correctness-atol / --correctness-rtol used to live here. atol/rtol
    # are now locked to correctness.DEFAULT_ATOL / DEFAULT_RTOL — see the
    # comment in scaffold.create_task() and in correctness.py.
    return parser


def main():
    parser = _make_arg_parser()
    args = parser.parse_args()

    # Resolve DSL (and via it, backend).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.settings import default_dsl
    from utils.hw_detect import (
        backend_for_dsl, derive_arch, fetch_worker_hardware,
    )
    from ar_vendored.op.verifier.adapters.factory import (
        get_dsl_adapter, get_framework_adapter,
    )

    args.dsl = (args.dsl or default_dsl()).lower()
    try:
        get_dsl_adapter(args.dsl)
    except Exception as e:
        print(json.dumps({"status": "error",
                          "error": f"unsupported --dsl {args.dsl!r}: {e}"}))
        sys.exit(1)

    try:
        args.backend = backend_for_dsl(args.dsl)
    except ValueError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

    try:
        get_framework_adapter(args.framework)
    except Exception as e:
        print(json.dumps({"status": "error",
                          "error": f"unsupported --framework "
                                   f"{args.framework!r}: {e}"}))
        sys.exit(1)

    # Hardware resolution: --devices XOR --worker-url.
    devices_list: list = []
    worker_urls: list = []
    args.arch = None

    if args.devices and args.worker_url:
        print(json.dumps({"status": "error",
                          "error": "--devices and --worker-url are mutually "
                                   "exclusive. Pick one (--devices for local "
                                   "eval, --worker-url for remote worker)."}))
        sys.exit(1)

    if args.devices:
        devices_list = [int(d.strip()) for d in args.devices.split(",")
                        if d.strip()]
        if not devices_list:
            print(json.dumps({"status": "error",
                              "error": "--devices parsed to an empty list"}))
            sys.exit(1)
        args.arch = derive_arch(args.backend, devices_list[0])
        if not args.arch:
            print(json.dumps({"status": "error",
                              "error": (f"could not derive arch from "
                                        f"{args.backend} device "
                                        f"{devices_list[0]} "
                                        f"(is the SMI tool on PATH?)")}))
            sys.exit(1)

    elif args.worker_url:
        worker_urls = [u.strip() for u in args.worker_url.split(",")
                       if u.strip()]
        status = fetch_worker_hardware(worker_urls[0])
        if not status:
            print(json.dumps({"status": "error",
                              "error": (f"worker {worker_urls[0]} unreachable "
                                        f"or /api/v1/status failed")}))
            sys.exit(1)
        worker_backend = str(status.get("backend", "")).lower()
        worker_arch = str(status.get("arch", "")).lower()
        if worker_backend and worker_backend != args.backend:
            print(json.dumps({"status": "error",
                              "error": (f"worker backend {worker_backend!r} "
                                        f"incompatible with --dsl {args.dsl!r} "
                                        f"(requires {args.backend!r})")}))
            sys.exit(1)
        args.arch = worker_arch or None
        if not args.arch:
            print(json.dumps({"status": "error",
                              "error": (f"worker /api/v1/status returned no "
                                        f"arch: {status}")}))
            sys.exit(1)

    else:
        print(json.dumps({"status": "error",
                          "error": "must pass exactly one of --devices "
                                   "(local eval) or --worker-url (remote)."}))
        sys.exit(1)

    if not args.op_name:
        print(json.dumps({"status": "error",
                          "error": "--op-name is required"}))
        sys.exit(1)

    if not os.path.isfile(args.ref):
        print(json.dumps({"status": "error",
                          "error": f"Reference file not found: {args.ref}"}))
        sys.exit(1)
    with open(args.ref, "r", encoding="utf-8") as f:
        ref_code = f.read()
    try:
        validate_ref(ref_code, args.ref)
    except ValueError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

    if not os.path.isfile(args.kernel):
        print(json.dumps({"status": "error",
                          "error": f"Kernel file not found: {args.kernel}"}))
        sys.exit(1)
    with open(args.kernel, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    # worker_urls / devices_list were resolved above.
    print(f"[scaffold] Creating task directory for {args.op_name}...", file=sys.stderr)

    task_dir = scaffold_task_dir(
        ref_code=ref_code,
        kernel_code=kernel_code,
        op_name=args.op_name,
        dsl=args.dsl,
        framework=args.framework,
        backend=args.backend,
        devices=devices_list,
        arch=args.arch,
        worker_urls=worker_urls,
        max_rounds=args.max_rounds,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        code_checker_enabled=not args.no_code_checker,
    )

    print(f"[scaffold] Task directory created: {task_dir}", file=sys.stderr)
    print(f"[scaffold] Files:", file=sys.stderr)
    for f in sorted(os.listdir(task_dir)):
        print(f"  {f}", file=sys.stderr)

    # Reference validation is now a single path through baseline.py: the
    # generated verify script splits ref-side and kernel-side try/excepts
    # and tags error_source on failure. Scaffold reads the resulting
    # progress.json after baseline and decides:
    #   - error_source == "ref"  → reject task (user must fix --ref)
    #   - everything else        → status=ok, task activates normally
    #     (kernel-side failures advance to PLAN via on_baseline_settled)
    # AST symbol presence was already checked earlier (validate_ref on
    # the source --ref file before copying), so import errors / missing
    # symbols never reach this point.
    if args.run_baseline:
        print(f"[scaffold] Running baseline eval...", file=sys.stderr)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_cmd = [sys.executable,
                        os.path.join(script_dir, "engine", "baseline.py"),
                        task_dir]
        if args.worker_url:
            baseline_cmd.extend(["--worker-url", args.worker_url])
        rc = subprocess.run(baseline_cmd).returncode
        # `_EXIT_FOR` in workflow/baseline.py maps EvalOutcome → exit code:
        #   5 = REF_FAIL              (reference is broken)
        #   4 = FRAMEWORK_ERROR       (eval framework crashed)
        #   3 = KERNEL_VERIFY_FAIL / KERNEL_PROFILE_CRASH (kernel bug)
        # REF_FAIL and FRAMEWORK_ERROR are both `STUCK_BASELINE_OUTCOMES`
        # — `PhaseController.on_baseline_settled` pins them at BASELINE
        # and `hooks/stop_save.py` allows early Stop. Sending the agent
        # toward plan->edit for these would burn rounds on something it
        # provably can't fix.
        if rc == 5:
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": ("reference.py failed during baseline — see "
                          "[baseline]/[eval] stderr above"),
                "hint": ("The file passed via --ref is broken (import / "
                         "forward / device-only bug). Fix the SOURCE file "
                         "and re-run /autoresearch from scratch. The task "
                         "directory is left in place for inspection but "
                         "MUST NOT be activated — reference.py is treated "
                         "as ground truth and the agent cannot fix it."),
            }))
            sys.exit(5)
        if rc == 4:
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": ("eval framework crashed during baseline — see "
                          "[baseline]/[eval] stderr above"),
                "hint": ("FRAMEWORK_ERROR: no per-shape data was produced, "
                         "so the seed kernel was not meaningfully "
                         "exercised. This is an operator-side issue "
                         "(eval.timeout, worker connectivity, device "
                         "contention, OOM before any case ran) — not "
                         "something the agent can fix via plan->edit. "
                         "Fix the underlying eval framework and re-run "
                         "`/autoresearch --resume <task_dir>` to retry "
                         "baseline. Phase will stay at BASELINE until a "
                         "kernel- or OK- outcome lands."),
            }))
            sys.exit(4)
        if rc != 0:
            # Kernel-side failure (KERNEL_VERIFY_FAIL / KERNEL_PROFILE_CRASH).
            # task_dir is left in place and will be activated normally;
            # the hook routes to PLAN so the agent rewrites the kernel
            # via plan->edit.
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": (f"baseline eval failed (exit {rc}); "
                          f"see [baseline]/[eval] stderr above"),
                "hint": ("Seed kernel failed baseline. Activate the task "
                         "(export AR_TASK_DIR=...) and proceed via the "
                         "standard plan->edit loop."),
            }))
            sys.exit(3)

    # Output
    print(json.dumps({"task_dir": task_dir, "status": "ok"}))


if __name__ == "__main__":
    main()
