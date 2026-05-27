"""Pre-flight verification for batch directories.

Tier 1 (default, no hardware): compile + import + required-symbol check
on ref.py (Model + get_inputs + get_init_inputs) and kernel.py (ModelNew).

Tier 2 (--full): LOCAL smoke test via `ar_cli.py verify --mode verify-only`.
Re-uses the same verify pipeline runtime calls, so batch pre-flight and
per-round eval share one correctness gate — no drift between them.

Each op runs in its own subprocess. Results: <batch_dir>/verify_results.json.

Usage:
    python .autoresearch/scripts/batch/verify.py <batch_dir>             # Tier 1
    python .autoresearch/scripts/batch/verify.py <batch_dir> --full      # Tier 1 + Tier 2
    python .autoresearch/scripts/batch/verify.py <batch_dir> --only op1,op2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.input_groups import resolve as _resolve_groups  # noqa: E402
from utils.json_io import parse_sentinel_line  # noqa: E402
from utils.settings import batch_verify_timeouts  # noqa: E402

VERIFY_RESULTS = "verify_results.json"
# Defaults overridden by `.autoresearch/config.yaml:batch_verify` via
# utils.settings. Tier 1 is compile/import only — 30s is generous. Tier 2
# includes cold-JIT pad-style kernels that can take many minutes; the
# 600s cap is per op (the eval-package itself scales by num_cases inside).
_timeouts = batch_verify_timeouts()
TIER1_TIMEOUT = _timeouts["tier1_timeout"]
TIER2_TIMEOUT = _timeouts["tier2_timeout"]

# Path to ar_cli.py — Tier 2 shells out to it instead of pulling the
# verify pipeline in-process. Same contract every other caller sees.
_AR_CLI = str(Path(__file__).resolve().parent.parent / "ar_cli.py")

# Reference must export Model + get_init_inputs + one of (get_inputs,
# get_input_groups). The "input provider" is checked separately (per
# input_groups.resolve duck-type) since either symbol satisfies it.
REF_REQUIRED = ("Model", "get_init_inputs")
REF_INPUT_PROVIDERS = ("get_inputs", "get_input_groups")
KERNEL_REQUIRED = ("ModelNew",)


# ---------------------------------------------------------------------------
# Subprocess workers (this same file is re-invoked with --tier-worker)
# ---------------------------------------------------------------------------
def _tier1_inspect(path: Path, required: tuple[str, ...]) -> dict:
    """Compile, import, check required attrs are present."""
    out: dict = {"path": str(path), "compile": "skip", "import": "skip",
                 "exports": "skip", "missing": [], "msg": ""}
    try:
        # utf-8-sig: PowerShell / Notepad on Windows tends to write source
        # files with a UTF-8 BOM; plain utf-8 leaves U+FEFF in the string and
        # compile() then dies with "invalid non-printable character U+FEFF".
        src = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        out["compile"] = "FAIL"
        out["msg"] = f"read error: {e}"
        return out
    try:
        compile(src, str(path), "exec")
        out["compile"] = "PASS"
    except SyntaxError as e:
        out["compile"] = "FAIL"
        out["msg"] = f"syntax error line {e.lineno}: {e.msg}"
        return out

    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            f"_verify_{path.stem}", str(path)
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not build spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out["import"] = "PASS"
    except Exception as e:
        out["import"] = "FAIL"
        out["msg"] = f"{type(e).__name__}: {e}"
        return out

    missing = [name for name in required if not hasattr(mod, name)]
    # Reference also needs an input provider: either get_inputs() or
    # get_input_groups(). Treat absence of both as missing.
    if required is REF_REQUIRED and not any(
            hasattr(mod, n) for n in REF_INPUT_PROVIDERS):
        missing.append(" or ".join(REF_INPUT_PROVIDERS))
    if missing:
        out["exports"] = "FAIL"
        out["missing"] = missing
        out["msg"] = f"missing: {', '.join(missing)}"
    else:
        out["exports"] = "PASS"
    return out


def _tier2_run(ref_path: Path, kernel_path: Path, op_name: str,
               dsl: str, worker_url: Optional[str] = None,
               device_id: Optional[int] = None) -> dict:
    """Run `ar_cli.py verify --mode verify-only` against (ref, kernel).

    Builds a minimal task_config JSON (op_name + dsl), points the CLI at
    the ref + kernel files via `@path`, parses the AR_VERIFY_RESULT:
    sentinel back. The verify pipeline materialises its own tempdir,
    runs eval_<op>.py twice (ref pass + kernel pass), and applies
    `utils.correctness` AND-of-maxima — exactly the path per-round eval
    takes, so batch pre-flight can't drift from runtime correctness.

    Tier 2's old return fields (max_abs_diff / per_case / atol / rtol)
    are derivable from the eval pipeline but not surfaced through the
    EvalResult dataclass today; we report `status` + a one-line `msg`,
    and stash the full sentinel payload under `verify_payload` for any
    consumer that wants to dig in.
    """
    out: dict = {"status": "skip", "msg": "", "verify_payload": None}

    task_cfg = {
        "name": op_name,
        "framework": "torch",
        "dsl": dsl or "triton_ascend",
        # backend / arch are derived from DSL by the verifier adapter —
        # passing only name + dsl + framework keeps Tier-2 callsite-blind
        # to whichever NPU / GPU the host happens to expose. eval_timeout
        # / warmup / run_times default in TaskConfig.
        "eval_timeout": TIER2_TIMEOUT,
    }
    cfg_fd, cfg_path = tempfile.mkstemp(prefix=f"_verify_cfg_{op_name}_",
                                         suffix=".json")
    try:
        with os.fdopen(cfg_fd, "w", encoding="utf-8") as f:
            json.dump(task_cfg, f)
        cmd = [
            sys.executable, _AR_CLI, "verify",
            "--task-config", f"@{cfg_path}",
            "--impl", f"@{kernel_path}",
            "--reference", f"@{ref_path}",
            "--mode", "verify-only",
        ]
        # Tier 2 must declare a transport: prefer worker_url (matches
        # what runtime eval does when manifest declares `worker.urls`);
        # otherwise fall through to a local device. Without either, the
        # CLI returns infra_fail.
        if worker_url:
            cmd += ["--worker-url", worker_url]
        else:
            cmd += ["--device-id", str(device_id if device_id is not None else 0)]
        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=TIER2_TIMEOUT, env=env)
        except subprocess.TimeoutExpired:
            out["status"] = "ERROR"
            out["msg"] = f"ar_cli.py verify timed out after {TIER2_TIMEOUT}s"
            return out
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass

    payload = parse_sentinel_line(proc.stdout, "AR_VERIFY_RESULT:")
    if payload is None:
        out["status"] = "ERROR"
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        out["msg"] = f"no AR_VERIFY_RESULT: sentinel (rc={proc.returncode}); {tail}"
        return out
    out["verify_payload"] = payload

    outcome = payload.get("outcome")
    if outcome == "ok" and payload.get("correctness"):
        out["status"] = "PASS"
        out["msg"] = "OK"
    elif outcome == "infra_fail":
        out["status"] = "ERROR"
        out["msg"] = (payload.get("error") or "infra_fail").splitlines()[0][:200]
    else:
        out["status"] = "FAIL"
        out["msg"] = (payload.get("error")
                      or "correctness mismatch (no diagnostics)").splitlines()[0][:200]
    return out


def _worker_main() -> int:
    """Subprocess entry point. Writes JSON to a sidecar path on stdout's last line.

    Tier 2 has no `--tier-worker` route: it shells out to `ar_cli.py verify`
    directly from `_verify_one`. The worker dispatch only handles the
    import-time Tier-1 checks (compile / import / required-symbol).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("1ref", "1kernel"), required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", default="")
    ap.add_argument("--sidecar", required=True)
    args = ap.parse_args(sys.argv[2:])  # skip the --tier-worker sentinel

    ref_path = Path(args.ref)
    kernel_path = Path(args.kernel) if args.kernel else None

    if args.tier == "1ref":
        result = _tier1_inspect(ref_path, REF_REQUIRED)
    else:  # tier == "1kernel"
        result = _tier1_inspect(kernel_path, KERNEL_REQUIRED)

    Path(args.sidecar).write_text(json.dumps(result), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_subprocess(*, tier: str, ref: Path, kernel: Path | None,
                    timeout: int) -> dict:
    sidecar = Path(os.environ.get("TMP", "/tmp")) / f"_verify_{os.getpid()}_{tier}_{ref.stem}.json"
    if sidecar.exists():
        sidecar.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--tier-worker",
           "--tier", tier,
           "--ref", str(ref),
           "--sidecar", str(sidecar)]
    if kernel is not None:
        cmd += ["--kernel", str(kernel)]

    env = os.environ.copy()
    # Default the Windows libomp/libiomp5md double-init workaround so users
    # don't see a wall of OMP error #15 on first run. No-op on Linux. Anyone
    # who wants the strict behavior can pre-set KMP_DUPLICATE_LIB_OK=FALSE.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "ERROR", "msg": f"timeout after {timeout}s",
                "elapsed_s": round(time.time() - t0, 2)}

    elapsed = round(time.time() - t0, 2)
    if not sidecar.exists():
        return {"status": "ERROR",
                "msg": f"no result; rc={proc.returncode}",
                "stderr_tail": (proc.stderr or proc.stdout)[-400:],
                "elapsed_s": elapsed}
    try:
        result = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "ERROR", "msg": f"parse sidecar: {e}",
                "elapsed_s": elapsed}
    finally:
        try:
            sidecar.unlink()
        except OSError:
            pass
    result["elapsed_s"] = elapsed
    return result


def _verify_one(case: dict, full: bool, dsl: str,
                worker_url: Optional[str] = None,
                device_id: Optional[int] = None) -> dict:
    op = case["op_name"]
    ref = Path(case["ref"])
    kernel = Path(case["kernel"])

    out: dict = {"op_name": op, "tier1_ref": None, "tier1_kernel": None,
                 "tier2": None}

    out["tier1_ref"] = _run_subprocess(tier="1ref", ref=ref, kernel=None,
                                       timeout=TIER1_TIMEOUT)
    out["tier1_kernel"] = _run_subprocess(tier="1kernel", ref=ref,
                                          kernel=kernel,
                                          timeout=TIER1_TIMEOUT)

    tier1_ok = (out["tier1_ref"].get("exports") == "PASS"
                and out["tier1_kernel"].get("exports") == "PASS")

    if full:
        if tier1_ok:
            t0 = time.time()
            tier2 = _tier2_run(ref, kernel, op_name=op, dsl=dsl,
                               worker_url=worker_url, device_id=device_id)
            tier2["elapsed_s"] = round(time.time() - t0, 2)
            out["tier2"] = tier2
        else:
            out["tier2"] = {"status": "skip",
                            "msg": "tier1 failed; skipping tier2",
                            "elapsed_s": 0}

    return out


_CONTENT_FAIL_FIELDS = ("compile", "import", "exports")


def _summary_status(record: dict, full: bool) -> str:
    """P/F/E/S single-letter. Compile/import/exports failures all map
    to F (matches the per-tier table column); runtime ERROR is E."""
    t1r = record["tier1_ref"]
    t1k = record["tier1_kernel"]
    t2 = record["tier2"]

    def _bad(t):
        return t and ("FAIL" in (t.get("compile"), t.get("import"), t.get("exports"))
                      or t.get("status") in ("FAIL", "ERROR"))

    def _content_fail(t):
        return t and any(t.get(f) == "FAIL" for f in _CONTENT_FAIL_FIELDS)

    if _bad(t1r):
        return "F" if _content_fail(t1r) else "E"
    if _bad(t1k):
        return "F" if _content_fail(t1k) else "E"
    if full and t2:
        if t2.get("status") == "PASS":
            return "P"
        if t2.get("status") == "FAIL":
            return "F"
        if t2.get("status") == "ERROR":
            return "E"
        return "S"
    return "P"


def _print_table(results: dict, full: bool) -> None:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for op, rec in results.items():
        t1r = rec["tier1_ref"]
        t1k = rec["tier1_kernel"]
        t2 = rec["tier2"]

        col_t1r = "PASS" if t1r and t1r.get("exports") == "PASS" else (
            "FAIL" if t1r and t1r.get("exports") == "FAIL" else (
                "FAIL" if t1r and (t1r.get("compile") == "FAIL"
                                   or t1r.get("import") == "FAIL") else "ERROR"))

        if t1k is not None:
            col_t1k = "PASS" if t1k.get("exports") == "PASS" else (
                "FAIL" if t1k.get("exports") == "FAIL" else (
                    "FAIL" if t1k.get("compile") == "FAIL"
                              or t1k.get("import") == "FAIL" else "ERROR"))
        else:
            col_t1k = "-"

        if full and t2 is not None:
            col_t2 = t2.get("status", "?")
        else:
            col_t2 = "-"

        # Pick the most informative message
        msg = ""
        for src in (t2, t1k, t1r):
            if src and src.get("msg") and src.get("msg") != "OK":
                msg = src["msg"]
                if "FAIL" in (src.get("compile"), src.get("import"),
                              src.get("exports")) or src.get("status") in ("FAIL", "ERROR"):
                    break
        rows.append((op, col_t1r, col_t1k, col_t2,
                     _summary_status(rec, full), msg[:70]))

    op_w = max(8, max(len(r[0]) for r in rows))
    headers = ("op", "t1_ref", "t1_kern", "t2", "ok", "note")
    print(f"  {headers[0]:<{op_w}}  {headers[1]:<6}  {headers[2]:<7}  "
          f"{headers[3]:<6}  {headers[4]:<3}  {headers[5]}")
    print(f"  {'-' * op_w}  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 3}  {'-' * 60}")
    for op, t1r, t1k, t2, ok, msg in rows:
        print(f"  {op:<{op_w}}  {t1r:<6}  {t1k:<7}  {t2:<6}  {ok:<3}  {msg}")


def run_verification(batch_dir: Path, *, full: bool = False,
                     only: str = "", dsl_override: str = "",
                     worker_url_override: str = "",
                     device_id_override: Optional[int] = None) -> int:
    """Run the verification loop programmatically (so prepare.py and other
    scripts can call us without subprocessing). Returns the same exit code
    main() would: 0 if everything passed, 1 if any FAIL/ERROR. All output
    still goes to stdout for the caller to surface.

    Tier 2 invokes `ar_cli.py verify --mode verify-only`; tolerances live
    in `utils.correctness` and are not exposed as flags here. `dsl_override`
    (CLI `--dsl`) wins over manifest.dsl when both are present; the field
    is required for Tier 2 because the verify pipeline picks a DSL adapter
    to build eval_<op>.py.

    Transport for Tier 2 (in priority order):
      1. `worker_url_override` (CLI `--worker-url`)
      2. `manifest.worker.urls[0]`
      3. `device_id_override` (CLI `--device-id`)
      4. local device 0 (with the usual ar_cli.py verify warning)
    """
    batch_dir = Path(batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    try:
        manifest_path = mf.find_manifest(batch_dir)
        manifest_data = mf.load_manifest(manifest_path)
    except mf.ManifestError as e:
        sys.exit(str(e))

    try:
        cases = mf.resolve_cases(batch_dir, manifest_data, "ref-kernel")
    except mf.ManifestError as e:
        sys.exit(f"manifest validation failed: {e}")

    dsl = (dsl_override or manifest_data.get("dsl") or "").strip()
    if full and not dsl:
        sys.exit("Tier 2 needs a DSL — declare `dsl:` in manifest or pass --dsl.")

    worker_block = manifest_data.get("worker") or {}
    manifest_worker_urls = worker_block.get("urls") or []
    if isinstance(manifest_worker_urls, str):
        manifest_worker_urls = [u.strip() for u in manifest_worker_urls.split(",")
                                if u.strip()]
    worker_url: Optional[str] = (worker_url_override.strip()
                                  if worker_url_override else None)
    if not worker_url and manifest_worker_urls:
        worker_url = manifest_worker_urls[0]

    only_set = {s.strip() for s in (only or "").split(",") if s.strip()}
    if only_set:
        cases = [c for c in cases if c["op_name"] in only_set]
        if not cases:
            sys.exit("--only filtered out all ops")

    transport_note = ""
    if full:
        if worker_url:
            transport_note = f"  worker={worker_url}"
        elif device_id_override is not None:
            transport_note = f"  device_id={device_id_override}"
        else:
            transport_note = "  device_id=0 (fallback)"
    print(f"verify  batch_dir={batch_dir}  "
          f"tier={'1+2' if full else '1'}  ops={len(cases)}"
          + (f"  dsl={dsl}{transport_note}" if full else ""))
    print()

    results: dict = {}
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        op = case["op_name"]
        sys.stdout.write(f"  [{i:>3}/{len(cases)}] {op} ... ")
        sys.stdout.flush()
        rec = _verify_one(case, full=full, dsl=dsl,
                          worker_url=worker_url,
                          device_id=device_id_override)
        results[op] = rec
        ok = _summary_status(rec, full=full)
        sys.stdout.write(f"{ok}\n")
        sys.stdout.flush()

    out_path = batch_dir / VERIFY_RESULTS
    out_path.write_text(json.dumps({
        "full": full, "dsl": dsl,
        "results": results,
    }, indent=2), encoding="utf-8")

    print()
    _print_table(results, full=full)
    print()

    n_pass = sum(1 for op in results
                 if _summary_status(results[op], full) == "P")
    n_fail = sum(1 for op in results
                 if _summary_status(results[op], full) == "F")
    n_err = sum(1 for op in results
                if _summary_status(results[op], full) == "E")
    print(f"  total={len(results)}  pass={n_pass}  fail={n_fail}  "
          f"error={n_err}  elapsed={time.time()-t0:.1f}s")
    print(f"  results: {out_path}")
    return 0 if (n_fail == 0 and n_err == 0) else 1


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--tier-worker":
        return _worker_main()

    ap = argparse.ArgumentParser(description="Pre-flight verify for batch directories.")
    ap.add_argument("batch_dir")
    ap.add_argument("--full", action="store_true",
                    help="also run Tier 2 (execute ref + kernel via "
                         "ar_cli.py verify --mode verify-only); needs the "
                         "same hardware /autoresearch eval would use")
    ap.add_argument("--only", default="",
                    help="comma-separated op names")
    ap.add_argument("--dsl", default="",
                    help="DSL for Tier 2; overrides manifest.dsl when set")
    ap.add_argument("--worker-url", default="",
                    help="Tier-2 transport: worker URL; overrides "
                         "manifest.worker.urls[0]")
    ap.add_argument("--device-id", type=int, default=None,
                    help="Tier-2 transport: local device id; used when "
                         "no worker URL resolved (default: 0)")
    args = ap.parse_args()

    return run_verification(
        Path(args.batch_dir),
        full=args.full, only=args.only, dsl_override=args.dsl,
        worker_url_override=args.worker_url,
        device_id_override=args.device_id,
    )


if __name__ == "__main__":
    sys.exit(main())
