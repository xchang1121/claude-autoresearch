"""Pre-flight verification for batch directories.

Tier 1 (default, no hardware): compile + import + required-symbol check
on ref.py (Model + get_inputs + get_init_inputs) and kernel.py (ModelNew).

Tier 2 (--full): LOCAL smoke test. Loads both modules on `torch.npu:0`
(or CPU), runs them, allclose via `utils.correctness`. NOT a proxy for
the batch eval path — `batch/run.py` defaults to a remote worker, and
a green --full says only "kernel runs locally".

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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf
# Reach up one level so we can import the shared correctness +
# input_groups modules the eval-package verify script also uses. Single
# source of truth for the allclose comparison — verify.py and
# autoresearch's eval can't drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.correctness import compare_outputs_per_case  # noqa: E402
from utils.input_groups import resolve as _resolve_groups  # noqa: E402

VERIFY_RESULTS = "verify_results.json"
TIER1_TIMEOUT = 30
# Cold JIT compile across many cases on triton-ascend can take several
# minutes for kernels with constexpr-driven specialisations. Bump the
# Tier-2 cap so multi-shape verifies aren't killed mid-loop.
TIER2_TIMEOUT = 600
# atol/rtol are imported from correctness — the single source of truth.
# Previously this file accepted --atol/--rtol CLI flags + a
# manifest.correctness_atol/_rtol override; those entry points were
# removed to eliminate drift between batch verify and the per-round
# worker verify.

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


def _tier2_run(ref_path: Path, kernel_path: Path) -> dict:
    """Run ref + kernel, compare outputs via the shared correctness module
    (autoresearch's eval calls into the same `compare_outputs`).

    Tolerances are per-dtype, derived from each ref tensor's dtype inside
    `compare_outputs_per_case` — no longer worker arguments. See
    `utils/correctness.py` for the table.
    """
    out: dict = {"status": "skip", "msg": "", "max_abs_diff": None}

    try:
        import torch  # type: ignore
    except ImportError as e:
        out["status"] = "ERROR"
        out["msg"] = f"torch import failed: {e}"
        return out
    try:
        import torch_npu  # type: ignore  # noqa: F401
    except Exception:
        pass  # not on Ascend; fine — kernel will pick its own device

    import importlib.util
    try:
        ref_spec = importlib.util.spec_from_file_location("_v_ref", str(ref_path))
        ref_mod = importlib.util.module_from_spec(ref_spec)  # type: ignore[arg-type]
        ref_spec.loader.exec_module(ref_mod)  # type: ignore[union-attr]
        kernel_spec = importlib.util.spec_from_file_location("_v_kernel", str(kernel_path))
        kernel_mod = importlib.util.module_from_spec(kernel_spec)  # type: ignore[arg-type]
        kernel_spec.loader.exec_module(kernel_mod)  # type: ignore[union-attr]
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"import: {type(e).__name__}: {e}"
        return out

    try:
        init_args = ref_mod.get_init_inputs()
        cases = _resolve_groups(ref_mod)
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"get_inputs/get_input_groups: {type(e).__name__}: {e}"
        return out

    # NPU when available; CPU otherwise. Kernels allocate output buffers
    # via torch.empty_like(x), so CPU input → CPU output buffer → garbage.
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None and getattr(npu_mod, "is_available", lambda: False)():
        device = torch.device("npu:0")
    else:
        device = torch.device("cpu")

    try:
        ref = ref_mod.Model(*init_args).to(device).eval()
        new = kernel_mod.ModelNew(*init_args).to(device).eval()
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"construct: {type(e).__name__}: {e}"
        return out

    def _to_list(x):
        if isinstance(x, (tuple, list)):
            return list(x)
        return [x]

    def _to_device(seq):
        return [t.to(device) if hasattr(t, "to") else t for t in seq]

    out_ref_per_case: list = []
    out_new_per_case: list = []
    try:
        with torch.no_grad():
            for case in cases:
                inp = _to_device(list(case))
                out_ref_per_case.append(_to_list(ref(*inp)))
                out_new_per_case.append(_to_list(new(*inp)))
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"forward: {type(e).__name__}: {e}"
        return out

    cmp = compare_outputs_per_case(out_ref_per_case, out_new_per_case)
    out["max_abs_diff"] = cmp["max_abs_diff"]
    out["num_cases"] = len(cases)
    out["per_case"] = cmp["per_case"]
    if cmp["correctness"]:
        out["status"] = "PASS"
        out["msg"] = f"OK (n={len(cases)})"
    else:
        out["status"] = "FAIL"
        # Surface the first failing diagnostic — the full list goes into the
        # JSON output for verify_results.json.
        bad = next((d for d in cmp["diagnostics"] if "OK" not in d), None)
        out["msg"] = bad or "correctness mismatch (no diagnostics)"
        out["diagnostics"] = cmp["diagnostics"]
    return out


def _worker_main() -> int:
    """Subprocess entry point. Writes JSON to a sidecar path on stdout's last line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("1ref", "1kernel", "2"), required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", default="")
    ap.add_argument("--sidecar", required=True)
    args = ap.parse_args(sys.argv[2:])  # skip the --tier-worker sentinel

    ref_path = Path(args.ref)
    kernel_path = Path(args.kernel) if args.kernel else None

    if args.tier == "1ref":
        result = _tier1_inspect(ref_path, REF_REQUIRED)
    elif args.tier == "1kernel":
        result = _tier1_inspect(kernel_path, KERNEL_REQUIRED)
    else:  # tier == "2"
        # atol/rtol are no longer worker arguments — locked to
        # correctness.DEFAULT_ATOL / DEFAULT_RTOL.
        result = _tier2_run(ref_path, kernel_path)

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


def _verify_one(case: dict, full: bool) -> dict:
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
            out["tier2"] = _run_subprocess(tier="2", ref=ref, kernel=kernel,
                                           timeout=TIER2_TIMEOUT)
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
                     only: str = "") -> int:
    """Run the verification loop programmatically (so prepare.py and other
    scripts can call us without subprocessing). Returns the same exit code
    main() would: 0 if everything passed, 1 if any FAIL/ERROR. All output
    still goes to stdout for the caller to surface.

    atol/rtol are not configurable — they're constants in correctness.py.
    Manifest's `correctness_atol` / `correctness_rtol` keys are now
    ignored; the CLI no longer exposes overrides."""
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

    only_set = {s.strip() for s in (only or "").split(",") if s.strip()}
    if only_set:
        cases = [c for c in cases if c["op_name"] in only_set]
        if not cases:
            sys.exit("--only filtered out all ops")

    print(f"verify  batch_dir={batch_dir}  "
          f"tier={'1+2' if full else '1'}  ops={len(cases)}  "
          f"tols: per-dtype (see utils/correctness.py)")
    print()

    results: dict = {}
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        op = case["op_name"]
        sys.stdout.write(f"  [{i:>3}/{len(cases)}] {op} ... ")
        sys.stdout.flush()
        rec = _verify_one(case, full=full)
        results[op] = rec
        ok = _summary_status(rec, full=full)
        sys.stdout.write(f"{ok}\n")
        sys.stdout.flush()

    out_path = batch_dir / VERIFY_RESULTS
    out_path.write_text(json.dumps({
        "full": full,
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
                    help="also run Tier 2 (execute ref + kernel, compare outputs); "
                         "needs the same hardware /autoresearch eval would use")
    ap.add_argument("--only", default="",
                    help="comma-separated op names")
    args = ap.parse_args()

    return run_verification(
        Path(args.batch_dir),
        full=args.full, only=args.only,
    )


if __name__ == "__main__":
    sys.exit(main())
