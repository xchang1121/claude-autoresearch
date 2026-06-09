"""Pre-flight verification for batch directories.

Tier 1 (default, no hardware): compile + import + required-symbol check
on ref.py (Model + get_inputs + get_init_inputs) and kernel.py (ModelNew).
Kernel.py also runs the DSL-aware static CodeChecker. Triton kernels must
use and launch @triton.jit kernels; CATLASS wrappers must call
torch.ops.catlass.* instead of replacing the kernel with torch compute.

Tier 2 (--full): FORMAL verify-only pass. It materializes a temporary
task directory and calls ``scripts/engine/eval_kernel.py --phases verify``,
so it reuses the same ``KernelVerifier`` + ``DSLAdapter`` chain as batch
eval / worker correctness, minus profiling.

For multi-file DSLs (ascendc_catlass), ``case["kernel"]`` is a directory
that gets passed to ``/autoresearch --kernel``, while
``case["kernel_module"]`` is the sibling ``kernel.py`` (or
``<op>_kernel.py``) that tier-1 imports and tier-2 verifies.

Each op runs in its own subprocess. Results: <batch_dir>/verify_results.json.

Usage:
    python scripts/batch/verify.py <batch_dir>             # Tier 1
    python scripts/batch/verify.py <batch_dir> --full      # Tier 1 + Tier 2
    python scripts/batch/verify.py <batch_dir> --only op1,op2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf
# Reach up one level for shared batch/eval helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.hw_detect import derive_arch  # noqa: E402
from utils.code_checker import CodeChecker  # noqa: E402
from utils.settings import (  # noqa: E402
    batch_tier1_timeout, batch_tier2_timeout, target_backend, target_dsl,
)

VERIFY_RESULTS = "verify_results.json"
# Timeouts (seconds) from config.yaml `batch:`. tier2 cold JIT across ~50
# cases on triton-ascend can take minutes for kernels with many constexpr
# specializations (maxpool3d, ...); warm runs land in tens of seconds.
TIER1_TIMEOUT = batch_tier1_timeout()
TIER2_TIMEOUT = batch_tier2_timeout()

# Reference must export Model + get_init_inputs + one of (get_inputs,
# get_input_groups). The "input provider" is checked separately (per
# input_groups.resolve duck-type) since either symbol satisfies it.
REF_REQUIRED = ("Model", "get_init_inputs")
REF_INPUT_PROVIDERS = ("get_inputs", "get_input_groups")
KERNEL_REQUIRED = ("ModelNew",)


# ---------------------------------------------------------------------------
# Subprocess pool (this same file is re-invoked with --tier-runner)
# ---------------------------------------------------------------------------
def _tier1_inspect(path: Path, required: tuple[str, ...]) -> dict:
    """Compile, import, check required attrs are present, and (for
    kernel files only) run the DSL-aware static checker."""
    out: dict = {"path": str(path), "compile": "skip", "import": "skip",
                 "exports": "skip", "validate": "skip", "missing": [], "msg": ""}
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
        return out
    out["exports"] = "PASS"

    # Step 4: DSL-aware static check (kernel-only). Refs legitimately use
    # framework-native code; kernels must satisfy the selected DSL's
    # anti-cheat rule.
    if required is KERNEL_REQUIRED:
        passed, error_msg, errors = CodeChecker(
            backend=target_backend(), dsl=target_dsl()
        ).check(src)
        if passed:
            out["validate"] = "PASS"
        else:
            out["validate"] = "FAIL"
            if errors:
                first = errors[0]
                out["msg"] = (
                    f"L{first.get('line', 0)} "
                    f"{first.get('error_type', '?')}: "
                    f"{first.get('detail', '')}"
                )[:160]
            else:
                out["msg"] = (error_msg.splitlines()[0] if error_msg
                              else "static check failed")[:160]
    return out


def _tier2_run(ref_path: Path, kernel_path: Path) -> dict:
    """Run the formal KernelVerifier-backed verify path on a temp task dir."""
    out: dict = {"status": "skip", "msg": "", "max_abs_diff": None}

    try:
        import yaml
        from eval.adapters.factory import get_dsl_adapter
        from task_config.task_layout import REF_FILE_DEFAULT
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"formal verify import failed: {type(e).__name__}: {e}"
        return out

    def _op_name_from_ref(path: Path) -> str:
        stem = path.stem
        return stem[:-4] if stem.endswith("_ref") else stem

    def _copy_ref_sidecars(src_ref: Path, task_dir: Path) -> list[str]:
        copied: list[str] = []
        ref_stem = src_ref.stem
        dst_ref_stem = Path(REF_FILE_DEFAULT).stem
        for sibling in sorted(src_ref.parent.iterdir()):
            if sibling.name.startswith(".") or not sibling.is_file():
                continue
            ext = sibling.suffix.lower()
            if ext not in (".json", ".pt", ".npz"):
                continue
            stem = sibling.stem
            if stem != ref_stem and not stem.startswith(ref_stem + "_"):
                continue
            dest_name = f"{dst_ref_stem}{ext}" if stem == ref_stem else sibling.name
            shutil.copy2(sibling, task_dir / dest_name)
            copied.append(dest_name)
        return copied

    op_name = _op_name_from_ref(ref_path)
    backend = target_backend()
    dsl = target_dsl()
    arch = derive_arch(0, backend=backend)
    if not arch:
        out["status"] = "ERROR"
        out["msg"] = "could not derive local arch via hardware probe"
        return out

    adapter = get_dsl_adapter(dsl)
    project_name = adapter.kernel_project_dir_name or "catlass_op"
    temp_root = Path(tempfile.mkdtemp(prefix=f"_batch_verify_{op_name}_"))
    task_dir = temp_root / "task"
    sidecar = task_dir / ".eval_result.json"
    try:
        task_dir.mkdir(parents=True, exist_ok=True)
        entry_name = adapter.entry_filename_template.format(op_name=op_name)
        shutil.copy2(ref_path, task_dir / REF_FILE_DEFAULT)
        shutil.copy2(kernel_path, task_dir / entry_name)

        editable_files = [entry_name]
        if adapter.kernel_arg_is_directory:
            project_src = kernel_path.parent / project_name
            shutil.copytree(project_src, task_dir / project_name)
            for rel in getattr(adapter, "kernel_project_files", []) or []:
                if rel not in editable_files:
                    editable_files.append(rel)

        data_files = _copy_ref_sidecars(ref_path, task_dir)
        task_yaml = {
            "name": op_name,
            "arch": arch,
            "editable_files": editable_files,
            "devices": [0],
            "eval": {"timeout": TIER2_TIMEOUT},
            "agent": {
                "ref_file": REF_FILE_DEFAULT,
                "max_rounds": 1,
            },
        }
        if data_files:
            task_yaml["data_files"] = data_files
        if dsl == "ascendc_catlass":
            task_yaml["catlass"] = {"op_dir": project_name}

        (task_dir / "task.yaml").write_text(
            yaml.safe_dump(task_yaml, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

        scripts_dir = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable,
            str(scripts_dir / "engine" / "eval_kernel.py"),
            "--task-dir", str(task_dir),
            "--op-name", op_name,
            "--kernel-file", Path(entry_name).stem,
            "--ref-file", Path(REF_FILE_DEFAULT).stem,
            "--device-id", "0",
            "--backend", backend,
            "--dsl", dsl,
            "--arch", arch,
            "--verify-timeout", str(TIER2_TIMEOUT),
            "--phases", "verify",
            "--output", str(sidecar),
        ]
        if dsl == "ascendc_catlass":
            cmd += ["--catlass-op-dir", project_name]

        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIER2_TIMEOUT + 30,
                env=env,
            )
        except subprocess.TimeoutExpired:
            out["status"] = "ERROR"
            out["msg"] = f"formal verify timeout after {TIER2_TIMEOUT + 30}s"
            return out

        if not sidecar.exists():
            out["status"] = "ERROR"
            combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
            out["msg"] = f"formal verify produced no sidecar; rc={proc.returncode}"
            out["stderr_tail"] = combined[-1000:]
            return out

        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"formal verify setup failed: {type(e).__name__}: {e}"
        return out
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    verify = raw.get("verify") if isinstance(raw, dict) else None
    verify = verify if isinstance(verify, dict) else {}
    per_case = list(verify.get("per_case") or [])
    out["max_abs_diff"] = verify.get("worst_max_abs_diff")
    out["num_cases"] = int(verify.get("num_cases") or len(per_case) or 0)
    out["error_source"] = verify.get("error_source")
    out["failed_indices"] = list(verify.get("failed_indices") or [])
    if per_case:
        out["per_case"] = per_case
    diagnostics = list(verify.get("diagnostics") or [])
    if diagnostics:
        out["diagnostics"] = diagnostics

    if raw.get("ok") and verify.get("correctness"):
        out["status"] = "PASS"
        out["msg"] = f"OK (n={out['num_cases']})"
        return out

    failure_kinds = {
        str(item.get("failure_kind") or "")
        for item in per_case
        if isinstance(item, dict)
    }
    kernel_failure = (
        verify.get("error_source") == "kernel"
        or "kernel_miss" in failure_kinds
        or any(isinstance(item, dict) and item.get("correctness") is False for item in per_case)
    )
    out["status"] = "FAIL" if kernel_failure else "ERROR"

    for item in per_case:
        if isinstance(item, dict) and item.get("error"):
            out["msg"] = str(item["error"])[:160]
            break
    else:
        if diagnostics:
            out["msg"] = str(diagnostics[0])[:160]
        else:
            errors = raw.get("errors") if isinstance(raw, dict) else None
            out["msg"] = str(errors or raw.get("ok") or "formal verify failed")[:160]
    return out

def _run_tier_subprocess() -> int:
    """Subprocess entry point. Writes JSON to a sidecar path on stdout's last line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("1ref", "1kernel", "2"), required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", default="")
    ap.add_argument("--sidecar", required=True)
    args = ap.parse_args(sys.argv[2:])  # skip the --tier-runner sentinel

    ref_path = Path(args.ref)
    kernel_path = Path(args.kernel) if args.kernel else None

    if args.tier == "1ref":
        result = _tier1_inspect(ref_path, REF_REQUIRED)
    elif args.tier == "1kernel":
        result = _tier1_inspect(kernel_path, KERNEL_REQUIRED)
    else:  # tier == "2"
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
           "--tier-runner",
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
                              encoding="utf-8", errors="replace",
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
    # Multi-file DSLs pass a project directory as case["kernel"] to
    # /autoresearch, but verification imports the sibling Python wrapper.
    kernel = Path(case.get("kernel_module") or case["kernel"])

    out: dict = {"op_name": op, "tier1_ref": None, "tier1_kernel": None,
                 "tier2": None}

    out["tier1_ref"] = _run_subprocess(tier="1ref", ref=ref, kernel=None,
                                       timeout=TIER1_TIMEOUT)
    out["tier1_kernel"] = _run_subprocess(tier="1kernel", ref=ref,
                                          kernel=kernel,
                                          timeout=TIER1_TIMEOUT)

    def _t1_pass(rec: dict | None) -> bool:
        if not rec:
            return False
        # validate is "skip" for ref-side records (only kernel runs it).
        return (rec.get("exports") == "PASS"
                and rec.get("validate") != "FAIL")
    tier1_ok = _t1_pass(out["tier1_ref"]) and _t1_pass(out["tier1_kernel"])

    if full:
        if tier1_ok:
            out["tier2"] = _run_subprocess(tier="2", ref=ref, kernel=kernel,
                                           timeout=TIER2_TIMEOUT + 60)
        else:
            out["tier2"] = {"status": "skip",
                            "msg": "tier1 failed; skipping tier2",
                            "elapsed_s": 0}

    return out


_CONTENT_FAIL_FIELDS = ("compile", "import", "exports", "validate")


def _summary_status(record: dict, full: bool) -> str:
    """P/F/E/S single-letter. compile/import/exports/validate failures
    all map to F (matches the per-tier table column); runtime ERROR is E."""
    t1r = record["tier1_ref"]
    t1k = record["tier1_kernel"]
    t2 = record["tier2"]

    def _bad(t):
        return t and ("FAIL" in (t.get("compile"), t.get("import"),
                                  t.get("exports"), t.get("validate"))
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
            if t1k.get("validate") == "FAIL":
                col_t1k = "FAIL"
            elif t1k.get("exports") == "PASS":
                col_t1k = "PASS"
            elif t1k.get("exports") == "FAIL":
                col_t1k = "FAIL"
            elif t1k.get("compile") == "FAIL" or t1k.get("import") == "FAIL":
                col_t1k = "FAIL"
            else:
                col_t1k = "ERROR"
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
    still goes to stdout for the caller to surface."""
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
          f"precision: allclose-style (per-dtype rtol+atol)")
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
        "precision": "npu-benchmark-mere-mare",
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
    if len(sys.argv) >= 2 and sys.argv[1] == "--tier-runner":
        return _run_tier_subprocess()

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
