"""Lightweight DSL-aware static checker used by autoresearch.

This keeps the local ``CodeChecker(backend, dsl).check(code)`` API shape
while using only in-tree validators. Triton delegates to the existing
validate_triton_impl checker; ascendc_catlass adds the core CATLASS
anti-cheat: ModelNew.forward must call torch.ops.catlass.* and must not
replace the kernel with hard torch compute operators.
"""
from __future__ import annotations

import ast
from typing import Optional

from eval.adapters.factory import get_dsl_adapter
from utils.validate_triton_impl import validate as validate_triton_impl

_TORCH_PREFIXES = (
    "torch",
    "torch.nn.functional",
    "F",
)
_HARD_TORCH_COMPUTE_OPS = frozenset({
    "matmul", "mm", "bmm", "addmm",
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "linear", "softmax", "layer_norm", "batch_norm", "group_norm",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "max_pool1d", "max_pool2d", "max_pool3d",
})


def _error(line: int, error_type: str, detail: str,
           suggestion: str = "", fix_strategy: str = "fix") -> dict:
    return {
        "line": line or 0,
        "error_type": error_type,
        "detail": detail,
        "suggestion": suggestion,
        "code_snippet": "",
        "fix_strategy": fix_strategy,
    }


def _format_errors(errors: list[dict]) -> str:
    lines = [f"CodeChecker found {len(errors)} issue(s):"]
    for err in errors:
        loc = f"L{err.get('line', 0)}"
        lines.append(
            f"- {loc} {err.get('error_type', 'error')}: "
            f"{err.get('detail', '')}"
        )
        if err.get("suggestion"):
            lines.append(f"  suggestion: {err['suggestion']}")
    return "\n".join(lines)


def _format_triton_report(result: dict) -> str:
    rtype = result.get("regression_type")
    type_desc = {
        1: "no @triton.jit kernel",
        2: "kernel defined but ModelNew.forward() never launches it",
        3: "forward() still uses torch compute",
    }.get(rtype, "unknown Triton regression")
    lines = [f"Triton regression check failed (type {rtype}: {type_desc})"]
    for name, sub in result.get("checks", {}).items():
        if sub.get("passed"):
            continue
        if sub.get("error"):
            lines.append(f"- {name}: {sub['error']}")
        for v in sub.get("violations", []) or []:
            lines.append(
                f"- L{v.get('line', '?')} {v.get('call', '?')}: "
                f"{v.get('reason', '')}"
            )
    if result.get("suggestion"):
        lines.append(f"suggestion: {result['suggestion']}")
    return "\n".join(lines)


def _triton_to_errors(result: dict) -> list[dict]:
    errors: list[dict] = []
    rtype = result.get("regression_type")
    for name, sub in result.get("checks", {}).items():
        if sub.get("passed"):
            continue
        if sub.get("error"):
            errors.append(_error(
                0,
                f"regression_type_{rtype}_{name}",
                sub["error"],
                result.get("suggestion", ""),
                "rewrite",
            ))
        for v in sub.get("violations", []) or []:
            errors.append(_error(
                v.get("line", 0),
                f"regression_type_{rtype}_{name}",
                f"{v.get('call', '?')}: {v.get('reason', '')}",
                result.get("suggestion", ""),
                "rewrite",
            ))
    return errors


def _dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


def _find_model_new(tree: ast.Module) -> Optional[ast.ClassDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            return node
    return None


def _find_forward(cls_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in cls_node.body:
        if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "forward"):
            return item
    return None


def _is_torch_call(call_name: str) -> bool:
    return any(call_name == p or call_name.startswith(p + ".")
               for p in _TORCH_PREFIXES)


def _check_catlass(tree: ast.Module, dsl: str) -> list[dict]:
    model = _find_model_new(tree)
    if model is None:
        return [_error(
            0,
            "missing_model_new",
            "DSL is ascendc_catlass but no ModelNew class was found.",
            "Expose a ModelNew class whose forward() calls torch.ops.catlass.*.",
            "rewrite",
        )]
    forward = _find_forward(model)
    if forward is None:
        return [_error(
            model.lineno,
            "missing_forward",
            "ModelNew has no forward() method.",
            "Implement ModelNew.forward() and call torch.ops.catlass.* there.",
            "rewrite",
        )]

    errors: list[dict] = []
    has_catlass_call = False
    hard_calls: list[tuple[int, str]] = []
    for node in ast.walk(forward):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            call_name = _dotted_name(node.func)
            if not call_name:
                continue
            if call_name.startswith("torch.ops.catlass."):
                has_catlass_call = True
                continue
            if _is_torch_call(call_name) and node.func.attr in _HARD_TORCH_COMPUTE_OPS:
                hard_calls.append((node.lineno, call_name))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            hard_calls.append((node.lineno, "@"))

    if not has_catlass_call:
        errors.append(_error(
            0,
            "no_catlass_call",
            f"DSL is {dsl}, but ModelNew.forward() does not call torch.ops.catlass.*.",
            "Call the compiled CATLASS kernel via torch.ops.catlass.<op_name>(...).",
            "rewrite",
        ))
    for line, call_name in hard_calls:
        errors.append(_error(
            line,
            "torch_api_instead_of_catlass_kernel",
            f"ModelNew.forward() uses hard torch compute op {call_name}.",
            "Move core compute into the CATLASS kernel and keep forward() as a wrapper.",
            "rewrite",
        ))
    return errors


class CodeChecker:
    """Small compatibility wrapper for AKG's CodeChecker API."""

    def __init__(self, backend: str, dsl: str, arch: str = "",
                 config: Optional[dict] = None):
        self.backend = (backend or "").lower()
        self.dsl = (dsl or "").lower()
        self.arch = (arch or "").lower()
        self.config = config or {}

    def check(self, code: str, task_info: Optional[dict] = None) -> tuple:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors = [_error(
                e.lineno or 0,
                "syntax_error",
                e.msg,
                "Fix the Python syntax error before running eval.",
            )]
            return False, _format_errors(errors), errors

        try:
            adapter = get_dsl_adapter(self.dsl)
        except Exception:
            adapter = None
        if adapter is not None and not adapter.static_check_via_python_ast:
            return True, "", []

        if self.dsl in ("triton_cuda", "triton_ascend", "triton-russia"):
            result = validate_triton_impl(code)
            if result.get("valid"):
                return True, "", []
            errors = _triton_to_errors(result)
            return False, _format_triton_report(result), errors

        if self.dsl == "ascendc_catlass":
            errors = _check_catlass(tree, self.dsl)
            if errors:
                return False, _format_errors(errors), errors

        return True, "", []
