"""Thin re-export of the in-tree validate_triton_impl AST checker.

The canonical implementation lives at `scripts/eval/validate_triton_impl.py`
— migrated in-tree with the rest of the eval package (see scripts/eval/SPEC.md).

Earlier revisions loaded this from an out-of-tree
`skills/triton/kernel-verifier/scripts/validate_triton_impl.py`; the eval
package is now the single source of truth, so callers do not have to
encode the skills-as-sibling layout assumption anymore.
"""
import os
import sys

from .external_paths import eval_dir

_EVAL_DIR = eval_dir()
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

# Star-import re-exports public names + module-level constants. The explicit
# re-imports below are insurance for the names autoresearch's callers reach
# for by name (`from utils.validate_triton_impl import validate as ...`).
from validate_triton_impl import *  # noqa: F401, F403
from validate_triton_impl import (  # noqa: F401
    validate,
    ALLOWED_TORCH_FUNCS,
    ALLOWED_TENSOR_METHODS,
    ALLOWED_TRITON_ATTRS,
    FORBIDDEN_TENSOR_METHODS,
    FORBIDDEN_PYTHON_STMTS,
)
