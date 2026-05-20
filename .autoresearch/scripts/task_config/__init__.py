"""task_config package — facade over single-concern submodules.

Layout:

    loader          TaskConfig dataclass + load_task_config (YAML parsing).
    metric_policy   EvalResult / EvalOutcome / is_improvement /
                    check_constraints / format_result_summary.
    package_builder DSL-adapter resolution, verify/profile script
                    generation, tar.gz assembly.
    eval_request    Case-count, timeout, sticky-baseline request building.
    eval_transport  Raw HTTP/local execution.
    eval_assemble   Raw response -> EvalResult metric semantics.
    eval_client     Public run_eval routing facade.

Only names imported from outside the package are re-exported here;
submodule helpers stay private.
"""
# fmt: off
from .loader import TaskConfig, load_task_config
from .metric_policy import (
    EvalOutcome, EvalResult, check_constraints, is_improvement,
    format_result_summary,
)
from .eval_client import run_eval, run_remote_eval, run_local_eval
# fmt: on
