"""task_config package — facade over three single-concern submodules.

Layout:

    loader            — TaskConfig dataclass + load_task_config (YAML
                        parsing). No internal deps; everyone else
                        consumes TaskConfig from here.
    metric_policy     — EvalResult, is_improvement, check_constraints,
                        format_result_summary. Pure data + arithmetic;
                        no I/O. Imported by keep_or_discard, dashboard.
    eval_client       — Local subprocess + remote HTTP transport,
                        result assembly. Depends on loader +
                        metric_policy. Local drives the static
                        `eval_kernel.py` via `eval_runner.local_eval`;
                        remote ships a `package_builder` tar.gz to a
                        worker `/api/v1/run` endpoint.
    package_builder   — task.yaml + ref + editable → tar.gz bytes,
                        for the remote transport. No deps outside loader.

This `__init__.py` re-exports only the names actually imported from
outside the package. Submodule-private helpers (operator tables,
internal result assembly) are not re-laundered through the facade —
reach into the submodule explicitly when you need them.
"""
# fmt: off
from .loader import (
    TaskConfig, load_task_config,
    REF_FILE_DEFAULT, py_stem,
)
from .metric_policy import (
    EvalOutcome, EvalResult, check_constraints, is_improvement, format_result_summary,
)
from .eval_client import run_eval
# fmt: on
