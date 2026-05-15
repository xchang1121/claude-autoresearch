"""task_config package — facade over four single-concern submodules.

The previous flat task_config.py was 1099 lines mixing YAML parsing,
verify/profile script code generation, tar.gz building, HTTP transport,
local subprocess execution, and metric arithmetic. Each grew its own
section over time but they didn't actually share data structures or
helpers — splitting them surfaces the dependency direction and lets
each layer be reasoned about (and changed) independently.

Layout:

    loader            — TaskConfig dataclass + load_task_config (YAML
                        parsing). No internal deps; everyone else
                        consumes TaskConfig from here.
    metric_policy     — EvalResult, is_improvement, check_constraints,
                        format_result_summary. Pure data + arithmetic;
                        no I/O. Imported by keep_or_discard, dashboard.
    package_builder   — DSL-adapter resolution, verify/profile script
                        generation, tar.gz assembly. Depends on loader.
    eval_client       — Worker URL discovery, HTTP transport, local-subprocess
                        transport, run_eval dispatcher, result assembly.
                        Depends on loader + metric_policy + package_builder.

This `__init__.py` re-exports only the names actually imported from
outside the package. Submodule helpers (script-template generators,
HTTP plumbing, metric arithmetic guards) stay private — reach into the
submodule explicitly when you need them. Earlier this file mirrored
the legacy flat-module API and re-exported every underscore helper "in
case"; that hid the real coupling story and meant submodule renames
rippled through here.
"""
# fmt: off
from .loader import (
    TaskConfig, load_task_config,
)
from .metric_policy import (
    EvalOutcome, EvalResult, check_constraints, is_improvement, format_result_summary,
)
from .eval_client import (
    run_eval, run_remote_eval, run_local_eval,
)
# fmt: on
