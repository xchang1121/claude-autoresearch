# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""task_config package facade.

Layout:

    loader         TaskConfig dataclass + task.yaml parsing.
    metric_policy  EvalOutcome/EvalResult and metric comparison helpers.
    eval_client    Public run_eval entry point. It delegates to
                   utils.akg_eval and the formal KernelVerifier chain.

Only names imported by outside packages are re-exported here. Historical
CA-only eval transport helpers were removed; callers should not import
private task_config modules for eval dispatch.
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
