# Copyright 2025 Huawei Technologies Co., Ltd
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

"""In-tree eval package (migrated from akg_agents.op.verifier).

Layout in this package — what akg_agents used to spread across multiple
subpackages now lives flat under `scripts/eval/`:

    kernel_verifier.py      <- akg_agents.op.verifier.kernel_verifier
    sol_verifier.py         <- akg_agents.op.verifier.sol_verifier
    profiler.py             <- akg_agents.op.verifier.profiler
    profiler_utils.py       <- akg_agents.op.verifier.profiler_utils
    roofline_utils.py       <- akg_agents.op.verifier.roofline_utils
    baseline_profiler.py    <- akg_agents.op.verifier.baseline_profiler
    data_cache.py           <- akg_agents.op.verifier.data_cache
    l2_cache_clear.py       <- akg_agents.op.verifier.l2_cache_clear
    kernel_verifier_patch.py <- akg_agents.op.verifier.kernel_verifier_patch
    validate_triton_impl.py <- skills/triton/kernel-verifier/scripts/validate_triton_impl.py
                               (pure-AST Triton regression checker; not part
                               of the verifier core but distributed with it)

    adapters/{factory,backend/*,dsl/*,framework/*}.py
                            <- akg_agents.op.verifier.adapters.*

    worker/{interface, local_worker, manager, remote_worker, device_pool}.py
                            <- akg_agents.core.worker.* + core.async_pool.device_pool

    config_utils.py         <- akg_agents.op.utils.config_utils
    process_utils.py        <- akg_agents.utils.process_utils
    triton_ascend_api_docs.py <- akg_agents.op.utils.triton_ascend_api_docs
    json_safe.py            <- akg_agents.op.utils.json_safe

    templates/*.j2          <- akg_agents.op.resources.templates
    compile/ascend/         <- akg_agents.utils.compile_tools.ascend_compile

`get_project_root()` returns this package directory so the Jinja2 + CMake
templates can locate their siblings without depending on the akg_agents
install layout.
"""
import os


def get_project_root() -> str:
    """Return the in-tree eval package root.

    Used by `kernel_verifier.py` / `sol_verifier.py` to locate template
    files at `<root>/templates/*.j2` and CMake/run.sh at
    `<root>/compile/ascend/`. In akg_agents this pointed at the
    `akg_agents/` package; here it points at `scripts/eval/`.
    """
    return os.path.dirname(os.path.abspath(__file__))
