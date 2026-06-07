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

"""In-tree eval package used by claude-autoresearch.

The verifier, profiler, adapters, worker helpers, templates, and Ascend
compile helpers all live under `scripts/eval/`. `get_project_root()`
returns this package directory so those resources can locate their
siblings without relying on an external package layout.
"""
import os


def get_project_root() -> str:
    """Return the in-tree eval package root.

    Used by `kernel_verifier.py` / `sol_verifier.py` to locate template
    files at `<root>/templates/*.j2` and CMake/run.sh at
    `<root>/compile/ascend/`.
    """
    return os.path.dirname(os.path.abspath(__file__))
