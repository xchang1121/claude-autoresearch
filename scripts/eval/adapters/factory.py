# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""Adapter factories: single source of truth for DSL adapter registration
and the per-framework/backend/arch-family support matrix.

``DSL_REGISTRY`` maps DSL names and aliases to adapter classes, and its
``support`` tuples drive ``eval.config_utils.VALID_CONFIGS``.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DSLEntry:
    module: str
    cls: str
    aliases: Tuple[str, ...] = ()
    support: Tuple[Tuple[str, str, str], ...] = ()


# family tag is interpreted by eval.config_utils. Keep framework dimensions
# explicit: torch/910 support does not imply mindspore/910 support.
_TORCH_ASCEND_910 = (("torch", "ascend", "910"),)
_TORCH_ASCEND_310 = (("torch", "ascend", "310"),)
_MINDSPORE_ASCEND_910 = (("mindspore", "ascend", "910"),)
_MINDSPORE_ASCEND_310 = (("mindspore", "ascend", "310"),)
_NUMPY_ASCEND_310 = (("numpy", "ascend", "310"),)
_TORCH_CUDA = (("torch", "cuda", "any"),)
_TORCH_CPU = (("torch", "cpu", "any"),)


# Adding a new DSL = one entry here. ``module`` is the submodule under
# ``eval.adapters.dsl``; ``cls`` is the adapter class name inside it.
DSL_REGISTRY: dict = {
    "triton_cuda": DSLEntry(
        "triton_cuda", "DSLAdapterTritonCuda", support=_TORCH_CUDA),
    "triton_ascend": DSLEntry(
        "triton_ascend", "DSLAdapterTritonAscend",
        aliases=("triton-russia",),
        support=_TORCH_ASCEND_910 + _MINDSPORE_ASCEND_910,
    ),
    "swft": DSLEntry(
        "swft", "DSLAdapterSwft",
        support=(
            _TORCH_ASCEND_310
            + _MINDSPORE_ASCEND_310
            + _NUMPY_ASCEND_310
        ),
    ),
    "ascendc": DSLEntry(
        "ascendc", "DSLAdapterAscendC",
        support=_TORCH_ASCEND_910 + _TORCH_ASCEND_310,
    ),
    "ascendc_catlass": DSLEntry(
        "ascendc_catlass", "DSLAdapterAscendC_Catlass",
        support=_TORCH_ASCEND_910 + _TORCH_ASCEND_310,
    ),
    "cpp": DSLEntry("cpp", "DSLAdapterCpp", support=_TORCH_CPU),
    "cuda_c": DSLEntry("cuda_c", "DSLAdapterCudaC", support=_TORCH_CUDA),
    "tilelang_npuir": DSLEntry(
        "tilelang_npuir", "DSLAdapterTilelangNpuir",
        support=_TORCH_ASCEND_910,
    ),
    "tilelang_cuda": DSLEntry(
        "tilelang_cuda", "DSLAdapterTilelangCuda", support=_TORCH_CUDA),
    "torch": DSLEntry(
        "torch", "DSLAdapterTorch",
        support=_TORCH_ASCEND_910 + _TORCH_ASCEND_310 + _TORCH_CUDA,
    ),
    "pypto": DSLEntry(
        "pypto", "DSLAdapterPypto", support=_TORCH_ASCEND_910),
}

_DSL_ALIAS_MAP: dict = {}
for _name, _entry in DSL_REGISTRY.items():
    _DSL_ALIAS_MAP[_name.lower()] = _name
    for _alias in _entry.aliases:
        _DSL_ALIAS_MAP[_alias.lower()] = _name


def get_dsl_adapter(dsl: str):
    """Get DSL adapter by name (or alias)."""
    canonical = _DSL_ALIAS_MAP.get(dsl.lower())
    if canonical is None:
        raise ValueError(f"Unsupported DSL: {dsl}")
    entry = DSL_REGISTRY[canonical]
    module = __import__(f"eval.adapters.dsl.{entry.module}",
                        fromlist=[entry.cls])
    return getattr(module, entry.cls)()


def get_framework_adapter(framework: str):
    """Get framework adapter by name (torch / mindspore / numpy)."""
    framework_lower = framework.lower()
    if framework_lower == "torch":
        from .framework.torch import FrameworkAdapterTorch
        return FrameworkAdapterTorch()
    if framework_lower == "mindspore":
        from .framework.mindspore import FrameworkAdapterMindSpore
        return FrameworkAdapterMindSpore()
    if framework_lower == "numpy":
        from .framework.numpy import FrameworkAdapterNumpy
        return FrameworkAdapterNumpy()
    raise ValueError(f"Unsupported framework: {framework}")


def get_backend_adapter(backend: str):
    """Get backend adapter by name (cuda / ascend / cpu)."""
    backend_lower = backend.lower()
    if backend_lower == "cuda":
        from .backend.cuda import BackendAdapterCuda
        return BackendAdapterCuda()
    if backend_lower == "ascend":
        from .backend.ascend import BackendAdapterAscend
        return BackendAdapterAscend()
    if backend_lower == "cpu":
        from .backend.cpu import BackendAdapterCpu
        return BackendAdapterCpu()
    raise ValueError(f"Unsupported backend: {backend}")
