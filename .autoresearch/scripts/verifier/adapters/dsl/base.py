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

"""Base class for DSL adapters.

Each concrete DSL (triton_ascend, tilelang_cuda, ascendc, ...) lives in
its own module under ``dsl/`` and registers itself via ``@register_dsl``.
``factory.get_dsl_adapter(name)`` looks adapters up in ``_DSL_REGISTRY``
after triggering one auto-import sweep — adding a new DSL is therefore a
single-file change: drop ``dsl/foo.py`` with ``@register_dsl("foo")``,
no factory edit required.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Type


_DSL_REGISTRY: Dict[str, Type["DSLAdapter"]] = {}


def register_dsl(name: str):
    """Class decorator: register the decorated DSLAdapter subclass under ``name``."""
    key = name.lower()

    def _decorate(cls: Type["DSLAdapter"]) -> Type["DSLAdapter"]:
        existing = _DSL_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"DSL name {name!r} already registered to {existing.__name__}; "
                f"refusing to overwrite with {cls.__name__}."
            )
        _DSL_REGISTRY[key] = cls
        return cls

    return _decorate


class DSLAdapter(ABC):
    """Abstract base class for DSL adapters."""

    # ------------------------------------------------------------------
    # Code generation hooks (called by the verify / benchmark templates)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_import_statements(self, framework: str) -> str:
        """Return the import block injected at the top of generated eval code."""
        pass

    @abstractmethod
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return ``from <op>_<dsl>_impl import …`` line (or empty string)."""
        pass

    def create_impl_module(self, framework: str,
                           framework_adapter: Any,
                           init_params_var: str = "init_params",
                           device_var: str = "device") -> str:
        """For ModelNew-style DSLs: code that instantiates impl_model.

        Function-style DSLs (e.g. tilelang_npuir) return ``""``.
        """
        return ""

    @abstractmethod
    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str,
                  data_dir: Optional[str] = None,
                  framework_output: Optional[str] = None) -> str:
        """Code that runs the implementation once and binds ``impl_output``."""
        pass

    @abstractmethod
    def needs_binary_io(self) -> bool:
        """True if outputs are exchanged via binary files (SWFT)."""
        pass

    @abstractmethod
    def needs_compilation(self) -> bool:
        """True if AOT compilation is required at runtime (AscendC)."""
        pass

    @abstractmethod
    def benchmark_impl(self, impl_func_name: str, inputs: str,
                       warmup: int, runs: int, backend: str, op_name: str,
                       case_idx: int = 0, framework_model: Optional[str] = None,
                       framework_adapter: Optional[Any] = None,
                       device_id: Optional[int] = None) -> str:
        """Code that profiles the implementation and binds
        ``execution_time_ms`` + ``method``.
        """
        pass

    def get_special_setup_code(self) -> str:
        """Per-DSL one-shot setup (cache clear, monkey-patch, …)."""
        return ""

    def get_autotune_info(self, case_idx: int) -> Optional[Dict]:
        """Triton-only autotune side-channel."""
        return None

    def get_binary_io_functions(self) -> str:
        """SWFT-only binary IO glue."""
        return ""

    # ------------------------------------------------------------------
    # Capabilities (queried by verifier / profiler / hw routing)
    # ------------------------------------------------------------------

    @abstractmethod
    def default_backend(self) -> str:
        """Backend implied by this DSL name (``ascend`` / ``cuda`` / ``cpu``).

        Single source of truth; ``hw_detect`` derives the DSL→backend map
        from each adapter at lookup time, so adapters that ship in this
        repo can't drift from a hand-maintained table.
        """
        pass

    def supported_backends(self) -> Set[str]:
        """All backends this DSL can target. Defaults to just ``default_backend``."""
        return {self.default_backend()}

    def l2_clear_kernel_name(self) -> Optional[str]:
        """Name of the L2-cache-clear kernel the profiler should both invoke
        and filter out of timing stats.

        ``None`` → no dedicated kernel; profiler falls back to
        ``tensor.zero_()`` for the clear and filters ``ZerosLike`` from
        stats (less precise; risks false-positive filtering when the
        kernel itself uses ``zeros_like``).

        Only ``triton_ascend`` ships a dedicated kernel today
        (``AR_l2cache_clear``).
        """
        return None

    def compile_command(self, src_path: str, out_path: str,
                        arch: str) -> Optional[list]:
        """AOT compile command (argv) for DSLs that produce a build artifact
        before run-time. ``None`` for JIT DSLs (triton, tilelang, torch,
        cpp_inline). Reserved for future MLIR / Pallas / AscendC pipeline
        unification — today AscendC still compiles inline inside its
        generated benchmark code.
        """
        return None

    def runtime_artifact_kind(self) -> str:
        """What ``call_impl`` ultimately invokes. One of
        ``python_module`` (default; KernelBench ModelNew + triton/tilelang),
        ``shared_object`` (compiled .so loaded via cpp_extension / ctypes),
        ``kernel_binary`` (raw NPU/CUDA kernel blob).
        """
        return "python_module"
