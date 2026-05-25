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

"""Factory for creating adapters.

DSL adapters are looked up from ``dsl.base._DSL_REGISTRY``, populated
when each ``dsl/<name>.py`` module is first imported (``@register_dsl``
class decorator). ``_ensure_dsls_loaded`` triggers that sweep once on
the first ``get_dsl_adapter`` call so new DSLs only need a single new
file in ``dsl/`` — no edit here.

Framework and backend keep explicit branches: there are only three of
each, they don't churn, and the import pattern (``framework/X.py``
imports heavyweight runtimes like ``torch`` / ``mindspore`` at module
load) means lazy imports inside the branch are valuable.
"""

from typing import Tuple

from .dsl.base import _DSL_REGISTRY, DSLAdapter


def get_framework_adapter(framework: str):
    framework_lower = framework.lower()

    if framework_lower == "torch":
        from .framework.torch import FrameworkAdapterTorch
        return FrameworkAdapterTorch()
    elif framework_lower == "mindspore":
        from .framework.mindspore import FrameworkAdapterMindSpore
        return FrameworkAdapterMindSpore()
    elif framework_lower == "numpy":
        from .framework.numpy import FrameworkAdapterNumpy
        return FrameworkAdapterNumpy()
    else:
        raise ValueError(f"Unsupported framework: {framework}")


_dsls_loaded = False


def _ensure_dsls_loaded() -> None:
    """Import every ``dsl/*.py`` once so ``@register_dsl`` decorators fire."""
    global _dsls_loaded
    if _dsls_loaded:
        return
    import importlib
    import pkgutil
    from . import dsl as dsl_pkg

    for _, name, _ in pkgutil.iter_modules(dsl_pkg.__path__):
        if name.startswith("_") or name == "base":
            continue
        importlib.import_module(f"{dsl_pkg.__name__}.{name}")
    _dsls_loaded = True


def get_dsl_adapter(dsl: str) -> DSLAdapter:
    _ensure_dsls_loaded()
    key = dsl.lower()
    cls = _DSL_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unsupported DSL: {dsl}. Known: {', '.join(sorted(_DSL_REGISTRY))}"
        )
    return cls()


def list_dsls() -> Tuple[str, ...]:
    """Sorted tuple of all DSL names registered in ``dsl/``.

    Single source of truth for any user-facing DSL menu (scaffold --help,
    parse_args missing-fields payload, slash-command docs).
    """
    _ensure_dsls_loaded()
    return tuple(sorted(_DSL_REGISTRY))


def get_backend_adapter(backend: str):
    backend_lower = backend.lower()

    if backend_lower == "cuda":
        from .backend.cuda import BackendAdapterCuda
        return BackendAdapterCuda()
    elif backend_lower == "ascend":
        from .backend.ascend import BackendAdapterAscend
        return BackendAdapterAscend()
    elif backend_lower == "cpu":
        from .backend.cpu import BackendAdapterCpu
        return BackendAdapterCpu()
    else:
        raise ValueError(f"Unsupported backend: {backend}")
