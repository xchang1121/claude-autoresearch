"""Vendored evaluation stack for claude-autoresearch.

Layout:

    ar_vendored/
        op/verifier/
            adapters/{dsl,backend,framework}/*.py, factory.py
            profiler.py, l2_cache_clear.py        # device-side timing
        op/utils/
            triton_autotune_patch.py, tilelang_compile_patch.py
        worker/server.py                          # single /run endpoint
"""

import os


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))
