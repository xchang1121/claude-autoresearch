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

"""Eval entry point: CA shim into the WA-compatible formal eval path.

``run_eval(task_dir, config, device_id=None, worker_urls=None)`` is the
contract baseline.py and pipeline.py call. The implementation delegates to
``utils.akg_eval.eval_kernel``, which builds ``eval.KernelVerifier`` and
routes local/remote execution through the worker manager. There is a single verification/profile path for foreground, batch, and worker execution.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from .loader import TaskConfig
from .metric_policy import EvalOutcome, EvalResult


_OUTCOME_VALUES = {e.value for e in EvalOutcome}


def _resolve_worker_url(worker_urls: Optional[list],
                        config: TaskConfig) -> Optional[str]:
    """Pick the first non-empty URL.

    Worker registration and device scheduling are handled downstream by
    ``utils.akg_eval`` / the worker manager, so this shim does not keep a
    second scheduler.
    """
    candidates = worker_urls or getattr(config, "worker_urls", None) or []
    for u in candidates:
        if u and str(u).strip():
            return str(u).strip()
    return None


def _resolve_device_arg(device_id: Optional[int], config: TaskConfig,
                        worker_url: Optional[str]):
    if device_id is not None:
        return int(device_id)
    devices = getattr(config, "devices", None)
    if devices:
        parsed = [int(d) for d in devices]
        return parsed if worker_url else parsed[0]
    if worker_url:
        return None
    print(
        "[akg_eval] WARNING: no device specified (no device_id arg, "
        "no `devices` field in task.yaml). Defaulting to local device 0.",
        file=sys.stderr,
    )
    return 0


def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None,
             worker_urls: Optional[list] = None) -> EvalResult:
    """Route eval through ``utils.akg_eval.eval_kernel`` and convert the
    returned dict into the ``EvalResult`` shape consumed by the workflow.
    """
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from utils.akg_eval import eval_kernel  # noqa: E402

    worker_url = _resolve_worker_url(worker_urls, config)
    dev_id = _resolve_device_arg(device_id, config, worker_url)

    try:
        raw = eval_kernel(task_dir, config,
                          device_id=dev_id,
                          worker_url=worker_url)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return EvalResult(
            outcome=EvalOutcome.INFRA_FAIL,
            error=f"akg_eval.eval_kernel raised {type(e).__name__}: {e}",
            error_source="infra",
        )

    outcome_str = raw.get("outcome") or "infra_fail"
    if outcome_str not in _OUTCOME_VALUES:
        outcome_str = "infra_fail"
    return EvalResult(
        outcome=EvalOutcome(outcome_str),
        metrics=raw.get("metrics") or {},
        error=raw.get("error"),
        raw_output=str(raw.get("raw_output_tail") or ""),
        error_source=raw.get("error_source"),
    )
