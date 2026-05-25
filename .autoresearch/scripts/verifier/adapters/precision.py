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

"""精度容差描述符。

替代原先 framework adapter 上单一 float 返回的 ``get_limit()`` API。
``rtol`` 字段与旧 ``get_limit()`` 的语义等价（相对容差上限）；新增
``atol`` 和 ``extra`` 用于未来 mixed-precision / 量化场景。

verifier 内部当前的比较逻辑仍按单值 ``rtol`` 运作，本次重构只升级数据
结构，不改比较语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PrecisionSpec:
    rtol: float
    atol: float = 0.0
    extra: Mapping[str, Any] = field(default_factory=dict)
