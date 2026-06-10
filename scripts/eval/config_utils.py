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

import logging
import re
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single source of truth — SKU enumeration + per-family DSL whitelists.
# Mirrors akg-hitl's op/utils/config_utils.py scheme.
#
# Validation strategy differs by backend:
#   - ascend uses **explicit SKU tuples** — Ascend SKUs are discrete real
#     products; enumeration gives crisp error msgs.
#   - cuda / cpu use **family regex** — model names are parametric
#     (``rtx<N>``, ``[ahvltb]<N>``, ``x86_64`` / ``aarch64`` / ...);
#     a new RTX 5090 / B300 / H200 is automatically accepted when
#     ``hw_detect`` extracts it from ``nvidia-smi``.
#
# Adding a new ascend SKU: one line in the matching ``_*_SKUS`` tuple.
# Adding a whole new cuda generation: zero code change (regex covers it).
# Adding a new DSL across a family: one ``support`` entry in
# ``eval.adapters.factory.DSL_REGISTRY``.
# ---------------------------------------------------------------------------

_ASCEND_910B_SKUS = (
    "ascend910b1", "ascend910b2", "ascend910b2c", "ascend910b3", "ascend910b4",
)
_ASCEND_910_93_SKUS = (
    "ascend910_9362", "ascend910_9372", "ascend910_9381",
    "ascend910_9382", "ascend910_9391", "ascend910_9392",
)
_ASCEND_950_SKUS = (
    "ascend950dt_95a",
    "ascend950pr_950z", "ascend950pr_9572", "ascend950pr_9574", "ascend950pr_9575",
    "ascend950pr_9576", "ascend950pr_9577", "ascend950pr_9578", "ascend950pr_9579",
    "ascend950pr_957b", "ascend950pr_957d", "ascend950pr_9581", "ascend950pr_9582",
    "ascend950pr_9584", "ascend950pr_9587", "ascend950pr_9588", "ascend950pr_9589",
    "ascend950pr_958a", "ascend950pr_958b", "ascend950pr_9591", "ascend950pr_9592",
    "ascend950pr_9599",
)
_ASCEND_910_FAMILY = _ASCEND_910B_SKUS + _ASCEND_910_93_SKUS + _ASCEND_950_SKUS
_ASCEND_310_SKUS = ("ascend310p3",)

# CUDA family regex — accepts any token ``hw_detect`` normalizes from
# ``nvidia-smi --query-gpu=name``: rtx<N> / gtx<N> / [ahvltb]<N> with an
# optional trailing variant letter (``l40s`` / ``v100`` / ``h200`` etc.).
_CUDA_ARCH_PAT = re.compile(
    r"^(?:rtx\d{3,4}[a-z]?|gtx\d{3,4}[a-z]?|[ahvltb]\d{1,4}[a-z]?)$"
)
_CPU_ARCH_PAT = re.compile(r"^(?:x86_64|aarch64|riscv64|ppc64le)$")


def _build_dsl_table():
    """Family → DSL whitelist, keyed by (framework, backend, family-tag),
    derived from DSL_REGISTRY support tuples (single source of truth)."""
    from eval.adapters.factory import DSL_REGISTRY
    table: dict = {}
    for name, entry in DSL_REGISTRY.items():
        for fbf in entry.support:
            dsls = table.setdefault(fbf, [])
            if name not in dsls:
                dsls.append(name)
        for alias in entry.aliases:
            for fbf in entry.support:
                dsls = table.setdefault(fbf, [])
                if alias not in dsls:
                    dsls.append(alias)
    return {fbf: tuple(dsls) for fbf, dsls in table.items()}


_DSL_TABLE = _build_dsl_table()
_ALL_DSLS = frozenset(
    dsl for dsls in _DSL_TABLE.values() for dsl in dsls
)


def _family_of(backend: str, arch: str) -> Optional[str]:
    """Return the family tag for (backend, arch), or None if the arch
    isn't recognized under that backend. Ascend uses explicit SKU
    membership; cuda / cpu use family regex."""
    if backend == "ascend":
        if arch in _ASCEND_310_SKUS:
            return "310"
        if arch in _ASCEND_910_FAMILY:
            return "910"
        return None
    if backend == "cuda":
        return "any" if _CUDA_ARCH_PAT.match(arch) else None
    if backend == "cpu":
        return "any" if _CPU_ARCH_PAT.match(arch) else None
    return None


def arch_hint(backend: str) -> str:
    """User-facing hint for what arch values ``backend`` accepts."""
    if backend == "ascend":
        return "/".join(_ASCEND_310_SKUS + _ASCEND_910_FAMILY)
    if backend == "cuda":
        return ("rtx<N> / gtx<N> / [ahvltb]<N> family "
                "(e.g. a100, v100, h100, h200, l40s, b200, rtx4060, rtx5090)")
    if backend == "cpu":
        return "x86_64 / aarch64 / riscv64 / ppc64le"
    return ""


def check_backend_arch(backend: str, arch: str):
    """验证后端与架构的匹配关系（family 驱动，跟 akg-hitl 同方案）。"""
    if backend not in ("ascend", "cuda", "cpu"):
        raise ValueError("backend must be ascend, cuda or cpu")
    if _family_of(backend, arch) is None:
        raise ValueError(
            f"{backend} backend does not recognize arch={arch} "
            f"(accepted: {arch_hint(backend)})"
        )


def supported_dsls(framework: str, backend: str, arch: str) -> Optional[tuple]:
    """Return the DSL whitelist for ``(framework, backend, arch)``, or
    None if the combination is not supported. Single canonical lookup —
    every other validator in this module routes through this."""
    family = _family_of(backend, arch)
    if family is None:
        return None
    return _DSL_TABLE.get((framework, backend, family))


def normalize_dsl(dsl: str, backend: str = None) -> str:
    """
    规范化DSL类型，将通用的triton根据backend转换为triton_cuda或triton_ascend
    
    Args:
        dsl: 实现类型
        backend: 硬件后端名称(ascend/cuda/cpu)，用于自动转换triton
        
    Returns:
        规范化后的DSL类型
        
    Raises:
        ValueError: 如果dsl为"triton"但backend未提供或无效
    """
    dsl = dsl.lower()
    
    # 如果已经是规范化的类型，直接返回
    if dsl in _ALL_DSLS:
        return dsl
    
    # 如果是通用的triton，需要根据backend转换
    if dsl == "triton":
        if backend is None:
            raise ValueError(
                "dsl='triton' is no longer supported. Please use 'triton_cuda' (for CUDA backend) "
                "or 'triton_ascend' (for Ascend backend) explicitly. "
                "Alternatively, provide backend parameter for automatic conversion."
            )
        backend = backend.lower()
        if backend == "cuda":
            return "triton_cuda"
        elif backend == "ascend":
            return "triton_ascend"
        else:
            raise ValueError(
                f"dsl='triton' cannot be used with backend='{backend}'. "
                "Please use 'triton_cuda' (for CUDA) or 'triton_ascend' (for Ascend) explicitly."
            )
    
    # 其他情况直接返回
    return dsl


def check_dsl(dsl: str):
    """
    验证实现类型
    Args:
        dsl: 实现类型(triton_cuda/triton_ascend/triton-russia/swft/torch/pypto等)
    """
    valid_dsls = sorted(_ALL_DSLS)
    if dsl not in valid_dsls:
        raise ValueError(
            f"dsl must be one of {valid_dsls}. "
            "Note: 'triton' is no longer supported. Use 'triton_cuda' or 'triton_ascend' instead."
        )


def check_task_type(task_type: str):
    """
    验证任务类型
    Args:
        task_type: 任务类型(precision_only/profile)
    """
    if task_type not in ["precision_only", "profile"]:
        raise ValueError("task_type must be precision_only or profile")


def check_task_config(framework: str, backend: str, arch: str, dsl: str):
    """
    统一验证配置参数之间的依赖关系（family 驱动，跟 akg-hitl 同方案）。
    Args:
        framework: 框架类型
        backend: 硬件后端名称
        arch: 硬件架构名称
        dsl: 实现类型（会自动转换triton为triton_cuda或triton_ascend）
    Returns:
        规范化后的 DSL，供调用者使用。
    """
    normalized_dsl = normalize_dsl(dsl, backend)
    dsl_list = supported_dsls(framework, backend, arch)
    if dsl_list is None:
        raise ValueError(
            f"({framework}, {backend}, {arch}) is not a supported combination "
            f"(accepted arch for {backend}: {arch_hint(backend)})")
    if normalized_dsl not in dsl_list:
        raise ValueError(
            f"({framework}, {backend}, {arch}) does not support "
            f"dsl={normalized_dsl} (accepted: {sorted(dsl_list)})")
    return normalized_dsl


def collect_and_save_all_examples(
    arch: str,
    dsl: str,
    project_root_path: Path,
    source_dirs: Dict[str, Path],
) -> Optional[Path]:
    """
    汇总所有examples并保存到统一目录 database/all_examples/{arch}/{dsl}
    
    此函数将来自不同源目录的示例文件统一复制到目录中，Python文件和其他文件分别保存。
    - Python文件保存到: database/all_examples/{arch}/{dsl}/code/
    - 其他文件保存到: database/all_examples/{arch}/{dsl}/docs/
    
    Args:
        arch: 硬件架构名称
        dsl: DSL类型
        project_root_path: 项目根路径
        source_dirs: 源目录字典，格式为 {"prefix": Path("source_dir")}
                    文件会被复制并重命名为 "{prefix}_{原文件名}"
                    例如: {"user": Path("user_examples/"), "local": Path("local_examples/")}
    
    Returns:
        Path: 统一保存目录的根路径，如果失败则返回None
    """
    if not arch or not dsl:
        logger.warning("arch或dsl为空，无法汇总示例代码")
        return None
    
    # 创建统一的保存目录
    base_dir = project_root_path / "database" / "all_examples" / arch / dsl
    code_dir = base_dir / "code"
    doc_dir = base_dir / "docs"
    
    code_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    saved_code_count = 0
    saved_doc_count = 0
    
    # 遍历所有源目录
    for prefix, source_dir in source_dirs.items():
        if not source_dir or not isinstance(source_dir, Path):
            logger.warning(f"跳过无效的源目录: {prefix} -> {source_dir}")
            continue
            
        if not source_dir.exists():
            logger.warning(f"源目录不存在，跳过: {source_dir}")
            continue
        
        try:
            # 如果是目录，复制其中的所有文件
            if source_dir.is_dir():
                for file_path in source_dir.glob("*"):
                    if not file_path.is_file():
                        continue
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if content:
                            # 根据文件扩展名选择目标目录
                            if file_path.suffix == ".py":
                                target_dir = code_dir
                                saved_code_count += 1
                            else:
                                target_dir = doc_dir
                                saved_doc_count += 1
                            
                            # 目标文件名：前缀_原文件名
                            save_path = target_dir / f"{prefix}_{file_path.name}"
                            with open(save_path, "w", encoding="utf-8") as f:
                                f.write(content)
                    except Exception as e:
                        logger.warning(f"复制文件 {file_path} 失败: {e}")
            
            # 如果是单个文件，直接复制
            elif source_dir.is_file():
                try:
                    with open(source_dir, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        # 根据文件扩展名选择目标目录
                        if source_dir.suffix == ".py":
                            target_dir = code_dir
                            saved_code_count += 1
                        else:
                            target_dir = doc_dir
                            saved_doc_count += 1
                        
                        save_path = target_dir / f"{prefix}_{source_dir.name}"
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(content)
                except Exception as e:
                    logger.warning(f"复制文件 {source_dir} 失败: {e}")
                    
        except Exception as e:
            logger.warning(f"处理源目录 {prefix}:{source_dir} 时发生错误: {e}")
    
    logger.info(f"汇总完成，共保存 {saved_code_count} 个Python文件到: {code_dir}")
    logger.info(f"汇总完成，共保存 {saved_doc_count} 个其他文件到: {doc_dir}")
    return code_dir
