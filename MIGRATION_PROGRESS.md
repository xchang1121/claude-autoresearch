# CATLASS Migration Progress

This file is the migration checkpoint. Update it before moving to a new
phase and after each commit/smoke test so context compaction cannot erase
the current state.

## Ground Rules

- Treat this file as the source of truth for migration state.
- Keep commits small and run smoke tests on `npu` after each commit.
- Do not rely on conversation context to remember what has landed.
- Avoid complicated file transfer/compression flows; prefer git-native sync.

## Completed

1. Target triple config
   - Local commit: `ec2839e Add configurable target triple`
   - NPU commit: `e6b6a57 Add configurable target triple`
   - Smoke: passed on `npu` as part of Step 2/3 follow-up.

2. DSL adapter extension hooks
   - Local commit: `09e1f40 Add DSL adapter extension hooks`
   - NPU commit: `6bf9cf9 Add DSL adapter extension hooks`
   - Smoke: `adapter protocol smoke ok` on `npu`.

3. AscendC CATLASS adapter
   - Local commit: `f11b708 Add AscendC CATLASS adapter`
   - NPU commit: `2f25816 Add AscendC CATLASS adapter`
   - Smoke: `catlass adapter npu smoke ok` on `npu`.

## In Progress

4. Multi-file DSL task/scaffold/package flow
   - Goal: allow `ascendc_catlass` tasks to pass `catlass_op/` as the
     kernel handoff while keeping single-file DSL behavior unchanged.
   - Current focus:
     - `scripts/scaffold.py`
     - `scripts/task_config/loader.py`
     - `scripts/task_config/package_builder.py`
     - `scripts/utils/eval_runner.py`
     - `scripts/worker/server.py`
     - `scripts/batch/*`
   - Split plan:
     - 4a: scaffold + TaskConfig loading for CATLASS task directories.
     - 4b: packaging / worker / batch path resolution for multi-file DSLs.
   - 4a edit status:
     - `scripts/scaffold.py` updated to use the DSL adapter for `--kernel`
       path interpretation and project-tree materialization.
     - `scripts/task_config/loader.py` updated with `catlass_root` and
       `catlass_op_dir` fields.
     - Local compile/diff checks are pending because shell execution is
       currently blocked by the approval/usage-limit gate.

## Pending

5. Multi-DSL CodeChecker/static checks
6. End-to-end local + NPU verification and final migration notes

## Latest Known Repo State

- Local `claude-autoresearch` head: `f11b708`
- NPU `/home/yyz/cxy/claude-autoresearch` head: `2f25816`
- NPU tracked files were clean after Step 3; only runtime dirs were untracked:
  `.autoresearch/`, `.session_tasks/`, `.task_dir_pointers/`, `extra-info/`.




## SUGGESTION 1
可以迁移，但不建议把 workspace_autoresearch 直接覆盖到 claude-autoresearch。更稳的判断是：claude-autoresearch 已经是一次“独立化剥离”的结果，应以它为主线，把 AKG 侧更新过的能力按模块合并进去。

我只做了读取和比对，没有改文件。当前基线干净：akg 为 e8ae797c，claude-autoresearch 为 d5e11e4。

核心结论
claude-autoresearch 已经内置了 AKG 外部依赖中的大头：scripts/eval/**、scripts/eval/worker/**、scripts/worker/server.py、scripts/ar_cli.py、skills/**。所以迁移重点不是“搬 workspace”，而是把 AKG 版最近新增的以下能力补进独立项目：

ascendc_catlass 多文件 DSL 支持。
AKG 新版 DSLAdapter 扩展协议。
CodeChecker 的多 DSL 静态检查。
defaults.backend/framework/dsl target triple 配置。
scaffold.py / batch/manifest.py 的 DSL-aware 文件或目录 kernel 解析。
remote package/worker 对 catlass_op/ 项目树的支持。
文档、skills、命令说明中的 AKG 路径改写。
依赖比对
AKG 版 workspace_autoresearch 仍依赖目录外组件：

AKG 外部依赖	AKG 用途	claude-autoresearch 现状
akg_agents.op.verifier.*	KernelVerifier、adapter、templates、profile	已迁到 scripts/eval/**，但版本落后于 AKG 新扩展
akg_agents.core.worker.*	local/remote worker manager	已迁到 scripts/eval/worker/** + scripts/worker/server.py
akg_agents.op.utils.code_checker.CodeChecker	triton/pypto/catlass/autotune/A5 静态检查	未完整迁移，目前主要是 validate_triton_impl
akg_agents.op.autoresearch.framework.git_repo.GitRepo	task 内 git commit/rollback	不需要迁，独立项目已有纯 git subprocess 实现
akg_agents.op.resources.skills	skills 根目录	已 vendor 到 skills/**，但缺 AKG 新的 ascendc-catlass 等内容
akg_agents.cli.service.remote_dispatch	worker SSH/tunnel	已由 scripts/ar_cli.py 替代
akg_agents.op.utils.catlass_paths	CATLASS_ROOT / CMake patch	缺失，必须迁
最重要差异
claude-autoresearch 不能被 AKG 版 task_config/eval_client.py、utils/akg_eval.py 回退覆盖。独立项目现在的 eval client 更完整：支持 worker 选择、任务打包、sticky baseline、ref/kernel 分进程。AKG 版只是桥到 akg_agents verifier。迁移时应保留独立项目 eval 主线。

真正要合并的是 AKG 版这些新设计：DSLAdapter 的 materialize_impl / prepare_config / read_kernel_source / materialize_project_tree / post_iteration_cleanup 等 hook，以及 ascendc_catlass.py、catlass_paths.py、多文件 batch/scaffold 逻辑。

迁移计划

冻结基线
在 claude-autoresearch 新建迁移分支，记录 akg:e8ae797c、claude-autoresearch:d5e11e4。先不要大面积复制，按模块提交。

同步 adapter 协议
把 AKG 的 scripts/eval/adapters/dsl/base.py 新扩展点迁入独立项目；同步修改 scripts/eval/kernel_verifier.py、sol_verifier.py、eval/worker/local_worker.py，让它们调用 prepare_config、materialize_impl、expected_artifacts、profile_via_python_script、benchmark_requires_l2_clear。

迁移 CATLASS
新增 scripts/eval/adapters/dsl/ascendc_catlass.py，更新 scripts/eval/adapters/factory.py 和 config_utils 的合法 DSL 列表。把 catlass_paths.py 改成独立项目路径语义，默认不再依赖 <akg-root>/thirdparty/catlass，建议支持：task.yaml catlass.root、CATLASS_ROOT、可选 <claude-autoresearch>/thirdparty/catlass。

打通 scaffold / TaskConfig / batch
合并 AKG 版 scaffold.py 的 DSL-aware --kernel 解析：单文件 DSL 读 .py，ascendc_catlass 读 catlass_op/ 并寻找 sibling kernel.py。
TaskConfig 加回 catlass_root、catlass_op_dir。
batch/manifest.py 支持 kernel 是目录、kernel_module 是 wrapper .py 的双字段。

改造 remote package
package_builder.py 当前只打包文件。CATLASS 需要支持 project tree，至少要把 catlass_op/ 中构建所需文件打进 tar。worker 解包后，eval_kernel.py 要能把 catlass_op 传给 adapter 的 task_info/config。

迁移 CodeChecker
建议把 AKG CodeChecker 迁为独立项目本地模块，例如 scripts/eval/code_checker.py 或 scripts/utils/code_checker.py，并把 akg_agents.* import 改为 eval.*。然后让 quick_check.py 和 batch/verify.py 从“只验 Triton”升级为按 defaults.dsl 做多 DSL 静态检查。

配置统一
在 claude-autoresearch/config.yaml 加回：
defaults.backend: ascend、defaults.framework: torch、defaults.dsl: triton_ascend。
settings.py 加 target_backend()、target_framework()、target_dsl()，让 scaffold、quick_check、eval_runner、eval_kernel 都读同一个 target triple。

同步 resources / docs
将 AKG 新 skills，尤其 skills/ascendc-catlass/**，合并到独立项目 skills/**。
文档中统一把 akg_cli 改成 python scripts/ar_cli.py，把远端 repo_path 改成 claude-autoresearch 根目录，删掉 AKG_AGENTS_AR_SKILLS_ROOT 依赖。

验证计划
先做无硬件验证：python -m compileall scripts、scaffold 单文件 Triton 任务、quick_check、batch manifest 单文件/catlass_op fixture、package_builder tar 安全测试、rg "akg_agents|AscendOpGenAgent|workspace_autoresearch|akg_cli" 确认无运行时依赖残留。

有 NPU 后再做：本地 /autoresearch baseline、remote ar_cli worker --start/status/stop、remote /run、sticky baseline、最后跑一个最小 ascendc_catlass compile/profile 任务。

建议优先级
先迁 adapter 协议和 target triple，再迁 CATLASS，再迁 package/worker，最后迁 CodeChecker 和文档。这样每一步都能保持独立项目可运行，不会在半路把已有的 eval/worker 独立化成果弄回 AKG 依赖。