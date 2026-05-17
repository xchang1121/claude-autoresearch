# AutoResearch

基于 Claude Code 的算子迭代优化框架。给定一对 `(reference.py, kernel.py)`，
Claude 自动跑 **plan → edit → eval → KEEP/DISCARD** 循环把 kernel 性能调优，
连续失败自动 DIAGNOSE，预算耗尽自动收尾出报告。整套阶段机由 hook 强约束，
Claude 不能跳步、不能改 plan.md、不能手写 phase。

支持 DSL：`triton_ascend` / `triton_cuda` / `ascendc` / `cuda_c` / `cpp` /
`tilelang_cuda` / `tilelang_npuir` / `pypto` / `swft` / `torch`。

## Quick Start

把 `(<op>_ref.py, <op>_kernel.py)` 放 [workspace/](workspace/)，然后：

```bash
cd claude-autoresearch
claude
```

在 Claude 里粘一条 slash 命令：

```
/autoresearch --ref workspace/sinkhorn_ref.py --kernel workspace/sinkhorn_kernel.py \
  --op-name sinkhorn --dsl triton_ascend --devices 5 --max-rounds 200
```

scaffold + 首轮 baseline 原子完成 → 进 PLAN → 自动迭代到 FINISH。

**实时监控**（另开终端）：

```bash
python .autoresearch/scripts/dashboard.py
```

## `/autoresearch` 命令

入参语义：

| 形式 | 行为 |
|------|------|
| `--ref X.py --kernel Y.py ...` | 新建任务（scaffold + 首次 baseline 原子完成） |
| `<task_dir 路径>` | resume 该目录 |
| `--resume` | resume 最近活跃 task |
| 无参数 | 交互式询问 |

新建任务的 flag：

| flag | 必填 | 说明 |
|------|------|------|
| `--ref <file>` | ✅ | reference.py 路径，要有 `Model` / `get_init_inputs` / `get_inputs`(或 `get_input_groups`) |
| `--kernel <file>` | ✅ | seed kernel 路径，要有 `ModelNew` |
| `--op-name <name>` | ✅ | 算子名（决定 task_dir 命名） |
| `--dsl <name>` | ✅ | 见上方 DSL 列表 |
| `--devices <N[,M,...]>` | ✅/XOR | 本地 NPU/GPU 下标，和 `--worker-url` 二选一 |
| `--worker-url <host:port>` | ✅/XOR | 远端 worker daemon，和 `--devices` 二选一 |
| `--framework` |  | `torch` / `mindspore` / `numpy`，默认 `torch` |
| `--max-rounds N` |  | 优化总轮数预算，默认 20 |
| `--eval-timeout S` |  | 单次 eval 超时（秒），默认 120 |
| `--no-code-checker` |  | 关闭静态 CodeChecker。当前规则只覆盖 `triton_*`，其他 DSL 建议加 |

派生项（用户不写）：`backend` 由 DSL 决定；`arch` 由 `--devices` 本地探测（`npu-smi` / `nvidia-smi`）
或从 worker `GET /api/v1/status` 自报。

> seed kernel 跑 baseline 失败也直接进 PLAN —— 第一批 plan items 会用于改写
> seed，不再有"生成 kernel"的独立阶段。

## 主循环

```
BASELINE  (scaffold 跑 seed，无论 PASS / FAIL 都进 PLAN)
   ▼
PLAN  → EDIT  → quick_check → eval → KEEP/DISCARD → settle
              │
              ├─ consecutive_failures ≥ 3   → DIAGNOSE → PLAN → EDIT
              ├─ 当前 plan 全部 settle      → REPLAN   → PLAN → EDIT
              └─ eval_rounds == max_rounds  → FINISH (report.md)
```

每个 plan item (`pN`) 要么在 `history.jsonl` 里 settle 成 KEEP / DISCARD / FAIL，
要么在 REPLAN/DIAGNOSE 边界被丢弃；pid 单调推进、不复用。

阶段产物：

| 阶段 | 触发动作 | 产物 |
|------|----------|------|
| BASELINE | `baseline.py` | `seed_metric` → progress.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py` | plan.md（含 ACTIVE 标记 + 全局 pN）|
| EDIT | Edit `kernel.py` → `pipeline.py` | history.jsonl + 可选 git commit |
| FINISH | (auto) `pipeline.py` → `report.py` | report.md（含内嵌 SVG 曲线 + 表格）|

## Dashboard

```bash
python .autoresearch/scripts/dashboard.py             # 当前任务，默认刷新
python .autoresearch/scripts/dashboard.py <task_dir> --watch 2
```

键位：`↑` / `↓` / `PgUp` / `PgDn` / `Home` / `End` 滚动 history，`q` / `Esc` 退出。
顶栏：task 名 / 阶段 / plan 版本 / budget / Baseline / Seed / Best / 改进比；
下栏：history 表 + 当前 plan。

## 批量跑

要对 10+ 个 op 一起跑：见 [claude-autoresearch-batch.md](claude-autoresearch-batch.md)。
工作流：

```bash
# 1. 备 ref/kernel
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels
cp my_ops/*_ref.py    $BATCH_DIR/refs/
cp my_ops/*_kernel.py $BATCH_DIR/kernels/

# 2. discover + Tier 1 verify
python .autoresearch/scripts/batch/prepare.py $BATCH_DIR --dsl triton_ascend

# 3. 起 worker daemon
python .autoresearch/scripts/ar_cli.py worker --start --port 9111 --bg

# 4. tmux 后台跑全批
tmux new -d -s ar_batch \
  "python -u .autoresearch/scripts/batch/run.py $BATCH_DIR --worker-url 127.0.0.1:9111"

# 5. 另开终端监控
python .autoresearch/scripts/batch/monitor.py $BATCH_DIR
```

## 远程 Worker

远端 NPU / CUDA 通过 SSH tunnel 接入，eval 提交到远端跑。HTTP server 自带
（[ar_vendored/worker/](.autoresearch/scripts/ar_vendored/worker/)），worker 端依赖：
`fastapi` + `uvicorn`、`torch`（+ `torch_npu` / CUDA runtime）、按 DSL 追加
`triton` / `pandas` / `msprof` / `nsys`。

```bash
# worker 机器
python .autoresearch/scripts/ar_cli.py worker --start \
    --backend ascend --arch ascend910b3 --devices 0 \
    --host 127.0.0.1 --port 9111 --bg

# 本地建隧道
ssh -f -N -L 127.0.0.1:9111:127.0.0.1:9111 \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 npu_host
curl http://127.0.0.1:9111/api/v1/status
# {"status":"ready","backend":"ascend","arch":"ascend910b3","devices":[0]}
```

`backend` / `arch` / `devices` 默认都是 `auto`（按 `npu-smi` / `nvidia-smi` 自动检），
显式传值永远优先。`ar_cli.py worker` 还支持 `--stop` / `--status` / 前台模式，详见 `--help`。

## 精度

每轮 worker 端在 device 上重算 reference，按 case 跟 `ModelNew` 输出比对：

1. `Model(*get_init_inputs())(*get_inputs())` 拿 ref
2. `ModelNew.forward()` 拿实测
3. AND-of-maxima 检查（对齐 `akg/akg_agents` 的精度标准）：

```
abs_diff = |ref - sol|
rel_diff = abs_diff / (|ref| + 1e-8)
PASS iff  max(abs_diff) <= atol  AND  max(rel_diff) <= rtol
```

solution 端任何 NaN/Inf 直接判 fail；非 Tensor 输出也判 fail。这比
`torch.allclose` 的逐元素和式 `|a-b| <= atol + rtol*|b|` 更严，因为
不允许"单个元素用绝对误差换相对误差"。

容差固定在 [`utils/correctness.py`](.autoresearch/scripts/utils/correctness.py)：
`DEFAULT_ATOL = DEFAULT_RTOL = 1e-2`。要调精度直接改这一个文件。

ref 时延（用于 speedup 计算）和 kernel 时延都在 worker 端单个
`eval_<op>.py` 子进程里跑（verify warm 了 JIT/autotune 缓存，profile
直接复用），结果落在 `eval_result.json` sidecar 里，不经 stdout。
单一 endpoint：`POST /api/v1/run`。

## 输出

每个 task 的产物落在 `<repo>/ar_tasks/<op>_<ts>_<uuid>/`：

```
ar_tasks/<op>_<ts>_<uuid>/
├── kernel.py          ← 性能优化后的 kernel（最佳版本）
├── reference.py       ← scaffold 拷过来的 ref
├── task.yaml          ← dsl / arch / metric / editable_files 配置
└── .ar_state/
    ├── .phase         ← 当前 phase（结束时是 FINISH）
    ├── progress.json  ← eval_rounds / baseline_metric / best_metric
    ├── plan.md        ← agent 优化历史 + settle 记录（权威态）
    ├── history.jsonl  ← 每轮 decision / metrics / commit
    └── report.md      ← 最终报告（含 SVG 收敛曲线 + 表格）
```

每轮 KEEP 都有一次 git commit；想 diff 哪轮做了什么、用 `git log` 即可。

## Skills 库

`skills/` 提供 DSL 优化素材，按 DSL 名字组织：
`skills/triton-ascend/` / `skills/triton-cuda/` / `skills/cuda-c/` / `skills/cpp/` /
`skills/tilelang-cuda/` / `skills/pypto/`。

PLAN 阶段 hook 会提示 Claude `Glob("skills/<dsl>/**/*.md")`，把命中的 SKILL id
写进 plan item rationale 里做溯源。

## 依赖

- Python ≥ 3.10
- `pip install pyyaml torch`
- [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI（或 VS Code 扩展）
- 按 DSL 追加：
  - `triton_ascend` / `tilelang_npuir`：`torch_npu` + `triton` + CANN
  - `triton_cuda` / `tilelang_cuda` / `pypto`：`triton` + CUDA runtime
  - `ascendc`：CANN toolkit（`msprof` 在 PATH）
  - `cuda_c`：Nsight Systems（`nsys` 在 PATH）
  - `_profile_via_msprof` / `_profile_via_nsys` 还要 `pandas`
- 远端 worker（可选）：SSH tunnel + rsync 项目过去

## 内部机制

外部接口稳定（slash 命令、`task.yaml` 字段、`.ar_state/` 路径）。
想动内部时的入口：

| 想了解 | 看哪里 |
|--------|--------|
| Phase 流转规则 / Bash gate | [phase_machine/phase_policy.py](.autoresearch/scripts/phase_machine/phase_policy.py) |
| Hook 接线 | [.claude/settings.json](.claude/settings.json) + [hooks/](.autoresearch/scripts/hooks/) |
| Plan / history / progress 写入 | [phase_machine/state_store.py](.autoresearch/scripts/phase_machine/state_store.py) |
| DSL adapter（profiler / autotune） | [ar_vendored/op/verifier/adapters/factory.py](.autoresearch/scripts/ar_vendored/op/verifier/adapters/factory.py) |
| 本地 vs 远端执行路由 | [utils/local_worker.py](.autoresearch/scripts/utils/local_worker.py) / [ar_vendored/worker/server.py](.autoresearch/scripts/ar_vendored/worker/server.py) |
| CodeChecker 规则 | [utils/code_checker.py](.autoresearch/scripts/utils/code_checker.py) + `.autoresearch/code_checker.yaml` |
| 不变量（plan 权威态 / pid 单调 / DIAGNOSE 契约 / 等） | [CLAUDE.md](CLAUDE.md) |
| 子代理 prompt（DIAGNOSE 用） | [.claude/agents/ar-diagnosis.md](.claude/agents/ar-diagnosis.md) |
