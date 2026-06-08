# AutoResearch

Claude Code 驱动的 Triton-Ascend 算子自动优化框架。

**主循环**：scaffold → BASELINE → PLAN → EDIT → eval → KEEP/DISCARD → 直到 `max_rounds` → FINISH。

第一次用：按下表选一条「操作」路径走完即可，配置说明都在「参考」里按主题归档。

| 目标 | 跳转 |
|---|---|
| 第一次用，本机就是 NPU 机 | [操作 A](#a-单机本机--测评机) |
| 第一次用，本机没 NPU，要用远端 NPU | [操作 B](#b-双机本机跑-claude无-npu-远端-npu-机跑-eval) |
| 一次跑很多算子 | [操作 C](#c-批量) |
| 续跑、重置、查历史 | [操作 D](#d-续跑--重置--查询) |
| 查 CLI 参数 | [参考 §1](#1-cli-参数) |
| 算子/文件该怎么命名 | [参考 §2](#2-命名契约) |
| 单轮 phase 怎么走 | [参考 §3](#3-主循环--状态机) |
| eval 怎么跑、worker 怎么联 | [参考 §4](#4-eval-执行链) |
| 精度判定标准 | [参考 §5](#5-精度容差) |
| 状态文件在哪 | [参考 §6](#6-文件与状态布局) |
| env.sh 该写什么 | [参考 §7](#7-envsh-契约) |
| 远程 worker 怎么配置 | [参考 §8](#8-ar_cli-worker) |
| Triton 调优 markdown 库 | [参考 §9](#9-skills-库) |
| 内部机制入口 | [参考 §10](#10-hook-与内部机制) |
| 并发跑 / 多 session / 共用卡 | [参考 §11](#11-并发与冲突) |

---

# 操作

## A. 单机：本机 = 测评机

开发与 eval 同一台 Ascend NPU 机，配置最少。

### A.1 准备 env.sh

前置：本机有 Ascend NPU，`npu-smi info` 列得出设备；本机 Python 已能 `import torch_npu, triton`；Claude Code CLI 已装。

写 `~/env.sh`，使非交互 shell `source` 后即可 `python -c "import torch_npu, triton"`。模板与禁项见 [参考 §7](#7-envsh-契约)。

验证：
```bash
source ~/env.sh && python -c "import torch_npu, triton" && npu-smi info
```

### A.2 写 ref 和 kernel

在 `autoresearch/workspace/` 下放两个文件。命名硬约束见 [参考 §2](#2-命名契约)。

`workspace/<op>_ref.py`：PyTorch 标准答案，暴露 `class Model(nn.Module)`、`get_inputs()`（或 `get_input_groups()`）和 `get_init_inputs()`。

`workspace/<op>_kernel.py`：种子 kernel，暴露 `class ModelNew(nn.Module)`，forward 内实际调用目标 DSL kernel。命名与接口细节见 [参考 §2](#2-命名契约)。

### A.3 启动

```bash
cd autoresearch
source ~/env.sh
claude
```

在 Claude 输入框里：

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <NPU-id>
```

`--devices` 必填。其他可调 flag（`--max-rounds`、`--eval-timeout` 等）见 [参考 §1.1](#11-autoresearch)。

### A.4 看进度

Claude 全自动按 `scaffold → BASELINE → PLAN → EDIT → ...` 推进，无须人工干预。另开终端实时查看：

```bash
cd autoresearch
python scripts/dashboard.py <task_dir> --watch
```

阶段流转与产物见 [参考 §3](#3-主循环--状态机)。

### A.5 拿结果

达到 `--max-rounds` 自动进入 FINISH，落盘：
- `<task_dir>/kernel.py`：最佳 kernel
- `<task_dir>/.ar_state/report.md`：报告（含内嵌 SVG）

每次 KEEP 对应一次 git commit。

---

## B. 双机：本机跑 claude（无 NPU）+ 远端 NPU 机跑 eval

本机没有 NPU，eval 委托远端 Ascend 机，orchestrator 留本机。worker 是 HTTP daemon，绑 `127.0.0.1`，访问通过 `ar_cli` 自动建立的 SSH tunnel，不暴露公网。

### B.1 [两端] 准备环境

| 在哪 | 做什么 |
|---|---|
| 远端 NPU 机 | 负责 worker/eval。准备 claude-autoresearch checkout 作为 `<repo_path>`；准备 `env_script`，source 完毕后 `python -c "import torch_npu, triton"` 退出码为 0；`npu-smi info` 能列出设备。**若使用 `ascendc_catlass` DSL**，在 `<repo_path>` 下跑 `bash scripts/download_catlass.sh`（一键 clone CATLASS C++ 库到 `thirdparty/catlass`，operator 不用配 `CATLASS_ROOT`；解析顺序见 [`config.yaml: catlass`](config.yaml)）。 |
| 本机 | 负责 Claude Code / orchestrator。安装 Python ≥ 3.10、PyYAML、Claude Code CLI；`~/.ssh/config` 配好 alias（下文以 `my-npu` 为例）+ 密钥免密登录远端。 |

<details><summary><code>~/.ssh/config</code> 怎么配 + 跑前自检</summary>

**第一步：本机 `~/.ssh/config` 配远端别名**。下面用 `my-npu`。

```text
# 文件路径：本机 ~/.ssh/config（没有就新建，注意是 config 不是 config.txt）
Host my-npu                            # 本地 SSH 别名，可按团队习惯命名
    HostName 192.168.x.x               # 远端机真实 IP 或域名
    User <remote-user>                 # 登录远端用的账号，比如 root
    # 私钥放在本机非默认位置时填写：
    # IdentityFile <本机私钥绝对路径>
    # 如需跳板机：
    # ProxyJump bastion
```

`Host` 别名通常与 B.2 的 `remote_worker.hosts.<alias>` 保持一致；需要不同名称时，在 B.2 填 `ssh_alias`。

**第二步：跑前自检**。在本机执行：

```bash
# 命令在本机敲。ssh 后面单引号里那一串都是在远端执行的，所以路径里的 <远端账号> 指远端机
# 的用户名，env.sh 是远端机上的脚本。
ssh my-npu 'source /home/<远端账号>/env.sh \
    && python -c "import torch_npu, triton" \
    && npu-smi info | head -3'
```

这条命令同时验证 SSH、`env_script`、`torch_npu/triton` import 和 NPU 可见性。失败时优先修对应的 SSH alias、env.sh 或 CANN/Python 环境。

</details>

### B.2 [本机] 配 worker host

`autoresearch/config.yaml`：

通用模板：

```yaml
remote_worker:
  hosts:
    my-npu:
      repo_path:  /home/<user>/claude-autoresearch
      env_script: /home/<user>/env.sh
```

`repo_path` 指远端 claude-autoresearch checkout。远端命令会在 `source env_script && cd repo_path` 之后运行 `python scripts/ar_cli.py worker ...`；解释器由 `env_script` 配好的 PATH 决定。

字段语义见 [参考 §8](#8-ar_cli-worker)。

### B.3 [本机] 检查并启动远端 worker（ar_cli 自动开 SSH tunnel）

状态检查：

```bash
cd autoresearch
python scripts/ar_cli.py worker --remote-host my-npu --status \
    --backend ascend --dsl ascendc_catlass
```

`--status` 能连上 worker 时输出 JSON；连不上时输出远端 env、import、NPU、磁盘和端口诊断。

启动远端 worker：

```bash
python scripts/ar_cli.py worker --remote-host my-npu --start \
    --backend ascend --devices <NPU-id> --dsl ascendc_catlass
```

一条命令做两件事：(1) SSH 到远端启动 daemon (2) 本机起 `ssh -L 9111:127.0.0.1:9111`。后续所有调用透传 tunnel。

省略 `--arch` 时由远端 CLI 用 `npu-smi info` 自动推断。如需覆盖，加 `--arch ascend910b<N>` 即可。

验证：
```bash
python scripts/ar_cli.py worker --remote-host my-npu --status
# → {"status":"ready", ...}
```

### B.4 [本机] 写 ref 和 kernel

同 [A.2](#a2-写-ref-和-kernel)。

### B.5 [本机] 启动 /autoresearch

```bash
cd autoresearch && claude
```

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <远端 NPU-id> --worker-url 127.0.0.1:9111
```

`--devices` 取**远端**卡下标。`--worker-url` 由 scaffold 透传写入 `task.yaml: worker.urls`，后续每轮 eval 自动走远端。

### B.6 [本机] 看进度 / 拿结果

同 [A.4](#a4-看进度)、[A.5](#a5-拿结果)。

### B.7 [本机] 停 worker（不再用时）

```bash
python scripts/ar_cli.py worker --remote-host my-npu --stop
```

会同时杀掉远端 daemon 和本机 tunnel。

---

## C. 批量

`autoresearch/scripts/batch/` 对一批 `(ref.py, kernel.py)` 任务循环跑 `/autoresearch`。批级状态写 `<batch_dir>/batch_progress.json`，round 级仍在 `ar_tasks/`。

### C.1 标准流程

约束：cwd 在 `autoresearch/` 内；长跑用 tmux/screen 包；`--devices` 指定本机或远端卡。

```bash
BATCH_DIR=/tmp/batch_001
DEVICE=0

# 0. 进入子目录
cd autoresearch

# 1. 放 ref/kernel（命名必须严格 <op>_ref.py / <op>_kernel.py，见参考 §2）
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels
cp workspace/*_ref.py    $BATCH_DIR/refs/
cp workspace/*_kernel.py $BATCH_DIR/kernels/

# 2. Tier-1 预检：纯静态（语法 / import / 必备 export），任意机器可跑，仅做 Python 静态检查
python scripts/batch/prepare.py $BATCH_DIR

# 3.（可选）Tier-2 预检：在有 NPU 的机器上跑 ref vs kernel
python scripts/batch/verify.py $BATCH_DIR --full

# 4. 后台跑（tmux 默认非 login shell，须 bash --login + source env.sh）
tmux new -d -s ar_batch \
    "bash --login -c 'source ~/env.sh && \
     python -u scripts/batch/run.py $BATCH_DIR --devices $DEVICE 2>&1 | tee -a $BATCH_DIR/batch.log'"

# 5. 另开终端监控
python scripts/batch/monitor.py $BATCH_DIR

# 6. 结束后汇总
python scripts/batch/summarize.py $BATCH_DIR
```

### C.2 配合远端 worker 跑批（本机无 NPU）

如果本机不是 NPU 机（即 B 章场景），批跑改动很小，整体仍按 [C.1](#c1-标准流程) 操作，但有两处差异：

1. **先按 [B.3](#b3-本机-启动远端-worker-ar_cli-自动开-ssh-tunnel) 启动远端 worker 并建立本机 tunnel**（每次开新 batch 前确认 tunnel 还活着：`python scripts/ar_cli.py worker --remote-host my-npu --status`）。
2. **`run.py` 加 `--worker-url 127.0.0.1:9111`**，本机 tmux 直接跑 `run.py`；eval 全走远端 worker。

第 4 步的 tmux 命令变为：

```bash
tmux new -d -s ar_batch \
    "python -u scripts/batch/run.py $BATCH_DIR \
       --worker-url 127.0.0.1:9111 --devices $REMOTE_DEVICE \
       2>&1 | tee -a $BATCH_DIR/batch.log"
```

`--devices` 在有 `--worker-url` 时仍要给，但取值是**远端**卡下标（透传到每个 op 的 task.yaml）。

其它步骤（prepare / monitor / summarize）和 [C.1](#c1-标准流程) 一样在本机直接跑。Tier-2 verify 需要 NPU 环境，完整预检可在远端执行 `python scripts/batch/verify.py <batch_dir> --full`；常规批跑也会在每轮 eval 中完成 verify。

### C.3 监控工具一览

| 工具 | 数据源 | 用途 |
|---|---|---|
| `monitor.py` | progress + ar_tasks/ | 实时队列 + phase + heartbeat |
| `monitor.py --dashboard` | 同上 | `execvp` 至 `dashboard.py` 看当前 task TUI |
| `summarize.py` | 仅 progress JSON | 离线汇总，复制粘贴友好 |
| `tail -f batch.log` | claude stdout | 查 hook 输出、Edit、Bash |

### C.4 断点续跑 / 重试

- `done` 保持完成状态
- `error` 默认跳过；`--retry-errored` 重新纳入
- `pending` 自动续
- `running`（终止瞬间在跑的 op）下次启动时自动降级为 `error`
- transient API/network 失败且 task 状态完整时，batch supervisor 会按 `config.yaml: batch.transient_retries` 接续

同一 batch_dir 由 `.batch.lock` 串行化；已退出进程残留的 lock 会在下次启动时自动判活清理。

完整 batch CLI 参数见 [参考 §1](#1-cli-参数)。

---

## D. 续跑 / 重置 / 查询

A/B/C 均适用：

| 操作 | 命令 |
|---|---|
| 续跑最近 task | `/autoresearch --resume` |
| 续跑指定 task | `/autoresearch --resume <task_dir>` |
| 重新开始 | 删除 `ar_tasks/<task_dir>/`，再 `/autoresearch --ref ... --kernel ...` |
| 查每轮记录 | `<task_dir>/.ar_state/history.jsonl`（一行一轮）|
| 查 plan 当前状态 | `<task_dir>/.ar_state/plan.md` |

---

---

# 参考

操作部分已经涵盖正常使用。下面按主题归档配置和机制说明。

## 1. CLI 参数

### 1.1 `/autoresearch`

四种调用形态由参数决定：

| 形态 | 触发条件 | 行为 |
|---|---|---|
| 新建任务 | `--ref X.py --kernel Y.py --op-name N` + (`--devices` 或 `--worker-url`) | scaffold 建 task_dir → 跑 baseline → 进 PLAN |
| 续跑最近 | `--resume`（无目录参数）| 自动找最近活跃的 task |
| 续跑指定 | `--resume <task_dir>` 或直接 `<task_dir>` | 续指定目录 |

新建任务的完整参数：

| flag | 类型 | 必填？ | 默认 | 说明 |
|---|---|---|---|---|
| `--ref` | 路径 | ✅ | — | PyTorch ref 文件（`<op>_ref.py` 或任意 `.py`，详见 [§2](#2-命名契约)）|
| `--kernel` | 路径 | ✅ | — | 种子 kernel 文件 |
| `--op-name` | 字符串 | ✅ | — | op 名，决定 task_dir 前缀 |
| `--devices` | 整数或逗号列表（`5` 或 `0,1,2,3`）| 二选一 | — | 本机 NPU 卡 id；如果给了 `--worker-url` 则可不填，由 worker 自带卡列表 |
| `--worker-url` | `host:port[,host:port]` | 二选一 | — | 走远端 worker（写入 `task.yaml: worker.urls`）|
| `--max-rounds` | int | ❌ | 30（[config.yaml](config.yaml): `defaults.max_rounds`） | 优化轮数上限，触发后进 FINISH |
| `--eval-timeout` | int 秒 | ❌ | 600（同上 `defaults.eval_timeout`） | 单 shape 的 verify+profile wall-clock 上限 |
| `--output-dir` | 路径 | ❌ | `ar_tasks` | task_dir 父目录 |
| `--no-code-checker` | flag | ❌ | (启用) | 关掉 [`engine/quick_check.py`](scripts/engine/quick_check.py) 的 Triton 退化 AST 检查（一般用于调试种子 kernel）|

注：`arch`（如 `ascend910b3`）由 `--devices` 选中的卡经 `npu-smi info` 自动推断，写进 `task.yaml` 仅供 dashboard / report 显示，用户可省略。

---

### 1.2 `scripts/ar_cli.py worker`（worker 启停 + tunnel）

`worker` 三种模式互斥：

| 模式 | 用法 |
|---|---|
| `--start` | 启动 worker daemon。如配合 `--remote-host`，会先 SSH 到远端启 daemon，再在本机起 `ssh -L <port>:127.0.0.1:<port>` tunnel |
| `--stop` | 停 `--port` 上的 daemon（POSIX-only）；配合 `--remote-host` 时还会 SIGTERM 本机 tunnel |
| `--status` | 查询 `127.0.0.1:<port>/api/v1/status`（含 ready / 卡 / 空闲 slot）；不可达时输出本机或远端诊断 |

所有 flag：

| flag | 类型 | 必填？ | 默认 | 说明 |
|---|---|---|---|---|
| `--start` / `--stop` / `--status` | flag | ✅ 三选一 | — | 互斥 |
| `--backend` | `ascend` / `cuda` / `cpu` | `--start` 时必填 | — | 硬件后端 |
| `--arch` | 字符串（如 `ascend910b3`） | ❌（`--start` 时不传则由 `npu-smi info` 自动推断） | 自动推断 | 写入 worker 自报的 arch |
| `--devices` | 逗号分隔（如 `2,5`） | `--start` 时必填 | — | worker 管理的卡集合 |
| `--dsl` | 字符串（如 `ascendc_catlass` / `triton_ascend`） | ❌ | — | 目标 DSL；`--status` 诊断会据此决定 triton 缺失是 warning 还是 fatal |
| `--port` | int | ❌ | 9111（[config.yaml](config.yaml): `worker.port`） | TCP 端口 |
| `--force` | flag | ❌ | (校验) | `--stop` 时跳过 worker 进程 cmdline 校验 |
| `--remote-host` | SSH alias | ❌ | (本机) | 通过 SSH 在 `config.yaml: remote_worker.hosts.<alias>` 定义的远端机执行同样命令；`--start` 时额外打通本机 tunnel |

---

### 1.3 `scripts/batch/run.py`

位置参数 `<batch_dir>` 必填（指向 [§6](#6-文件与状态布局) 的 batch 目录）。

| flag | 类型 | 必填？ | 默认 | 说明 |
|---|---|---|---|---|
| `--devices` | 整数或逗号列表 | ✅（除非给了 `--worker-url`）| — | 透传给每个 op 的 `/autoresearch --devices` |
| `--worker-url` | `host:port[,host:port]` | ❌ | — | 透传 `--worker-url`，整批走远端 worker，本机可无 NPU |
| `--mode` | `ref-kernel` / `ref` | ❌ | `manifest.mode` 或 `ref-kernel` | 整批是否要 kernel；`ref` 模式下 kernel 由 agent 现写 |
| `--max-rounds` | int | ❌ | 同 `/autoresearch` | 透传 per-op |
| `--eval-timeout` | int 秒 | ❌ | 同上 | 透传 per-op |
| `--timeout-min` | int 分钟 | ❌ | 180（[config.yaml](config.yaml): `batch.run_timeout_min`）| 单 op wall-clock 上限，超时杀 `claude --print` 子进程 |
| `--only` | 逗号分隔 op 名 | ❌ | — | 队列过滤：只跑这些 |
| `--limit` | int | ❌ | — | 队列过滤：跑前 N 个 |
| `--retry-errored` | flag | ❌ | (跳过) | 把 `error` / `stale running` 重新纳入队列（下一次批跑时生效）|
| `--cooldown-sec` | int 秒 | ❌ | 5（[config.yaml](config.yaml): `batch.cooldown_sec`）| 两个 op 之间的间隔 |
| _无 CLI flag_ | int | — | 3（[config.yaml](config.yaml): `batch.transient_retries`）| 单 op 内 transient 自动续：claude.exe 异常退但 framework progress 完整时，supervisor 自动 `/autoresearch --resume --force` 接续最多 N 次。见 [§C.4](#c4-断点续跑--重试)。|
| `--claude-bin` | 路径 | ❌ | `claude` | Claude CLI 可执行文件路径（Windows 下需指向 `claude.cmd`）|
| `--model` | 字符串 | ❌ | (空) | 透传到 `claude --model`，覆盖默认模型 |
| `--extra-claude-arg` | 字符串（可重复）| ❌ | — | 任意额外 flag 透传给 `claude --print`（可叠加：`--extra-claude-arg --foo --extra-claude-arg --bar`）|

## 2. 命名契约

| 项 | 约束 |
|---|---|
| ref 文件名 | 单跑任意；批跑严格 `<op>_ref.py` |
| kernel 文件名 | 单跑任意；批跑严格 `<op>_kernel.py` |
| ref 暴露 | `class Model(nn.Module)` + `get_init_inputs()`，并二选一：`get_inputs()` 单 shape / `get_input_groups()` 多 shape |
| kernel 暴露 | `class ModelNew(nn.Module)`；forward 内调用目标 DSL kernel |
| 同目录数据文件 | 与 ref 同名或以 ref basename + `_` 开头的 `.json` / `.pt` / `.npz` 会随任务打包。例如 `31_IOU.py` 可配 `31_IOU.json` 或 `31_IOU_cases.json` |
| task_dir 命名 | scaffold 生成 `ar_tasks/<op_name>_<unix_ts>_<uuid6>/`，供 resume、dashboard、batch 关联 |

## 3. 主循环 / 状态机

单轮：`PLAN → EDIT → quick_check → eval → KEEP/DISCARD → settle`。
连续 N 次 FAIL 转 `DIAGNOSE`（N 默认 3，配 [`config.yaml`](config.yaml) `defaults.consecutive_fail_threshold`）；plan 全部 settle 转 `REPLAN`；预算耗尽转 `FINISH`。DIAGNOSE 子代理对同一 plan_version 最多重试 `defaults.diagnose_max_attempts` 次（默认 5），用尽后转人工 fallback。

```
INIT
  │  /autoresearch --ref X.py --kernel Y.py
  ▼
BASELINE  (scaffold --run-baseline 原子完成，运行 seed kernel)
  │
  ▼
PLAN  (BASELINE PASS 或 FAIL 均进入 PLAN；FAIL 时首批 plan items 改写 seed)
  │  create_plan.py 校验
  ▼
   ┌────────────────────────────────── EDIT ◀────────────┐
   │  pipeline.py:                                       │
   │    quick_check → run_eval → keep_or_discard         │
   │   ├─ KEEP    : git commit (editable_files)，更新 best│
   │   ├─ DISCARD : 回滚 editable_files                   │
   │   └─ FAIL    : consecutive_failures++，回滚         │
   │                                                     │
   │   ├─ failures ≥ N (config)     ─→ DIAGNOSE ─→ PLAN ─┤
   │   ├─ plan 全部 settle           ─→ REPLAN  ─→ PLAN ─┤
   │   └─ eval_rounds == max_rounds ─→ FINISH
   └─────────────────────────────────────────────────────┘
```

DIAGNOSE / REPLAN 不绕回 PLAN：`create_plan.py` 校验通过后 hook 直接写 `phase = EDIT`。每个 plan item 在 `history.jsonl` 中持有 KEEP/DISCARD/FAIL 终态，或在 REPLAN/DIAGNOSE 边界被丢弃；pid 单调递增不复用。

阶段产物：

| 阶段 | Claude 操作 | 产物 |
|---|---|---|
| BASELINE | `baseline.py` | seed_metric 写入 state.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py` | plan.md（含 (ACTIVE) 标记）|
| EDIT | Edit kernel.py 后跑 `pipeline.py` | history.jsonl，可选 git commit |
| FINISH | (auto) `pipeline.py` → `report.py` | report.md（含内嵌 SVG）|

## 4. Eval 执行链

`baseline.py` 与 `pipeline.py` 进程内直接调用 `task_config.run_eval`，按 `worker_urls` 是否非空分两条路径。

**本地路径**（默认）：

```
baseline.py / pipeline.py
 └─ task_config.run_eval(task_dir, config, device_id)
     └─ utils.eval_runner.local_eval                       # 2-pass driver
          ├─ Popen eval_kernel.py --phases profile_base    # ref 单独跑
          └─ Popen eval_kernel.py --phases verify,profile_gen
```

**远程路径**（`--worker-url` 或 `task.yaml: worker.urls` 非空）：

```
baseline.py / pipeline.py
 └─ task_config.run_eval(..., worker_urls=[...])
     ├─ package_builder.build_package                      # tar.gz: task.yaml + ref + kernel + data_files
     └─ HTTP POST /api/v1/run → worker                     # multipart
                                  └─ safe_extract →
                                     utils.eval_runner.local_eval（同上）
```

两条路径在 worker 内部共用同一份 `local_eval`，确保本地与远程行为不漂移。eval_kernel 调用拆成 ref / kernel 两个独立子进程，避免 kernel 端 SIGKILL 或设备不可用污染已落盘的 ref 时延。

verify 失败时 ref 时延由 `profile_base` 同进程单独测得，与 verify 解耦，dashboard 顶栏始终显示 PyTorch baseline。Sticky baseline 写定（`baseline_metric` 与 `baseline_source=ref` 写入 state.json）后，后续轮 phase 列表不再包含 `profile_base`。

**直接调试 eval_kernel.py**（薄 CLI，包装 [`scripts/eval/`](scripts/eval/) 的 KernelVerifier）：
```bash
cd <task_dir>
python <repo>/scripts/engine/eval_kernel.py \
    --task-dir . --op-name <op> --kernel-file kernel --ref-file reference \
    --device-id 0 --phases verify,profile_gen
```

可选项（默认对齐 autoresearch 当前 NPU+Triton+PyTorch 场景）：`--backend ascend` `--dsl triton_ascend` `--arch ascend910b4` `--framework torch` `--task-id 0` `--current-step 0` `--verify-timeout 300`。`--arch` 在框架调用时由 [`eval_runner.local_eval`](scripts/utils/eval_runner.py) 从 `task.yaml: arch` 自动透传。

**客户端断连自动 cancel**：worker `/api/v1/run` 同时 await eval task 与 disconnect watch：
- eval 先完成 → 取消 watch，返回结果
- 客户端先断开（例如 `claude --print` wall-clock 终止）→ 取消 eval、subprocess group SIGTERM、释放 device，返回 HTTP 499

避免客户端进程终止后 eval 继续占用 device。非有限浮点（inf/-inf/nan）在序列化前递归改写为 `null`，避免 FastAPI 默认 JSON encoder 拒收返 HTTP 500。

## 5. 精度容差

verify（Tier 2 预检）与 `/autoresearch` 每轮 verify 共用 [`correctness.py`](scripts/utils/correctness.py)，对齐已迁移至 [`scripts/eval/adapters/framework/torch.py`](scripts/eval/adapters/framework/torch.py) 的 CANN MARE/MERE-aligned 分层容差表：

- 按 ref dtype 取 `(rtol, atol, outlier_rtol, outlier_atol, outlier_ratio)`：fp32 → `(1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)`、fp16 → `(9.77e-4, 1e-3, 9.77e-3, 1e-2, 0.005)`、bf16 → `(7.81e-3, 1e-2, 7.81e-2, 1e-1, 0.010)`，未知 dtype 回落 fp32
- 元素级双带判定：`strict_tol = atol + rtol·|ref|`，`relaxed_tol = outlier_atol + outlier_rtol·|ref|`；`hard_fail`（超 relaxed）> 0 即 FAIL，或 `outlier`（在 strict 与 relaxed 之间）超 `outlier_ratio · total` 即 FAIL
- 额外硬性检查：NaN 位置一致、Inf 位置 + 符号一致、bool/int 精确匹配
- 容差表通过 [`config.yaml:precision`](config.yaml) 可覆盖，运行时由 [`utils/settings.py`](scripts/utils/settings.py) `precision_table()` 注入

## 6. 文件与状态布局

| 路径 | 用途 |
|---|---|
| `workspace/<op>_ref.py` / `<op>_kernel.py` | 候选 ref/kernel 输入 |
| `task.yaml` | name / arch / editable_files / ref_file / devices / eval_timeout / max_rounds / metric / `data_files` / `worker.urls` |
| `.ar_state/state.json` | **单文件控制态**：phase / owner / progress / pending_settle / 一致性 expected_* |
| `.ar_state/plan.md` | 规划与结算历史（权威态）|
| `.ar_state/history.jsonl` | 每轮 decision / metrics / commit |
| `.ar_state/plan_items.xml` | PLAN/DIAGNOSE/REPLAN 写给 `create_plan.py` 的 XML |
| `.ar_state/diagnose_v<N>.md` | DIAGNOSE 结构化诊断报告（CLAUDE.md 不变量 #10）|
| `config.yaml` | `hallucinated_scripts`（脚本名容错）+ `remote_worker.hosts`（SSH alias → repo_path / env_script）|
| `.claude/settings.json` | Hook + 权限（提交至仓库）|
| `.claude/settings.local.json` | API key / model 覆盖（不提交至 git）|

`.ar_state/` 内除 `plan_items.xml` 与 `diagnose_v<N>.md` 外均由 hook 与脚本管理，Claude 不可手写。

**Batch 目录布局**：

```
<batch_dir>/                         ← 批级
  manifest.yaml                      # prepare.py 写
  batch_progress.json                # run.py 写：每个 op 的 status / task_dir / metrics
  batch.log                          # run.py 写：claude --print 的全部 stdout
  verify_results.json                # prepare.py / verify.py 写
  refs/<op>_ref.py                   # 文件名必须严格为 <op>_ref.py
  kernels/<op>_kernel.py             # 文件名必须严格为 <op>_kernel.py

<repo>/ar_tasks/<op>_<ts>_<uuid>/    ← round 级（由 /autoresearch 维护）
  kernel.py / reference.py / task.yaml
  .ar_state/{state.json, plan.md, history.jsonl, report.md}
```

`batch_progress.json:cases.<op>.task_dir` 连接批级与 round 级。

## 7. env.sh 契约

适用场景：tmux 非 login shell、`ar_cli worker --remote-host` 远端启动、batch `bash --login -c 'source env.sh && python ...'`。三处共用同一脚本，路径填入 `config.yaml: remote_worker.hosts.<alias>.env_script`。

**唯一职责**：source 完成后，PATH 上的 `python` 执行 `python -c "import torch_npu, triton"` 不抛异常。

模板（按本机 Python + CANN 安装方式选一）：

```bash
# 例 1：conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env 名>
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 例 2：venv
source ~/.venvs/<env 名>/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 例 3：系统 Python，仅加载 CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**写法约定**：env.sh 负责环境变量和 Python 环境初始化；工作目录由 ar_cli 切换到 `repo_path`。

验证：`source ~/env.sh && python -c "import torch_npu, triton"` 退出码 0。

## 8. ar_cli worker

Worker 是个 FastAPI HTTP daemon，跑 eval 子进程。`scripts/ar_cli.py worker` 负责本机/远端起停、ssh -L tunnel 维护、探活。

### 常用示例

```bash
scripts/ar_cli.py worker --remote-host my-npu --start --backend ascend --devices 0,1 --dsl ascendc_catlass
scripts/ar_cli.py worker --remote-host my-npu --status
scripts/ar_cli.py worker --remote-host my-npu --stop
```

本机模式使用同一组 `worker` 命令，省略 `--remote-host <alias>` 即可。

`--start` 幂等：daemon 已在跑时返回现有状态；tunnel 抖了再跑一次 `--start` 即可恢复。`--status` 能连上 worker 时输出 JSON；连不上时输出本机或远端诊断。

### 参数

`--start` / `--stop` / `--status` 三选一，互斥。

| flag | 类型 | 默认 | 必填 | 说明 |
|---|---|---|---|---|
| `--backend` | `ascend` / `cuda` / `cpu` | — | `--start` 必填 | 硬件后端 |
| `--devices` | csv，如 `0,1,2` | — | `--start` 必填 | device id 列表 |
| `--arch` | 字符串，如 `ascend910b3` | auto（`npu-smi info` 推断）| 可省 | 硬件 arch |
| `--dsl` | 字符串，如 `ascendc_catlass` / `triton_ascend` | — | 可省 | 目标 DSL；`--status` 诊断按 DSL 确定 triton 策略 |
| `--port` | int | `config.yaml: worker.port`，否则 `9001` | 可省 | TCP 端口 |
| `--remote-host` | alias | — | 可省 | 走 SSH 远端模式；alias 在 `config.yaml: remote_worker.hosts.<alias>` 定义 |

`config.yaml: remote_worker.hosts` 字段：

```yaml
remote_worker:
  hosts:
    my-npu:
      repo_path:  /abs/path/to/repo   # 必填
      env_script: /abs/path/env.sh    # 必填
      ssh_alias:  my-npu              # 可省，默认 = key
```

`repo_path` 指远端 claude-autoresearch checkout。远端入口固定为 `python scripts/ar_cli.py`；`python` 来自 `env_script` source 后的 PATH。

## 9. Skills 库

根目录：`skills/`，按 DSL 分区：`triton-ascend/`、`triton-cuda/`、`pypto/`、`cpp/`、`cuda-c/`、`tilelang-cuda/`。每个 DSL 下进一步分为：

- `fundamentals/` — API rules、hardware constraints、basics、debugging、grid-config、memory、optimization
- `guides/` — attention、matmul、reduce、elementwise、elementwise-reduce-fused 等场景指南
- `cases/` — 单算子题型 case
- `examples/` — 端到端可运行示例
- `evolved-fix/`、`evolved-improvement/` — 故障排查与性能优化的演化知识

PLAN 阶段 hook 提示 Claude Glob `../skills/<dsl>/**/SKILL.md`、Read 1-3 个最相关的，将文件名写入 plan item rationale。

## 10. Hook 与内部机制

外部接口（slash 命令、`task.yaml`、`.ar_state/` 路径）保持稳定。修改内部实现的入口：

| 主题 | 入口 |
|---|---|
| Bash gate（命令在何 phase 合法）| [phase_policy.py](scripts/phase_machine/phase_policy.py) 头部注释 |
| Hook 接线 | [.claude/settings.json](.claude/settings.json)；脚本位于 [hooks/](scripts/hooks/)，`guard_*.py` / `post_*.py` / `stop_*.py` |
| phase 转移 | [phase_machine/state_store.py](scripts/phase_machine/state_store.py) 阶段常量；`compute_next_phase` / `compute_resume_phase` 在 [phase_policy.py](scripts/phase_machine/phase_policy.py) 末尾 |
| 测时 + verify/profile | [eval_kernel.py](scripts/engine/eval_kernel.py)（静态脚本，CLI 接 `--phases`）|
| Eval 执行链 | [task_config/eval_client.py](scripts/task_config/eval_client.py) `run_eval` → 本地 [eval_runner.py](scripts/utils/eval_runner.py) `local_eval` → subprocess [eval_kernel.py](scripts/engine/eval_kernel.py)；远程 [package_builder.py](scripts/task_config/package_builder.py) → HTTP POST → [worker/server.py](scripts/worker/server.py) → 同一份 `local_eval` |
| Remote worker SSH 调度 | [ar_cli.py](scripts/ar_cli.py) `worker --remote-host` 分支；config 在 [config.yaml](config.yaml) `remote_worker.hosts` |
| Triton 退化静态检查 | 规则与 AST：[`scripts/eval/validate_triton_impl.py`](scripts/eval/validate_triton_impl.py)（canonical，已迁移进 in-tree eval 包）；EDIT 前入口 [`engine/quick_check.py`](scripts/engine/quick_check.py)（其 docstring 列出三类退化模式）；[`scripts/utils/validate_triton_impl.py`](scripts/utils/validate_triton_impl.py) 仅为 re-export shim |
| DIAGNOSE 契约 | [CLAUDE.md](CLAUDE.md) 不变量 #9（canonical-form bash）+ #10（DIAGNOSE artifact）|
| 子代理 | [.claude/agents/ar-diagnosis.md](.claude/agents/ar-diagnosis.md) |

Hook 通过 [.claude/settings.json](.claude/settings.json) 接入，运行时围绕 `state.json` 维护 phase、owner 和 progress。日常使用主要通过 `/autoresearch`、`scripts/batch/*` 和 `scripts/ar_cli.py worker` 入口操作；内部改动按上表定位。

## 11. 并发与冲突

运行约定：

| 资源 | 推荐做法 |
|---|---|
| batch 目录 | 一个 `<batch_dir>` 同时跑一个 `run.py`，状态由 `.batch.lock` 串行化 |
| task 目录 | 一个 `<task_dir>` 同时由一个 Claude session 驱动，owner/heartbeat 写在 `state.json` |
| NPU 卡 | 一张卡交给一个本地 eval 或一个 worker；批量任务按卡分流，或统一走同一个 worker 队列 |
| worker / config | 修改 `scripts/` 或 `config.yaml` 后重启 worker，使 daemon 读取新代码和配置 |

常用恢复入口：

| 场景 | 操作 |
|---|---|
| batch 被旧 lock 卡住 | 确认对应进程已退出后清理 `.batch.lock` |
| task 被旧 owner 卡住 | 等 `resume.heartbeat_fresh_seconds` 过期，或确认旧 session 已退出后清理 `state.owner` |
| tunnel 不通 | 先 `ar_cli worker --remote-host ... --status`，再按需 `--stop` / `--start` |
| worker 返回源码漂移提示 | 重启 worker |
