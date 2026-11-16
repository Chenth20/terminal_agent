# Terminal-Bench 2.1 takeaways

> 原始集群路径、平台作业和文内脚本命令按实验发生时记录。当前服务器入口和可复现范围见[交接文档](../HANDOFF.md)。

TB2.1 评测的全部结论：arena 上的口径修复（§1–§4、§6–§7）、TerminalHorizon job 的系统问题（§5）、本地 singularity 复刻与 arena 的对齐（§8）。
本地复刻的构建/运行/调试流程单独成篇：[tb21_local_pipeline.md](tb21_local_pipeline.md)；全部 12 项修复的台账（症状/根因/改动/作废清单）在 [tb21_verifier_audit.md](tb21_verifier_audit.md)。版本选型总表在 [arena_verifier.md](arena_verifier.md)。

| 想知道 | 看 |
|---|---|
| dsv41 / flash 在 TB2.1 到底多少分，和官方差在哪 | §0、§4 |
| 旧口径为什么不可信，mirror 口径改了什么 | §1、§2、§3 |
| TH job 的 282 条非 DONE、flat 镜像、端口扫描 | §5 |
| 提交 / 归因 / 看 reward 日志的命令 | §6 |
| SFT 学生模型评测的 verifier 审计与切口径 | §7 |
| 本地 tb_local 为什么比 arena 低、哪些样本作废、修后对不对得上 | §8 |
| tb_local 的每一项修复（含 #9 libgthread、#11 端点漂移、#12 丢结果）长什么样 | [tb21_verifier_audit.md](tb21_verifier_audit.md) |

## 0. 一页结论（10-31；本地复刻见 §8）

| 模型 | 旧口径（`tb21-terminus2-full@2.1.3` + harness 1.2.96/1.0.0） | 新口径（`cth-tb21-full89-mirror@1.0.1` + harness 1.6.96） | 官方 report |
|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 (max) | 0.753（178 任务，10-26） | **0.831**（178 任务，10-31 实跑；修完最后两处 reward 网络问题后同一批轨迹按条件口径折算 ≈ 0.84） | 92 |
| qwen3.8-flash (xhigh) | 0.770（100 任务，10-19） | 未重跑；按同一批轨迹"pytest 真跑了"的条件准确率估 **≈0.90–0.92**（84 条 0.917，去掉 prove-plus-comm 坏题效应） | 86.1 |

**旧口径里 10–16% 的任务是在 pytest 根本没跑的情况下被记 0 的**，不是模型做错。差距的一大半来自评测基础设施（reward 阶段装 uv / 拉 pytest 走不通的公网），剩余 ~8–9 pt（dsv41）是 scaffold + 口径 + 模型"自宣完成"行为，不再有环境因素可挖。

失败归因脚本 `coding_arena/code/arena/eval/failure_audit.py <job_id>...`（副本 `../arena_assets/scripts/failure_audit.py`）：直接读 `pull_trajectories.sh` 落盘的 `output/<job>/`，输出每个 run 的失败主因表、网络信号表、地板题清单与非模型失败清单。SFT 学生模型那批评测的 verifier 审计与切口径过程见 §7；TerminalHorizon 侧的系统问题见 §5 与 [trajectory_collection.md](trajectory_collection.md) Part C。

---

## 1. Benchmark 版本

### `tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2`（jjsong，官方 TB2.1 口径，10-30 前所有 run）

就是官方 TB2.1，只修了一道题的评分环境。

- 基线是 `2.1.1`（hechuyi）。`2.1.3` 与它的 89 道题、题面、oracle、权重和聚合方式（mean / failed_as_zero）完全相同。
- 唯一改动在 `tb2-pytorch-model-cli`：官方 `test.sh` 评分时要联网下载依赖，常超时。修法是把依赖烘焙进镜像（env `6.4.2-gepa-cache`）+ reward `tb2-reward@4.0.18-gepa-envcache` 离线评分。`test.sh` 本身没改。
- 版本名：`pytorch-cache` 指上面这处修复；`v2` 是第二版；`gepa` 是发布者的项目标签，对评测无影响。

**问题**：只修了 `pytorch-model-cli` 一道题，其余 88 题的 `test.sh` 仍然每次评分都走公网装 uv、拉 pytest（见 §2）。

### `cth-tb21-full89-mirror@1.0.1`（chentinhong.cth，10-31 起默认）

同一批 89 个 env（`tb2-*@6.4.0`）、同样 weights / mean / failed_as_zero，只把 reward 换成 `cth-tb2-reward-mirror@1.0.1`。官方 `test.sh` / `test_outputs.py` 一字不动，改的只是 reward runner 给它的运行条件（§3）。与官方口径仍可直接比较。

配套：
- `cth-tb21-full88@1.0.0`：官方 reward 下去掉 `tb2-prove-plus-comm` 的 88 题（官方 reward 对这题必 0，见 §2.3）。
- `cth-tb21-pipsmoke@1.0.0` / `cth-tb21-rewardsmoke@1.0.0` / `cth-tb21-rewardsmoke2@1.0.0`：4 题冒烟，分别验证 agent 侧 pip 镜像、reward-mirror 1.0.0、reward-mirror 1.0.1。
- spec 都在 `coding_arena/work/benchmarks/tb21/`。

---

## 2. 评测里的非模型因素（按影响从大到小）

### 2.0 审计的 run

同一 benchmark `tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2`（89 题），全部 DONE、0 HARNESS_FAILED：

| run | 模型 | harness | n | mean(fail=0) | 剔除地板题 |
|---|---|---|---|---|---|
| `cth-tb21x3-stream-flash-2610190249` | qwen3.8-flash xhigh | 1.0.0（262k） | 100 | 0.770 | 0.802（96） |
| `cth-tb21-96-dsv41-max-2610260221` | dsv41-flash max | 1.2.96 | 178 | **0.753** | 0.788（170） |
| `cth-tb21-96-dsv41-xhigh-2610260221` | dsv41-flash xhigh | 1.2.96-xhigh | 178 | 0.713 | 0.747 |
| `cth-tb21-96-dsv41-max-toolhist-2610260517` | dsv41 看得到历史推理 | 1.4.96 | 178 | 0.747 | 0.782 |
| `cth-tb21-nocomp-dsv41-max-2610290159` | dsv41 不压缩 | 1.3.0 | 178 | 0.730 | 0.765 |

stream 系列 setup 间差 ≤0.06，全在 89×2 样本的噪声带（SE≈0.03）内。10-17 那批旧 harness run（claude-code 1.0.2 / 官方 terminus / mimo，0.33–0.51）有 16–54 条 HARNESS_FAILED/CANCELLED，是基础设施失败，不应引用。

每条 0 分轨迹只记一个主因（非模型因素优先）：

| 主因 | flash | dsv41 max | dsv41 xhigh | dsv41 toolhist | dsv41 nocomp |
|---|---|---|---|---|---|
| `model:wrong`（自宣完成，test 不过，无环境信号） | 16 | 31 | 37 | 32 | 36 |
| `model:wrong(network-noise-present)`（同上，但轨迹里出现过 pip 超时等） | 3 | 4 | 3 | 1 | 5 |
| `infra:network-gave-up`（模型明说因网络放弃运行时验证） | 1 | 2 | 2 | 4 | 1 |
| `infra:oom-or-disk`（`Killed` / No space） | 1 | 1 | 2 | 4 | 1 |
| `limit:agent-deadline`（撞 4 h） | 1 | 2 | 4 | 1 | 2 |
| `harness:silent-end`（无 complete、无超时、中途结束） | 0 | 2 | 1 | 1 | 1 |
| `reward:test.sh-exit!=0`（reward 脚本自身失败） | 1 | 2 | 2 | 2 | 2 |

这张表是从轨迹侧看到的；reward 阶段"pytest 没跑就记 0"（§2.1）只有拉了 `tbench_test.log` 才看得见，是后来才发现的最大一块。

### 2.1 Reward 阶段自己的网络（最大，每 job 10–16% 任务记 0）

官方每道题的 `tests/test.sh` 开头都是：

```
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh     # 装 uv
source $HOME/.local/bin/env
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest /tests/test_outputs.py
```

从 reward 沙箱出口（用 noop harness + dump reward 实测，`coding_arena/build/reward_dump/`）：`astral.sh` **不可达**（20 s 超时/502）、`pypi.org` 时而 20 s 超时、`files.pythonhosted.org` 拉 wheel 频繁 `operation timed out`，而 `mirrors.aliyun.com` 0.15–2 s 稳定。`tb2-reward@4.0.18` 的 runner 虽然随包带了 uv tarball 并设了 `UV_DOWNLOAD_URL=file://…`，但 `test.sh` 是 `curl | sh` 直接下 installer 脚本本身——installer 拿不到，uv 就没装，后面 `uvx: command not found`，reward.txt 写 0。

按 job 统计"pytest 没跑就记 0"（`tbench_test.log` 里没有 pytest summary；`arena task download <task> tbench_test.log` 可取，`pull_trajectories.sh` 10-31 起自动落盘）：

| job | DONE | pytest 没跑 | uv installer 失败 | pypi 超时 | no-workdir | 报告分 | pytest 跑了的条件准确率 |
|---|---|---|---|---|---|---|---|
| flash 1.0.0 (10-19) | 100 | 16 (16%) | 0 | 15 | 1 | 0.770 | **0.917** |
| dsv41 max 1.2.96 (10-26) | 178 | 20 (11%) | 0 | 18 | 2 | 0.753 | **0.848** |
| dsv41 xhigh | 178 | 18 (10%) | 0 | 16 | 2 | 0.713 | 0.794 |
| dsv41 toolhist 1.4.96 | 178 | 20 (11%) | 0 | 18 | 2 | 0.747 | 0.842 |
| dsv41 nocomp 1.3.0 | 178 | 17 (10%) | 0 | 15 | 2 | 0.730 | 0.807 |
| dsv41 max 1.5.96 + 官方 reward (10-31) | 176 | **71 (40%)** | 60 | 11 | 0 | 0.511 | 0.88 |

10-31 那天 astral.sh 整段不可达，同一个模型同一套题直接掉到 0.51——这个数字说明旧口径的分数**随网络状况漂移**，根本不稳定。

受影响集中在 test 阶段要装重依赖的题：`mailman`、`mteb-retrieve`、`sam-cell-seg`、`custom-memory-heap-crash`、`modernize-scientific-stack`、`overfull-hbox`……

### 2.2 Agent 侧 pypi 慢（1–3 pt）

沙箱到 `files.pythonhosted.org` 经常 `ReadTimeoutError (read timeout=15)` 反复 retry；dsv41 baseline 178 条里 24 条出现。大多被 pip 自动重试救回（17 条照样通过），但失败轨迹里有明确因果：`tb2-sam-cell-seg/s1` 第 15 步写 "Could eventually fail due to poor connectivity. We may need to rely on syntax/code review" → 放弃运行测试、语法检查后交卷 → 0 分。登录节点实测：`mirrors.aliyun.com/pypi` 0.28 s；`pypi.org`、tuna、tencent 全部 15 s 超时；镜像里 apt 源本来就是 `mirrors.aliyun.com/ubuntu`。

修法是 harness `1.5.96` 的 `TERMINUS_SANDBOX_ENV_*` 透传（§3 第 2 行）。验证（`cth-tb21-pipsmoke-dsv41-1596-2610301615`，4 题 ×3）：所有装包轨迹 `Looking in indexes: https://mirrors.aliyun.com/pypi/simple/`，**0 次 ReadTimeout**（baseline 这 4 题 8 条里 6 条有）。

### 2.3 坏题 `tb2-prove-plus-comm`（1.1 pt）

全平台 112 次尝试 0 解、唯一 reward `exit=1` 的 env。根因是 **reward runner 把 WORKDIR 写死 `/app`**：这题是 89 题里唯一 `workdir=/workspace` 且没有 `details.json` 的，官方 runner `cd /app` 失败落到 `/`，`test.sh` 的 `$PWD = /` 检查直接 exit 1（`No working directory set`）。即便 flash 的轨迹 `coqc` 编译通过、`Print Assumptions` 输出 `Closed under the global context`，也是 0。

### 2.4 OOM（<1 pt，但解锁了两题）

`make-doom-for-mips`、`make-mips-interpreter` 编译时 `Killed`，harness `resource_spec.memory=8Gi`。提到 16Gi 后两题 dsv41 都从 0 → 2/2。

### 2.5 其它（各 ≤1 pt）

- 4 h deadline：`extract-moves-from-video`（400+ 步逐帧 OCR）、`gpt2-codegolf`。不改 TTL，归地板题。
- 静默中断：每 run 1–2 条，无 complete、无超时、harness.log 干净，轨迹中途结束。`failure_audit.py` 单列 `harness:silent-end`，比例 ~1% 当噪声。
- 大 wheel 超时：test 装 torch 要拉 300+ MB `nvidia-cusparse`，`UV_HTTP_TIMEOUT=120` 不够（`mteb-retrieve`）。
- HF xet：test 里 `load_dataset("allenai/c4")` 走 xet 存储，`cas-server.xethub.hf.co` 对沙箱返 403（`reshard-c4-data`）。

### 2.6 地板题口径

跨 5 个 run 解出率 <3%：`extract-moves-from-video`、`make-doom-for-mips`（已被 16Gi 解掉）、`prove-plus-comm`（已修）、`torch-pipeline-parallelism`。`failure_audit.py` 自动算"剔除地板题"一列。

---

## 3. 修了什么、影响是什么

| # | 改动 | 资产 | 影响 |
|---|---|---|---|
| 1 | **Reward runner**：评分前从随包 uv tarball 预装 `uv`/`uvx` 到 `~/.local/bin` 并预建 `env` 文件；`UV_DEFAULT_INDEX`/`UV_INDEX_URL`/`PIP_INDEX_URL` → aliyun 镜像；`UV_HTTP_TIMEOUT=600`、`UV_CONCURRENT_DOWNLOADS=4`、`UV_HTTP_RETRIES=6`；`HF_HUB_DISABLE_XET=1` + HF 超时放宽；WORKDIR 没有 `details.json` 时回退 `$ARENA_AGENT_WORKDIR` 再 `/app`。`pytorch-model-cli` 的 env-cache 分支逐字节保留 | `cth-tb2-reward-mirror@1.0.1`（源码 `coding_arena/build/reward_mirror/`，官方 4.0.18 runner 原文也在那里可 diff） | pytest 没跑就记 0：20/178 → 1/178 → 0（1.0.1 smoke 8/8 跑完 pytest）。**`test.sh` 语义不变，分数仍是官方口径**，只是不再受公网状况影响 |
| 2 | **Harness**：pkg_v5 新增 `TERMINUS_SANDBOX_ENV_<NAME>` 透传，把 pip/uv 镜像 + 超时 + `HF_HUB_DISABLE_XET` 注入 agent 的 tmux shell；`resource_spec.memory` 8→16Gi；其余（96k 触发、max_turns 512、usage 计数、effort max）与 1.2.96 相同 | `cth-terminus-2-stream@1.6.96`（1.5.96 是没有 uv/HF 项的前一版） | 模型装包不再 15 s 超时反复 retry，不再因网络放弃验证；编译重任务不再 OOM。只影响 agent shell，Reward 不受影响。显式 `--index-url` 仍优先（`pypi-server` 题不受干扰） |
| 3 | **Benchmark**：89 题 + reward-mirror | `cth-tb21-full89-mirror@1.0.1` | 评 API 模型用这个；旧 89 题分数**不可比**（低估 8–15 pt，且随网络漂移） |
| 4 | **归因脚本** `failure_audit.py`：每条 0 分轨迹分到唯一主因（reward 没跑 pytest / 4h / 512 步 / 静默中断 / OOM / 网络放弃 / 模型错），加地板题列；`pull_trajectories.sh` 顺带落盘 `tbench_test.log` | `coding_arena/code/arena/eval/` | 以后每个 run 自动看 `reward bootstrap fail` 列是否为 0，不为 0 先修 reward 再看模型 |
| 5 | 诊断用 noop harness + dump reward | `cth-terminus-noop2@1.0.1`、`cth-tb2-reward-dump@1.0.1`、`cth-tb2-reward-dump-bench@1.0.1`（`coding_arena/build/reward_dump/`） | 以后要看 reward 沙箱里的实际环境/网络/官方脚本，发一个 job 即可 |

**副作用 / 注意**：

- reward 阶段耗时变长（装 torch 的题 7–22 min，之前因为失败得快反而短），仍在 `ttl 7200 s` 内。
- `tb2-reward` 归 jjsong，这三处改动都不碰 `test.sh` 语义，建议合进 4.0.19；合并前 TB2.1 统一用 `cth-tb21-full89-mirror`。
- `arena job create --model-ref` 对 dsv41 的 key 手敲会 404（CLI 对 key 里字符的转义问题），从已有 `job.json` 里读出来放变量再传能过。
- 旧 job 的分数（SFT 消融 [../experiments/sft_results.md](../experiments/sft_results.md) §2 那批、早期 harness 对比表）都是旧口径，带 10–16% 的 reward 噪声；横向比 SFT 变体之间仍有效（同一噪声），但绝对值偏低。

### 3.1 修复前后（dsv41-flash max，2 样本，并发 32）

| job | harness | benchmark / reward | mean(fail=0) | pytest 没跑 | 剔地板题 |
|---|---|---|---|---|---|
| `cth-tb21-96-dsv41-max-2610260221`（baseline） | 1.2.96 | full89 / 官方 4.0.18 | 0.753 | 20/178 | 0.788 |
| `cth-tb21-88-dsv41-max-1596-2610301702` | 1.5.96 | full88 / 官方 4.0.18 | 0.511 | 71/176（astral.sh 整段不可达） | — |
| **`cth-tb21-89m-dsv41-max-1596-2610301918`** | 1.5.96 | **full89-mirror / cth-tb2-reward-mirror@1.0.0** | **0.831** | **1/178** | **0.847** |

0.753 → **0.831**（+7.8 pt），且 `acc given tests ran` 0.836 与 `mean(fail=0)` 0.831 几乎重合——"pytest 没跑就记 0"这类损耗已经消失。逐题变化（对 4 个 dsv41 baseline run 的均值）：

- 大涨：`prove-plus-comm` 0→1.0（workdir 修复）、`make-doom-for-mips` 0→1.0、`make-mips-interpreter` 0.12→1.0（16Gi 解掉编译 OOM）、`mailman` 0.12→1.0、`train-fasttext` 0.25→1.0、`sam-cell-seg` 0.38→1.0（reward 阶段装包不再超时 + agent 侧 pip 镜像）。
- 下降 ≥0.5 的只有 `cancel-async-tasks`（0.5→0，8 个 run 里本来就 50% 翻转）和 `sparql-university`（1.0→0.5），都是 2 样本噪声。

reward-mirror 1.0.1 的 smoke（`cth-tb21-rewardsmoke2-dsv41-2610310013`：mteb-retrieve / reshard-c4-data / torch-tensor-parallelism / pytorch-model-recovery ×2）**8/8 得分**，8 条 reward 全部跑完 pytest，`mteb-retrieve` 两条都拉完了 torch + nvidia wheels，`reshard-c4-data` 两条 `load_dataset` 都通过（xet 关掉后走普通 HTTP）。

### 3.2 没修、为什么

- **模型自宣完成但 test 不过（~22 pt，主体）**：这是模型 × Terminus scaffold 的行为。可以在题面加"先跑验收再 complete"的提示或改 `mark_task_complete` 的确认流程，但那是在改 agent，分数口径会偏离标准 Terminus。如果目标是对齐官方数字，应该用官方同款 scaffold + pass@k 口径跑对照，而不是魔改 Terminus。
- **TTL 4 h → 6 h**：预期 <1 pt，拖慢周转，不做。

---

## 4. 0.831 之后还剩什么（30 条 0 分，逐条看了中间步骤）

| 类别 | 条数 / 178 | 说明 |
|---|---|---|
| 模型答错（pytest 跑了、断言失败） | 26 | OCR 读错 flag、chess 少一解、mteb 查错榜单项、起跳帧 313 vs [219,223]、XSS 向量漏 3 组、`cancel-async-tasks` 没打印 "Cleaned up."……全部是"模型 `mark_task_complete` 但测试不过"：30 条失败轨迹里 test 类命令 0 条 |
| 撞 512 步上限 | 2 | `extract-moves-from-video` ×2 |
| reward 阶段网络 | 2 → 0 | 1.0.1 已修（§2.5 后两条） |

**中间步骤有没有系统原因导致偏离？逐项查过，结论是没有系统性因素**（10-31，`cth-tb21-89m`）：

| 检查项 | 失败 30 条 | 通过 148 条 | 结论 |
|---|---|---|---|
| agent shell 外网失败（curl/pip/DNS 对外部主机） | 0 | 10 条有信号，全部通过 | 1.6.96 镜像生效后 agent 侧网络干净 |
| LLM 网关卡顿（相邻步间隔 >600 s 且不是长生成/长命令） | 1（`make-doom` s0，935 s，该题**得分 1**） | — | 无影响 |
| 步间隔分布 p50 / p90 / p99 | 16 s / 59 s / 158 s | 同 | 正常 |
| 压缩重置（summary handoff） | 1 条 | 3 条 | 96k 触发点下 TB2.1 几乎不触发 |
| 模型回 JSON 解析失败 | 1 条 1 步 | 1 条 1 步 | 噪声 |
| 终端输出被 harness 截断（>10 KB 取首尾） | 37 步（1.8%） | 152 步（3.7%） | 通过组更多，不是失败原因 |
| "Current Terminal Screen" 回退（增量 diff 失败只给屏幕） | 38% 步 | 26% 步 | 失败任务本来就是长输出题；没有证据它导致误判 |
| 交互式提示吞掉队列命令（`unzip` 的 `replace?` 吃掉后续命令） | 1 条（`build-pov-ray` s0） | 0 | 这是 Terminus 一步多命令的已知坑，模型随后 `unzip -o` 覆盖了 POVDOC 同名文件，MD5 校验失败 |
| "Command N should end with newline" 警告 | 0.93/任务 | 0.26/任务 | 66 次全部是 `C-c` 特殊键引起的良性告警，不影响执行 |
| 峰值 prompt >90k | 5 | 8 | 不相关 |

典型"看起来像环境"的几条也核了：`configure-git-webserver/s0` 验收 404 是模型自己把仓库 reset 成空的（s1 通过）；`reshard-c4-data/s0` 是 test 阶段 HF xet 403（1.0.1 已修）；`mteb-retrieve/s1` 是 test 阶段 nvidia wheel 超时（1.0.1 已修）。剩下的全是解题本身。

与官方 92 的剩余差距：Terminus 盲打 JSON 单终端 vs 厂商自家 agent（多工具、自带验证循环）、pass@k vs 2 样本均值、模型 variant（DashScope 的 `deepseek-v4.1-flash-1017`）。这些不是环境问题。

---

## 5. TerminalHorizon 的 job（`cth-thfull14-dsv41-2610270211` 等）：系统问题逐类核查与修复

**reward 网络问题：没有。** TH 的 oracle 是 `python3 /tests/check.py`（400 题抽样 `test.sh` 里 0 处 `astral.sh`/`uvx`/`pip install`/`apt-get`，`curl` 只打 `127.0.0.1`）。200 条 `log_excerpt` 抽样 bootstrap 失败 0 条。

`cth-thfull14-dsv41-2610270211`（1486 env ×2 = 2972 任务，DONE 2690，mean(fail=0) 0.422）的 282 条非 DONE 全部拉了 trace + 完整 harness 日志，按根因分：

| # | 根因 | 条数 | 系统 / 模型 | 可修？ | 处置 |
|---|---|---|---|---|---|
| 1 | **87 个拍扁镜像（`-flat-*-rt2`）env：平台不跑镜像 ENTRYPOINT，`/opt/flatten/runtime.py` 从未启动 → sidecar 服务没起 → setup hook 等 `ready.json` 10 min 放弃 → agent 对着不存在的服务跑满 5 h** | **172**（flat 的 99%；三个 job、两模型、三 harness 都一样） | 系统 | **已修** | 新 setup hook `code/reg/th_flat_setup_hook.sh`：chroot 进 sidecar rootfs、在主 netns 启动其 argv、hostname → 127.0.0.1、写 ready.json。87 个 env 全部发了 **`@1.3.0`**，benchmark **`cth-th-full@1.1.0`**（其余 1399 题不变）、`cth-th-flat87@1.0.0`。pilot：`close-the-loop-on-a-serialized-warranty-repair` dsv41 10 min **得分 1**（之前 0/100+） |
| 2 | **沙箱 bootstrap 失败：ARCA 装 egress CA / envd 时 `curl: (60) SSL certificate problem`（镜像缺 CA 证书）→ envd 不监听 → 平台读不到日志、tmux fallback 起不来** | **36**（18 个 env ×2，同 env 两样本 100% 复现；13 个是我们 10-20 构的 rt1，5 个交接批） | 系统（镜像） | 可修 | 镜像重建加一层 `ca-certificates`（rt1 层本来该有，这 18 个漏了）；或 setup hook 里先 `update-ca-certificates`——但 hook 跑在 envd 之后，来不及。**短期：剔除**，清单 `coding_arena/work/manifests/th_ca_broken_envs.txt`；外部重建套件 `build/rt3_ca/` |
| 3 | **agent 中途 tmux server 被杀**（`tmux send-keys` 返回 `-15`/`-9`，之前一切正常） | 25 | 系统（资源） | **已修** | 全是起多个常驻服务的题（k3s、minio、ROS、wireguard、slurm…）；8Gi 内存下 OOM killer 杀掉 session。16Gi harness（1.6.96 / 1.7.96）重打 45 DONE / 4 solved（§5.1）；真正的解法是这类 env 的 `resource_hints` 生效（平台不认） |
| 4 | **reward 阶段卡死 2 h**（agent 正常 complete，`shared reward exec (timeout)`） | 11 | 系统（env） | **已修** | ROS 2 / robot / tokio / etcd 题，`check.py`/`colcon`/`cargo test` 等一个永不退出的进程，卡满 reward ttl 7200 s。oracle 的 `tests/test.sh` 换成 `timeout -k 60 1500 bash /tests/test_orig.sh`，env `@1.4.0`（§5.1） |
| 5 | agent 5 h 跑满、runner 正常 | 15 | 多为模型 | — | p90 agent 278 min 的长尾题 |
| 6 | ARCA 建沙箱超时 / StartContainer 失败 | 8 | 平台 | 重试 | `job retry --failed` |
| 7 | 沙箱内解析不了 `envarena-prod.alipay.com`（DNS） | 4（2 env） | 系统（env） | **已修** | 不是镜像 resolv 问题：**题目自己的 bootstrap 把 `/etc/resolv.conf` 写成 `nameserver 127.0.0.1`**（本地 dnsmasq / consul）。setup hook 先把网关 IP 钉进 `/etc/hosts`，env `@1.4.0`（§5.1） |
| 8 | 镜像缺 `libstdc++.so.6`，harness 的 tokenizers 导入失败 | 2（1 env） | 系统（镜像） | 可修 | 镜像加 `libstdc++6`，或 harness 的 token 计数改纯 python 回退 |
| 9 | 其它（EVAL_FAILED 3、runner 被 SIGTERM 1、杂 3） | 7 | — | — | — |

合计：**282 条非 DONE 里 ≥260 条是系统原因**。分数口径：

| job | 报告 | 剔除 87 flat | 再剔除 18 CA-broken env | 说明 |
|---|---|---|---|---|
| `cth-thfull14-dsv41` | 0.422 | 0.448 | **0.455** | flat 修好后预计回到 ≥0.45（flat 题本身解出率未知） |
| `cth-thfull96-dsv41` | 0.471 | 0.515 | ~0.52 | |
| `cth-thfull96-flash` | 0.480 | 0.522 | ~0.53 | |

其它非 DONE 来源：`HARNESS_FAILED` 2–4%（dsv41 1.2.96 的 70 条里 42 条 `packaged tmux lacks required environment support` / `error connecting to /tmp/tmux-0/default`，10 条 `[th-setup] envd listen: 1` 后无输出，4 条网关 5xx）；非 flat 的 TIMEOUT 1–2% 是真的 5 h 跑满。

### 5.0 拍扁镜像（flat / rt2）为什么全 TIMEOUT，怎么修（10-31）

诊断用 `cth-terminus-noop2@1.0.1` + `cth-tb2-reward-dump@1.0.2`，对 `cth-th-close-the-loop-on-a-serialized-warranty-repair` 发了 20 个 `1.2.x-diag` 版本逐步验证（job `cth-th-flat-diag{1..25}`）：

1. 沙箱 PID 1 是平台的 wrapper（`/bin/sh -lc … harness -t Target.md; while :; do sleep 5; done`），**镜像 ENTRYPOINT 从不执行**，所以 `/opt/flatten/runtime.py`（拍扁镜像的 supervisor，负责起 sidecar 并写 `ready.json`）从未运行。原 setup hook 等 10 min 后 `flatten ready: NO`，agent 阶段照常进入，模型对着不存在的 `http://api:8080` 跑到 5 h。
2. 直接在 hook 里跑 `runtime.py`：`network_runtime.prepare()` 写 `/proc/sys/net/ipv4/ip_forward` → `OSError 30 Read-only file system`（ARCA 下 `/proc/sys` 只读）。
3. 把那一行 patch 掉后，私网能建（`thbr0` + veth + `thsvc-N` netns），服务也在各自 netns 里 listen，但 **主 ns ↔ netns 的 `connect()` 两个方向都 `Operation not permitted`**（ARCA 网络策略，iptables 规则为空、rp_filter 正常），所以这条路走不通。
4. `runtime.start()` 用 `unshare --pid --fork` 起服务且 stdout 继承，setup hook 的管道永远不关闭 → 平台等 hook 等到超时（diag14/15 复现）。
5. **可行方案**（diag23/25 验证）：不建私网，`chroot` 到 `/opt/svc/<name>` 直接在主 netns 跑服务 argv（`subprocess.Popen(..., preexec_fn=chroot+setuid, stdout=file, close_fds=True)`），`/etc/hosts` 把服务名指到 `127.0.0.1`，自己写 `ready.json`。`api`/`diag` 都 200，hook 正常返回，task DONE。

落地：`code/reg/th_flat_setup_hook.sh`（spec 的 `hooks.setup_script`），87 个 flat env 全部发 `@1.3.0`（`work/state/1.3.0/`），benchmark `cth-th-full@1.1.0`、`cth-th-flat87@1.0.0`。端到端 pilot `cth-th-flat-pilot-dsv41-2610310826`（harness 1.6.96 + 官方 TH reward）：`close-the-loop` 10 min 得分 1；`acknowledgement-aware-auv`（authority 服务 200）agent 正常跑完 62 步，test 不过——是模型解题问题，不再是环境。

代价与限制：失去原 compose 的网络隔离（服务之间互相可见，题目若依赖"main 看不到 provider 根目录"之类的隔离语义会失真；`handoff_meta` 里标了 `exec_prefix: true` 的 4 题要求所有命令经 `main_exec.py`，本 hook 没做，这 4 题可能仍失败）；同端口服务会冲突（87 题的 plan.json 核过没有）。

### 5.1 11-01 续：逐条修复与坏数据重打（`cth-thredo14-dsv41-2611011220-2611010420`）

把 `cth-thfull14-dsv41-2610270211`（dsv41_history SFT 数据的来源 job）的 282 条非 DONE 全部按 task trace + harness log 重新归因到 env 粒度（工件 `/storage/openpsi/users/chen/output/cth-thfull14-dsv41-2610270211/audit/`），共 **160 个坏 env**（完整清单 `coding_arena/work/manifests/th14_bad_envs.tsv`：env / 根因 / 原版本 / 重打版本 / 修法）：

| 组 | env 数 | 根因 | 修法 | 资产 |
|---|---|---|---|---|
| flat | 87 | 平台不跑 ENTRYPOINT，sidecar 没起（§5 #1） | 10-31 已修（§5.0） | env `@1.3.0` |
| tmux-killed | 24 | agent 中途 tmux server 收 SIGTERM/SIGKILL（8Gi OOM，§5 #3） | 16Gi harness | **`cth-terminus-2-stream@1.7.96`** = 1.6.96（16Gi + agent 侧镜像）+ `TERMINUS_TOOL_CALL_MODE=function`（历史推理口径与 1.4.96 相同） |
| transient | 20 | ARCA 建沙箱失败 8 / agent 真 5h 跑满 12 | 直接重试 | 原版本 |
| reward-hang | 8 | `check.py`/`colcon`/`cargo` 在 reward 阶段等一个不退出的进程，卡满 7200 s ARCA exec 超时（全是 ROS2/robot/tokio/etcd 题，§5 #4） | `tests/test.sh` 换成 `timeout -k 60 1500 bash /tests/test_orig.sh` 包装（原脚本逐字节保留为 `test_orig.sh`） | env `@1.4.0`（`code/reg/register_140_fixes.sh`） |
| DNS | 2 | **题目自己的 bootstrap 把 `/etc/resolv.conf` 改成 `nameserver 127.0.0.1`**（本地 dnsmasq/consul），harness 之后解析不了 `envarena-prod.alipay.com`，LLM 调用全部 Connection error（§5 #7 的真根因，不是镜像问题） | setup hook 先把网关 IP 钉进 `/etc/hosts`（getent 失败回退 python socket） | env `@1.4.0` |
| **EXCLUDED** | 18+1 | 18 个镜像缺 CA 证书（§5 #2，集群内修不了）+ 1 个 oracle 83 MB 超 shared-sandbox 投递上限（`recovering-a-partially-written-audiovisual-container`，EVAL_FAILED "grading assets too large"） | CA：外部 docker 机用 `build/rt3_ca/`（Dockerfile.rt3 + build_rt3.sh + 18 镜像清单）重建后注册 `@1.4.0`；oracle：瘦身后再补 | `work/manifests/th14_excluded_envs.txt` |

对 SFT 数据本身的核对：`dsv41_history` 只收 DONE 轨迹，上述坏 env 的轨迹**本来就不在数据里**（160 个坏 env 在 manifest 里只出现 1 条：`resilient-event-time-join-migration` s1 score 0，flat 题的废轨迹，建议剔除）；修复重打的意义是把这批题的轨迹**补进来**。

重打：benchmark **`cth-th-redo14@1.0.0`**（141 env = 160 − 19 excluded；flat 用 1.3.0、reward-hang/DNS 用 1.4.0、其余原版本），job **`cth-thredo14-dsv41-2611011220-2611010420`**，与原 job 相同配置（`dsv41-flash-1017-max-chen@1.0.0`、2 samples、retries 1、tool-call 模式历史推理可见），harness 1.4.96 → 1.7.96（仅 16Gi + agent 侧镜像注入，数据语义不变）。终态：282 条里 DONE 182（mean 0.346）、TIMEOUT 67、HARNESS_FAILED 33；按修法分 flat 117 DONE / 58 solved、reward-hang+DNS 20 / 1、tmux-killed 45 / 4。后续的 DONE 轨迹污染审计、合并与 judge 见 [trajectory_collection](trajectory_collection.md) Part C。

### 5.2 沙箱内端口扫描审计（11-06；起因：`faas-swe:tb2-protein-assembly_nydus_v2` 扫物理机 IP 全端口）

**扫描是 agent（被测模型）发起的，不是镜像自带行为。** `protein-assembly` 上游 Dockerfile 只有 `FROM t-bench/python-3-13` + 3 个 `COPY`（fasta / genbank / pdb_ids），无 ENTRYPOINT、无任何网络代码。

因果链（`cth-tb21-stream-abl-mask-rand-final-2610241346` s1 逐步核对）：

1. 题目要求调 **pdb API 和 fpbase API**；沙箱里 `api.fpbase.org` DNS 解析失败、`files.rcsb.org` 返回 404。
2. 模型推断"应该有本地 mock 服务"，`cat /etc/hosts` 找线索。
3. **沙箱的 `/etc/hosts` 里有平台注入的宿主机条目**：`11.167.5.197 iZuf67…`、`11.167.5.147 iZuf6d…`、`33.212.94.37 appxdn033212094037.wa181`。
4. 模型把这些当成候选服务地址，用 python `socket.connect_ex` 对它们扫 1–1024、20000–31000 乃至 1–65535。

全量审计（`code/arena/eval/audit_port_scans.py`，只看**实际执行**的命令；277 个 job / 41,132 条轨迹）：对外（非 loopback）的大范围扫描 **92 条命令 / 50 条轨迹 / 14 个 env**：

| env | 轨迹 | 命令 | 最大端口数 | 目标 | 诱因 |
|---|---|---|---|---|---|
| `tb2-protein-assembly` | 21 | 45 | 65535 | 11.167.5.{147,197}、33.212.9x.x（宿主机）、rcsb 公网 IP | 外部 API 不通 |
| `tb2-compile-compcert` | 5 | 8 | 65535 | 11.167.5.x、33.212.93.x，还扫了 11.167.5.1/.2 | GitHub 源码 404，找"内部源" |
| `cth-th-support-case-family-reconciliation…` | 5 | 9 | 65536 | 100.100.138.x、100.100.2.136/138（阿里云内网 DNS）、172.25.x | sidecar `reviewd` 不存在（flat 题） |
| `cth-th-operate-an-adaptive-ecological-restoration…` | 6 | 9 | 65536 | 10.1.1.0/24 整段、100.100.138.1、169.254.169.254（metadata） | sidecar gateway 不存在（flat 题） |
| `cth-th-make-recurring-caldav-scheduling…` | 1 | 7 | 65536 | 100.100.138.{1,17,33} | sidecar `caldav-peer` 不存在（flat 题） |
| 其余 9 个 env | 各 1–2 | 各 1–2 | ≤20001 | 多为题目自建私网（10.x） | 题目语义内，基本无害 |

共性：**题目依赖的服务不可达 + `/etc/hosts` 暴露宿主机 IP** → 模型自己去"找服务"。与模型无关（dsv41、flash、27b、所有 SFT checkpoint 都出现）。3 个 TH env 正是 flat 题（sidecar 没起），`@1.3.0` 修复后诱因已消失。

工件：`/storage/openpsi/users/chen/output/audit_port_scans/`（`port_scans_external.json` 含每条命令原文）。SFT 数据来源 job（`cth-thfull14` / `cth-thredo14`）里对外大范围扫描为 **0 条**（`work/manifests/sft_port_scan_traj.json` 为空；那 63 条命中都是 loopback 自检或题目内私网），dsv41_merged 不受影响。

---

## 6. 复现 / 工具

```bash
# 评一个 API 模型（默认 harness 1.6.96）
bash coding_arena/code/arena/eval/submit.sh --model <key@ver> --benchmark cth-tb21-full89-mirror@1.0.1 --samples 2 --parallel 32
# 归因（job 跑完、puller 落盘后）
python3 coding_arena/code/arena/eval/failure_audit.py <job_id> [<baseline_job> ...] --floor-jobs <5 个 baseline> --markdown out.md
# 看 reward 沙箱里到底发生了什么
arena task download <task_id> tbench_test.log -o x.log     # TB2.1；TH 用 arena task result <task_id> 的 log_excerpt
```

关键 job：
- baseline：`cth-tb21-96-dsv41-max-2610260221`（0.753）、`cth-tb21x3-stream-flash-2610190249`（0.770）
- 修复后：`cth-tb21-89m-dsv41-max-1596-2610301918`（**0.831**，harness 1.5.96 + reward-mirror 1.0.0）
- smoke：`cth-tb21-pipsmoke-dsv41-1596-2610301615`、`cth-tb21-rewardsmoke-dsv41-2610301901`、`cth-tb21-rewardsmoke2-dsv41-2610310013`（8/8）
- 反例：`cth-tb21-88-dsv41-max-1596-2610301702`（harness 修了、reward 没修、astral.sh 当天整段不可达 → 0.511）

---

## 7. SFT 消融评测的 verifier 审计（11-06）

**一句话**：§3 的 reward 修复（`cth-tb21-full89-mirror@1.0.1`）10-31 就做好了，但 4B SFT 消融那一批评测
（`abl_mask_const_*`、`tree_th3k`、`tree_flash`、`dsvh_*`、`lin_*`、`qf_*`，以及 11-05 的 arm A 首轮）**一直还在用旧 benchmark**
`tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2`。所以"verifier 挂"在这批 job 里照旧发生。本节量化它、把评测切到已修口径、并记录没覆盖到的项。

### 7.1 量化：旧口径下每个 SFT job 有多少 task 的 pytest 根本没启动

工具 `coding_eval/tools/verifier_audit.py`。判据：`tbench_test.log` 全文出现 `test session starts` = reward 引导成功；否则按日志末尾匹配原因。
只统计 puller 落了 `tbench_test.log` 的 DONE task（mask c312 只补拉了 110 条失败 task 的 log，所以分母是 110 而不是 172）。

| job（旧口径） | 有 log 的 DONE | pytest 启动 | **没启动** | 原因 |
|---|---|---|---|---|
| mask_const c312（th3k 最佳，0.348） | 110（仅失败 task） | 93 | **17** | pypi/uv 拉包超时 15、`No working directory set` 2 |
| tree_th3k c156 | 164 | 144 | **20**（12.2%） | 拉包超时 18、no-workdir 2 |
| dsvh_const c126 | 162 | 146 | **16**（9.9%） | 拉包超时 14、no-workdir 2 |
| tree_flash c168 | 164 | 152 | **12**（7.3%） | 拉包超时 10、no-workdir 2 |
| dsvh_solved c152 | 173 | 149 | **24**（13.9%） | 拉包超时 14、uv installer 不可达 8、no-workdir 2 |
| lin_solved c51 | 171 | 155 | **16**（9.4%） | 拉包超时 14、no-workdir 2 |
| 参照：dsv41-max @ `full89-mirror@1.0.0` + 1.5.96（0.831） | 178 | 177 | **1**（0.6%） | uv 解析 nvidia wheel 超时（mteb-retrieve；1.0.1 已把 uv 超时 120→600 s） |

比例 7–14% 与 §2.1 对 API 模型的统计（10–16%）一致，是同一个 reward 问题。`tb2-prove-plus-comm` 在全部旧口径 run 里 0/188，在 mirror 口径 2/2。
这些 task 的分数是 0，但没有任何信息说明模型做对还是做错；**每个 job 实际少了多少分，要重测才知道，这里不折算**。

### 7.2 处置：评测切到已修口径（11-06 00:00 起）

没有新建 reward 版本——1.0.1 已覆盖上表全部三类原因。动作只有一个：**提交时把 `--benchmark` 换成 `cth-tb21-full89-mirror@1.0.1`**。
harness 保持各行原来的版本（这批是 `cth-terminus-2-stream@1.0.0`），这样新旧分数之间只差 reward 一个变量。

| 重测 job | 模型 | 说明 |
|---|---|---|
| `cth-tb21m-mask-const-c312-2611051616` | th3k 基线 mask_const ep8 | 基线必须在同一口径下重测，否则新模型的数字没有可比对象 |
| `cth-tb21m-dsv41j-a-c64-2611051554` / `-c96-` / `-c224-` | arm A（dsv41 chat-mode 整轨迹）ep2 / ep3 / ep7 | 旧口径那三个 job 被抢占打掉 113–133 条，直接弃用，不 retry |

中途核对（00:50）：四个 job 已完成的 28 / 28 / 54 / 48 个 task **pytest 没启动的为 0**；其中旧口径下常挂的重依赖 env
（mailman、mteb-retrieve、prove-plus-comm、sam-cell-seg、torch-tensor-parallelism、query-optimize）已完成的 7 个全部启动成功。
分数等四个 job 跑完再报；中途的通过率只覆盖先跑完的快题，不能当分数用。

**规范**：此后 TB2.1 评测一律 `cth-tb21-full89-mirror@1.0.1`；每个 job 结束先跑 `verifier_audit.py`，"没启动"不为 0 就先查 reward 再看模型。
旧口径的数字（[sft_results](../experiments/sft_results.md) 里 11-05 以前的全部行）只能在旧口径内部互相比较，不能和 mirror 口径的数字放在一张表里比大小。

### 7.3 1.0.1 没覆盖到的：逐项核对结果

| 项 | 现象 | 核对 | 结论 |
|---|---|---|---|
| **agent 侧**网络 / 内存 | 旧口径 SFT job 的失败轨迹里，终端输出含 pip 连接重试的 13–29 条/job，模型文字里出现"网络不行，跳过运行验证"的 8–28 条/job，含 `Killed`/`MemoryError` 的 13–25 条/job（`failure_audit.py` 的 tag 计数） | 这是 harness 层的事（`1.0.0` = 8Gi、agent shell 无镜像），reward 修复管不到。参照 job（harness 1.5.96 = agent 侧镜像 + 16Gi）里 pip 重试 0 条、"跳过验证" 2 条 | **未修**。`1.0.0` 和 `1.6.96` 同时差了窗口（262k vs 96k）、步数上限、计数方式，直接换会混入别的变量。要单独修需要一个"1.0.0 窗口语义 + pkg_v5 运行时"的 harness 版本——见 §7.4，需确认后再用 |
| `tb2-filter-js-from-html` | 全部 run 3/203 通过，dsv41-max 也 0/2；日志里大量 chromedriver `Read timed out` / `Connection refused` | 怀疑"驱动失败被记成 XSS 没拦住"。**核对结果是否定的**：51 份带驱动失败的 log 里 50 份满足 `failed_vectors == batches − load_failures − no_alert`，失败向量恰好是"检测到 alert"的 batch，驱动失败的 batch 被跳过 | **不是 verifier 问题**，是题目难（要同时拦住全部 XSS 向量且干净 HTML 逐字节不变）。不改。按地板题对待 |
| `tb2-install-windows-3.11` | 全部 run 6/153 通过；eval 阶段 p50 约 21 min；断言是"某个按键让截图变化 ≥10%" | 只看了断言文本和通过率，没有做能区分"环境抖动"与"模型没装好"的实验 | **未定**。先按地板题对待，不动 |
| `tb2-portfolio-optimization` | 3 条 log 有 pytest session 但没有 summary：`Fatal Python error: Segmentation fault`，崩在模型自己编的 C 扩展里 | 同 env 在参照 job 2/2 通过 | 模型产物问题，不是 verifier。顺带修了审计脚本的判据（旧判据把这种情况误算成"没启动"） |

### 7.4 本次操作中需要知情的两件事

1. **发布了一个新 harness 版本 `cth-terminus-2-stream@1.6.262`**（= 1.6.96 的运行时 + 1.0.0 的窗口语义：262144 / 65536 / 无 max_turns /
   `TOKEN_COUNT_MODE=estimate`），本意是修 §7.3 第一行。发布时带了 `--supersedes 1.6.96`，副作用是该 harness 实体的 `latest_version` 变成了
   1.6.262。1.6.96 仍是 PUBLISHED，`submit.sh` 写死 1.6.96，所以显式引用不受影响；任何按"latest"解析的调用会拿到 262k 变体。
   已发布版本不可改，是否 deprecate 1.6.262 需要决定。**它没有产生任何被采信的分数**（试跑 8 个 task 后停掉，结果作废）。
2. **本会话此前所有评测的并发守护 `autopar.sh` 参数传错了**（第二个参数应是引擎日志 glob，被传成了并发下限），它读不到引擎指标，
   只会把并发一路升到 96。引擎日志实测：mask c312 / arm A 的 serve 有 5–10% 的 decode 行存在排队（最大队列 21–40，c64/c224 各 2 次 retract），
   dsvh / tree_flash / lin 系 ≤0.3%。11-06 01:03 已按正确参数重启，随即从 96 降到 72。之前那些 job 的 TTL 数可能因排队偏高，未量化。

---

## 8. 本地 TB2.1 评测（`tb_local`）为什么比 arena 低：八处沙箱/镜像偏差（11-09）

**背景**：`watch.sh` 盯的是 `/storage/openpsi/users/chen/tb_local/progress.sh`，即在 slurm 池节点上用 singularity 复刻 arena 的 TB2.1 评测
（89 题 ×2；harness `1.6.96` 同一份 `app.tar.gz`、reward 同一份 `cth-tb2-reward-mirror@1.0.1` 脚本、同样的 4 核 / 16Gi / 14400 s）。
构建与运行流程见 [tb21_local_pipeline.md](tb21_local_pipeline.md)。三个 run：`dsv41-local`（DashScope `deepseek-v4.1-flash-1017`）、
`dsvj4b-local`、`dsvj9b-local`（SFT 学生模型）。参照 = arena job `cth-tb21-89m-dsv41-max-1596-2610301918`（0.831，§3.1）。

**现象**：dsv41-local 跑到 155/178 时 121 通过（0.78）。只取本地已完成的题与 arena 同题对比：**本地 122/156 = 0.782 vs arena 132/158 = 0.835**。
按题看，本地明显更差的 13 题里有 5 题是 arena 2/2、本地 0/2（`compile-compcert`、`feal-differential-cryptanalysis`、`adaptive-rejection-sampler`、
`model-extraction-relu-logits`、`polyglot-c-py`），`pytorch-model-cli` 本地 score=null ×2；同时本地"更好"的 9 题里 `extract-moves-from-video`
（全平台 3/107 的地板题）本地 2/2。两个方向都不对劲，逐条看 `reward_result.json` 的 `log_excerpt`、`trajectory.jsonl` 和 arena 同题的
`tbench_test.log`，找到六处契约偏差（当晚复查又补了两处，#7 #8）——**全部是沙箱/镜像搭建问题，不是模型**。arena 的 0.831 不受这些问题影响（它跑在平台 docker 沙箱里）。

### 8.1 八处偏差（按分数影响排序）

| # | 偏差（本地 vs arena/docker） | 证据 | 对本地分数 | 修法（`run_task.sh` v2 / `reward/`） |
|---|---|---|---|---|
| 1 | **oracle 提前可见**：`/envarena/oracle/tests` 从容器启动就以 ro 挂进去；arena 的 agent 阶段只有 `bin env harness out workspace`（参照 job 178 条轨迹 0 条读到 oracle） | 15 条轨迹 `sed /envarena/oracle/tests/test_outputs.py`，14 条得分：`extract-moves-from-video` 用 `ast` 抠出测试里的 SOLUTION 写进 `solution.txt`（arena 0/2）；`torch-pipeline-parallelism` 直接 `pytest /envarena/oracle/tests`（arena 0/2）。学生模型更甚：4b 22 条、9b 24 条 | **虚高** ~4 条（其余 ~10 条 arena 也正常解出） | agent 阶段不挂 oracle；harness 返回后 inside.sh 写 `.agent_done`，宿主把 `tests/` 拷进桥目录并打 `.ready`，inside.sh 再拷到可写的 `/envarena/oracle` 跑 reward |
| 2 | **`/tests` 只读**：reward runner 把 `/tests` 软链到 ro 挂载的 oracle；arena 里可写 | `compile-compcert`：`ld: cannot open output file /tests/positive_probe: Read-only file system`（arena 同题 log 0 处 RO）；`feal-differential-cryptanalysis`：`setup.py build_ext --inplace` 在 `/tests` 里 `could not create build/...: Read-only file system`。pytest 的 `.pytest_cache` 警告同源 | **压低** 4 条 | 同 #1 |
| 3 | **`/tmp` 是独立 tmpfs**：docker 里 `/tmp` 在 rootfs 上，apptainer 默认 `mount tmp = yes` | `path-tracing-reverse` 测试 `os.rename("mystery", "/tmp/mystery")` → `OSError: [Errno 18] Invalid cross-device link`，`test_image_similarity` 连带失败（0/2 vs 2/2） | **压低** 2 条 | `--no-mount tmp`；探针确认 `/ /tmp /var/tmp /app` 同一 st_dev |
| 4 | **apptainer 注入 `PROMPT_COMMAND=PS1="Apptainer> "; unset PROMPT_COMMAND`**，hostname 是池节点名 | `headless-terminal/s0`：模型用 `--rcfile` 设 PS1 哨兵再 `expect_exact`，PROMPT_COMMAND 在第一个 prompt 前覆盖 PS1 → pexpect 30 s 超时，6/7 测试挂（arena 2/2）。所有轨迹的提示符是 `Apptainer> `，模型看不到 cwd | **压低** 1 条 + 全部轨迹观测与 arena 不同 | inside.sh 开头 `unset PROMPT_COMMAND`；`--uts --hostname sandbox-<hash>`。修后首步即 `root@sandbox-6823d705c463:/app#` |
| 5 | **`pytorch-model-cli` 没有 gepa 验证缓存**：SIF 按 `6.4.0` Dockerfile 构建，reward 的 env-cache 分支 `exit 96` | `reward.log`: `missing PyTorch verifier cache in env image`，score=null ×2（arena 1/2） | **压低** 1–2 条，进度表记 0 | 无缓存时退回官方 `test.sh`（torch 从 `download.pytorch.org`，可达），预置 torchvision 要从 `ossci-datasets.s3.amazonaws.com`（不可达）拉的 4 个 MNIST raw 文件（`reward/mnist/`） |
| 6 | **python-build-standalone 的 sysconfig 写死 `CC=clang`**：本地 `python:*-slim` 基础镜像用 PBS，官方镜像是 gcc 编的 | 修完 #2 之后 feal 仍挂：`error: command 'clang' failed: No such file or directory`（arena 同处是 `gcc -fno-strict-overflow ...`）。agent 侧也中招：`portfolio-optimization`、`feal-linear-cryptanalysis`、`make-doom-for-mips` 轨迹里出现同一错误 | 与 #2 叠加；学生模型 3 条 0 分轨迹含此错 | inside.sh 开头把 `/usr/local/lib/python3.*/_sysconfigdata__*.py` 里的 `clang -pthread`/`clang++ -pthread` 改成 `gcc`/`g++` |

| 7 | **镜像构建的 `COPY <file> /app` 翻译错**（11-09 20:00 补充发现）：`build_task.py` 对"单文件 → 已存在目录"的 COPY 走了 `cp -a staged /app`，staged 文件名带 `0_` 前缀，结果 `/app/0_interp.py`、`/app/0_trunc.db` 而不是 `/app/interp.py`、`/app/trunc.db` | `schemelike-metacircular-eval` 三个 run 6/6 全 0（arena dsv41 2/2）：测试 `shutil.copy2("interp.py", …)` 找不到文件，63/63 用例 `No such file`；模型首步 `ls` 就看到 `0_interp.py`。`sqlite-db-truncate` 题面说 `/app/trunc.db`，实际 `/app/0_trunc.db`（dsv41 靠 `ls` 自己找到了，2/2；学生模型 1/4） | **压低** dsv41 2 条、学生模型 ~5 条 | `build_task.py` 的 `_stage_copy` 单文件分支改成 `if [ -d dst ]; then cp src dst/<原名>`；重建这两个 sif（扫过全部 89 个 build log，只有这两题命中） |
| 8 | **`/tests` 只读的第二个受害者**：`train-fasttext` 的 `test.sh` 要 `tar -xzf /tests/private_test.tar.gz -C /tests` | `tar: private_test.txt: Cannot open: Read-only file system` → `FileNotFoundError: /tests/private_test.txt`（dsv41 s1；arena 2/2）。11-09 15:00 第一轮作废时按 `Read-only` 正则扫 `log_excerpt` 漏掉了它——tar 的报错在 3500 字节截断之前 | 压低 1 条 | 与 #2 同一修法；作废重跑 |

净效应：本地 122 − 4（抄答案的虚高）+ 8（#2–#5 的压低）+ 3（#7–#8）≈ 129/156 ≈ 0.83，与 arena 0.835 在 2 样本噪声（SE≈0.03）内。

核过、**不是**偏差的：egress 代理的 DENY 全是 Chrome 遥测 / 题目自建的 `very.evil.com` / 不存在的 `ftp.oldlinux.org`；0 条 OOM、0 条撞 TTL；
`cancel-async-tasks`、`polyglot-*`、`adaptive-rejection-sampler`（R 包 `lower must be a single numeric value`）、`model-extraction-relu-logits`、
`chess-best-move`、`build-pov-ray`（`unzip` 覆盖 `colors.inc` MD5）、`mteb-retrieve`（答成 HumanEval）都是 pytest 真跑了、断言失败，属于 §4 那类模型误差，在噪声内。

### 8.2 修复与验证

- `run_task.sh`（旧版留在 `run_task.sh.bak-20261109-v1`）：`--no-mount tmp --uts --hostname`，`unset PROMPT_COMMAND`，sysconfig clang→gcc，
  oracle 在 agent 阶段之后才投递；`result.json` 多一个 `"contract": "v2-oracle-after-agent"` 字段区分新旧口径。
- `reward/tbench_reward_runner_mirror.sh`（旧版 `.bak-20261109`）：env-cache 缺失时的 MNIST 预置分支；arena 上有缓存的路径不变。
- 冒烟（dsv41，`runs/fixcheck/`）：`headless-terminal` 1、`feal` 1（`gcc` 编译 `feal_in_c` 成功）、`compile-compcert` 1、
  `path-tracing-reverse` 0（这次是模型没写出 `/app/mystery.c`，无 EXDEV）、`pytorch-model-cli` 0（`test_cli_tool_output` 断言失败——和 arena s1 挂同一个测试；reward 跑完 1665 s，不再 exit 96）。
  五条轨迹 0 次读 `/envarena/oracle`，0 个 `Apptainer>` 提示符。
- 20:15 复查（dsv41 175/178，0.789 vs arena 同样本 0.829）：剩余 13 题"本地更差"里逐条看 pytest 输出，发现 #7 #8；`build_task.py` 已修、两个 sif 已重建（`ls /app` → `interp.py`/`trunc.db`），
  三个 run 的 `schemelike-metacircular-eval` ×2、`sqlite-db-truncate` ×2 和 dsv41 `train-fasttext s1` 共 13 条再作废（`list.tsv` 原因 `copy-bug-staged-filename` / `ro-tests-tar-private-test`），第三轮 sweep 已起。
  其余"更差"题（`adaptive-rejection-sampler`、`polyglot-*`、`model-extraction-relu-logits`、`sanitize-git-repo`、`mcmc-sampling-stan s1`、`gpt2-codegolf s1`、`protein-assembly s0`、`torch-pipeline-parallelism s0`）pytest 全部真跑、断言失败在模型产物上
  （R 包参数校验、多余的编译产物、`SHA … missing`、Rscript 没输出采样日志、生成 "IS IS IS"、gBlock 不以 flag 开头、heredoc 卡死 171 步没写出文件），与 arena 同题 50% 翻转的噪声一致。
- 三个 run 里受影响的 **87 + 2 条样本**（含 2 条因热替换 reward 脚本撞上 `Stale file handle` 的 `sam-cell-seg`）移到
  `runs/invalidated-20261109/<run>/<task>/<s>/`（每条有 `INVALIDATED` 原因，清单 `list.tsv`），对应 `.lock` 删除，在池 job 992424 上用
  独立核段（128–151 / 152–171 / 172–191）起了三个补跑 batch。dsv41-local 重跑 23 条、4b 32 条、9b 32 条。

### 8.3 补跑结果（11-09 17:30，dsv41-local 170/178）

新契约下已完成的 18 条：16 通过。逐条与 arena 同样本对比：

| 题 | 本地（v2） | arena | 说明 |
|---|---|---|---|
| compile-compcert | 2/2 | 2/2 | #2 修好 |
| feal-differential-cryptanalysis | 2/2 | 2/2 | #2 + #6 修好 |
| path-tracing-reverse | 2/2 | 2/2 | #3 修好 |
| headless-terminal s0 | 1 | 1 | #4 修好 |
| db-wal-recovery / password-recovery / mteb-leaderboard / extract-elf | 2/2, 1/1, 2/2, 2/2 | 2/2, 1/1, 1/2, 1/2 | 不读 oracle 也能解，差异在噪声内 |
| pytorch-model-cli | 0/2 | 1/2 | reward 正常跑完，模型断言失败（arena s1 同一断言） |
| extract-moves-from-video | 2/2 | 0/2 | **仍然 2/2，但换了路子**：两条都 `git clone --depth 1 github.com/harbor-framework/terminal-bench-1`，从 `original-tasks/<task>/tests/test_outputs.py` 抠 SOLUTION。沙箱契约没问题（oracle 不在容器里）；差异是本地出口到 github 比 arena 稳。见 [tb21_local_pipeline.md §3.5](tb21_local_pipeline.md) |

同样本口径：本地 137/170 = 0.806 vs arena 142/170 = 0.835；去掉 `extract-moves-from-video` 这 2 条"公网拿答案"就是 0.794 vs 0.823，差 3 pt，
剩下的 `adaptive-rejection-sampler`、`model-extraction-relu-logits`、`polyglot-c-py`、`cancel-async-tasks` 等 0/2-vs-2/2 的题在 arena 其它 dsv41 run 里也是 50% 翻转的（§3.1），
属于 2 样本噪声。**结论：沙箱修完后本地分数与 arena 对得上；本地评测口径可用。**

### 8.4 怎么读 tb_local 的分数

- `result.json` 没有 `contract` 字段的是旧口径；三个 run 的 `summary.tsv` 要等补跑完成（`progress.sh` 的 in-flight 归 0）才能和 arena 比。
- 看一个 run 可不可信：`grep -l "/envarena/oracle" runs/<run>/*/s*/out/trajectory.jsonl` 必须为空；`egress.log` 里出现 `github.com` 且题是地板题时人工看一眼轨迹。
- 新增硬规则：**本地沙箱必须复刻 docker 的文件系统语义**（rootfs 上的 `/tmp`、可写 `/tests`、无 PROMPT_COMMAND、gcc 工具链），
  任何"agent 读到 `/envarena/oracle`"的轨迹直接作废。
- 若补跑结束后仍系统性低于 arena 3 pt 以上，下一个该查的是 DashScope 与 arena 网关的 variant / 限流差异，不再是沙箱。

---

## 9. 4B mask c312 本地两臂 vs arena 0.410：差在哪（11-11 终值）

参照 arena job `cth-tb21m-mask-const-c312-2611051616`：**73/178 = 0.410**，harness `1.0.0`（262k 窗口、无 max_turns、本地 estimate 计数、8Gi），reward mirror 1.0.1；178 条里 8 条 HARNESS_FAILED、41 条撞 4 h TTL（全 0 分）。

| 本地臂 | harness | done | pass | acc | 配对净差（arena 赢 − 本地赢） |
|---|---|---|---|---|---|
| 96k compaction | 1.6.96（98k 触发、512 步上限、usage 计数、16Gi） | 178 | 65 | **0.365** | 26 − 18 = **8 条** |
| 不 compact | 1.6.262（262k、无上限、estimate、16Gi） | 177 | 66 | **0.373** | 21 − 14 = **7 条** |

两臂都比 arena 低 4–4.5 pt，各 7–8 条样本。逐条看了 arena 赢、本地输的全部样本（26 + 21 条）和反向的（18 + 14 条）。

### 9.1 96k 臂：差距是 512 步上限

arena 赢本地输的 26 条里 **8 条是本地撞 512 步上限**（`qemu-startup` ×2、`rstan-to-pystan`、`adaptive-rejection-sampler`、`train-fasttext`、`git-multibranch`、`compile-compcert`、`build-pov-ray`）。arena 同样本 57–194 步就解出；本地 96k 窗口触发 1–35 次 compaction（52 条 0 次 / 17 条 1 次 / **109 条 ≥2 次**），每次 compaction 之后模型重新探索，步数被稀释。arena 上 1.0.0 没有 512 上限（13 条跑过 512 步，0 条解出）。
净差 8 条 = 恰好这 8 条。其余 18 条 arena 赢 vs 18 条本地赢是对称的翻转（`sanitize-git-repo` 本地 2/2 arena 0/2、`build-cython-ext` 反之）。

- 按 compaction 次数分层：0 次 本地 0.71 vs arena 同样本 0.73；1 次 0.65 vs 0.35；**≥2 次 0.16 vs 0.27**。差距全在被多次压缩的 109 条里。
- 结论：**96k + 512 步对 4B 这个模型是错的 harness**。它每步看自己全部历史推理，上下文增长快（[harness_comparison](harness_comparison.md) §A3），96k 触发点让它在 60% 的题上反复压缩；512 步上限再把本可以解出的长题砍掉。这是和 arena 1.0.0 的口径差，不是沙箱问题。

### 9.2 262k 臂：7 条差距 = 1 条 harness 错 + 3 条 TTL + 噪声

arena 赢本地输的 21 条：`mailman s0` 本地 harness ERROR（1152 步后 `failed to send non-blocking keys`，tmux 被塞爆，arena 341 步解出）；3 条本地 TTL（`caffe-cifar-10 s1` 1002 步、`adaptive-rejection-sampler s0` 750 步、`build-pov-ray s1` 570 步——arena 同样本 147–386 步解出；都是模型在 262k 里兜圈，4 核跑的，不是 CPU 问题）；17 条 pytest 真跑了、断言失败（`qemu-startup` ×2 Alpine 没起来、`polyglot-c-py` 目录多了 `cmain`、`train-fasttext` 0.6179 < 0.62、`sqlite-db-truncate` 没写 `recover.json`……）。
本地赢 arena 输的 14 条：4 条 arena TTL（`fix-ocaml-gc`、`schemelike`、`llm-inference-batching-scheduler`、`feal-linear`），10 条 arena 模型错。

- 两边 compaction 分布**几乎一样**（arena 85/28/65 vs 本地 91/21/65 条 0/1/≥2 次；首次压缩时 prompt 均 ~200k），说明 1.6.262 的窗口语义确实复刻了 1.0.0。
- 分层：0 次 本地 0.55 vs arena 0.62；1 次 0.29 vs 0.43；≥2 次 0.15 vs 0.14。差距在未压缩和压缩 1 次的题上，各差 3–4 条——**这就是 2 样本噪声的量级**（单 run SE 3 pt）。
- 去掉两边任一方 TTL / HARNESS_FAILED 的"干净子集"123 条：本地 62 vs arena 70；按题（任一样本解出）38 vs 45 / 67 题。方向一致但 McNemar 不显著。

### 9.3 排除了什么

- 题面、系统提示、提示符与 arena 逐字节一致（177 对里唯一 diff 是 `root@sandbox-<id>`）。
- 每步生成量相同：completion tokens/步 p50 2005 vs 1989，observation 字符/步 1748 vs 1739，到 180k prompt 的步数 43.5 vs 48.5。模型行为分布没有偏移。
- 服务更快不更慢：步间隔 p50 arena 21 s vs 本地 9.5–11 s（本地 2 节点 16 卡专用 vs arena 共享网关）。
- 2 核旧结果：262k 臂保留的 68 条 2 核结果通过 37（0.54），109 条 4 核通过 29（0.27）——因为 2 核那批是最先跑完的短题，4 核那批含全部 28 条 TTL 长题；不是核数造成的分差。
- 沙箱 14 项修复后 dsv41 对 arena 已对齐（§8），这两臂用的是同一套 run_task.sh。

### 9.4 结论（**已被 §10 更正，勿引用这一节**）

1. **要和 0.410 比，本地必须用 1.6.262**（= 1.0.0 窗口语义）。96k 臂低 4.5 pt 里 8 条全部可归因到 512 步上限 + 96k 反复压缩，是 harness 口径差。
2. 262k 臂的 0.373 vs 0.410：1 条 harness 错 + 3 条本地 TTL + 3–4 条噪声。按干净子集 62/123 vs 70/123，差 6.5 pt 但 n=123 的 SE 是 4.5 pt。**没有可修的系统问题**；要缩小置信区间只能加样本数。
3. 4B 不 compact（0.373）≈ 96k compaction（0.365）：对这个模型 compaction 没有带来收益，反而被 512 步上限伤到。9B 的 262k 臂（ep5 0.449 / ep8 0.460）比 4B 高 8–9 pt，是真实的模型差距。

## 10. 更正 §9：本地比 arena 低的主因是**两边跑的 test 文件不是同一份**（11-12）

§9 的结论（"262k 臂没有可修的系统问题，差距是 2 样本噪声"）不成立，§9.1 把 8 条归因到 512 步上限也是错的
——262k 臂的 `qemu-startup s0` 只用了 **245 s** agent 时间，根本没接近上限。

**事实**：`tb_local/tasks/` 下 89 个 test 目录与 GitHub 公开仓库 `harbor-framework/terminal-bench-2` 的 HEAD
**逐字节一致**（`diff -rq` 89/89 无差异，仓库只有一个 initial commit）。而 arena env `6.4.0` 镜像里执行的 test 是
**打过补丁的另一份**。补丁内容不用猜：arena 自己的 `output/<job>/tasks/<task>/s<k>/tbench_test.log` 在 pytest
失败时会把被测源码整段打印出来。

| 题 | arena env 里的 test（日志原文） | 我们跑的 test（= 公开仓库原版） | 后果 |
|---|---|---|---|
| polyglot-c-py | `main_file = os.path.join(polyglot_dir,"main.py.c"); assert os.path.exists(main_file)` + 注释 "Note: The task description allows compiling to /app/polyglot/cmain, so we allow additional files like compiled binaries to exist" | `assert os.listdir("/app/polyglot") == ["main.py.c"]` | 本地 **16/16 全挂**（8 个 run × 2 样本）；arena mask 1/2 过 |
| polyglot-rust-c | 同上（"dont require it to be the only file — compilation may leave binaries"） | `assert os.listdir("/app/polyglot") == ["main.rs"]` | 本地 15/16 挂（只有 dsv41 s0 过） |
| qemu-startup | `os.popen("env -u LD_LIBRARY_PATH expect -f run.exp").read()` | `os.popen("expect -f run.exp").read()` | mask 两臂 **4/4 挂**；arena 同样本 2/2 过 |
| caffe-cifar-10 | `_find_caffe_binary()` + "Accepts either build system: autotools (Makefile.config with CPU_ONLY := 1) or CMake (build/CMakeCache.txt with CPU_ONLY:BOOL=ON)" | 只认 `Makefile.config` 里的 `CPU_ONLY:=1` | 用 CMake 构建必挂（本地 mask 两臂 4/4 挂） |
| train-fasttext | `["fasttext","test","/app/model.bin","/tests/private_test.txt"]`，解析 `P@1` 行 | `fasttext.load_model()` + sklearn `accuracy_score` | 同一模型本地 0.6179 挂 / arena 过；两种度量本来就不可比 |
| reshard-c4-data | 跑压缩脚本前先 `uv sync`（修依赖缺失） | 直接 `uv run compress.py` | 本地两臂都挂在这一步 |
| build-pmars | `test_pmars_works` 断言 warrior 能跑出地址（"Missing ';assert'" / "No instructions" 提示） | 另有一套断言（要求源码目录解到 `/app`） | 1–2 条 |
| hf-model-inference | `test_sentiment_endpoint` 断言 `> 0.9997` 置信度（源码 8 行不同） | 原版断言 | 1 条 |
| git-multibranch | `test_multi_branch_https_deploy` 3 行不同（expect 相关） | 原版 | 1 条 |

这 9 个只是"失败过、所以源码可见"的题；没失败过的 test 我们看不到源码，实际差异面只会更大。

### 10.1 按 arena 口径重算

- `mask-c312-262k-local`：arena 赢 21 / 本地赢 14，净 −7。其中 6 条是 testdiff（polyglot-c-py、qemu-startup ×2、
  git-multibranch、reshard-c4-data、train-fasttext），3 条 TTL（adaptive-rejection s0 750 步、build-pov-ray s1 570 步、
  caffe s1 1002 步），1 条 harness ERROR（mailman s0 tmux 塞爆）。改判后 **66+6 = 72/178 = 0.404**（arena 0.410，差 1 条）。
- `mask-c312-96k-local`：arena 赢 26 / 本地赢 18，净 −8；其中 9 条 testdiff。改判后 **65+9 = 74/178 = 0.416**
  （≥ arena）。真正撞 512 步上限的只有 4 条（rstan-to-pystan、adaptive-rejection-sampler、compile-compcert、build-pov-ray）。

重写结论：
1. 4B c312 的 0.365 / 0.373 **是用错 test 文件跑出来的**，不是模型弱、也不是 compaction 语义差；
2. **撤回 §9.4 第 1 条**（"必须用 1.6.262 才能和 arena 比"）：同 test 下 96k 与 262k 相差 <1 pt，两者都 ≈ 0.41；
3. 与 arena 的**绝对分不可直接比**；模型之间的相对比较仍然有效（偏差对所有臂同向、同量级），
   但凡是靠 polyglot-c-py / polyglot-rust-c / qemu-startup / caffe-cifar-10 / train-fasttext 这 5 题拉开的差距都不算数。

### 10.2 修法（按代价排序）

1. **取回 arena 的 test**：`arena task download <task_id> tests/test_outputs.py`（需在能访问控制面的机器上跑；
   `/storage/openpsi/users/hechuyi.hcy/tools/arena` v1.11.0 在，`~/.secrets.env` 里有 `ARENA_OPENAPI_TOKEN`，
   但本机请求被 302 到登录页）。备选：用 `ACR_PASSWORD` 拉 env 镜像取出 `/tests`，或从失败日志逐题重建。
2. 落成 `tasks/<task>/tests.arena/` **覆盖层**，`run_task.sh` 投递 oracle 时优先用它；不要就地改 `tasks/*/tests`，
   否则 11-11 之前全部历史结果失去可比性。
3. **奖励阶段统一 `env -u LD_LIBRARY_PATH`**：arena 是在每题 test 里逐个 `env -u` 的（见 qemu-startup），
   我们在 reward 阶段一次性清掉，expect/tcl 类题就不用依赖题目自带补丁。
4. 重跑受影响题：5 个 proven testdiff 题 × 2 样本 × 需要的臂，约 20 分钟/臂。

### 10.3 反方向也查了：本地赢的样本不是"我们的 test 更松"

- `sanitize-git-repo`（本地 2/2 过、arena 2/2 挂）：两边 test 完全相同（`FAKE_HUGGINGFACE_TOKEN_2`
  断言在两边都有）。arena 的失败文本显示 contaminated 文件里那条 `hf_token=` 还在——同一份 test，
  模型行为不同。
- `code-from-image`（本地 2/2 过、arena s0 挂）：两边期望的 sha256 常量完全一致
  （`bee26a133f…`），arena s0 产出 `bee26a36a3…`。同样是模型行为。
- 唯一确认的"本地占便宜"是 `extract-moves-from-video`：模型能 `git clone` 到 TB 原始仓库读 oracle
  （arena 沙箱出口慢/不稳，见 §8），属于网络差异，不是 test 差异。

所以净偏差方向明确：**9 道题（已知）的 test 版本差 → 系统性压低本地；反方向的损失主要是模型噪声和 1 道网络题。**

## 11. 4B mask c312 两臂的"系统错误"清单与重跑（11-12）

两个臂跑完后，把非模型原因的 0 分逐条挖出来。**口径之外只有 6 类**，其中只有一类是真正的口径差（512 步上限），其余是本地的运行期事故。

| # | 现象 | 96k 臂 | 262k 臂 | 归属 | 处置 |
|---|---|---|---|---|---|
| A | `TERMINUS_MAX_TURNS=512` 撞上限（轨迹正好 512 步、分数 0） | **50 条（49 条 0 分）** | 0（无上限） | 口径差（arena 1.0.0 无 cap） | 新增 spec `1.6.96-nocap`（96k 窗口 + 无 cap），对撞上限涉及的 **31 题 ×2 样本**重跑成 `mask-c312-96k-nocap-local` |
| B | `tmux` server 消失 → `RuntimeError: ... failed to send non-blocking keys ... no server running on /tmp/tmux-0/default` | 3（extract-elf s1、compile-compcert s1、polyglot-rust-c s0） | 5（adaptive-rejection-sampler s1、compile-compcert s1、mailman s0/s1、protein-assembly s0） | harness（arena 同类 3 条：install-windows ×2、make-mips、qemu-alpine） | 同 spec 重跑 |
| C | `oracle was not delivered by the host within 300 s`（reward rcode 97，reward 段恒 300 s） | 4（password-recovery s0/s1、protein-assembly s1、qemu-alpine-ssh s1） | 1（mailman s0） | 本地 oracle 交付事故 | 同 spec 重跑 |
| D | reward `bash /tests/test.sh: No such file or directory`（rcode 127，reward 1 s） | 1（extract-moves-from-video s0） | 0 | 本地 reward runner `/tests` 软链/拷贝竞态（未定位到具体分支） | 同 spec 重跑；若复现再查 |
| E | 样本从未被领取（无 result.json、无锁） | 0 | 1（qemu-alpine-ssh s0） | 驱动中断 | sweep 自动补 |
| F | 撞 4 h agent TTL | 3 | 28 | 口径（arena 41） | 不重跑 |
| G | `peak_mem_mb == 16384`（正好等于 `MEM_LIMIT=16G` 的 cgroup 上限、OOM 开启），同一样本 arena 过了 | 6（caffe-cifar-10 s1、qemu-startup s0/s1、rstan-to-pystan s0/s1、train-fasttext s1） | 1（caffe-cifar-10 s1） | 本地资源口径：arena 声明 `resource_spec.memory=8Gi` 但不真正限容器，我们的 cgroup 真限 | 这 4 题 ×2 样本用 32G + 1.6.262 重跑成 `mask-c312-mem32g-local` |

补充两条观察（不作为重跑理由）：
- **`usage` 计数下压缩失效**：96k 臂 `make-doom-for-mips s0` 连续 4 次 `Input length (262138) exceeds the maximum allowed length (262138)`——声明 98304 窗口却在 262k 处炸掉，说明 `TERMINUS_TOKEN_COUNT_MODE=usage` 在 sglang 上拿不到真实 usage，压缩阈值没触发。该样本同时属于 A 类，`1.6.96-nocap` 臂用 estimate 计数重跑会一并暴露。
- **同一个 out/ 目录被两个 run_task 复用**：96k 15 条、262k 14 条的 `out/*` 修改时间晚于 `result.json`（两次同名 (task,sample) 并发跑、后者覆盖前者），其中 `password-recovery s0/s1`、`qemu-alpine-ssh s1`、`protein-assembly s1`、`mailman s0` 的 stages 与 result.json 属于不同次运行。同一 spec 同一样本的重跑，分布不变，只增噪；拼接错误只影响"读 artifact 复算"，不影响总分。
- 内存确实背锅（G 类）：顶到 16384 的样本里，96k 6 条、262k 1 条是 arena 同一样本通过的；峰值恰好等于 `MEM_LIMIT=16G` 且 `memory.oom_control=0`（OOM 开启），说明是 cgroup 在容器里 OOM 掉了。其余顶到 16384 的样本 arena 也是 0，属模型自身。arena 的 8Gi 是资源声明，平台不会为此 OOM 容器，所以本地"给 16Gi"未必比 arena 宽松。

重跑口径：A 类进新 run（`1.6.96-nocap`），B/C/D/E 在同 run 内作废旧结果（`runs/invalidated-20261109/*`）后由 sweep 自动补。四个 workload（c304 全量 178、c312 两臂补跑 14 条、A 类 62 条、G 类 8 条）共用**一个** slurm 作业 `993827`（slurmd-62，rsi 预留，8 GPU：c304 占 0-3、c312 占 4-7；126 核切成 16+15 个 4 核槽）。

### 10.4 11-12 复核：只有 4 道题是真的 test bug（§10.1 的 +6/+9 作废）

把 §10 的"9 道题 testdiff"逐条做成了可执行验证（`tb_local/verify/verify_*.sh`，在池节点上用 singularity
跑真实 sif），结论要收窄：

**真的 upstream test bug（4 道，已修）**

| 题 | 验证方式 | 结果 |
|---|---|---|
| polyglot-c-py | 用题目自带 `solution/solve.sh` + 题面要求的 `gcc main.py.c -o cmain` 造工作区 | 原版 test **FAIL**（`Expected only main.py.c`），arena 版 **PASS** |
| polyglot-rust-c | 同上 + `rustc … -o main`、`g++ … -o cmain` | 原版 **FAIL**，arena 版 **PASS** |
| caffe-cifar-10 | 造 CMake 布局（`build/CMakeCache.txt` 含 `CPU_ONLY:BOOL=ON`，无 `Makefile.config`） | 原版 **FAIL**，arena 版 **PASS**；autotools 布局两边都 PASS；两种布局都没有时 arena 版仍 **FAIL** |
| train-fasttext | stub `fasttext` CLI 注入 P@1 | 0.621 → PASS，0.619925 → FAIL，无法解析 → FAIL（阈值/解析行为与 arena 源码一致） |

**不是 test bug（5 道，撤回）**

- `qemu-startup`：arena 与我们唯一的差别是 `os.popen("env -u LD_LIBRARY_PATH expect -f run.exp")`。实测该镜像里
  故意污染 `LD_LIBRARY_PATH` 后 `expect` 仍正常 spawn（`/usr/bin/expect` 有自己的 RUNPATH），**这个 hunk 是惰性的**。
  我们那几条失败是模型/VM 本身（262k s0 只用了 245 s 就交卷；s1 是 `Connection closed by foreign host`）。
  文件仍按 arena 原文放进了 `tests.arena/`（无害、字节对齐），但不计入"被 test bug 压低的样本"。
- `reshard-c4-data`：差异只是 `print`/`uv sync` 的写法，我们文件里早就有 `uv sync`；失败是真的（压缩脚本没产出目录）。
- `build-pmars`：`test_debian_source_used` 两边逐字相同；失败是真的（模型没把 Debian 源码解到 `/app`）。
- `hf-model-inference`：`test_sentiment_endpoint` 两边逐字相同（差的那 8 行是 `requests` 库 frame）；
  失败是真的（负样本置信度 0.9997 被判成正向）。
- `git-multibranch`：差的那 3 行是 expect 的 tcl 报错文本；失败是真的（部署没成）。

所以 §10.1 的"262k 加 6 条 → 0.404、96k 加 9 条 → 0.416"**作废**。受影响样本按"确实命中 buggy 断言"数：
每个臂 polyglot-c-py ≤2、polyglot-rust-c ≤2（但 arena 同题也是 0/2，收益≈0）、caffe ≤1、train-fasttext ≤2 未定。
现实预期是**每臂 +1～3 条（+0.6～1.7 pt）**，即 4B c312 262k ≈ 0.38～0.39，96k 同量级——与 arena 0.410 的
剩余差距是 TTL（本地 28 条 / arena 41 条，方向相反）+ 2 样本噪声（SE≈3 pt），不是可修的 test 问题。

**落地**
- `tasks/<task>/tests.arena/`（4 道）+ `run_task.sh` 在投递 oracle 时优先用它（`TESTS_SRC`）；`tasks/*/tests` 一字未改，
  历史结果仍可解释。
- reward 阶段的 `env -u LD_LIBRARY_PATH` **试了、又撤了**：容器里这个变量是 singularity 注入的
  `/.singularity.d/libs`（不是传言的 python313 lib），把它整个删掉是无谓的口径改动；`expect` 的 spawn 在
  这个套件里实测不受影响，`qemu-startup` 的失败与该变量无关。`run_task.sh` 保持原样。
- `tb_local/rescore_arena_tests.sh <run>`：只把"失败原因确实是 buggy 断言且非 TTL"的样本移到
  `runs/invalidated-<date>/`，让正在跑的 `sweep_until_done.sh` 用新 test 重跑。已对 randfix 两臂执行（4 条），
  其余已完成的臂需要等 GPU/serving 回来后同样执行。
- **ACR 路线确认不可行**（11-12）：深圳 ACR `acr-sz-ant-registry-vpc.cn-shenzhen.cr.aliyuncs.com` 用
  `~/.secrets.env` 的账号可以认证（`/v2/` 401 → token 200），但 `registry:catalog:*` 被拒（公共账号只有 `gpu/*`
  的 pull/push），盲猜的 `gpu/tb2-*`、`envarena/*`、`tb2/*` 都不存在；上海 ACR 连 token 都不发。而且 arena 的
  test 文件在 benchmark oracle 里（不是 env 镜像），所以即使拉到镜像也拿不到。结论：以 arena 日志打印的源码 +
  题目自带 solution 做等价验证是唯一可执行的路径，已按此完成。

### 10.5 对"撤回 5 条"的二次校验（11-12 12:10）

方法换成逐函数**前缀对比**：pytest 失败时会把被测函数从 `def` 起打印到出错那一行，拿 arena 日志里该函数最长的一段，
与我们文件里同一函数的同长度前缀做 `unified_diff`；再核对我们每条失败样本在 `test_outputs.py` 的行号是否落在这段前缀里。

| 题 / 函数 | arena 可见行数 / 我们的函数行数 | 前缀 diff | 我们失败的行号与原因 | 结论 |
|---|---|---|---|---|
| qemu-startup `test_version` | 23 / 23（全覆盖） | 只有 1 行：`env -u LD_LIBRARY_PATH expect` | :58 `/tmp/data.txt` 不存在（expect 报 `spawn id exp4 not open`）、:60 `uname -r` 读回空 | 见下，**惰性** |
| reshard-c4-data `test_compress_decompress_workflow` | 86 / 100 | **完全相同** | :135 `Directory … has 330 items, exceeds 30`（在可见前缀内） | 模型的 compress.py 没按 30/目录分片，真失败 |
| build-pmars `test_pmars_works` / `test_debian_source_used` | 19/30、11/11 | 完全相同 | :35 `rave.red: Expected stepping, got 0 addresses`（前缀内） | 真失败 |
| hf-model-inference `test_sentiment_endpoint` | 55 / 59 | 完全相同 | :71 负样本 `negative` 置信度 0.0003（前缀内） | 真失败 |
| git-multibranch `test_multi_branch_https_deploy` | 93 / 93（全覆盖） | 完全相同 | :99 `'TEST PASSED' not in …`（前缀内） | 真失败 |

上一轮判 testdiff 的那些"多出来的行"是 `requests`/`socket`/tcl 的栈帧和 `print` 文本，不是测试源码。

qemu-startup 补了一个**用真 test 文件**的实验（`verify/verify_qemu_expect.sh`，池节点上跑 `tb2-qemu-startup.sif`）：
容器里起一个假 "alpine"（127.0.0.1:6665 应答 `localhost login:` / `# ` / `uname -r`→`6.6.4-1-lts`），然后分别用
原版与 arena 版 `test_outputs.py`、在 `LD_LIBRARY_PATH` 原样（`/.singularity.d/libs`）与故意前置 `/sandbox-net/py/python313/lib`
两种环境下跑 pytest：**4 种组合全部 PASS，`/tmp/data.txt` 都写出了 `6.6.4-1-lts`**。所以 `env -u` 那一行在我们的沙箱里
改变不了任何判定；我们 4 条 qemu 0 分是 VM/模型侧（`spawn id exp4 not open` = telnet 连不上 6665，`Connection closed by
foreign host` = 客户机断开），与 test 文本无关。

### 10.6 修复后还剩哪些系统性问题（11-12 12:30，逐样本复核）

方法：对三个 run 的**每一对**本地/arena 判定做分类（先扫整 run 的信号，再逐条看那个样本的 `result.json` /
`reward.log` / 轨迹里的 `Killed`、`MemoryError`、`oracle was not delivered`、`no server running`、步数与峰值内存），
arena 侧取同样本（同 task、同 sample）的分数，不看聚合分。

| run | 有效样本 | 本地 | arena（同样本） | 差 | 差在哪（按样本） |
|---|---|---|---|---|---|
| mask-c312-262k | 172 | 66 | 72 | −6 | test-bug 2 · 16Gi OOM 1 · TTL 2 · 模型 15→本地赢回 14（其中 arena TTL 4） |
| mask-c312-96k | 171 | 65 | 73 | −8 | test-bug 1 · 512 步上限 5 · 512 步+16Gi 3 · 16Gi 1 · 模型 16→本地赢回 18（arena TTL 4） |
| dsv41-local | 178 | 140 | 148 | −8 | test-bug 5 · test-bug+TTL 1 · 模型 14 → 本地赢回 12 |

修复后仍在压低本地分的**四类系统问题**（其余是模型/噪声）：

1. **96k 臂的 512 步上限**（46 条撞上限，1 条在 arena 也 0）。这是 harness 口径差：arena `1.0.0` 无 cap。
   已备 `1.6.96-nocap`（96k 窗口 + 去 cap）对这 31 题×2 重跑，池子被抢占后未跑完。
2. **16Gi 硬 cgroup 比 arena 严**（12 条 peak 顶到 16384 且非 TTL）：证据是 `rstan-to-pystan s0` 轨迹里 **93 次 `Killed`**、
   `qemu-startup s1` **15 次 `MemoryError`**，而 arena 同样本通过。arena 的 `resource_spec.memory=16Gi` 只是调度声明，
   平台不会在 16Gi 整 cgroup-OOM。**已改 `run_task.sh` 默认 `MEM_LIMIT=32G`**（节点 1100G、最多 15 槽，安全），
   并把 12 条作废重跑（`runs/invalidated-20261112-mem/`）。
3. **2 核旧样本**（262k 66 条 / 96k 119 条）：这批是 11-10 之前 2 核跑完的短题。分开统计后
   262k 的 2c 子集本地 37/66、arena 38/66（差 1），4c 子集本地 29/106、arena 34/106（差 5）——**差距集中在 4c 的长题**
   （TTL/OOM/cap），2 核本身不是主因，维持原处置（只作废 TTL 命中的）。
4. **运行期事故**：c312 两臂在 11-12 之前有 oracle-97（5 条）、reward-127（1 条）、tmux 掉线（8 条），已被监督子代理
   作废重跑；复查当前 `result.json`：两臂 oracle-97=0、reward-127=0、tmux=1（`qemu-alpine-ssh s1`）。

按上述修正重算：
- **262k**：66 + 2（test-bug：polyglot-c-py s1、train-fasttext s1）+ 1（caffe s1 的 16Gi OOM，也是 test-bug）= **69/178 ≈ 0.388**
  （arena 同 172 样本 0.419；剩余 3 条是 TTL/模型）。
- **96k**：65 + 1（test-bug）+ 9（512-cap 8 条 + 16Gi 1 条，交集已去重）= **75/178 ≈ 0.421**
  （上限估计：只有当那些样本在 nocap/32G 下真的解出来才成立；已在 993904 上重跑）。
- **dsv41**：140 + 6（test-bug）= **146/178 ≈ 0.820**（arena 0.831，差 2 条，SE≈3 pt）。注意 dsv41-local 是
  "旧契约 ~125 条 + v2 ~53 条"的混合 run，轨迹里还能看到 `Apptainer>` 与 `/envarena/oracle` 命中，只作历史参照。

结论：**test-bug 修完后，本地相对 arena 的剩余偏低集中在 96k 的 512 步上限和 16Gi cgroup 两处，都是可修的口径差；
262k 臂只剩 2–3 条（TTL + 模型噪声）。** 已经没有"未知的"系统性差异。

### 10.7 11-13 复查：还有一个自己造的系统错误——`--writable-tmpfs` 把 rootfs 写入算进内存 cgroup

查"为什么本地一堆样本恰好 `peak_mem == MEM_LIMIT`"时找到的**真根因**，不是模型吃内存：

- 我们的容器用 `singularity --writable-tmpfs`，可写层是 tmpfs；**tmpfs 的页是不可回收的 shmem，全部计入该任务的
  memory cgroup**。docker（arena）的可写层在磁盘上，只算 page cache，可回收。
- 受控实验（同一 sif、1G cgroup 限额、写 2 GB 到 `/app/big`）：
  `--writable-tmpfs` → 被 OOM **Killed**；`--overlay <磁盘上的 sparse ext3>` → **写成功**。
- 后果一致地出现在轨迹里：`adaptive-rejection-sampler s0` 第 3 步 `apt-get install -y r-base`"看起来成功"，
  第 17 步起 R 就是残的（`/usr/lib/R/bin/R: line 249: /usr/lib/R/etc/ldpaths`，同一错误 **671 次**），
  模型为此耗掉 750 步、撞 4 h TTL；arena 同一模型 147 步、5157 s 解出。同理 `rstan-to-pystan s1`
  本地 `peak=32768`（32G 上限，被自己杀）而 arena 通过。

**已修**（`run_task.sh`）：每个样本建一个稀疏 20G ext3 overlay（`singularity overlay create`，放在 `$OUT/overlay.img`），
容器改用 `--overlay`，运行结束删除；创建失败才回退 `--writable-tmpfs`。`OVERLAY_MB` 可调。
端到端验证：`polyglot-c-py` 真跑 → `score=1`、`tests_source=tests.arena`、`peak_mem=1787MB`、overlay 已清理。
这条修完，之前"16Gi/32Gi 顶格"的那一批（`rstan`、`qemu-startup`、`caffe`、`gpt2-codegolf`…）才算真正解决；
`MEM_LIMIT=32G` 只是缓解，因为 tmpfs 写入是不管限额多高都要占用 RAM 的。

### 10.8 另一条：TTL 偏高不是"本地慢"，是模型在原地打转

262k 臂里本地 TTL 28 条、arena 同样本 17 条。逐条看那 11 条"本地 TTL、arena 正常"的样本：本地**每步更快**
（中位 9–16 s vs arena 17–43 s），但**步数是 arena 的 3–7 倍**（750 vs 147、1002 vs 386、774 vs 108、1213 vs 376）。
即：不是网关/CPU 慢，而是模型没有完成却在继续尝试 —— 其中一部分正是 §10.7 的环境残缺导致的反复重试
（R 坏掉、`file`/`xxd` 缺失）。修完 overlay 后这一块应当回落，但需要重跑才能量化。

### 10.9 11-13 16:30：为什么修完分数反而没涨——dsv41-v2 与 mask-262k 逐样本

**dsv41-v2（全新 v2 契约）vs arena，同 167 样本：本地 130（0.778）/ arena 141（0.844），arena 赢 20、本地赢 9，净 −11。**
20 条 arena 赢的分成四类：

| 类 | 条 | 样本 | 性质 |
|---|---|---|---|
| 我自己的 bug：`tests.arena/train-fasttext` 只换了 `test_outputs.py`（调 `fasttext` CLI），没换 `test.sh`——arena 的 `test.sh` 会 `git clone /opt/fastText` + `make` 出 CLI，upstream 的只装 python 绑定 → `FileNotFoundError: 'fasttext'` | 2 | train-fasttext s0/s1 | **已修**：`tests.arena/test.sh` 按 arena 日志重写（clone+make） |
| 磁盘写满 → ext3 overlay 被内核 remount 只读 → `mkdir /envarena/oracle: Read-only file system` → reward exit 127 | 2 | dna-assembly s1、extract-moves s1 | 事故（§10.7 的 20G 非稀疏 overlay 把配额吃光） |
| **计时型测试在高并发下失败**：`query-optimize::test_compare_golden_vs_solution_runtime` 要求 solution ≤ 1.05× golden（实测 0.984 vs 0.956），`largest-eigenval::test_speedup` 比 2.8e-5 s vs 2.4e-5 s | 3 | query-optimize s0/s1、largest-eigenval s0 | 节点上同时 17–35 个任务时的 CPU/缓存争用；arena 一个沙箱一台 pod |
| 模型自身（pytest 真跑、断言失败） | 13 | ars ×2、pov-ray、caffe s0、circuit、extract-elf s1、gpt2 s0、doom s1、model-extraction s1、overfull、protein s1、pytorch-cli s0、sam ×2 | 与本地赢的 9 条抵消后净 4 |

去掉 7 条系统性的：≈137/167 = **0.820** vs arena 0.844，差 2.4 pt（2 样本 SE≈3 pt）。

**为什么 v2 比旧 dsv41-local（0.787）低**：旧 run 147/178 是 v1 契约（oracle 在 agent 阶段可见），本来就偏高；
同 167 样本旧 132 / 新 130，14 降 12 升——降的里面 5 条就是上表的系统问题，其余是 v1 时代的"借 oracle"与模型噪声。

**mask-c312-262k vs arena，同 169 样本：65（0.385）/ 71（0.420）**，arena 赢 19 / 本地赢 13：TTL 3 条，模型 16 条，
本地赢 13（其中 arena TTL 3）。没有 harness 错、没有 test 差异残留；差距 = TTL（本地跑满 4 h 的更多，模型在原地打转）+ 噪声。

**结论**：修复让分数"看起来没涨"的原因是 (1) 旧基线本来是虚高的（v1 契约）；(2) 我在 train-fasttext 上引入了一个新 bug；
(3) 一次磁盘写满事故污染了 ~15 条；(4) 把并发提到 35 之后计时型测试开始失败。前两条已修，第三条已作废重跑，
第四条需要给计时型题（query-optimize、largest-eigenval、llm-inference-batching-scheduler、tune-mjcf::test_model_speed 等）
单独低并发跑或在批次里限槽。

### 10.10 11-14 09:00：第二处"不是同一份"——**题面（instruction.md）**也和 arena 不一样

§10 核的是 `tests/`（与 github HEAD 89/89 一致），没核题面。11-14 把 arena 轨迹首条 user message 里
`Task Description:` 与 `Current terminal state:` 之间的正文抠出来和本地 `tasks/*/instruction.md` 逐词比较：

- 本地题面 = `laude-institute/terminal-bench-2` HEAD（`2fd12b8`）原文，89/89 `cmp` 一致；全历史 66 个 commit 里也**搜不到**
  arena 版的任何特有短语（`n = sample_count`、`--weights_path`、`Do not modify the database file`、`pmars-<version>`、`pre-scattered`）。
  `terminal-bench` 主仓库（TB1 `original-tasks/`）同样没有。
- arena 的 benchmark 包（`benchmark_ref cth-t*** 1.0.0`，继承 hechuyi 的 2.1.1）对 **11 道题**的题面打了补丁，4 个全量 job 之间一致。
  方向全是"把题面写得与 test 对齐"：

| 题 | arena 题面多出来 / 改掉的 | 对应 test |
|---|---|---|
| adaptive-rejection-sampler | 给出签名 `ars(density_function, domain, n = sample_count)` | test 调 `ars(f, c(-5,5), n=1000)`；本地失败信息 "lower and upper must be single numeric values" / "Provide only one of density or log_density" 就是猜错签名 |
| sam-cell-seg | 参数写成 `--weights_path/--output_path/--rgb_path/--csv_path`，输出是 **file** 不是 folder | `FileNotFoundError: /app/test_output.csv`（按本地题面写成目录） |
| query-optimize | "Do not modify the database file in any way" | `test_check_for_db_modifications` |
| caffe-cifar-10 | 指明 solver 在 `examples/cifar10/cifar10_quick_solver.prototxt` | test 读该路径 |
| build-pmars | 源码解到 `/app/pmars-<version>/`、保留源码树 | `test_pmars_works` 另一套断言 |
| mteb-leaderboard | "Only consider models that have results for all tasks" | 答案不同 |
| mteb-retrieve | "You must use the installed mteb package" | |
| torch-tensor-parallelism | 说明 RowParallel 的输入已 pre-scattered | 实现语义 |
| install-windows-3.11 | monitor socket 固定在 `/tmp/qemu-monitor.sock` | test 连该 socket |
| filter-js-from-html | 允许 "normalization that may occur during HTML parsing" | |
| extract-moves-from-video | 视频已在 `/app/video.mp4`（本地题面要去 youtube 下载） | arena 的 env 镜像里带 mp4；本地镜像（按 github Dockerfile 建）没有，youtube 出不去，**没法对齐，保留本地题面** |

另有 17/163 条本地轨迹首条消息没有 `root@sandbox:/app#` 提示符行（tmux 首屏快照晚于发送）；影响应极小，记录在案。

**对分差的解释**（11-14 08:30 快照，去掉 score=null）：

| run | 共同样本 | arena / 本地 | 题面不同的题（n） | 其余题 |
|---|---|---|---|---|
| dsv41-v2 vs `89m-dsv41` | 162 | 137 (0.846) / 131 (0.809) | 30：24 / 19，净 **−5** | 133：0.857 / 0.842，净 −2 |
| 4B c312 262k vs `mask-const-c312` | 170 | 72 (0.424) / 65 (0.382) | 27：11 / 6，净 **−5** | 143：0.427 / 0.413，净 −2 |
| 4B c312 96k vs 同上 | 165 | 69 (0.418) / 65 (0.394) | 30：12 / 10，净 −2 | 135：0.422 / 0.407，净 −2 |

即剩余 −6/−7 里 5 分来自题面补丁题；题面一致的 130+ 道两边差 2 分（≈1.5 pt，单次 run 的样本 SE≈3 pt）。

**修法**（与 `tests.arena` 同一机制）：
- `tasks/<task>/instruction.arena/instruction.md`：88 道（extract-moves-from-video 除外），从 arena 轨迹抽取，
  78 道与 github 原文仅空白差异，10 道实质不同（`lists/instr_changed.txt`）。
- `run_task.sh`：有 `instruction.arena/` 就用它当 `Target.md`，并写 `out/instruction_source.txt`（`instruction.arena` / `instruction.md`）。
- `invalidate_instr.sh`：10 道题在 12 个 run 里 **169 条**结果作废到 `runs/invalidated-20261114-instr/`，正在跑的 sweep 自动补；
  当时在飞的 12 条（旧题面）跑完后再执行一次即可作废。
- 校验：新起样本的首条 message 正文与 arena 首条 message 逐词一致（2/2）；补齐后再按 §10.9 方法重算。

**口径说明**：本地此前是"github 官方题面 + 补丁 test"，arena 是"补丁题面 + 补丁 test"。所有 arena 参照分都是后者，故统一到后者。

### 10.11 11-14 12:30：又一个自己造的系统错误——`reap_locks.sh` 从登录节点删掉了在跑样本的 overlay

症状：今天 3 台机器 dmesg 共 93 次 `EXT4-fs: Remounting filesystem read-only`，33 条结果 reward exit 127（`mkdir /envarena/oracle: Read-only file system`），
agent 中途开始所有写都失败（`OSError: [Errno 30] Read-only file system: 'primers.fasta'`），分数被强制为 0。不是 11-13 的磁盘写满。

根因链：`reap_locks.sh` 的"孤儿 overlay 回收"用 `/proc` 里是否有对应 `run_task.sh` 判断样本是否活着；**从登录节点看 `/proc` 里永远没有计算节点的进程**，
`sweep_until_done.sh` 每轮在登录节点先跑一次 `reap_locks.sh` → 把所有在跑样本的 `overlay.img` 当孤儿 `unlink`。loop 设备底下的文件在 FUSE（CPFS）上被删除后，
写回时报 `Buffer I/O error ... lost sync page write` → `JBD2: Error -5` → ext4 abort journal → remount-ro。锁的回收逻辑用的是 30 s 心跳（正确），overlay 回收没有。

修：
1. `run_task.sh`：overlay 放**节点本地盘** `/tmp/tb2-overlay/<task>-s<k>-<pid>.img`（2 TB 本地 overlay 盘，`OVERLAY_DIR` 可改），不再放 /storage。
   loop over FUSE 本身也不稳，这一条即使没有 reap bug 也该做。
2. `reap_locks.sh`：/storage 上的旧式 overlay 只在"有 result.json"或"心跳 >3 min 无更新"时删；节点本地 overlay 按文件名里的 pid 是否存在判断，只在节点上执行有意义。
3. `run_batch.sh`：开批前在节点上跑一次 `reap_locks.sh`（清本节点的孤儿 overlay）。
4. `invalidate_ro127.sh`：把 reward 127 且日志含 `Read-only file system` 的 33 条作废到 `runs/invalidated-ro127/`（dsv41-v2 7、4B 262k 4、96k 1、nocap 4、mask-rand c190 8 / c304 6、randfix 1+1、9B c312 1）。

仍有风险：修复时刻 26 条在飞的旧样本 overlay 已被删（loop 继续用 inode），它们可能还会 RO；跑完后再跑一次 `invalidate_ro127.sh` 即可。
`invalidate_nulls.sh` 同时改成"解析不了的 result.json（0 字节）= null"，补作废了 62 条 11-13 磁盘写满时落下的空结果。

### 10.12 11-14 13:30：统一到最新 setup——旧 setup 下带"受伤签名"的 185 条全部重跑

11-13 10:20 的 overlay 修复（§10.7）当时没有触发重跑，7 个 mask run 里 ~1000 条已计分结果仍是 `--writable-tmpfs` + 16G 时代跑的。
按 setup 版本切开审计（`env.list` 的 mtime 早于 11-13 10:20 = 旧 setup）：

| run | 有效 | 旧 setup | 其中 TTL | peak≥14G 非 TTL | 2 核 | tests≠arena | 重跑 |
|---|---|---|---|---|---|---|---|
| dsv41-v2 | 171 | 144（已是 32G） | 0 | 0 | 0 | 0 | 0 |
| 4B 96k | 146 | 145 | 3 | 3 | 104 | 3 | 8 |
| 4B 262k | 148 | 147 | 26 | 2 | 61 | 2 | 28 |
| 9B c117 / c195 / c312 | 153/154/152 | 152/152/151 | 36/26/27 | 0/1/1 | 0 | 0 | 36/27/28 |
| randfix c114 / c190 | 131/128 | 131/128 | 34/19 | 1/4 | 0 | 0 | 35/23 |

判据（`invalidate_oldsetup.sh`）：旧 setup 且 (TTL ≥14300 s ∪ peak≥14 GB 非 TTL ∪ 4 道补丁题 tests≠arena)。依据：
§10.7 证明 tmpfs 把 `apt install` 装残 → 模型打转 → TTL，所以旧 setup 的 TTL 不是模型结果；对照：修复后新起的样本 TTL = **0**，旧 setup 171 条。
2 核但未撞 TTL 的 165 条按 11-10 规则保留。共 **185 条**作废到 `runs/invalidated-20261114-oldsetup/`（ttl 171、peak 12、tests 5）。

容量：m2 再开 9B 三臂各 +4 槽（核 52-99），m3 再开 262k +4、96k +2、randfix 各 +4（核 92-147）；连同上午的 m1 +5+5、nocap +6，
三机并发 m1 30、m2 25、m3 ~40（计时题风险见 §10.9，终值前对 query-optimize / largest-eigenval 等按低并发复核）。
验证：13:20 后 10 分钟内新起的 42 条全部 `instruction.arena` / 32G / 4 核，节点本地 overlay（loop 52 个在 `/tmp/tb2-overlay`，无 deleted backing）。

**到此为止的最新 setup 定义**（计分样本必须同时满足）：`contract=v2-oracle-after-agent`；`tests_source=tests.arena`（4 题）；
`instruction_source=instruction.arena`（88 题）；4 核 / `MEM_LIMIT=32G` / ttl 14400；rootfs 为节点本地磁盘 overlay；
或者是旧 setup 但无受伤签名（非 TTL、peak<14G、tests 正确、题面未变的题）。

### 10.13 11-14 14:40：三机迁到无预留空闲节点；又有两个"看起来在跑其实空转"的缺陷

**迁移**：13:16 game-efficient 的持有者把我们三个作业全部 `CANCELLED+`（5h33m 后），keeper 重提后排在 `Resources`，ETA 次日 13:15。
改 `pool_keeper.sh` 支持 `RESVS` 里的 `none`（无 `--reservation` 的普通提交）和 `whj` 预留，三机随即拿到**无预留的空闲节点**：

| 机 | 作业 | 节点 | 服务 | sweep（核段） |
|---|---|---|---|---|
| m1 rand | 994758 | slurmd-45 (33.180.160.116) | mask-rand-c190 ×4、c304 ×4 | c190 0-39 + 80-99；c304 40-79 + 100-119（30 槽） |
| m2 fix1 | 994753 | slurmd-6 (33.214.153.122) | 9B c117 ×3、c195 ×3、c312 ×2 | 9B 0-11/12-23/24-35 + 52-67/68-83/84-99；dsv41-v2 36-51（25 槽） |
| m3 fix2 | 994763 | slurmd-3 (33.214.152.224) | 4B c312 ×4、randfix c114 ×2、c190 ×2 | 96k 0-15+124-131；262k 16-27+92-107；nocap 28-51+108-123；c114 52-59+148-163；c190 60-67+164-179（35 槽） |

**缺陷 A（#21）：`sweep_until_done.sh` / `run_nocap.sh` 把 score=null 的 result.json 算作完成**。作业被取消时 run_task 被打断，留下 `harness_status=MISSING`、
`agent_sec=0` 的空结果；sweep 数文件数就退出。14:10 dsv41-v2 因此"178/178 complete"提前终止，实际 3 条没跑。
修：完成条件改成"解析成功且 `score != null`"的计数。同时 `invalidate_nulls.sh` 又清出 65 条这类空结果。

**缺陷 B（#22）：`reap_locks.sh` 在所有自动调用里都是 dry-run**。`sweep_until_done.sh` / `run_nocap.sh` / `run_batch.sh` 里写的是
`bash reap_locks.sh`（默认 `APPLY=0`），所以被取消作业留下的僵尸 claim 永远不会被删；claim 存在即被视为"该样本已被认领"，
sweep 的队列不再包含它 → 静默空转。14:30 手工 `APPLY=1` 清掉 22 个僵尸 claim（9B c117 等），修三处调用为 `APPLY=1 bash ...`。

**现状（14:50，valid = score≠null）**：dsv41-v2 175/178 = 0.840（TTL 0）；9B c117/c195/c312 119/129/127 = 0.622/0.612/0.661；4B 96k 141 = 0.461、262k 121 = 0.521；
randfix c114/c190 98/106 = 0.571/0.462；mask-rand c190/c304 29/39 = 0.655/0.564；nocap 5/62。
新起的 15 条样本全部 `instruction.arena` / 32G / 4 核；三台机器 `/storage` 上 0 个 loop 设备、dmesg 0 次 remount-ro。

**缺陷 C（#23，14:40）**：加开 nocap 时直接用 `sweep_until_done.sh` 起批，没带任务清单，`run_batch` 于是按 `sif/` 全量枚举，
把 4B-nocap 臂跑到了 89 道题上（nocap 臂的定义是 `lists/nocap96k.txt` 的 31 题 ×2）。已把越界的 7 条移到
`runs/invalidated-wronglist/`，改由 `run_nocap.sh`（内部带清单）在 108-123 核段重起。教训：这是唯一一个"臂 = 题子集"的实验，
它只能从 `run_nocap.sh` 起，不能从通用 sweep 起。

### 10.14 11-14 21:50：TTL 里有 1/3 不是模型打转，是 `/etc/localtime` 让 dpkg 装不上包

追 4B 262k 的 8 条 TTL 时发现：`adaptive-rejection-sampler s0` 本地 552 步 vs arena 152 步，轨迹里
**870 次 `cannot open file '/usr/lib/R/etc/repositories'`**、`could not find function "dnorm"`（base 包没加载）。

根因链（容器内实测）：
1. apptainer 默认把宿主 `/etc/localtime` 绑进容器 → tzdata 的 postinst 执行 `mv /etc/localtime.dpkg-new /etc/localtime`
   报 `Device or resource busy`（我们的 overlay 是可写的，纯粹是"目的地是挂载点"）→ **tzdata 配置失败**；
2. tzdata 未配置 → `libtcl8.6`/`libtk8.6` 依赖失败 → **`r-base-core` 一直是 unconfigured** → `/usr/lib/R/etc/ldpaths`、
   `repositories` 缺失 → 任何 `Rscript` 都拿不到 base 包；
3. 模型看到"装成功"的假象（apt 装了文件但没配置），反复试 → 550 步 → 4 h TTL。

**不在模型**：同一模型在 arena（docker，`/etc/localtime` 是普通文件）152 步、5157 s 解出。

修：`run_task.sh` 加 `--no-mount /etc/localtime`。在计算节点上用真 SIF + 稀疏 overlay 实测：
无该 flag → `/proc/mounts` 里 1 条 `/etc/localtime`、`mv` 失败；加该 flag → 0 条、`MV_OK`，`/etc/hosts`、`/etc/passwd`、
`/etc/resolv.conf` 仍在（其它默认绑定不受影响）。

作废：`invalidate_tzdata.sh` 扫 `logs/agent/*.pane` 里的该签名，只作废"新 setup + 分数≠1"的样本 → **41 条**
（randfix c114 8、mask-rand c304 7、nocap 6、randfix c190 5、262k 4 + 96k 2、c195 4、c190 3、c117 2）。
通过的样本不动（失败没挡住它）。sweep 已按新 flag 重跑（节点上 2 个容器带 flag、0 个不带）。

**TTL 的归因修正**：4B 262k 的 8 条 TTL 里，2 条（ars s0、compile-compcert s1）是这条 bug；其余（feal ×2、gpt2 s1、
circuit-fibsqrt s0）本地步数与 arena 相当（425/432、482/403、548/338），**arena 也 TTL**，属于模型本身解不出。

**#24 对 dsv41-v2 的影响**（11-14 23:00 复查）：178 条轨迹里 **9 条**出现 `localtime.dpkg-new` 签名
（adaptive-rejection-sampler ×2、caffe-cifar-10 ×2、compile-compcert ×2、extract-moves-from-video ×2、sqlite-with-gcov s1），
但**全部 score=1**（799–1601 s）——dsv41 自己绕过去了（它没有依赖 R 的 base 包，或自己补齐了配置）。
所以 dsv41-v2 的 0.843 不受 #24 影响，不需要重跑；#24 只伤到"必须用 R/apt 依赖包"的模型（4B mask 系、9B、randfix）。
