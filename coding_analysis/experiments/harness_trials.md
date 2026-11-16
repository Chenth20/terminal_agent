# Harness 对比评测结果

> 所有数字来自 `/storage/openpsi/users/chen/output/` 下落盘的结果文件。
> 口径：**acc** = 完成 task 的均分；**acc(fail=0)** = 全部 task，失败记 0（即平台 `failed_as_zero` 口径）；
> **pass@2** = 每题 2 个样本至少一个得分；**fail** = `HARNESS_FAILED` + `TIMEOUT` 等非正常终态。

## 1. Harness 对比（主实验）

设置：benchmark `cth-tbench-filtered-100@1.0.0`（terminal-bench filtered 集中取 100 题，`image_offline`，reward `tb2-reward@4.0.0`），每题 2 样本，并发 32，retries 1。
结果目录：`/storage/openpsi/users/chen/output/tbench100_harness_cmp/<job>/`（`tasks.json`、`job.json`、每 task 的 jsonl、`summary.md`），汇总在 `harness_comparison.json`。

| 模型 | harness | acc | acc(fail=0) | pass@2 | done | fail | turns | tool calls | total tok | compl tok |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8-27b | **terminus** | **0.885** | **0.845** | **0.910** | 191 | 9 | 22.5 | 45.3 | 1.32M | 45.5k |
| qwen3-8-27b | mimo | 0.845 | 0.820 | 0.870 | 194 | 6 | 29.1 | 36.9 | 1.89M | 38.6k |
| qwen3-8-27b | claude | 0.810 | 0.555 | 0.650 | 137 | 63 | 23.4 | 29.3 | 1.13M | 39.9k |
| qwen3-8-flash | **terminus** | **0.919** | **0.855** | **0.890** | 186 | 14 | 23.5 | 46.4 | 1.04M | 31.1k |
| qwen3-8-flash | mimo | 0.889 | 0.845 | **0.890** | 190 | 10 | 38.8 | 46.4 | 2.20M | 32.1k |
| qwen3-8-flash | claude | 0.798 | 0.670 | 0.690 | 168 | 32 | 19.6 | 22.2 | 599k | 23.7k |
| DSv4-Flash | terminus | 0.725 | 0.555 | 0.600 | 153 | 47 | 12.6 | 31.8 | 195k | 13.8k |

token 列为每 task 均值（total = prompt + completion，累计全部轮次）。

| job | job_id |
|---|---|
| 27b × terminus | `cth-tb100-terminus-qwen-2610160346` |
| 27b × mimo | `cth-tb100-mimo-qwen-2610160859` |
| 27b × claude | `cth-tb100-claude-qwen-2610160859` |
| flash × terminus | `cth-tb100-terminus-flash-2610160912` |
| flash × mimo | `cth-tb100-mimo-flash-2610160912` |
| flash × claude | `cth-tb100-claude-flash-2610160912` |
| dsv4 × terminus | `cth-tb100-terminus-dsv4-2610160346` |

### 结论：推荐 Terminus 2

- **准确率最高**：两个 qwen 模型上 acc 和 acc(fail=0) 都第一；27b 的 pass@2 0.910 也最高。
- **稳定**：27b 只有 9 个失败（Claude Code 63 个）。
- **成本低**：token 比 MiMo 少 30–50%，轮数也更少。
- **MiMo 是接近的第二名**：失败数最少（6/10），flash 上 pass@2 与 terminus 持平，但 token 多 1.4–2×、轮数多（flash 38.8 vs 23.5）。
- **Claude Code 当前不适用**：失败率高（27b 32%，flash 16%），0.81 的 acc 是在幸存样本上算的，有选择偏差；acc(fail=0) 比 terminus 低 0.19–0.29。27b 上还有部分正常结束但得 0 分的 run。
- DSv4 + terminus 失败 47 个，acc(fail=0) 0.555，低于两个 qwen；轮数和 token 都最少，说明它更早放弃或更早出错，需要看失败原因后再下结论。

### 注意事项

- 单次实验，每题 2 样本，没有置信区间；差距小于约 0.03 的不宜过度解读。
- flash 走 DashScope API，27b 走 slurm 自建服务，两者的延迟/超时特性不同，**跨模型比较不完全公平**。
- 本表不考虑超时问题（按用户要求）。

## 2. Terminal-Bench 2.1 历史结果

benchmark `tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2`（89 题），harness `terminus-2-tb21@1.0.12`。

| 日期 | job_id | 模型 / 档位 | 配置 | 结果 |
|---|---|---|---|---|
| 09-04 | — | `qwen35-27b-base-tp1-sglang0706@1.0.1`（基线） | 89×1 | 0.4045 |
| 10-02 | — | Qwen3.5-27B 单卡（slurm） | smoke | 0.5 |
| 10-03 | `qwen38-27b-chen-ais-tb21-full-2610030612` | Qwen3.8-27B，AIS 6 节点 | 89×1 | **0.5843** |
| 10-14 | `qwen38-27b-chen-ais-4n-r2-tb21-full-2610140739` | Qwen3.8-27B，AIS 4 节点 | 89×1 | 0.4944（73 ok / 16 fail；15 个网关 502） |
| 10-15 | `qwen3-8-27b-router-xhigh-chen-tb21-smoke-2610151807-2610151007` | qwen3-8-27b-router / xhigh | smoke | headless-terminal 1，pypi-server 1，git-multibranch 0 |
| 10-15 | `qwen3-8-27b-router-xhigh-chen-tb21-full-2610151818-2610151019` | qwen3-8-27b-router / xhigh | 89×8，par 32 | **未收口**：本地落盘 100 个 task，其中 79 CANCELLED、3 HARNESS_FAILED、18 DONE |
| 10-15 | `deepseek-v4-flash-0731-max-chen-tb21-full-2610151819-2610151019` | DSv4-Flash / max | 89×8，par 32 | **未收口**：本地落盘 100 个 task，86 DONE、12 HARNESS_FAILED、2 运行中；落盘部分 fail=0 均分 0.56 |

10-14 的 0.4944 vs 10-03 的 0.5843：差距主要来自 15 个网关 502（`failed_as_zero` 记 0），不是模型能力变化。
10-15 两个 full job 的最终分数需要重新 `job get` 核对，本地文件只是 puller 中断前的快照。

## 3. TB2.1 × 8 三 harness 对比（10-17，已停止）与 harness error 修复验证

### 3.1 三 harness 全量（中途停止）

`tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2` × 8，模型 `qwen3-8-flash-xhigh-chen@1.0.0`，par 32。跑了约 9.6 h 后停止，原因是 harness error 集中且确定性复现，继续跑没有意义。停止时快照：

| harness | job_id | 已结束 / 712 | HARNESS_FAILED | acc（完成） |
|---|---|---|---|---|
| terminus `terminus-2-tb21@1.0.12` | `cth-tb21x8-terminus-flash-2610170432` | 272 | 34（全是 `:7002` 502） | 0.819 |
| mimo `cth-mimo-agent@1.0.3` | `cth-tb21x8-mimo-flash-2610170432` | 371 | 1（502） | 0.711 |
| claude `cth-claude-code-swe@1.0.2` | `cth-tb21x8-claude-flash-2610170432` | 529 | 94（84 个 `python3: command not found`） | 0.740 |

- 三个 job 完成的子集不同（terminus 只完成 1/3，偏易），**这些均分不能互相比较**，也不能和 100 题集的 0.9 比较。
- 失败按 env 高度集中：terminus 的 34 个落在 7 个 env，claude 的 94 个落在 18 个 env（7 个 env 8 样本全挂）。按 env 集中、而不是随机分布，本身就说明是 harness 或镜像问题，不是模型或抖动。
- 当时的故障涉及 wrapper 的 Python 运行时依赖，以及非流式请求超过网关时限。

### 3.2 修复验证（`cth-tb21-harnesserr@1.0.0`）

把上面所有出过 HARNESS_FAILED 的 env（23 个，env 版本和 reward 与 TB2.1 一致）单独做成 benchmark，每题 8 样本，共 184 task。用小集合复现问题、对照验证修复，比重跑全量快一个数量级。

| harness × 模型 | job_id | 完成 | HARNESS_FAILED | acc |
|---|---|---|---|---|
| `terminus-2-tb21@1.0.12` × xhigh（对照） | `cth-fixchk-terminus-base-2610171923` | 72（已停） | 22 | 0.833* |
| **`cth-terminus-2-stream@1.0.0` × xhigh** | `cth-fixchk-terminus-stream-2610180739` | **184** | **0** | **0.799** |
| `terminus-2-tb21@1.0.12` × capped 2000/2000 | `cth-fixchk-terminus-cap2k-2610172201` | 184 | 0 | 0.598 |
| `cth-claude-code-swe@1.0.3` × xhigh | `cth-fixchk-claude-native-2610171922` | 182 | 2（进程被 kill） | 0.637 |

\* 只算幸存样本，偏高。最容易 502 的 4 个 env（adaptive-rejection-sampler、circuit-fibsqrt、dna-assembly、cancel-async-tasks）：对照 30 个结束里 18 个失败；stream 版 32/32 完成。

结论：**流式 Terminus 不降推理预算即可消除 harness error**；限推理预算（capped）虽然也能消除，但均分掉 0.2，已废弃。TB2.1 全量用流式版重跑尚未进行。

## 4. MiMo-SWE 批次（当时中止，原因已查明）

- 目标：50 题 MiMo-SWE（镜像 `xiaomimimo__mimo-v2.6-rl-oss`），三个 harness 对比。
- 已完成：50 个 env 全部发布（45 个新注册）、benchmark spec、模型 `@1.0.2/@1.0.3`。
- 当时放弃的原因：沙箱启动后约 30 s 日志停更，task 卡满 3 h 20 m。当时判断为"容器消失 / 镜像问题"。**事后确认是误判**：真正原因是镜像里 `arca_envd` 没启动（`:11452` 无监听），日志断流只是因为平台读日志也要经过它。原平台运维指南未随仓库分发。
- 现状：Env 加 setup hook 或用 `terminus-2-mimo@1.0.1` 后能正常出分（003048 得 1 分）。三个 harness 在 MiMo-SWE 上的对比尚未重跑。
- 资产：`opencode/mimo_001457/`（`make_task.py`、`tasks/`、`bench50-spec.json`、`status50.sh`）。

## 5. 推理服务性能（Qwen3.8-27B，SGLang vs vLLM）

完整报告：[推理性能报告](teacher_serving_performance.md)（2026-08-26，旧镜像 `glm-5-2-2fbb8e.sif`，**不代表**当前 `deepep-base-85b53914.sif`）。要点：

- 240 case 全部有结果：204 成功，24 KV 容量不支持（SGLang T1D8 MTP-on），12 实际失败（vLLM MTP-on，CUDA illegal memory access）。
- SGLang prefill/TTFT 比 vLLM 慢约 8%，但 T2D4 下 decode TPOT 更快（MTP-on −13.7%）。
- T2D4 比 T1D8 更适合长上下文：TTFT 改善 17–31%，输入吞吐 +29–50%。
- MTP 主要改善 decode TPOT，不提高输入吞吐；收益依赖并行策略和上下文长度。
- 这也是当前 slurm 档案选 **tp2 × dp4 + EAGLE** 的依据。
