# SFT 学生模型评测结果（Terminal-Bench 2.1）

> **读数顺序**　9B mask_const c312 于 11-12 在本地 1.6.262 留有 81/178 的快照；11-14 作废部分旧 setup 样本，此值已撤回，缺少修正后的终值。4B c312 在 Arena mirror＋1.0.0 为 73/178（41.0%）。旧 verifier 下的 4B base 与 4B c312 才可直接对照。各实验条件和数据可用性见[实验索引](README.md)及[本地运行记录](local_tb21_runs.md)。
>
> 归档的 1.0.0 配置为 T=1、top_p=0.95、top_k=50、reasoning_effort=max，未设置 max_turns。早期报告另记 top_k=20/xhigh，各次请求的实际生效参数需查原始请求或服务日志。[归档配置](../arena_assets/harness/configs/1.0.0.env)

Qwen3.5-4B / 9B 在 TerminalHorizon 系列数据上做 SFT 后，在 TB2.1 上的全部分数与消融结论（11-07 离线整理 + 10-21 → 11-03 的逐系分析）：基线、th3k 数据系消融（bad-action / 超长 / packing / lr / epoch / force-JSON / tree）、其它数据系（dsv41_history、tree_flash、qwen38_flash、arm A）各自为什么低、题级分析、TB4、教师采集 job、已知问题与可信度。**"为什么非 th3k 数据都输"的轨迹级解剖在 [trajectory_failures](trajectory_failures.md)**，本文只给分数和每个数据系的症状摘要。

评测口径的变化要记住：**10-31 前所有 TB2.1 job 是旧口径**（`tb21-terminus2-full@2.1.3` + 官方 reward），每 job 7–16% 任务 pytest 没跑就记 0，绝对值偏低且随网络漂移；变体间优先采用同口径配对，环境故障仍可能影响横比。11-05 起切到 mirror 口径（`cth-tb21-full89-mirror@1.0.1`），th3k 最佳 0.348 → 0.410。两个口径的数字不能放在一张表里比大小。详见 [tb21_verifier_audit](../methods/tb21_verifier_audit.md) §2、§7。base 模型的 compaction 触发点扫描在 [context_sweeps](context_sweeps.md) §B8。

---

> **数据来源**：Arena 平台自 11-06 16:00 起不可达，全部数字来自本地快照
> `/storage/openpsi/users/chen/output/<job>/job.json`（puller 最后一次 `arena job get` 的结果；276 个 job 可用），
> 汇总表 `/storage/openpsi/users/chen/output/_summary/all_jobs.{tsv,json}`（生成脚本 `coding_eval/tools/offline_results.py`）。
> 快照时间 ≠ 当前时间：标注 RUNNING 的 job 是"快照时仍在跑"，其后的变化不在本文内。
>
> **口径**：pass@all = 通过样本数 / 全部样本数（HARNESS_FAILED、TIMEOUT 等非 DONE 记 0）；TTL = agent 用满 4 h 被杀；
> ×2 = 89 题 × 2 样本 = 178；×3 = 267。早期服务记录的 Qwen3.5 设置：sglang 85b53914、tp1、SSM fp32、EAGLE MTP、T=1/top_p .95/top_k 20。

---

## 0. 一页结论

| | 结论 | 置信度 |
|---|---|---|
| Arena mirror 已评的最佳 4B | **mask_const c312**（th3k 整轨迹，ep8）：TB2.1 **0.348**（旧 verifier）/ **0.410**（修复 verifier 后重测，harness **1.0.0**） | 两个 verifier 口径，各有一次测量。11-10 起在本地 tb_local 上用 1.6.96（96k compaction）与 1.6.262（不 compact）各测一臂：0.365 / 0.373，见 §2.2b 与 [tb21_verifier_audit](../methods/tb21_verifier_audit.md) §6 |
| 同期 Arena mirror 4B 对照 | **dsv41j arm A c224**（dsv41 整轨迹 JSON-mode，ep7）：**0.326**（修复 verifier） | 单次；171 DONE，6 HARNESS_FAILED、1 TIMEOUT，分母 178 |
| 此批早期 4B 数据系 | 全部 ≤ 0.26，不如 th3k | 高（每系 ≥3 个点） |
| TB4.0 | 仅有少量已评分样本：4B 模型 0/12，API 模型 1/25、3/24，缺少可比较的全量分数 | 低（25% 完成，被中断） |
| 老 verifier 的系统性低估 | 7–14% 的失败任务 pytest 根本没启动（pypi 超时 / 无 WORKDIR）；修复后 mask c312 +5 题确证归因，其余 +6 题在噪声内 | 高 |
| 评测噪声 | 同模型同配置 ×2=178 两次：4B base 0.185/0.191，9B base 0.287/0.247 → **±0.03–0.04** | 高 |

---

## 1. 基线模型（TB2.1）

| 模型 | harness | ×N | pass@all | TTL | 备注 |
|---|---|---|---|---|---|
| Qwen3.5-4B（bf16 SSM） | 1.0.0 | ×3 | 0.150 | 23 | **作废**：SSM bf16 掉分，之后全部改 fp32 |
| **Qwen3.5-4B（fp32 SSM）** | 1.0.0 | ×3 | **0.187** | 10 | 4B 基线 |
| Qwen3.5-4B | 1.0.0-json（已删） | ×3 | 0.165 | 1 | force-JSON 无收益 |
| Qwen3.5-4B | 1.2.{64,96,128,256} r1 | ×2 | 0.163 / 0.185 / 0.180 / 0.180 | | 压缩点扫描 |
| Qwen3.5-4B | 1.2.{64,96,128,256} r2 | ×2 | 0.180 / 0.191 / 0.185 / 0.157 | | 同上重复，**噪声 ±0.03** |
| Qwen3.5-4B patchtpl | 1.0.0 | ×3 | 0.199 | 17 | 模板补丁，无显著差异 |
| Qwen3.5-4B | 1.2.0–1.2.4（dshmin 系列） | ×3 | 0.165（1.2.4 完成） | 8 | 另一流程的 harness 迭代 |
| **Qwen3.5-9B** | 1.0.0 | ×3 | **0.247** | 18 | 9B 基线 |
| Qwen3.5-9B | 1.2.{64,96,128,256} r1 | ×2 | 0.225 / 0.286 / 0.253 / 0.230 | | |
| Qwen3.5-9B | 1.2.{64,96,128,256} r2 | ×2 | 0.225 / 0.247 / 0.264 / 0.253 | | 96k 两次 0.287/0.247 → ±0.04 |
| Qwen3.5-9B patchtpl | 1.0.0 | ×3 | 0.187 | 14 | 低于 1.0.0 基线，原因未查 |
| Qwen3.5-9B | 1.2.4 | ×3 | 0.221 | 5 | |

**基线结论**：4B ≈ 0.19、9B ≈ 0.25；压缩触发点对 base 影响在噪声内（96k 略优）。

### 1.1 API 教师模型（TB2.1，作为上限参考）

| 模型 | harness | ×N | pass@all | 备注 |
|---|---|---|---|---|
| Qwen3.8 Flash xhigh | 1.0.0 | ×3 | **0.771** | 10-19，267/267 DONE |
| DeepSeek V4.1 Flash max | 1.2.96 | ×2 | 0.753 | |
| DeepSeek V4.1 Flash xhigh | 1.2.96-xhigh | ×2 | 0.714 | effort 从 max 降到 xhigh −0.04 |
| DeepSeek V4.1 Flash max | 1.4.96（tool-call 模式） | ×2 | 0.747 | 历史推理可见，无差异 |
| DeepSeek V4.1 Flash max | 1.3.0（1M 窗口 = 不压缩） | ×2 | 0.730 | 不压缩不加分 |
| DeepSeek V4.1 Flash max | 1.0.0 | ×2 | 0.719（171/178 DONE 时） | |
| **DeepSeek V4.1 Flash max** | **1.5.96 + 修复 verifier（cth-tb21-full89-mirror）** | ×2 | **0.832** | **verifier 修复后 +0.08**：pytest 启动失败 1/178（旧 benchmark 上 7–14%） |
| Qwen3.8-27B router xhigh | 1.0.12 | ×1 | 0.494 | 10-14，另一流程 |

---

### 1.2 base 与 SFT 横比的注意事项

- **SFT 4B（0.287–0.348 @1.0.0）> 9B base 最优档（0.267 @1.2.96）> 4B base 最优档（0.188 @1.2.96）**。即使给 base 最有利的触发点，SFT 4B 仍领先 9B base 3 个点以上。
- base 的压缩触发点扫描（[context_sweeps](context_sweeps.md) §B8）用 1.2.x，SFT 消融用 1.0.0；base 行按 1.2.96 读才是公平值：4B 0.188（vs 1.0.0 的 0.187）、9B 0.267（vs 0.247）。
- 并发 / 部署差异（1 AIS 节点 8 引擎、slurm 2 卡或 4 卡）不影响单题行为（每个 episode 是串行 agent），各行可比。
- 所有 non-DONE 任务都读过 harness 日志：除 1 条网关 503 外全是模型侧（episode 1770 s 被 SIGTERM、模型发 `C-d` 退 shell、TIMEOUT、上下文打满后断流），集中在同一批题（git-multibranch、install-windows-3.11、configure-git-webserver、rstan-to-pystan、make-mips-interpreter），不重跑。


## 2. SFT 消融：th3k 数据系（TerminalHorizon-3K，2441 条整轨迹，DeepSeek-V4-Pro 教师）

训练配方（全系一致）：Qwen3.5-4B-text、full SFT、lr 1e-5、AdamW、bf16、ZeRO-3、cutoff 262k、64 轨迹/步、bad-pair mask。

### 2.1 第一轮：bad-action / 超长 / 采样 / lr 消融（harness 1.0.0，×2；10-21 → 10-26）

两组实验同一 benchmark `tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2`（89 题），同一套 Qwen3.5 serving（sglang 85b53914、tp1 引擎、SSM fp32、EAGLE MTP、T=1 / top_p 0.95 / top_k 20 / xhigh、ctx 262144）。口径：**acc(all) = pass / 全部任务，失败计 0**。每题 2 次采样时一个任务 = 0.0056，<0.02 的差距不可分辨。

| # | 变体 | 改动 | acc(all) | pass | 解出 ≥1 次 / 89 | 2/2 都解出 | 单题 agent 分钟 |
|---|---|---|---|---|---|---|---|
| — | Qwen3.5-4B base | — | 0.187 | 50/267 | 30 | 4（of 3） | 47 |
| — | Qwen3.5-9B base | — | 0.247 | 66/267 | 37 | 8（of 3） | 38 |
| D1 | mask_const | lr 1e-5 常数（warmup 10%），3 epoch；续训到 8 epoch 见 §2.2 | 0.298 | 53 | 37 | 16 | 109 |
| A1 | base | 丢掉含坏动作的轨迹（2201 条） | 0.287 | 51 | 33 | **18** | 114 |
| A2 | mask | 坏动作留在上下文、loss=0（2441 条）— 默认 | 0.287 | 51 | 35 | 16 | 114 |
| D2 | mask_lr2e5 | lr 2e-5 cosine | 0.287 | 51 | 35 | 16 | 119 |
| D3 | mask_lr5e6 | lr 5e-6 cosine | 0.270 | 48 | 33 | 15 | 138 |
| A3 | keep | 坏动作也监督 | 0.258 | 46 | 33 | 13 | 126 |
| B2 | mask_cut | 超长轨迹 256K 前缀截断 | 0.242 | 43 | 30 | 13 | 132 |
| C2 | mask_rand | 随机采样器（非 group_by_length） | 0.242 | 43 | 28 | 15 | 119 |
| B3 | mask_tail | 超长轨迹 = 任务 prompt + 最长后缀 | 0.236 | 42 | 29 | 13 | 118 |
| C3 | mask_pack | neat packing（63 步） | 0.236 | 42 | 30 | 12 | 123 |
| A1 | base ckpt-70 | | 0.242 | 43 | 32 | 11 | 90 |
| A2 | mask ckpt-70 | | 0.236 | 42 | 30 | 12 | 87 |
| A3 | keep ckpt-70 | | 0.236 | 42 | 29 | 13 | 111 |

**结论**

1. **SFT 有效**：4B 0.187 → 0.287–0.298，超过 9B base（0.247）。提升主要来自**稳定性**：SFT 模型 2/2 都解出的题 13–18 个，
   base 4B/9B 只有 4/8。代价是单题时长 2.5–3×（SFT 模型每轮看到自己全部历史推理，上下文增速 2.7×，见 [harness_comparison](../methods/harness_comparison.md) §A3）。
2. **Stage A 坏动作**：mask = base（0.287）> keep（0.258）。监督坏输出扣 5 题；mask 多保留的 240 条轨迹没有可测收益。mask 仍为默认。
3. **Stage B 超长轨迹**：cut 0.242 / tail 0.236，都比 mask 低 8–9 题。**超长轨迹直接丢**，截断或开窗都有害。
4. **Stage C 采样/packing**：rand 0.242 / pack 0.236，都比 mask 低 8–9 题。**保留 group_by_length、不开 packing**。packing 的
   loss 曲线与 C1 一致、collator 隔离已验证，掉分最可能来自 batch 组成变化和优化步数减半（63 vs 117）。
5. **Stage D lr/schedule**：lr 5e-6 明显差（−3 题）；2e-5 = cosine 1e-5；**常数 lr 0.298** 比 cosine 多 2 题（噪声内）但同时并列
   解出题数最多（37）、是单行最优。117 步的 schedule 上 cosine 衰减没有帮助 → 默认改为 `constant_with_warmup`。
   要分辨 const vs mask 的 0.011 需 4–8 samples 重跑这两个。
6. **epoch 越多越好，到 8 还没拐点**：const 续训到 5 / 8 epoch（resume，前段 step 完全一致）：3 ep 0.298 → 4 ep 0.298 → 5 ep 0.320 → 6 ep 0.331 → 7 ep 0.298 → **8 ep 0.348**（62/178，比 3 ep 多 9 题；解出 41 题、2/2 都解出 21 题）。训练 loss（epoch 均值）1.075 → 0.953，harness 分数跟着走；epoch 7 是唯一的凹点（−6 题，夹在两个更好的 checkpoint 之间，噪声量级）。单题时长也从 109–122 min 降到 95–100 min。epoch 1–2 没测。逐 epoch 全表 + 按题矩阵：`/storage/openpsi/users/chen/output/tb21_const_epoch_sweep.md`。
7. **分组**：const / base / mask / lr2e5（0.287–0.298）一组；keep / cut / tail / rand / pack / lr5e6（0.236–0.270）明显更差。
   训练 loss（1.09–1.15）对这个排序没有任何预测力。

按题的全表、job id、失败审计：`/storage/openpsi/users/chen/output/tb21_ablation_summary.md`；训练侧：`coding_analysis/experiments/sft_results.md` §5。

**11-07 离线复核对上面结论的两点修正**：

- 评测噪声实测 ±0.03–0.04（同模型同配置重复测量，§0），折合 178 任务里 5–7 题。上面"低 8–9 题"的 cut / tail / rand / pack 只是刚出噪声带，"差 2–5 题"的对比（const vs mask、lr 2e-5 vs 1e-5、keep vs mask）都在噪声内。稳妥的读法是：**3 epoch 下所有变体落在 0.24–0.29，mask 不输任何变体，选为默认**；真正拉开差距的是 epoch 数（§2.2）。
- `mask_pack` 的 0.236 来自 10-26 的 job；11-07 的本地快照里该变体只有 3/178 条记录（标为"跑得很慢，作废"）。两处记录不一致，原因未查，这一行不要引用。

各变体 final 的 TTL 条数（4 h 跑满，178 任务）：base 56、mask 49、keep 72、cut 75、tail 56、rand 58、lr2e5 56、lr5e6 74；ckpt-70：base 46、mask 39、keep 53。

补充两行（历史 think 不可见训练的变体，@1.0.0 / @1.0.1 两个 harness 各测一次）：

| 变体 | pass@all | TTL |
|---|---|---|
| mask_nothink（历史 think 不可见训练） | 0.292 @1.0.0 / 0.242 @1.0.1 | 64 / 62 |
| keep_nothink | 0.242 @1.0.0 / 0.247 @1.0.1 | 72 / 75 |

1.0.1 = usage 计数 + 不回传历史推理（[context_sweeps](context_sweeps.md) §C2）。mask_nothink 换到正确计数后掉 9 题，在噪声边缘。

### 2.2 第二轮：mask_const（constant lr，8 epoch，39 步/epoch），harness 1.0.0，×2

| ckpt | epoch | pass@all | TTL | 状态 |
|---|---|---|---|---|
| c117（= const_final） | 3 | 0.298 | 50 | DONE |
| c156 | 4 | 0.298 | 56 | DONE |
| c195 | 5 | 0.320 | 42 | DONE |
| c234 | 6 | 0.332 | 51 | DONE |
| c273 | 7 | 0.298 | 46 | 快照 RUNNING（2 条未完） |
| **c312** | **8** | **0.348** | 43 | 快照 RUNNING（1 条未完）；终值 62/178 |
| c312 force-JSON | 8 | 0.330（×3） | 91 | harness 1.0.0-json，无收益，已删 |
| **c312 @ 修复 verifier** | 8 | **0.410**（73/178） | 41 | benchmark cth-tb21-full89-mirror@1.0.1，11-05 重测 |

**0.348 → 0.410 的分解**（逐题对比两次 run）：旧 verifier 从未启动 pytest 的题 +5/−0（prove-plus-comm +2、mailman +1、build-pmars +1、compile-compcert +1）；其余题 +15/−9 = +6，属 T=1 重采样噪声。**可归因于 verifier 的只有 +5（0.348→0.376），"0.410" 带 ±0.07 的 n=178 噪声。**

### 2.2b 同族五个点在本地 `tb_local` 口径下的结果（11-10 → 11-12）

原集群 11-12 的历史快照，以当时有效的样本计分。下表的分数在后续题面、测试和沙箱修正前取得。

| 模型 | 臂 | n | pass | acc | TTL | harness ERROR | 题级解出 |
|---|---|---|---|---|---|---|---|
| 4B mask c312 ep8 | 96k compaction（1.6.96） | 170 | 65 | **0.382** | 3 | 0 | 40/88 |
| 4B mask c312 ep8 | 262k 不 compact（1.6.262） | 172 | 66 | **0.384** | 28 | 0 | 41/88 |
| 9B mask c117 ep3 | 262k（1.6.262） | 178 | 72 | **0.404** | 43 | 3 | 45/89 |
| 9B mask c195 ep5 | 262k（1.6.262） | 178 | 79 | **0.444** | 27 | 5 | 48/89 |
| 9B mask c312 ep8 | 262k（1.6.262） | 178 | 81 | **0.455** | 30 | 3 | 46/89 |

11-14 的作废与补测改变了各臂的有效样本集合。4B、9B 的规模差异、96k 对 262k 的比较，需用修正后一致的题面、资源和有效分母重新估计。详见[本地实验账本](local_tb21_runs.md)。

### 2.3 force-JSON（`response_format=json_object`）— epoch 8 模型 + 4B base（10-27 / 10-28）

harness `cth-terminus-2-stream@1.0.0-json`（= 1.0.0 语义 + `TERMINUS_FORCE_JSON_OBJECT=true`，每次请求带 `response_format={"type":"json_object"}`），c312 权重与 4B base 各每题 ×3（267 任务）：

| | pass@all | passed | 解出 ≥1 | 每次都解出 | 单题 agent 分钟 |
|---|---|---|---|---|---|
| plain 1.0.0 ×2 | **0.348** | 62/178 | 41 | 21（2/2） | 98 |
| force-JSON ×3 | **0.330** | 88/267 | 43 | 14（3/3） | 125 |
| 4B base plain 1.0.0 ×3 | **0.187** | 50/267 | 30 | 4（3/3） | 47 |
| 4B base force-JSON ×3 | **0.169** | 45/267 | 29 | 5（3/3） | — |

三次采样 30/29/29，噪声很小。分数只差 0.018，但 episode 结局分布变了：**跑满 4 h TTL 从未宣告完成的比例 24% → 34%**（错误宣告完成 35% → 29% 抵消了一部分），单题 98 → 125 min。原因（266 vs 178 条轨迹）：

1. **约束让模型吐"合法但错误"的 JSON**：977 步（3.9%，88 条轨迹）内容是占位对象 `{"status":"final"}` ×79、`{"parsed_json":true}` ×52、`{"parsed":false}` ×40、`{}` ×31 …——think 里已经写好了下一步命令（938/977 以"Let's run…"结尾），但 `{` 被强制后模型给了个最小对象。harness 回 "Missing required fields"、重发，一步白费。plain 下 26 446 步 0 次解析失败，约束根本没有可修的东西。而且它会自我强化：P(占位 | 上一步占位)=0.43 vs 0.023，最长连续 10 步；含占位的轨迹通过率 16% vs 42%，平均 194 vs 86 min。
2. **xgrammar 约束解码慢 ~30%**：引擎日志 running=1 时单请求 217 vs 313 tok/s（MTP accept len 不变 2.2）；轨迹侧每步 48 vs 98 tok/s，每步墙钟 40 vs 23 s。30 min episode 里回合数减半，直接变成 "跑满 TTL" 那 +9 个点。
3. 压缩子代理也被约束成 JSON，压缩次数 2.49 → 1.29/条，交接文本变短变硬。

正常回合的行为没变（每步命令数 2.39 vs 2.46、按键长度、analysis/plan 长度都一样），丢掉的题（caffe-cifar-10 2/2→0/3 等）不是占位造成的，是跑不完。4B base 同样无收益：单采样 pass 14/11/20（plain 15/16/19），25 题涨跌两向无规律；差别只在非 DONE 更多（14 HF + 1 TIMEOUT vs 4 + 1，都是长题的流式中断，8 次 retry 仍失败），即使按 plain 通过率补回也只多 1–2 题。**结论：base 与 SFT 都不开 force-JSON；harness `1.0.0-json` 已 deprecate + 删除，后续只用 1.0.0。** 全文 `/storage/openpsi/users/chen/output/tb21_c312_force_json.md`。

### 2.4 tree_th3k（同 th3k 数据，tree attention = 历史 think 不可见训练，39 步/epoch）

| ckpt | epoch | @1.0.0 | @1.0.1 nothink | @1.2.96 |
|---|---|---|---|---|
| c117 | 3 | 0.202 | 0.213 / 0.202（两次） | 0.197 |
| c156 | 4 | 0.292 | 0.253 | — |
| c195 | 5 | 0.247 | 0.281 | — |

tree 同 step 比 mask 低 1–17 题；nothink 评测不翻转结论。tree 的失败模式是 0 TTL 但"交卷后通过率 32%"（mask 50%）——学会了快交没学会交前验。

### 2.5 th3k_solved（重放判分后只留 660 条 solved）

训练完成（72 步 = 8 ep × 9），**未评测**（优先级让给其他实验）。

---

## 3. SFT 消融：其他数据系（全部不如 th3k）



### 3.1 dsv41_history（DeepSeek V4.1 Flash tool-call 模式，96k 压缩**段**，历史推理可见）— harness 1.2.96，×2

| 变体 | ep3 | ep5 | ep7 | ep8 | 备注 |
|---|---|---|---|---|---|
| const（全量 11,146 段，42 步/ep） | c126 **0.242** | c210 **0.258** | c294 0.213 | c336 0.219 | 平台 ~0.25，0 TTL |
| rand（随机段） | c126 0.152 | c210 0.157 | c294 0.219 | c336 0.197（修复后 0.264†） | rand≈const，段选法无关 |
| solved（4,228 段，19 步/ep） | c57 0.174 | c95 0.191 | — | c152 0.174 | **solved-only 更差** |
| lin_solved（线性整轨迹 solved，1114 条） | c51 0.174 | — | — | — | **线性化不救 solved** |

† dsvh_rand c336 在修复 verifier 上重测 0.264（177/178 时）。
症状：think 坍缩到 1.1–1.6k 字符/步（mask 4–5k）→ 每步 1.6 条命令 → 步数膨胀 → 28–39 条撞 512 步上限。

**为什么低（11-02 分析）**：

1.2.96 ×2：const ep3 **0.242** / ep5 **0.258**（0 TTL）/ ep7 0.213；rand ep3 0.152 / ep5 0.157 / ep7 0.219。const 三点平台在 0.25，epoch 不是瓶颈。
与 qf 系相反**没有 TTL 问题**（正常交卷、dur 2.1 s），输在 think 坍缩（1052 字符/步 vs mask 4114）+ 每步 1.6 条命令 → 步数 152 vs 92、
30 条撞 512 上限。根因在数据：dsv41_history 的 think p50 只有 511 字符（th3k 2769）、76% 的段无交卷尾、const 臂 62% 监督 token 来自失败轨迹。
rand≈const 说明段选法无关。统一图景：**分数排序 = 数据"每步推理长度 × 交卷结构完整性"**，教师绝对能力次要（dsv41 本尊 0.48 → 蒸馏 0.26）。
全文 `/storage/openpsi/users/chen/output/tb21_dsv41_history_analysis.md`。

### 3.1b dsv41 数据的三种筛选：不 judge 全量 / 不 judge 剔除未解决 / judge 全量（对比总表 + 评测口径 + 变量控制，11-10）

同一份 dsv41 tool-mode 采集，三种数据筛选方式训出来的 4B 模型放在一起比。**三组的评测口径不同**，表里单列一栏，口径之间的换算放在表后。先把三段设定写全：

**数据怎么来的（三组同源）**：教师 = DashScope `deepseek-v4.1-flash-1017`（effort max），在 TerminalHorizon 1,486 题 ×2 上采集，job `cth-thfull14-dsv41-2610270211`，采集 harness `cth-terminus-2-stream@1.4.96`（96k compaction 触发、max_turns 512、usage 计数、**tool-call 模式**即历史推理可见、8Gi、无 pypi 镜像）。轨迹按 compaction 边界切段（`build_multi_sft.py`：段 0 = 任务 prompt → 首次压缩，段 i = 任务 prompt + handoff 摘要 → 下次压缩），一段 = 一个训练样本，段长被 96k 触发点封顶（max ~140k token），`--overlong drop` 实际 0 丢弃。三组的差别只在选哪些轨迹/步：

| | A `dsv41_history` | B `dsv41_history_solved` | C `dsv41_judged` |
|---|---|---|---|
| 轨迹来源 | th14 全部 2,690 DONE | 同左，只留 score=1 的 1,253 条 | th14 2,365 + **redo14 133 + thredopol 43**（重打轨迹用 1.7.96 = 1.4.96 语义 + 16Gi + 镜像采集）；剔 101 条环境污染 + Opus 4.8 judge DROP 53 + OVERLONG 196 未评 |
| 步级处理 | 原样 | 原样 | **另剔 13,737 个静默失败步**（tool JSON 非法、harness 没执行）+ 65 个 parse-error 步 |
| 规模 | 11,146 段 | 4,228 段 | 8,876 段 / 2,541 轨迹（1,251 solved） |
| 文件 | `data/terminal_horizon_dsv41_history/` | `…_history_solved/` | `…_judged/`（源 `code_sft_data/dsv41_judged/`） |

**怎么训的（三组 + th3k 参照同配方，`*_const_e8` 系；只有 batch 顺序不同）**：Qwen3.5-4B-text（C 另有 9B）全参 SFT；窗口 262,144、一段一行、不 packing、不截断；loss = 每个 assistant 轮全部监督（`<think>` + JSON 动作），非法动作轮留在上下文但 loss=0（bad-pair mask）；有效 batch = 64 条轨迹。**batch 顺序有两套实现**：A/B/C（段数据）走自定义 `build_traj_sched.py`，一条轨迹的所有段固定在同一 optimizer step，A/B/C 用 `--order global_sort`（整轨迹 token 全局降序切 64，无 RNG），A' 用 `--order random`（每 epoch 轨迹级随机置换）；**R1 th3k 参照（整轨迹，一行一条）走 HF Trainer 原生 `group_by_length`**（`LengthGroupedSampler`，**每 epoch 重新 randperm** 后每 200 行局部降序，accelerate 按 64 行一步喂），不是全局排序，细节见 A' 注释末尾；lr 1e-5 `constant_with_warmup`（10%）、AdamW(0.9, 0.95)、wd 0.1、clip 1、bf16、ZeRO-3、FA2、liger、seed 42；8 epoch。步数：A 42 步/ep ×8=336，B 19 步/ep，C 39 步/ep ×8=312（4B AIS 2×8 卡 ga=40；9B 4×8 卡 ga=20）。checkpoint 在 `checkpoints/abl_dsv41_{history,history_solved,judged}_const_e8_*`。

**怎么评的**：89 题 TB2.1 ×2 采样 = 178，pass@all（非 DONE 记 0）。serving 全部 sglang（tp1、SSM fp32、EAGLE MTP、T=1 / top_p 0.95 / top_k 20 / xhigh、ctx 262144）。三组用的口径与 harness **不同**（这是本节最大的坑）：

| 组 | 平台 | benchmark + reward | 评测 harness | 说明 |
|---|---|---|---|---|
| A、B（及 A' 旧分） | arena | **[旧]** `tb21-terminus2-full@2.1.3` + 官方 `tb2-reward@4.0.18`（7–16% 任务 pytest 没跑记 0） | `1.2.96`（96k / 512 步 / usage / **8Gi 无镜像**） | 10-31 至 11-04 |
| A' c336 重测 | arena | **[镜]** `cth-tb21-full89-mirror@1.0.1` + `cth-tb2-reward-mirror@1.0.1` | `1.2.96` | 11-06 |
| C（4B / 9B）、th3k 本地参照 | **tb_local**（slurm singularity） | **[本]** 同一份 mirror reward 脚本，本地 runner | `1.6.96` spec（96k / 512 步 / **16Gi + pypi 镜像**），effort max | 11-09 起；C 的 132/136 条是修复前 v1 沙箱契约 |

#### 主表（TB2.1，89 题 × 2 样本 = 178，pass@all；底座 Qwen3.5-4B，另注 9B）

| 行 | 筛选方式 | 训练数据 | 轨迹 / 段 | solved | ep3 | ep5 | ep7 | ep8 | 口径 | 评测 harness |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 不 judge · 全量 | `dsv41_history` (const) | 2,690 / 11,146 | 47% | 0.242 | **0.258** | 0.213 | 0.219 | [旧] | 1.2.96 |
| A' | = A，采样器 rand | `dsv41_history` (rand) | 同 A | 47% | 0.152 | 0.157 | 0.219 | 0.197 [旧] · **0.264 [镜]** | [旧] + [镜] | 1.2.96 |
| **B** | 不 judge · 剔除未解决 | `dsv41_history_solved` | 1,253 / 4,228 | 100% | 0.174 | 0.191 | — | 0.174 | [旧] | 1.2.96 |
| B' | = B，不切段 | `dsv41_hist_lin_solved` | 1,114 / — | 100% | 0.174 | — | — | — | [旧] | 1.2.96 |
| **C** | judge · 全量 | `dsv41_judged` | 2,541 / 8,876 | 49% | — | — | — | **0.298** | [本] | 1.6.96 |
| C-9B | = C，底座 9B | `dsv41_judged` | 同 C | 49% | — | — | — | **0.343** | [本] | 1.6.96 |
| D | judge · 剔除未解决 | `dsv41_judged-solved` | 数据已生成 | 100% | 未训练 | | | | — | — |
| R1 | 参照：th3k | `th3k` 2,441 整轨迹 | 2,441 / — | — | | | | 0.348 [旧] · 0.410 [镜] · 0.354 [本] | 三口径 | 1.0.0 / 1.0.0 / 1.6.96 |
| R2 | 参照：dsv41 chat 整轨迹 | `dsv41_json262k` (arm A) | 2,037 / — | 60% | 0.219 | — | 0.326 | 未评 | [镜] | 1.0.0 |

#### 注释

**行**

- **A 不 judge · 全量**：`cth-thfull14-dsv41` 的 2,690 条 DONE 轨迹全部进训练，不做任何质量筛选，按 compaction 边界切成 11,146 段。这是 dsv41 段数据的基线。
- **A'**：与 A 同一份数据（同 11,146 段、同 tokenized cache）、同配方，**唯一差别是训练时哪 64 条轨迹放进同一个 optimizer step**（`build_traj_sched.py --order`）。先说共同点：训练样本的最小单位是"段"（一条轨迹按 compaction 边界切成 1–14 段，一段一行），但 batch 的单位是"整条轨迹"——脚本先把 cache 里属于同一 (job, env, sample) 的所有段归到一起，再以轨迹为单位分组，**每个 step 恰好 64 条完整轨迹、一条轨迹的所有段一定在同一 step**；这一点 A 和 A' 都一样，A' 并不会把同一轨迹的段打散。差别只在"哪 64 条轨迹同一步"：
  - **A = `global_sort`**：把 2,690 条轨迹按整轨迹 token 数降序排一次，顺序切 64 条一步，**每个 epoch 顺序完全相同、无随机数**。每一步是一个很窄的长度桶（步内最长/最短轨迹之比 p50 1.1），第 0 步 = 64 条最长的轨迹（62.6M token、960 行），最后一步 = 64 条最短的（1.7M token）。rows/step 最大 960 → `ga = 60`，slot 填充率 28%（大量 dummy 行）。好处：确定性、OOM 第一步就暴露、每步 loss 对应一个长度桶（后来 Loss-Length Replication 的探针用的就是它）；坏处：step 内长度高度同质，epoch 间无 shuffle。
  - **A' = `random`**：每个 epoch 用 `seed+epoch` 做一次**轨迹级**随机置换，再顺序切 64 条一步。步内长度混杂（最长/最短之比 p50 37.9，最大 375），每步 token 均匀在 ~14M，rows/step 最大 352 → `ga = 22`，填充率 75%。epoch 间顺序不同。
  - 两者尾处理相同（`--tail drop`：2,690 % 64 = 2 条最短轨迹每 epoch 丢弃），都是 42 步/epoch × 8。
  - **一条轨迹走一遍**（数字取自两份 schedule 的 `manifest.json`）：`cth-th-adapt-a-battlecode-bot-across-a-replayable-opponent-pool` s1，score=1，压缩 3 次 → 4 段，prep 后是连续 4 行：段 0（任务 prompt → 第 1 次压缩，102 条消息 / 356k 字符）、段 1（prompt + handoff 1 → 第 2 次压缩，112 / 239k）、段 2（prompt + handoff 2 → 第 3 次压缩，4 / 12k）、段 3（prompt + handoff 3 → task_complete，50 / 96k），合计 107,644 token = 这条轨迹的"长度"。脚本按 (job, env, sample) 把这 4 行归成 trajectory #13，之后一切分组都针对 #13，**4 行永远同进同出一个 step**。
    - A gsort：#13 按长度排在第 28×64 附近，**8 个 epoch 都在第 28 步**，同步的 63 条邻居长度全在 103k–109k，这一步 158 行；G=60 → 960 slot，填 802 个 dummy 行。
    - A' random：epoch 0 第 3 步、epoch 1 第 34 步、epoch 2 第 32 步……每轮邻居都换；epoch 0 那一步邻居长度 30k–565k，207 行；G=22 → 352 slot，填 145 个 dummy。
    - step 内的行都按行长度降序填进 16 个 rank 的 slot（rank = j % 16，micro-batch = j // 16），不够的 slot 填 8-token dummy（labels 全 −100，对 loss 和 num_items_in_batch 贡献 0）。
  - **速度差**（`trainer_log.jsonl`，稳定的 epoch 1，42 步）：gsort 144 min / epoch，rand **121 min**，rand 快 ~16%（8 epoch 省 ~3 h）。gsort 的 sec/step 从 692（64 条最长轨迹，62M token，960 行）单调降到 76，吞吐 22k–90k tok/s；rand 每步 10–17M token，135–246 s，吞吐稳定 71k–87k。两个原因：① 长序列集中在一步，ZeRO-3 下每 rank 串行 60 个 ~262k 的 micro-batch，attention 二次开销被放大；② gsort 填充率 28%（G 按最长步定 60，p50 步只 199 行），dummy 行仍要走 forward/backward 和 rank 同步。两 run 在不同集群（AIS vs slurm），绝对秒数有硬件差，但 epoch 内吞吐随 tokens/step 变化的形态是 schedule 决定的。gsort 仍是默认，因为确定性、第 0 步暴露 OOM、每步 loss 对应一个长度桶可做探针。
  - **第三种 batch 方式：th3k 参照 R1 用的 HF `group_by_length`，loss 曲线有周期 3 的锯齿**。`abl_mask_const_e8_slurm2n`（th3k 整轨迹、一行一条，不走 traj schedule）用 HF Trainer 自带的 `LengthGroupedSampler(batch_size = per_device_bs × ga = 1 × 4 = 4)`：先全局 randperm，再切成 `50 × 4 = 200` 行的 mega-batch，每个 mega-batch 内**按长度降序**；accelerate 再把这串顺序按 64 行一步（16 卡 × ga 4）喂给 optimizer。200 行 / 64 行 ≈ 3.1 步，所以 tokens/step 呈**周期 3 的锯齿**（epoch 0：12.7M → 8.2M → 5.1M → 11.5M → 8.1M → 5.4M …；用 cache 的行长度 + 同一 seed 模拟得到 12.0 / 7.8 / 4.9 / 12.0 / 8.6 / 5.7，完全复现），每 ~19 步因为 200 ≠ 64×3 而漂一格（跳升点在 3,6,9,12,**16**,19,22,25,28,31,34）。loss 残差与 tokens/step 的相关 0.40，谱峰 3.0–3.2 步；长步（11M）的 loss 比短步（5.5M）高 0.07（1.083 vs 1.004）——长轨迹本身 loss 高，不是训练不稳。**注意：randperm 每个 epoch 重做一次**（transformers 4.42 的 `LengthGroupedSampler` 以 `generator=None` 构造，每次 `__iter__` 从全局 RNG 取新置换）。8 个 epoch 的 tokens/step 序列相关 0.97–0.99 是排序结构造成的（200 行降序 + 周期 3），**不是同一批行**：同一位置的 tokens/step 跨 epoch 平均差 0.33M，用两个不同 seed 的 randperm 模拟得 0.34M、同一步的行重合率仅 7%；loss 残差（去长度效应后）跨 epoch 相关只有 0.53–0.79，randfix 是 0.98–1.00。所以 R1 实际上是"每 epoch 重洗 + 局部长度分组"，和 A' random 同属重洗类；gsort / randfix 是固定顺序类。11-11 之前的文档把 R1 误记为"一次 randperm"，已更正。
  - **四种 batch 顺序哪个更好（11-11 结论）**。分数上 gsort vs random 噪声内（0.219 vs 0.197 旧口径），所以看训练动力学和成本。grad norm（warmup 后 step>40，spike = >3× 中位数）：gsort **20 次 spike、max 4.1、中位 0.71**，全部落在每个 epoch 的头 3 步（64 条最长轨迹，62M/47M/40M token）和尾 3 步（最短，3.7M/3.1M/1.7M）；random **0 次、max 1.27、中位 0.45**；HF group_by_length（R1）0 次、max 1.68；judged gsort 7 次、max 2.9。gsort 的 epoch 内 token 量从 62M 单调降到 1.7M，等于每个 epoch 做一次"长→短"的隐式课程再从头来，epoch 边界处分布突变是 spike 的来源。速度：random 比 gsort 快 ~16%（填充率 75% vs 28%）。五篇参考论文（TMax、OpenThoughts-Agent、FACET、Terminal-Universe、TerminalHorizon）没有一篇用长度排序或课程；唯一讨论顺序的 OpenThoughts-Agent 关 inter-epoch shuffle（`disable_shuffling=True`）且发现随机 vs 顺序呈现无差。**结论（11-11 晚修正）：生产用 `random`（每 epoch 重洗）。** 平坦 token 量的好处（无边界突变、grad norm 稳、比 gsort 快 16%）random 和 random_fixed 都有；但 randfix 训完后发现**固定顺序的 run 后期 loss 下降持续变慢**（ep7→8 每 epoch 降 0.016 vs 重洗类 0.023–0.025，gsort 更差到 0.001），见 §3.1c "randfix 为什么比 R1 高"。random_fixed 只在需要每步 loss 跨 epoch 严格可比时用（探针）；gsort 保留给 LLR 探针（每步 = 一个长度桶）。HF group_by_length 本质是重洗 + 局部分组，效果上和 random 一类，周期 3 锯齿无害但没有理由保留。
  - 为什么要有 A'：当时怀疑 gsort 的"同长度桶一步"会让梯度在 epoch 内系统性地从长轨迹走向短轨迹；结果旧口径 ep7/ep8 两者 0.219/0.197 vs 0.213/0.219，噪声内（§3.1 "rand≈const"）。A' 单独列一行只是因为 **A 组里只有 A' 的 c336 在 mirror 口径重测过**（0.264），后文口径换算用它代表 A。
- **B 不 judge · 剔除未解决**：A 的数据只留 reward=1 的 1,253 条轨迹（4,228 段），其余全部丢掉。回答"失败轨迹是不是噪声"。
- **B'**：B 的数据不切段，整条轨迹线性拼接（`build_multi_sft.py --linear`，compaction handoff 内联成一条对话），1,114 条 ≤262k 进训练。回答"solved-only 差是不是切段造成的"。只评了 ep3（c51）。
- **C judge · 全量**：A 的源轨迹经过完整清洗流水线——① 并入 176 条修好坏 env 后用 1.7.96 重打的轨迹（redo14 133 + thredopol 43）；② 剔 101 条含环境污染（eperm-net / oom / pip-net / gave-up）的轨迹；③ Opus 4.8 按 TerminalHorizon 三准则盲评，DROP 53（泄漏/篡改/纯浏览），OVERLONG 196 未评不进；④ 段化时另剔 13,737 个"tool JSON 非法、harness 没执行"的静默失败步。得 2,541 条 / 8,876 段，**solved 和失败都保留**（49% solved）。注意 C 和 A 的差不只是 judge，①②④ 也捆在一起（见变量控制表）。
- **C-9B**：C 同一份数据，底座换 Qwen3.5-9B（4 节点 ga=20，其余配方同）。174/178 done，按 178 计。
- **D judge · 剔除未解决**：C 的数据只留 solved。`dsv41_judged-solved.jsonl` 已生成，**未训练**，是 2×2 矩阵缺的最后一格。
- **R1 参照 th3k**：公开 TerminalHorizon-3K（DeepSeek-V4-Pro 教师，整轨迹不切段，2,441 条进训练），同配方 `mask_const` ep8。三个口径都有分，是口径换算的锚。
- **R2 参照 arm A**：dsv41 **chat-mode** 采集（非 tool-mode）的 2,037 条整轨迹，同配方。与 A/B/C 的差别是采集模式 + 不切段，列出来是为了看"整轨迹 chat 数据"的上限（mirror ep7 0.326）。

**列**

- **轨迹 / 段**：段 = 按 compaction 边界切出的训练样本，一段一行；"—" 表示整轨迹训练不切段。
- **solved**：训练数据里 reward=1 轨迹的占比。
- **ep3 / ep5 / ep7 / ep8**：checkpoint 对应关系——A/A' 42 步/epoch（c126 / c210 / c294 / c336），B 19 步/epoch（c57 / c95 / — / c152），C 39 步/epoch（c312 = ep8），R1 39 步/epoch（c312），R2 31 步/epoch（c96 ≈ ep3，c224 ≈ ep7）。加粗 = 该行最高。
- **口径**：
  - **[旧]** = arena，benchmark `tb21-terminus2-full@2.1.3` + 官方 `tb2-reward@4.0.18`。reward 阶段 `curl astral.sh | sh` 装 uv 和拉 pytest 走公网常超时，每 job 7–16% 任务 pytest 没跑就记 0，**系统性低估 3–8 pt**。10-31 前所有 job 都是这个口径。
  - **[镜]** = arena，benchmark `cth-tb21-full89-mirror@1.0.1` + `cth-tb2-reward-mirror@1.0.1`（uv 预装、pypi 走 aliyun、WORKDIR 回退；`test.sh` 不动）。**正式口径**。
  - **[本]** = tb_local，slurm 池节点上 singularity 复刻 arena（同一份 harness `app.tar.gz`、同一份 mirror reward 脚本、4 核 / 16Gi / 14400 s）。比 [镜] 系统性低 3–5 pt；C / C-9B 的 132 / 136 条是 11-09 修复前的 v1 沙箱契约（oracle 提前可见、`/tests` 只读、`/tmp` tmpfs、PROMPT_COMMAND、clang），净方向压低。详见 [tb21_verifier_audit](../methods/tb21_verifier_audit.md) §8。
- **评测 harness**：`1.2.96` = 96k compaction / 512 步 / usage 计数 / **8Gi、无 pypi 镜像**；`1.6.96` = 同窗口语义 + **16Gi + pypi/HF 镜像注入 agent shell**；`1.0.0` = 262k 窗口 / 无 max_turns / 本地计数 / 8Gi。A/B 用 1.2.96 意味着 agent 侧还可能吃到 pip 超时和 OOM，C 用 1.6.96 没有——这是口径之外又一个对 A/B 不利的变量。

**每行的失败形态**（为什么分数低）

| 行 | TTL（4 h） | 撞 512 步 | 未宣告完成 | 主要死法 |
|---|---|---|---|---|
| A / A' | 0 | 28–39 | — | think 坍缩到 1.1–1.6k 字符/步 → 每步 1.6 条命令 → 步数膨胀 |
| B / B' | 0 | — | — | 同 A，且训练量只有 A 的 38% |
| C（4B） | 0 | **70** | **44%** | 与 A 同形态，步数 p50 304；judge 只剔 2% 轨迹，没改变 tool-mode 段数据"think 短 + 段无交卷尾"的分布 |
| C-9B | 0 | 52 | 32% | 同上，程度轻 |
| R1 th3k | 42 | 1 | 27%（[本]） | 在 24 个无人能解的 env 上烧满 4 h；PASS 轨迹 30 步 |
| R2 arm A | 74 | — | — | 卡住后 think 爆到 50k+/步、改环境、/tmp 试验（[trajectory_failures](trajectory_failures.md) §2） |

#### 评测口径补充：三个口径之间怎么换算

同一个模型在不同口径下的实测差：

| 行 / 模型 | [旧] | [镜] | [本] | 旧→镜 | 镜→本 |
|---|---|---|---|---|---|
| R1 th3k mask_const c312 | 0.348 | 0.410 | 0.354（162 done；同题配对 0.387 vs 0.436） | **+0.062** | **−0.049**（163 题配对） |
| A' dsvh_rand c336（代表 A 组） | 0.197 | 0.264 | — | **+0.067** | — |
| dsv41-flash 教师 | 0.753 | 0.831 | 0.787（同题 0.806 vs 0.835） | +0.078 | −0.029 |

- **旧 → 镜 约 +0.06~0.08**，来源是 reward 阶段 pytest 没跑的任务被救回（对装包重依赖的产物影响更大，rand c336 +12 题 > c312 +11 题）。
- **镜 → 本 约 −0.03~0.05**，来源：① tb_local 修完 8 处沙箱偏差后仍剩的系统性差（怀疑 AIS/DashScope 路由与 arena 网关的差异，未定）；② 学生模型的本地 run 里 **132/136 条是 11-09 修复前的 v1 契约**（oracle 提前可见 / `/tests` 只读 / `/tmp` tmpfs / PROMPT_COMMAND / clang），v1 的净方向是压低（对教师 −3 pt，对学生未单独量化）；th3k c312 的本地 run 是全 v2。
- **所以把 A、B、C 折到同一口径时，C 的 [本] 0.298 大致相当于 [镜] 0.32~0.35、[旧] 0.26~0.29**；A 的 [旧] 0.258 相当于 [镜] ~0.32。换算后 **A ≈ C**，差距在 ±0.03 的噪声内。这个换算是用其它模型的口径差外推的，不是测量；要定论见下面补齐计划 ②。

#### 同题配对（只在同一口径内部配对才有意义）

| 对比 | 口径 | n | A | B | diff | wins / losses | z |
|---|---|---|---|---|---|---|---|
| **C vs R1 th3k c312** | [本]，都 1.6.96 | 162 | 0.321 | 0.389 | **−0.068** | 9 / 20 | **−2.04** |
| C-9B vs R1 th3k c312（4B） | [本] | 160 | 0.375 | 0.394 | −0.019 | 19 / 22 | −0.47 |
| C vs C-9B | [本] | 174 | 0.305 | 0.351 | −0.046 | 11 / 19 | −1.46 |
| C [本] vs A' c336 [镜] | **跨口径** | 177 | 0.299 | 0.266 | +0.034 | 20 / 14 | +1.03（不可信，口径差 ≈ 0.05） |
| C [本] vs R2 arm A c224 [镜] | **跨口径** | 177 | 0.299 | 0.328 | −0.028 | 14 / 19 | −0.87（同上） |
| B vs A（各取 best epoch） | [旧]，同 harness | 178 | 0.191 | 0.258 | **−0.067** | — | 约 −12 题，超出 ±0.03 噪声 |

#### 三组放在一起的读法

| | A 不 judge 全量 | B 不 judge 剔未解决 | C judge 全量 |
|---|---|---|---|
| 分数（折到 [镜]，估） | ~0.32 | ~0.25 | ~0.32–0.35 |
| 相对 A | — | **−0.067**（实测，同口径） | **≈ 0**（跨口径外推，±0.03） |
| 相对 R1 th3k 同口径 | −0.146（[镜] A' 0.264 vs 0.410；A 本身无 [镜] 分，用 A' 代） | ≈ −0.15（外推） | **−0.068**（[本] 配对，z −2.0） |
| 失败形态 | 撞 512、不交卷 | 撞 512、不交卷 | 撞 512、不交卷（4B 70 条 / 44% 未宣告） |
| 结论 | 基线 | **剔除未解决是负收益**：失败轨迹占 62–73% 监督 token，"失败但过程可靠"是有效监督；th3k 自己重放也只 25% 过测试 | **judge 没有逆转段数据的劣势**：只剔 2% 轨迹，不改变 think 坍缩 + 段无交卷尾的分布；可能有 0~+3 pt 的小收益，但在口径差内看不出来 |

#### 变量控制检查

把 judge 版（`dsv41_judged`）和无 judge 版（`dsv41_history`）逐项对齐，列出哪些是真正的自变量、哪些是漏控的：

| 变量 | `dsv41_history`（无 judge） | `dsv41_judged`（judge） | 是否受控 |
|---|---|---|---|
| 源 job | `cth-thfull14-dsv41`（2,690 DONE） | `cth-thfull14` 2,365 条 + **`cth-thredo14` 133 + `cth-thredopol` 43** | ✗ **多了 176 条重打轨迹**（修好的 flat / tmux / reward-hang env，69 个新 env） |
| 环境污染剔除 | 无 | 剔 101 条（eperm-net / oom / pip-net / gave-up） | ✗ 与 judge 捆绑 |
| judge | 无 | Opus 4.8 KEEP（DROP 53 + OVERLONG 196 未评 + 6 其他） | ✓ 自变量 |
| 采集 harness | 1.4.96（8Gi、无镜像、tool-call） | 1.4.96 同批 + 重打用 **1.7.96**（16Gi + 镜像） | ✗ 重打部分采集条件不同 |
| 段化 / 格式归一化 | `build_multi_sft.py` 段切 | 同脚本，**但多剔了 13,737 个"命令发了但工具 JSON 非法、harness 什么都没执行"的静默失败步**（[trajectory_collection](../methods/trajectory_collection.md) §C6.1） | ✗ 与 judge 捆绑，且这一项直接影响"不发命令"行为 |
| 训练配方 | 262k、bad-pair mask、overlong drop、const lr 1e-5、8 ep、64 轨迹/步 | 完全相同 | ✓ |
| 训练步数 | 42 步/ep × 8 = 336 | 39 步/ep × 8 = 312 | ≈ |
| 评测 benchmark | 旧口径（`tb21-terminus2-full@2.1.3`，pytest 未启动 7–14%） | tb_local（mirror reward 脚本，v1/v2 两种沙箱契约） | ✗ **口径不同，不能直接比绝对值** |
| 评测 harness | 1.2.96（8Gi、无镜像） | 1.6.96（16Gi、pypi 镜像） | ✗ |
| 评测环境 | arena docker 沙箱 | singularity，**132/178 条是 v1 契约**（oracle 提前可见、/tests 只读、/tmp tmpfs、PROMPT_COMMAND、clang） | ✗ |

上表里 ✗ 的五项中，**评测口径**（[旧] vs [本]）的量级 ≈ 0.05~0.08，与 A、C 之间观测到的差距同量级，是最主要的混淆；**重打轨迹 + 污染剔除 + 静默失败步剔除**三项与 judge 捆绑在同一份数据里，即使口径对齐也拆不开，只能一起算作 judge 流水线的整体效应。

**tb_local 学生 run 的可信度**：4B 178/178、9B 174/178 done，但 132 / 136 条是 11-09 修复前的 v1 契约（压低项：只读 `/tests`、`/tmp` 跨设备、PROMPT_COMMAND、clang、COPY 文件名 bug；虚高项：oracle 提前可见）。按 `runs/invalidated-20261109/list.tsv`，4B 作废 58 条、9B 60 条（oracle-leak 23/24、llm-endpoint-connection-error 16/18、其余 ro-tests / copy-bug / apptainer-prompt），补跑的 v2 样本（46 / 38 条）已并入主表分数；未作废的 ~130 条 v1 样本仍带 PROMPT_COMMAND / clang 等系统性偏差，净方向压低（对教师 −3 pt，学生未单独量化）。对照 th3k c312 的本地 run 是全 v2，所以 C vs th3k 的 −0.068 是上界。

**补齐计划**（把 A、B、C 放到同一口径下；前 3 项只需评测，不需训练）：

| # | 做什么 | 权重 | 回答 |
|---|---|---|---|
| ① | 本地重跑 judged-4B 的 v1 契约样本（~130 条）为 v2 | ✅ `serve/dsvj-4b-c312` | 把 judged 版的本地分数做成纯 v2，和 c312 本地 run 同契约 |
| ② | 本地跑 A `dsv41_history const` ep8（c336）和 B `dsv41_history_solved` ep5（c95），同一套 v2 契约、1.6.96 | ✅ `abl_dsv41_history_const_e8_ais2n/checkpoint-336`、`…_solved_const_e8_ais2n/checkpoint-95` | **A / B / C 三组落到同一口径 [本]**；A vs C 的剩余混淆只剩重打 176 条 + 静默失败步剔除 |
| ③ | 本地跑 `abl_dsv41_hist_lin_full`（整轨迹全量、无 judge） | ✅ ckpt-64 | 把"整轨迹 vs 段"从 judge 效应里拆出来 |
| ④ | 训 + 评 `dsv41_judged-solved` | ❌ 数据已生成 | 2×2 的最后一格 |

顺序 ② → ① → ③ → ④。② 最关键也最便宜（两次评测）：没有它，A/B/C 的比较只能靠口径换算外推。

### 3.1c batch 顺序（sampler）对分数与 loss 曲线的影响：四种实现的全部实验（11-11）

四种"一步 64 条轨迹"的顺序实现：**HF `group_by_length`**（**每 epoch 重新 randperm**，每 200 行局部降序）、**`random`**（HF `train_sampling_strategy=random` 或 `build_traj_sched --order random`，每 epoch 重新置换）、**`global_sort`**（全局长度降序，每 epoch 相同）、**`random_fixed`**（一次置换，每 epoch 相同）。

#### 全部有分数的 run

| 数据 | 顺序 | run | 口径 / harness | ep3 | ep5 | ep7 | ep8 | 备注 |
|---|---|---|---|---|---|---|---|---|
| th3k mask（整轨迹） | HF group_by_length（cosine） | `abl_mask`（A2） | [旧] 1.0.0 | 0.287 | — | — | — | 3 ep 消融 |
| th3k mask | **HF random**（cosine） | `abl_mask_rand`（C2） | [旧] 1.0.0 | 0.242 | — | — | — | 3 ep 消融，比 A2 低 8 题 |
| th3k mask | HF group_by_length（const） | `abl_mask_const_e8_slurm2n`（R1） | [旧] 1.0.0 | 0.298 | 0.320 | 0.298 | **0.348** | mirror 0.410 / 本地 0.354 |
| th3k mask | **random_fixed**（const） | `abl_mask_randfix_const_e8_ais2n` | — | 训完（304 步，11-12 00:10） | | | | 待评 |
| th3k mask | **random**（const，每 epoch 重洗） | `abl_mask_rand_const_e8_ais2n` | — | 训完（304 步，11-13 13:25；两次 AIS 挂掉后续训） | | | | 待评；与 randfix 是唯一变量为"重洗 vs 固定"的干净对照，loss 结果见下 |
| dsv41_history 段 | global_sort（const） | `abl_dsv41_history_const_e8`（A） | [旧] 1.2.96 | 0.242 | 0.258 | 0.213 | 0.219 | |
| dsv41_history 段 | **random**（const） | `abl_dsv41_history_rand_const_e8`（A'） | [旧] 1.2.96 | 0.152 | 0.157 | 0.219 | 0.197（mirror 0.264） | |
| dsv41_judged 段 | global_sort（const） | `abl_dsv41_judged_const_e8`（C） | [本] 1.6.96 | — | — | — | 0.298 | |

#### 同配置只换顺序的两组直接对比

| 对比 | 数据 / 配方 | 顺序 A | 顺序 B | 分数 | 差 | 判断 |
|---|---|---|---|---|---|---|
| **A2 vs C2** | th3k mask，3 ep，cosine，1.0.0，×2 | HF group_by_length 0.287 | HF random 0.242 | −8 题 | −0.045 | 当时读成"保留 group_by_length"；现在看 **刚出 ±0.03–0.04 噪声带**，且 3 epoch 欠训练阶段对顺序更敏感，单次测量不足以定论 |
| **A vs A'** | dsv41_history 段，8 ep，const，1.2.96，×2 | global_sort 0.242 / 0.258 / 0.213 / 0.219 | random 0.152 / 0.157 / 0.219 / 0.197 | ep3/5 random 低 16–18 题；ep7/8 持平 | ep3 −0.090、ep5 −0.101、ep7 +0.006、ep8 −0.022 | **早期 epoch random 明显落后，训满后收敛到同一水平**。两个 run 在不同集群（AIS vs slurm），其余配方相同 |
| **R1 vs randfix** | th3k mask，8 ep，const，待同口径评 | HF group_by_length（每 epoch 重洗） | random_fixed（固定） | 待评 | — | 同数据、同卡数（16）、同 ga（4）；差三处：**重洗 vs 固定**、局部长度分组、warmup 12 vs 31 步（R1 的 scheduler 按 3 ep 建后续训）。loss 上 randfix 后期明显变慢（见下节） |

**两组已有对比指向同一个现象**：训练早期（≤5 epoch）随机顺序落后于长度相关的顺序 4–10 题，训满 8 epoch 后差距消失。loss 曲线上也看得到——A' random 的 epoch 均值 loss 在 ep2–7 比 A gsort 高 0.013–0.019，到 ep8 反而低 0.004（0.733 vs 0.737）；randfix 到 ep5 比 R1 高 0.012（1.035 vs 1.023）。可能的解释是长度相关顺序每个 epoch 内先学长（难）轨迹、再学短（易）轨迹，前期像一个课程；但 8 epoch 下这个前期优势被抹平，而 gsort 付出的是 grad norm spike（§A' 注释）和 16% 的速度。**判断不变：生产 run 用 random_fixed，但不要少于 8 epoch。** randfix 评完后补这一节。

#### loss 曲线的"epoch 阶梯"，以及 randfix 为什么比 R1 高、后期越来越慢

**阶梯是所有 run 的共性**。去掉每个 run 自己的"位置形状"（每个位置 k 在 ep2+ 的均值，或回归掉 log tokens/step）后，epoch 内的 loss 斜率都 ≈ 0：R1 −0.025…+0.017，randfix −0.017…+0.011，A' random −0.006，A gsort +0.008（每 epoch 累计）；下降全部发生在 epoch 边界。原因是每条轨迹每 epoch 只见一次，第 k 步的 loss 衡量的是"上次见到距今 ≈ 一个 epoch"的样本，epoch 内多学 64 条别的轨迹（<3%）对它几乎没有可测影响；无论顺序是否重洗都如此。曲线**看起来**是阶梯还是斜坡只取决于位置形状：R1 是无趋势的周期 3 锯齿（std 0.037，与 tokens/step 相关 0.87）所以平台 + 台阶清晰；randfix 的固定形状碰巧自带 −0.029/epoch 斜率（seed 42 的置换把偏长轨迹多排在前半，std 0.018）所以像斜坡；gsort 62M→1.7M 单调所以 epoch 内滑坡最陡。看进步一律同一位置跨 epoch 比。

**先排除配置/数据差异（11-12 核对）**。`abl_mask_const_slurm2n` → `_e5` → `_e8` 是同一个 run 两次续训（从 checkpoint-117、195 续），三个 slurm log 与 R1 的 loss 逐步相同（差 ≤ 0.0005），参照只有一条。数据：`sched16_randfix` 的 step-0 64 行与 flat cache 对应轨迹 `input_ids`/`labels` 逐行字节相同，监督比例 0.635 vs 0.634，每步 token 与 manifest 304/304 吻合、无重启丢步。`training_args.bin` 逐字段 diff 只有 `max_steps`/`num_train_epochs`/`save_total_limit`；DeepSpeed、lr、clip、wd、`average_tokens_across_devices` 等全同，同镜像同 yaml。

**randfix 比 R1 高多少**。最后一步 0.958 vs 0.908（+0.050）大半是口径：R1 的 step 312 是每 epoch 的 9 条尾批（2.3M token），每轮都比 epoch 均值低 0.045；R1 每步 token 2.2–13.2M 摆动、短批 loss 低（相关 0.58）。ep8 **token 加权**均值 0.967 vs 0.984（+0.017），逐 epoch 差呈 U 形：+0.021 / +0.008 / +0.003 / +0.001 / +0.002 / +0.005 / +0.010 / +0.017。

**U 形的来源：固定顺序后期收敛变慢，重洗顺序不变慢**。每 epoch 的 token 加权 loss 降幅（同位置跨 epoch）：

| 顺序类型 | run | ep1→2 | 2→3 | 3→4 | 4→5 | 5→6 | 6→7 | 7→8 |
|---|---|---|---|---|---|---|---|---|
| 重洗 | A' random（dsv41） | 0.138 | 0.044 | 0.032 | 0.027 | 0.025 | 0.024 | **0.025** |
| 重洗 + 局部长度分组 | R1 HF group_by_length（th3k） | 0.095 | 0.032 | 0.025 | 0.023 | 0.022 | 0.022 | **0.023** |
| **重洗（随机置换，与 randfix 同 seed/同 warmup）** | **rand（th3k，11-13 训完）** | 0.108 | 0.036 | 0.027 | 0.024 | 0.024 | 0.025 | **0.025** |
| 固定（随机置换） | randfix（th3k） | 0.109 | 0.036 | 0.027 | 0.023 | 0.019 | 0.017 | **0.016** |
| 固定（长度降序） | A gsort（dsv41） | 0.185 | 0.042 | 0.020 | 0.011 | 0.011 | 0.014 | **0.001** |

**rand vs randfix 是直接验证**：两者同数据、同 seed（ep1 置换相同）、同 warmup 31 步、同 304 步，只差后续 epoch 是否重洗。token 加权 epoch loss 到 ep5 完全重合（差 ≤ 0.001），从 ep6 分开：rand 每 epoch 继续降 0.024/0.025/0.025，randfix 降 0.019/0.017/0.016；ep8 rand 0.961 vs randfix 0.983（**−0.022**），且 rand 比 R1（0.967）还低 0.006——R1 多出的 17 个满 lr warmup 步被重洗的后期优势抵消。三个重洗 run 从 ep4 起稳定在每 epoch −0.022～−0.025 到 ep8 不减速；两个固定顺序 run 单调减速，randfix 到 0.016，gsort 到接近 0。ep1→2 randfix 比 R1 多降（0.109 vs 0.095）是 warmup 差异（R1 12 步、randfix 31 步，R1 的 scheduler 按 3 ep 建后续训到 8 ep），只影响 ep1，之后两者 lr 相同，解释不了后期分化。这是 shuffle-once vs random-reshuffling 的已知现象：固定顺序下每个 epoch 走同一串梯度，后期进入重复的局部轨迹、增量递减；重洗每 epoch 给出不同的组合，持续提供新梯度方向。gsort 更极端是因为它的顺序还和长度完全绑定。

**对判断的影响**：(1) `random_fixed` 不再是生产首选，改用 `random`（每 epoch 重洗，HF `train_sampling_strategy=random` 或 `build_traj_sched --order random`）；它同样有平坦 token 量和稳 grad norm，损失的是每步 loss 跨 epoch 可比——这个用途留给探针 run。(2) A vs A' 的"random 早期落后、训满持平"要重新读：A gsort 的早期优势来自长→短课程，但固定顺序让它 ep5 后几乎停止下降，A' 以恒定速度追上；再多训几个 epoch 可能反超。(3) 分数仍是最终裁判：randfix c304 评完后与 R1 c312 同口径对比填进上表；若 randfix 低 ≥ 2 pt 则 loss 结论得到分数确认。(4) 对照实验的 warmup 要写固定步数（`warmup_steps: 12`），不要用 ratio。

### 3.2 dsv41-96k-flash JSON-mode 段 → tree_flash（tree 训练，336 步 = 1 epoch）— harness 1.3.96（96k + nothink），×2

| ckpt | epoch | pass@all | TTL | 备注 |
|---|---|---|---|---|
| c126 | 0.375 | 0.146 | 18 | 36 条撞 512 |
| c168 | 0.5 | 0.202 | **74（45%）** | 数据 think 重尾（16% 的步 >20k 字符）→ 学生单步 3.4k tok、9.6% 的步 ≥16k tok，4 h 耗尽 |

训练只到半个 epoch，欠训练；但 TTL 45% 是数据分布问题，不是步数。

**为什么低 + 跨系列根因（11-03 分析）**：

1.3.96（96k + nothink，与 tree 训练条件一致）×2：c126（0.375 ep）**0.146**、c168（0.5 ep）**0.202**，但 c168 **TTL 45%**（74/165）是全场最高。
和 dsvh/qf 相反的死法——"想太多"：训练数据 think 重尾（p50 4.1k、16% 的步 >20k 字符，th3k 6.7%），学生 TTL 轨迹 think p50 10.7k、
单步 completion 3.4k tok、9.6% 的步 ≥16k tok，~90 tok/s 下 85 步耗尽 4 h。宣告完成后通过率 59% 却是最高（想明白的题能做对）。

五套数据指纹（各抽 250 条）把结论收敛成一句话：**th3k 是唯一同时满足 ① 整轨迹 100% 交卷尾 ② think 中等且窄（2.9k / p99 46k）③ 100% 以 `task_complete` 结尾（**不是** 100% solved——th3k 无 score 字段，重放只有 ~25% 过测试，见 §7 第 5 条）的数据**；
其余四套都是 96k 段（44–78% 从 handoff 开头、44–80% 无交卷尾）、think 要么太短（tool-mode 0.5k / qwen3-8-flash 1k）要么重尾（JSON-mode flash）、
62–73% 监督 token 来自失败轨迹。缺 ① → 撞 512 步（dsvh、treef-c126）；缺 ② 偏短 → 步数膨胀，偏长 → TTL（treef-c168）；qf 两者皆缺 + 长 sleep。
排除项：训练曲线全正常；compaction 重探索是均匀税（所有模型压缩后 5 步内 15–32% 命令是 ls/cat，mask 也 25%）；nothink 口径不翻转；rand≈const。
建议：段数据改回整轨迹拼接或只留含交卷尾的段族；先测已训完的 `abl_dsv41_history_solved` 与 `tree_flash_solved`；JSON-mode 数据对 >20k think 做 mask。
全文 `/storage/openpsi/users/chen/output/tb21_why_th3k_wins.md`；工具 `coding_eval/tools/{traj_stats,traj_compaction_cost,sft_data_fingerprint,classify_tb21_failures}.py`。

### 3.3 qwen38_flash（Qwen3.8 Flash 自蒸馏 96k 段）— harness 1.2.96，×2

| 变体 | ckpt | pass@all | TTL | 备注 |
|---|---|---|---|---|
| solved | c63 | 0.045 / 0.051 | 158 / 147 | **作废**：4 卡 par 128 饱和 retract |
| solved | c105 | 0.174 | 96 | |
| solved | c116 | 0.157 | 99（另一 job 165 HF） | |
| const | c125 | 0.101 | 109 | |
| const | c129 (ep3) | 0.135 | 126 | |
| const | c215 (ep5) | 0.101 | 138 | |
| const | c258 (ep6) | 0.124 | 127 | |

TTL 71–79%，epoch 不降 TTL。学生复刻教师风格：35% 命令是 `clear`、14.5% 带 ≥30 s sleep。

**为什么低（10-31 / 11-01 分析）**：

harness 1.2.96（96k + 历史推理可见）×2：c63 0.051（4 卡饱和，作废）/ c105 **0.174** / c116 **0.157** / const-c125 0.101（同族早期 ckpt）。
同一口径下 4B base 0.188、tree c117 0.197，1.0.0 口径下 mask c312 0.348。**主因是 52% 的 episode 跑满 4 h TTL 计 0**
（c105 96/173、c116 99/159、c125 109/166）；按非 TTL 的 36% 通过率把 TTL 补回来，c105 就是 0.354。宣告完成的样本通过率 36–46%，
与 mask 的 44–46% 持平，能力在，是不交卷。

根因链（逐层可测）：
1. **训练数据 29%（534/1819 段）没有交卷动作**——最后一个 assistant 轮还在写"下一步"，`commands` 还有命令，数据就结束了；
   reward=1 但 agent 从未说 `task_complete`。th3k（mask/tree 的数据）是 100% 以 `task_complete` 结尾。模型学到的就是"干到一半可以停"。
2. **步数 ×3**：199 步 p50（mask 63、tree 74、base 52）；每步 2.09 条命令（mask 2.60）；26% 的轮次声明 ≥30 s 的空等，
   sleep 占墙钟 36%（mask 11%）。同一题 mask 15 步解出、solved-c105 要 44 步。
3. compaction 5.31 次/条（base 0.87），每次 97k→5k 后约 10 步重建状态。
4. 训练量只有一半：solved 1284 条 → 168 步/8 epoch，th3k 2441 条 → 312 步；c105 = epoch 5 of 168。
5. think 坍缩：数据 p50 3323 / p90 26k 字符，rollout 只有 1212——学不会"想清楚再批量发命令"，只能小步慢走。

排除：serving（0 retract、queue 0）、沙箱（无 503，全部 `agent_ms` 顶满 4 h）、解析（失败率 0–0.3%）、任务难度（会做的题通过率与 mask 持平）。
全文 `/storage/openpsi/users/chen/output/tb21_qf_solved_analysis.md`。

**勘误 + 深挖（11-01）**：训练数据不是 dsv41 文件，是 `terminal_horizon_qwen38_flash{,_solved}` = **qwen3-8-flash 自己在 96k 下的 rollout 按 compaction 切段**
（const 11,001 段 / 2,732 条，73% score=0；solved 3,583 段 / 1,288 条）。段样本只有 23% / 36% 以 `task_complete` 结尾（th3k ~100%），
教师风格 = think p50 ~850 字符（th3k 2769）、cmd/步 2.0、≥30 s 声明等待占 20% 的步（th3k 4%）。训练曲线完全正常（loss 1.06→0.81 单调，
无 restart），模型忠实复刻了这套慢节奏，但没有教师的单步正确率 → 步数 ×3、TTL 52–66%。epoch 加到 5 也不降 TTL（solved c105）。
全文 `/storage/openpsi/users/chen/output/tb21_qwen38_flash_sft_analysis.md`。

### 3.4 dsv41j arm A（dsv41 **整轨迹** JSON-mode，2037 条，cap off）— harness 1.0.0，×2，**11-05/06**

| ckpt | epoch | 旧 benchmark | **修复 verifier（mirror）** | TTL | 备注 |
|---|---|---|---|---|---|
| c64 | 2 | 0.180（40 HF 路由中断） | 0.180 | 87 | |
| c96 | 3 | 0.213（STOPPED，18 HF） | 0.219 | 84 | |
| c224 | 7 | 0.270（27 HF） | **0.326**（1 条未完） | 74 | **第二好的模型** |
| c155 | 5 | 未开始（serving 未起） | — | — | |
| c248 | 8 | 未开始 | — | — | 训练完成，终 loss 0.079 |

arm A 的数据来源：6 个已有 dsv41 chat-mode job 重新转换（3593 → 2037 条，262k 内），**未去重**（883 env，多数 2–5 条/env）。
arm B（think-cap 20k）：训练 record 失败在 step 8，无评测。
arm A 的 TTL 74–87 条（42–49%），比 mask（41）高一倍——dsv41 教师轨迹的 think 比 th3k 长。

### 3.5 跨系横向（同口径尽量对齐）

| 数据 | 教师 | 样本结构 | 最好分（旧 verifier）| 最好分（修复 verifier）| 主要死法 |
|---|---|---|---|---|---|
| **th3k** | DeepSeek-V4-Pro | 整轨迹 2441 | **0.348** | **0.410** | 42 TTL（在 24 个无人能解的 env 上） |
| dsv41 整轨迹（arm A） | DSV4.1 Flash JSON | 整轨迹 2037 | 0.270 | **0.326** | 74 TTL |
| dsv41_history | DSV4.1 Flash tool | 96k 段 | 0.258 | 0.264（rand c336） | 撞 512 步 |
| dsv41_judged（dsv41_history 同源 + judge） | DSV4.1 Flash tool | 96k 段 | — | 0.298（tb_local，非 arena；9B 0.343） | 撞 512 步，与无 judge 同形态（§3.1b） |
| dsv41-96k-flash（treef） | DSV4.1 Flash JSON | 96k 段 | 0.202 | — | TTL 45% |
| qwen38_flash | Qwen3.8 Flash | 96k 段 | 0.174 | — | TTL 75% |
| 4B base | — | — | 0.187 | 0.165（1.2.4） | — |

**整轨迹 > 段** 在两个教师上都成立（dsv41：0.326 vs 0.264）；教师换 DSV4-Pro→DSV4.1-Flash 整轨迹还差 0.08。

---

## 4. 题级分析（89 题，所有 4B/9B 模型）

- **30 题无任何 4B/9B 模型解出过**（含 mirror 重测）；教师 DSV4.1 Flash 解出其中 22 题。这 30 题是 mask c312 的 TTL 主要来源。
- 所有 SFT 模型并集解出 **57 题**；mask c312 单独 41 题（21 题 2/2，20 题 1/2）。
- mask c312 的 110 个失败（旧 verifier）：TTL 42、交卷-部分过 39（19 条只差 1 个测试）、verifier 未启动 13、交卷-全挂 11、其他 5。
  六类归类与根因见 [trajectory_failures](trajectory_failures.md) §8。

---

## 5. TB4.0（Terminal-Bench 4.0 Arena 子集，48 题）— **全部未完成**

| 阶段 | 模型 | harness | 已完成样本 | 通过 | 备注 |
|---|---|---|---|---|---|
| 单题 benchmark ×48（11-05 23:59） | mask c312 / c234 | 1.0.0 | 8 / 10 | 0 / 0 | 4 h 只提交 7/48；全部取消 |
| 同上 | Qwen3.8 Flash | 1.0.0 | 2 | 0 | 取消 |
| 统一 benchmark `chenth-tb4-unified@1.0.0`（11-06 12:07） | mask c312 | 1.8.0-nocomp-4b | 6（另 31 HARNESS_FAILED）| **0** | 我停的：262k 窗口顶满 → 引擎 400 → router 熔断 503 |
| 同上 | mask c234 | 1.8.0-nocomp-4b | 6（另 24 HF）| **0** | 同上 |
| 同上 | DeepSeek V4.1 Flash | 1.8.0-nocomp-api（1M） | 25 | **1**（mvcc-lsm-compaction 1/2） | 16:00 被平台/他人停止 |
| 同上 | Qwen3.8 Flash | 1.8.0-nocomp-api | 24 | **3**（formal-crypto、html-js-filter 各 1/2，第三题记录丢失） | 同上 |

- TB4 单题 1–8 小时（时限 8 h），比 TB2.1 大 3–7 倍；取消前跑完的 cad-model：c312 304 步/222 min、c234 474 步/113 min。
- 统一 benchmark 的 reward 是占位（平台一个 benchmark 只绑一个 reward），真实分数靠 `score_tb4.py` 对每个 DONE 任务跑官方 per-task reward；只在 0 分样本上验证过与原生一致，**未在通过样本上验证**。
- "不压缩" harness 实际只是把压缩推到窗口满（runner 写死 `enable_summarize=True`）；对 4B 的 262k 窗口不可行。
- Qwen3.8 Flash 第三个通过样本的记录在平台断连后被评分脚本的空结果覆写，无法恢复。

---

## 6. 教师数据采集 job（TerminalHorizon 系列，非消融，但产生了训练数据）

| job | benchmark | 模型 | harness | n | pass@all | TTL | 用途 |
|---|---|---|---|---|---|---|---|
| cth-th500-stream-{27b,flash,dsv41api} | th-500 | 27B / Qwen3.8F / DSV4.1F | 1.0.0 | 499 | 0.515 / 0.593 / 0.583 | 71 / 13 / 29 | |
| cth-threst-stream-{flash,dsv41api} | th-rest | Qwen3.8F / DSV4.1F | 1.0.0 | 794 | 0.419 / 0.428 | 51 / 51 | arm A 源数据 |
| cth-thall-stream-{flash,dsv41api} | th-all | Qwen3.8F / DSV4.1F | 1.0.0 | 1289 | 0.494 / 0.483 | 54 / 79 | arm A 源数据 |
| cth-csweep-c{64,96,128,256,512}k-{flash,dsv41} | th-long128 | 两模型 | 1.2.x | 512 ×10 | flash 0.312→0.389、dsv41 0.353→0.383 随窗口增大 | flash 64k TTL 283 | 压缩点扫描；c128/256/512 dsv41 = arm A 源 |
| cth-thfull96-{dsv41,flash} | th-full-rest | 两模型 | 1.2.96 | 1948 | 0.471 / 0.480 | 30 / 195 | dsv41-96k-flash（treef）与 qwen38_flash 源数据 |
| cth-thfull14-dsv41 | th-full | DSV4.1F | 1.4.96 tool | 2972 | 0.422（2690 DONE） | 341 | dsv41_history 源数据（DashScope 限速期，52% TTL） |
| cth-thredo14-dsv41 | th-redo14 | DSV4.1F | 1.7.96 | 282 | 0.223（182 DONE 时） | 5 | 坏 env 重跑 |
| cth-l128-nothink-flash | th-long128 | Qwen3.8F | 1.3.96 | 512 | 0.307 | 201 | nothink 对照（vs 0.334 @1.2.96） |
| cth-th3k-replay{1,2} | th3k 重放 | 无模型 | replay 1.0.1 | 204+2556 | 40+620 = **660 solved / 2650** | — | **th3k 只有 ~25% 过测试** |

---

## 7. 已知问题与数据可信度标注

1. **旧 verifier 系统性低估**（10-31 前所有 TB2.1 job）：7–14% 失败任务 pytest 未启动。影响所有 §2/§3 旧 benchmark 数字，幅度约 +0.03（mask c312 确证 +5 题）。§3.4 与 mask c312 的 mirror 数字不受影响。
2. **autopar 参数顺序错误**（11-02 至 11-06 01:03 的所有 job）：并行度一路升到 96 不看负载；engine 日志显示 mask-c312 / arm A 的 serve 5–10% decode 行有排队、各 2 次 retract。TTL 可能略被放大，未量化。
3. **AIS 抢占**：dsvh c126、treef、arm A 的多个 job 经历过整批 route-down 失败后 retry；retry 后的 TTL 分布不受影响（任务重新计时），但 wall-clock 被拉长。
4. **噪声**：×2=178 的重复测量差 ±0.03–0.04；跨模型差 <0.05 不应下结论。
5. **th3k "100% solved" 是错的**：th3k 无 score 字段，100% 指以 task_complete 结尾；重放只有 ~25% 过测试。这反而支持"失败轨迹也是有效监督"（dsvh solved-only 更差）。
6. **TB4 全部为中间结果**，不构成任何结论。
7. mask_pack、th3k_solved、dsv41j arm A ep5/ep8、arm B、lin_full、lin_solved_tail：有权重无评分或未训完。

---

## 8. 文件索引与复现

- 仓库保留了 [276 个 Arena 作业的离线汇总](evidence/all_jobs_offline_2026-11-07.tsv)。逐样本原件在原存储机的 /storage/openpsi/users/chen/output/ 下。临时案例和调度日志属于原存储机，需挂载原存储后查看。
- [旧口径逐题消融](th3k_ablation_by_task.md)、[上下文实验](context_sweeps.md)、[轨迹级错误分析](trajectory_failures.md)、[mirror c312 人工复判](c312_failure_cases.md)。
- 汇总与审计脚本在仓库根目录下的 coding_eval/tools/。TB4 中途停止记录在 §5，原平台工具不随此仓库分发。
