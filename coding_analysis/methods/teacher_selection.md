# 数据选取：教师风格 → 学生行为的传递规律、compaction 归因、论文准则复核

> 训练数据清洗过程及数据量见[采集与清洗](trajectory_collection.md)，后续训练效果见[SFT 实验](../experiments/sft_results.md)。

日期：2026-11-10。承接 [trajectory_failures](../experiments/trajectory_failures.md)（为什么非 th3k 数据输）和 [trajectory_collection](trajectory_collection.md) Part C（dsv41 清洗流水线），回答四个问题：

- **Part A** qwen3-8-flash 数据能不能靠数据处理救？Part C 流水线对它有没有用？给出一条典型缺陷轨迹。
- **Part B** 模型学没学会处理 compaction？分数差里有多少能归咎到 compact？五个模型的行为分布。
- **Part C** 五篇论文（TH3K / Tmax / Data Recipes / FACET / Terminal-Universe）怎么选 teacher、怎么选轨迹，对我们的借鉴。
- **Part D** 用论文准则逐项度量四套真实训练输入（th3k / dsv41 armA / dsv41_final / qwen38 flash）与各自学生的行为，找出"哪些指标从数据传递到学生"；折扣项（clear / sleep / 轮询）排除后的对比。

工具：`coding_eval/tools/traj_compaction_behavior.py`（评测轨迹：行为分布 + compaction 窗口 + handoff 忠实度）、`coding_eval/tools/sft_paper_criteria.py`（训练数据：论文 Table 3 指标 + judge 三准则量化 + 折扣指标）。论文全文抽取在 `/tmp/opencode/26*.txt`（容器重建会丢，pypdf 从 `../paper/*.pdf` 重抽）。

**结论先行**

1. **qwen38 flash 数据不可救**：数据处理（含 Part C 全套）只能消掉"教坏"的部分（不收尾、长 sleep 监督、think 重尾），消不掉"没得教"的部分（零测试、低信息密度、`clear` 占命令 1/3）。Part C 的 judge 按准则"等待不加分不扣分"会 KEEP 它们——dsv41 走完流水线风格不变是前车之鉴。维持 [trajectory_failures](../experiments/trajectory_failures.md) §7.4 "qf 系弃用"。
2. **compaction 不是病因**：三家模型压缩边界后的行为都正常（重写已完成文件 4–10%、think 回到正常水平、命令密度上升），th3k 对 handoff 最忠实（0.58）、qf 最差（0.17）。没压缩的任务上 th3k / dsv41 / dsvh pass 完全相同（0.63）；崩塌只发生在 ncomp ≥3（pass ≤0.12），而那时任务已经卡了 200+ 步。th3k→dsv41 的 8.5 pt 差距里可归咎于 compaction 机制的 ≤1–2 pt。
3. **五篇论文的共识**：最强模型不是最好 teacher（Data Recipes 直接实验；我们 qf 教师 0.771 → 学生 0.174 是第三个复现）；失败轨迹该留（TH3K / Tmax / Data Recipes 都不按 solved 筛）；单源风格 > 混源数量（Terminal-Universe：模仿异构源 40.3 < base 46.2，同 teacher 重解 52.9）；扩数据扩任务不扩采样。
4. **论文 Table 3 的 teacher 选取指标必须折扣后再用**：turns / feedback chain / exact repeats 三项会被轮询刷高——qwen38 按这四个数字是三家最优（chain 78、重复 4.8%），折扣 `clear`/`sleep`/轮询后有效命令/turn 1.53 是三家最差。真正从训练数据传递到学生、且排序与分数一致的是：**折扣后命令构成、收尾率、think 重尾**；feedback chain 和 exact repeats 不传递。
5. **dsv41_final（待评测）的预测**：段口径收尾 30%、think p50 920，落在 dsvh（0.264）与 arm A（0.326）之间、更靠 dsvh。若实测如此，证明 judge（真实性过滤）的边际贡献小于结构项（收尾 + think 密度）。

---

# Part A. qwen3-8-flash 数据：能否靠数据处理避免 failure_analysis 中的问题

## A1. 两家的病不一样

| | qwen3-8-flash 系（qf_const / tree_flash） | dsv41 chat 系（arm A） |
|---|---|---|
| 评测死法 | TTL 122 条：**81 分钟/任务花在 sleep**，≥60 s 命令 39 条/任务；tree_flash 另有"每步 54 秒、completion 5.4k tok"的纯 think 肿胀 | TTL 70 条：卡住后 think 爆到 50k+/步、C-c 打断安装、改 sources.list、/tmp 堆试验文件 |
| 数据指纹 | duration p90 = 30 s、5.5% ≥60 s、**每轨迹声明等待 7.3 分钟**（th3k 1.6）；35% 命令是 `clear`；测试命令 ≈0；段仅 21–36% 以 task_complete 结尾 | 末段 think 增长 7.5×、14.3% 步 >20k、envfix 命令 6×、测试命令 0.4×、86% 监督字符是 think |
| 病的性质 | 教师风格缺陷（睡觉、清屏、不测试）+ 段不收尾 | 教师风格缺陷（卡住就长想、改环境）+ 流程放大（overlong 丢 43%、token 均值 loss） |

背景：flash 教师在 TB2.1 上是三位教师里最强的（0.771 vs dsv41 0.753），学生却最差（0.174 < 4B base 0.187）。教师能力和学生分数脱钩。

## A2. 哪些能修、哪些修不了

**数据处理可以修的**：
1. 段不收尾 → 只保留以 task_complete 结尾的段链，或非终止段末轮 loss=0。
2. think 重尾 → 按"相对首段增长 ≤4×、单步 ≤20k"做 mask；tree_flash 的另一半问题（历史 think 不可见训练、只训 0.5 epoch）是流程问题，改回 mask 式训练 + 训满即可。
3. 长 sleep → 形态 A 折叠（连续"等待→相同观察"只留首尾）、等待轮/duration 字段 loss=0、剔除 sleep 主导轨迹。但 flash 的等待是弥散的（每条 7.3 分钟、14.5% 命令带 ≥30 s sleep），按轨迹剔会剔空，按轮 mask 只能"不学等待"，学不到"该等多久"。

**数据处理修不了的（致命项）**：测试命令 ≈0、`clear` 占 35%、信息密度低。过滤只能删行为，不能注入不存在的行为。flash 轨迹里几乎没有"写完→跑测试→修"的样本可供上采样；删掉 sleep 和 clear 之后剩下的也不是 th3k 那种"一步读多文件、写完就测"的流程（§D 折扣后有效命令/turn 1.53 vs th3k 2.74）。

## A3. Part C 流水线对 flash 的适用性

| Part C 环节 | 对 flash 有没有用 |
|---|---|
| 系统坏 env 修复 + 重打 | 有用但答非所问：env 修复教师无关，重打出来的还是同一风格 |
| 环境污染剔除 | 检测器可直接复用，但不是 flash 的主要死因 |
| Opus 三准则 judge | **抓不到 flash 的病**：准则原文"等待不加分但不扣分"，th3k 自己 6.8% 轮询主导轨迹照样 KEEP；预期 KEEP ~98%，病原体全数通过 |
| §C3.7 形态清理 | 唯一直接打中症状的环节（轮询折叠、短 think 剔除），但只能去负不能补正 |

反证来自 Part C 自己：dsv41 走完全套后 `dsv41_final` 仍是"小步快跑"（think p50 579 vs th3k 2,963、84% 含 compaction），原文"这是采集配置决定的，清洗没改变风格"。

## A4. 典型缺陷轨迹（真实样本）

`/storage/openpsi/users/chen/code_sft_data/qwen3-8-flash-history/TerminalHorizon-flash-96k-SFT.jsonl` 第 1956–1958 行：
**`cth-th-qualify-an-atomistic-molecular-dynamics-ensemble-for-production-use` s0**（job `cth-csweep-c96k-flash-2610231040`，3 段，GROMACS 三副本任务，**score=1，solved 过滤筛不掉**）

| 段 | 步数 | 命令 | sleep 累计 | `clear` | ≥30 s 命令 | 空等待步 | think<100 字符 | think p50 | 段尾 task_complete |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 39 | 87 | 9.6 min | 24 | 22 | 7 | 2 | 1,166 | 否 |
| **1** | **91** | 235 | **174 min** | 42 | **117** | 36 | **43（47%）** | **120** | 否，末轮 think = `Wait.` |
| 2 | 28 | 112 | 0.9 min | 22 | 7 | 5 | 2 | 1,286 | 是 |
| 合计 | 158 | 434 | **184 min** | **88（20%）** | 146 | 48 | 47 | 375 | 测试命令 **0** |

段 1 摘录：

```
# 6  think(6843)  [60s] "clear; sleep 60; cat pipeline.status; gmx check ..." x4
# 7  think(205)   "40 frames/min, need 500 more => ~12.5 min. Poll."
                  [60s] "clear; sleep 180; ..." x3 + "sleep 120; ..."
# 8  think(215)   "The screen is weird; commands queued?"   [60s] '' ; [2s] '\n' ; [30s] "clear; ..."
# 9  think(253)   "It seems terminal is buffering?"          [60s] '' ; [60s] ''
#11  think(35)    "Now 881 and queued sleep 120. Wait."      [60s] '' x2
#37  think(27)    "598/841. Need ~6 min. Poll."              [60s] "clear; sleep 300; ..." ; [30s] ''
#52  think(5)     "Wait."                                    [60s] ''
#56  think(223)   [60s] "clear; sleep 600; cat ...; gmx check ..."
#62  think(249)   "Could send Ctrl-C? But if mdrun running?" [2s] 'C-c' ; [2s] '\n' ; "echo READY; ..."
#71  think(1126)  [60s] 'sleep 240; ...' x3        <- 一步排队 12 分钟 sleep
#82  think(152)   [60s] 'clear; sleep 300; ...' x3  <- 又排队 15 分钟
#83–90           think 5–119 字符："Queued; need wait ~15min." / "Wait." / "Wait." ...
#90  think(5)     "Wait."  -> 段在这里结束（compaction 切断），无 task_complete
```

五个缺陷同时出现：(1) `sleep 300/600` 写在 keystrokes 里 + duration 60 排队，一步声明 12–15 分钟等待——学生复刻成 81 分钟/任务；(2) sleep 堵死终端 → 命令排队 → 以为终端坏了 → C-c 和空回车，四轮循环（#8–11、#38–40、#53–62、#76–81）；(3) think 坍缩到 "Wait." 5 字符，47% 的步 <100 字符；(4) `clear;` 前缀 20% 命令，每步只看最后一屏，#3/#4/#5/#16/#30 反复 "missed ... screen scrolled" 重跑；(5) 段 1 最后一个监督轮是 `<think>Wait.</think>{"commands":[{"keystrokes":"","duration":60}]}`。对照 th3k 的等法：空命令 + 100 字符 think（[trajectory_failures](../experiments/trajectory_failures.md) §2 死法 a）。

同族备选：`cth-th-extend-a-mature-http-2-proxy-s-backpressure-and-stream-lifecycle` s0 段 3（156 步、`clear` 87 次、72 步短 think、score=0）；`cth-th-rust-crate-security-release-with-downstream-compatibility` s0 段 1（101 步、`clear` 146 次、score=1）。

## A5. 其他方法（含 dsv41）

- **零训练 / harness 侧**（对所有数据系有效，先做）：E1/E2 宣告确认清单；§6.4 止血注入（触发点可改用 ncomp ≥3，见 §B3）；§8.5 "两条采样取宣告 think 更短者" +6–7 pt。
- **dsv41 数据侧（主力）**：在 dsv41_final 上叠加收敛行为过滤（末段 think 增长 ≤4×、单步 ≤20k、含 ≥1 测试命令、无 sources.list/alternatives）；think 按相对增长 mask；overlong 用 tail 窗；非终止段末轮 loss=0；从 dsv41 轨迹上采样"apt/pip 装包 → 空命令等待 → 成功"段落补耐心教材；E3 对照臂。solved 过滤已证无效（−5~8 题），别走回头路。
- **flash 数据侧**：若一定要救，最小组合 = 段链收尾过滤 + 等待轮/duration loss=0 + think 相对增长 mask + 形态 A 折叠 + mask 式训练训满，上限是"不拖后腿"。更划算的是改采集配置重采（禁 keystrokes 内 sleep/clear、duration 上限 30 s、交卷前强制自验证）。Part B §B6 反对混用三套（思考预算差 3–4×）。

---

# Part B. compaction 归因：模型学会处理 compact 了吗？分数差有多少能记到它头上

数据：`cth-tb21m-mask-const-c312`（th3k 最佳）、`cth-tb21m-dsv41j-a-c224`（dsv41 最佳）——同 harness 1.0.0、同采样 index，严格可比；`cth-tb21m-dsvh-rand-c336`、`qf-solved-c105`、`qf-const-c129` 是 96k harness，压缩频率天然偏高，只作风格参照。工具 `traj_compaction_behavior.py`，compaction 边界用 trajectory.jsonl 的 `extra.context_management.type == "compaction"` 系统事件定位。

## B1. 五个模型的行为分布（178 任务）

| 指标 | **th3k c312** | **dsv41 c224** | dsvh c336 | qf c105 | qf c129 |
|---|---|---|---|---|---|
| pass@all（本扫描 / 文档值） | **0.429** / 0.410 | 0.341 / 0.326 | 0.266 / 0.264 | 0.172 / 0.174 | 0.138 / 0.135 |
| TTL ≥235 min | 41 (24%) | 73 (43%) | 0† | 120 (69%) | 126 (72%) |
| 宣告 task_complete / 宣告后通过率 | **74% / 0.58** | 55% / 0.59 | 56% / 0.43 | 31% / 0.44 | 23% / 0.50 |
| 步数 p50 / p90 | **67 / 486** | 166 / 530 | 296 / 512† | 92 / 227 | 227 / 457 |
| think 字符 p50 / p90 | **5621 / 23078** | 4418 / 20067 | 1175 / 7649 | 1280 / 11699 | 1060 / 8292 |
| think <100 字符步 | **2.9%** | 3.4% | 11.4% | 7.7% | 7.2% |
| think >20k 步 | 12.8% | 10.1% | 2.4% | 4.9% | 2.8% |
| 命令/步（非等待步） | **2.57** | 2.40 | 1.68 | 2.15 | 1.67 |
| 等待步（空按键） | **10.2%** | 8.6% | 8.3% | 6.5% | 5.4% |
| sleep 秒/轨迹；≥30 s 命令/轨迹 | 37; 7.9 | 51; 4.2 | 54; 3.8 | **119; 29** | **106; 61** |
| `clear`/轨迹；`C-c`/轨迹 | 2.1; 4.9 | 0.8; 6.9 | 1.4; 5.7 | **58.5**; 3.2 | **151**; 8.3 |
| 读文件类命令占比 | 40.8% | 32.8% | 35.4% | 32.8% | 23.0% |
| 测试命令/轨迹 | 1.1 | 1.9 | 2.1 | 1.0 | 3.4 |
| completion tok/步 p50 | **1989** | 1593 | 603 | 747 | 632 |
| compaction/轨迹；含 ≥1 次 | **2.66; 51%** | 3.51; 62% | 5.03; 74% | 2.45; 79% | 4.14; 89% |

† dsvh 在 96k harness 有 512 步上限，用步数耗尽代替时间耗尽。本扫描 c312 0.429 vs 文档 0.410：8 条 task.json 缺 `agent_ms` 被跳过，都是失败轨迹，不影响相对结论。

th3k 相对 dsv41 唯一质变的行为是"宣告率 74% vs 55%"和"步数 p50 67 vs 166"——宣告后通过率两者一样（0.58 / 0.59）。

## B2. 操作层面：学会了

每个 compaction 边界取前后 8 / 15 步窗口（th3k 452 个、dsv41 597、dsvh 886、qf 404–721）：

| compaction 边界行为 | th3k c312 | dsv41 c224 | dsvh c336 | qf c105 | qf c129 |
|---|---|---|---|---|---|
| 压缩后第一步命令与 handoff 计划的重合度 | **0.58** | 0.46 | 0.34 | 0.20 | 0.17 |
| 第一步与 handoff 零重合 | **19%** | 25% | 43% | 58% | **70%** |
| 15 步内重写"压缩前已写过的 /app 文件" | **4%** | 4% | 5% | 10% | 10% |
| 压缩后命令重复早先命令 | 24% | 20% | 21% | 18% | 14% |
| think：前 8 步 → 后 8 步（均值） | 18.9k → 9.2k | 16.0k → 6.0k | 6.6k → 4.6k | 16.2k → 9.6k | 8.9k → 5.9k |
| 命令/步：前 → 后 | 2.08 → **3.25** | 2.01 → 3.08 | 1.43 → 2.11 | 1.99 → 2.66 | 1.52 → 2.15 |
| 读文件占比：前 → 后 | 0.34 → 0.49 | 0.28 → 0.43 | 0.27 → 0.48 | 0.31 → 0.41 | 0.22 → 0.26 |
| [PASS] vs [TTL] 窗口的重复命令率 | 20% vs 24% | 20% vs 19% | — | 21% vs 19% | 17% vs 13% |

手读三条边界证实统计读数：
- **th3k PASS（adaptive-rejection-sampler s0）**：压缩前一步 think 89k 重写整个 `ars.R`；handoff 给 4 条验证命令；压缩后第一步原样执行这 4 条，第二步 `sed -n` 分块重读文件，3 步内回到修 bug 主线。
- **dsv41 TTL（rstan-to-pystan s0，第 2 次压缩）**：压缩后第一步同样忠实执行 handoff 的编译器状态检查——compaction 处理没问题；但 handoff 交接的主线已经是修环境死循环（第 185 步，装 clang 绕 apt 坏源），压缩只是忠实延续了死路。
- **qf TTL（mailman s1）**：压缩后第一步 think 2.6k 重新定位尚可，3 步后回到 `clear; sed -n` 翻页，第 206 步自己写 "I missed the outputs of the first three commands (screen scrolled)"。

结论：(a) 三家都没有"压缩后失忆乱跑"的硬故障；(b) 处理质量有梯度，th3k 最忠实、qf 最差——qf 训练数据 78% 含 compaction、75% 段以 handoff 开头，暴露最多却学得最差；th3k 数据只 12% 含 compaction 却处理最好，compaction 处理能力来自整体推理质量，不来自样本量；(c) PASS 和 TTL 轨迹的边界后行为几乎无差——压缩边界不是成败分水岭。

## B3. 定量归因：没压缩的任务上三家分数一样

| 条件 | th3k c312 | dsv41 c224 | dsvh c336 | qf c105 / c129 |
|---|---|---|---|---|
| ncomp=0 的 pass | **0.63** (84) | **0.63** (65) | **0.63** (46) | 0.43 (37) / 0.42 (19) |
| ncomp=1–2 的 pass | 0.50 (34) | 0.52 (21) | 0.21 (32) | 0.17 / 0.19 |
| ncomp≥4 的 pass / 宣告率 | 0.05 / 9% | 0.08 / 11% | 0.11 / 19% | 0.08 / 8% / 6% |

th3k / dsv41 / dsvh 在"一次都没压缩"的任务上 pass 完全相同——能力差为零；qf 即使不压缩也只有 0.42–0.45（风格税与 compaction 无关）。

**th3k vs dsv41 的 8.5 pt 分解**（164 对同采样 index，净差 −14 题）。pass = P(落入压缩区) × P(pass | 压缩区)：
- th3k：49.4% × 0.63 + 50.6% × 0.233 = 0.429；dsv41：38.2% × 0.63 + 61.8% × 0.162 = 0.341
- 反事实 1（给 dsv41 换上 th3k 的落入频率）→ 0.393：**"更常掉进压缩区"解释 5.2 pt（59%）**
- 剩下 3.6 pt（41%）来自压缩区内通过率更低（0.162 vs 0.233）

但两块都不等于"compaction 机制造成"。26 条 th3k 赢的配对里 6 条 dsv41 没压缩就输了；可精确对比的 12 条（th3k 零压缩解出、dsv41 压缩后失败）中，**dsv41 在第一次压缩之前的段落就已用掉超过 th3k 整条解题轨迹的步数（10/12）和写文件次数（10/12）**，压缩前段 obs 错误率 0.40（th3k 全程 0.22）、C-c 4.2 次（th3k 0.3）、测试 0.3 条（th3k 1.5）。典型：rstan-to-pystan 压缩前已 119 步修环境、polyglot 压缩前 15 步 obs-err 0.93、regex-log 压缩前 20 步 obs-err 0.70。**失败在第一次压缩之前就已成定局。**

真正能记到 compact 头上的：(1) 重建税——每次压缩后 ~8–15 步重读状态，th3k 2.66 次/轨迹 ≈ 每条压缩轨迹 20–40 步 / 10–20 分钟，只对贴着 4 h 线失败的任务有边际影响；(2) 压缩区内 th3k 仍优的 7 pt 里 handoff 忠实度（0.58 vs 0.46）可能有贡献，但与 PASS/TTL 窗口行为无差矛盾，更可能是上游 think 质量的投影。

**结论：8.5 pt 差距里可归咎于 compaction 机制/处理能力的 ≤1–2 pt；≥6.5 pt 是"不收敛 → 步数膨胀 → 被迫反复压缩 → TTL"这条链，压缩只是链上的可见症状。** 反向证据：ncomp=1–2 时 pass 还有 0.50–0.52（宣告率 89–96%），崩塌只在 ncomp≥3，而 ncomp≥3 的充分条件是"已经卡了 200+ 步"。

对训练的推论：不需要专门补 compaction 处理数据；止血注入的触发点可用 ncomp（第 3 次压缩）代替 think 长度，它是比"单步 think >12k"更干净的失败预警（ncomp≥3 → pass ≤0.12，覆盖几乎全部 TTL）。

---

# Part C. 五篇论文的 teacher / 轨迹选取与借鉴

论文：TH3K（2610.13166）、Tmax（2606.23321）、Data Recipes for Agentic Models（2606.24855）、FACET（2608.18580）、Terminal-Universe（2609.04148）。

## C1. 一览

| | **TH3K** | **Tmax** | **Data Recipes** | **FACET** | **Terminal-Universe** |
|---|---|---|---|---|---|
| Teacher | DeepSeek-V4-Pro-0813 | Qwen3.6-27B（SFT 仅为弱模型暖启动） | **GLM-4.7-AWQ**（5 选 1 消融） | DeepSeek-V4-Pro（无消融） | Qwen3.7-Max（单 teacher） |
| 选 teacher 依据 | **100 题子集比行为统计**：feedback chain 30.3 vs 19.2、exact repeats 13.2% vs 15.8%、pass 50% vs 44%（vs GLM-5.2） | 未明说；暗示格式兼容 | **5 个 teacher 各训一个学生直接比**；最强的 GPT-5.3-Codex 是**最差** teacher（TB2.0 差 ~5 pp） | 无 | 无；§6.1 证明"同 teacher 重解"远胜"模仿异构源轨迹" |
| 成功过滤 | **不用**（judge 看不到 reward） | **不用**（沿用 Nemotron 先例） | **不用** | **只留 solved**（1270/6066 = 20.9% → 取 1200） | **用 verifier**；多轮会话"剪尾部连续失败轮，保留失败→修复" |
| 过程质量过滤 | **LLM judge 三准则**，~93% KEEP | 只滤 tool-call 解析失败（~6%） | **min-turns ≥5**（compute 对照 +3.5 pp） | 无（质量控制全在任务合成侧：22.77 测试/题） | 无 judge；verifier 过滤消融 ±噪声 |
| 规模 → 学生收益 | 2,852 条 → Qwen3.5-27B +13.85 | 16.5k 条 → 帮 Qwen3-8B（+5~7），**伤 Qwen3.5-9B（−6~−10）** | 100k 条 → Qwen3-32B TB2.0 26.2 | **1.2k 条短轨迹（11.9 turns）→ 4B +7.1 / 9B +8.2 / 27B +6.8** | 32k 条 → 27B +11.9 |

TH3K judge 三准则原文要点（App. F / J.1）：(1) 轨迹有效性——截断、协议损坏、泄漏/篡改/绕过才 DROP，孤立错误可恢复不算；(2) 环境接地——用文件/日志/错误/测试结果指导决策，"排除不可行方案、定位根因也是进展"；(3) 多轮训练价值——"**Waiting, repeated polling, and mechanical loops receive no additional credit merely for increasing the number of turns**"，"Do not judge training value solely from trajectory length, command count, the presence of mistakes, or final task success"。

## C2. 跨论文共识

1. **最强模型 ≠ 最好 teacher，行为统计才是选取标准。** Data Recipes 直接实验；TH3K 的做法是操作化（100 题比 chain / repeats / 长度）。我们是第三个独立复现：qf 教师 0.771（最高）→ 学生 0.174（最差）。
2. **风格一致 > 数量，混源有害。** Terminal-Universe §6.1：SFT 异构源原始轨迹 35.8k 条 → 40.3 < base 46.2；同 teacher 重解 → 52.9。与我们 B6 "不混用"、dsvh 0.264 同构。
3. **失败轨迹该留，前提是过程可靠。** TH3K / Tmax / Data Recipes 都不按 solved 筛；我们 solved-only 全部 −5~8 或零收益。FACET 是反例，但条件我们不具备（题均 22.77 个 pytest 断言、11.9 turns 短轨迹）。
4. **轨迹级过滤的有效维度是多轮交互密度，不是对错。** Data Recipes 唯一做 compute 对照的是 min-turns ≥5；TH3K 准则 3 是同一思想的 LLM 化。但 **turns 类指标可被轮询刷高**（Part D）。
5. **SFT 对强学生可能负收益。** Tmax：所有 SFT 混合都伤 Qwen3.5-9B，只帮 Qwen3-8B → 强模型直接 RL。对我们 9B 路线的预警。
6. **扩数据扩任务不扩采样。** Data Recipes 同任务多 rollout 在 31.6k→100k 平台化；Terminal-Universe 2× 环境（56.0）> 2× query（53.8）≈ 2× 采样（53.9）。

## C3. 对我们的借鉴

- **A. teacher 行为准入筛**（借 TH3K Table 3 + Data Recipes Table 6）：100 题固定子集跑候选 teacher，量 feedback chain、exact repeats、末 1/4 ÷ 首 1/4 think 增长、单步 think p99、测试命令率、**sleep/clear/长 duration 占比、折扣后有效命令/turn**、收尾率。qf 当场出局。成本 = 100 题采集，比一个 SFT run 便宜一个量级。**注意 Part D：只抄 Table 3 四个指标会选错 teacher，折扣项必须在筛子里。**
- **B. 坏数据重解，不洗**（借 Terminal-Universe intent-recovery）：qf 覆盖的 TH 题用过筛 teacher 重打，redo pipeline 现成。
- **C. judge 之上叠行为指纹过滤**：judge 管"真不真"（抓作弊/废轨迹），指纹管"像不像好 teacher"（末段 think 增长 ≤4×、无 envfix、含测试命令、剔 <5 turns 平凡段）。
- **D. 失败轨迹剪尾不整剔**（借 Terminal-Universe multi-round 规则）：dsv41 TTL 轨迹头部留诊断-修复链、尾部剪 think 爆炸死循环，与 overlong tail 窗互补。
- **E. FACET 的提示**：若给 TH 题补 FACET 式密集分项 verifier（题均 20+ 断言），成功过滤才会变成可用信号，E3 的判据也从代理变成真信号。
- **F. 9B/27B 风险对冲**（借 Tmax）：dsv41_judged_9b 评测以"是否低于 9B base 最优档 0.267"为止损线；不涨则转 RL（E5 KTO 同向）。

---

# Part D. 论文准则复核：四套真实训练输入 → 学生行为的传递规律

训练输入：`terminal_horizon_th3k_mask/th3k_mask.jsonl`（2441 整轨迹）、`code_sft_data/dsv41_json/dsv41-json-262k.jsonl`（arm A 源，3593 行 / 2037 条入窗）、`terminal_horizon_dsv41_judged/dsv41_judged.jsonl`（dsv41_final 切段 8876）、`terminal_horizon_qwen38_flash/qwen38_flash.jsonl`（11001 段）；另 `code_sft_data/flash_linear/flash-linear.jsonl`（qwen38 2732 整轨迹）做口径对照。整轨迹全量扫描，段文件抽样 ~500–700。工具 `sft_paper_criteria.py`。

**折扣规则**（TH3K judge 准则 3 的量化）：`clear`、`sleep N`、轮询类命令（tail / ps / cat *status|log* / gmx check / echo CHECK|AFTER|PING|READY / wc -l）和 duration ≥30 s 的命令不计为有效动作；一个 turn 至少含一条非轮询命令才算"有效 turn"。

## D1. 训练数据：表面指标 vs 折扣后指标

| | **th3k**（2441 整轨迹） | **dsv41 armA**（整轨迹） | **dsv41_final**（段） | **qwen38 flash**（段） | qwen38 整轨迹（对照） |
|---|---|---|---|---|---|
| **论文 Table 3 表面指标** | | | | | |
| turns / 样本 | 40（p90 65）† | 85（p90 184） | 25（p90 49） | 37（p90 61） | 147（p90 337） |
| feedback chain（最长连续动作） | 24.7 | 40.1 | 19.2 | 29.7 | **78.1** |
| 命令级 exact repeats | 13.3%（论文 DSV4-Pro 13.2%） | 11.3% | — | — | **4.8%** |
| teacher pass | ~47%（重放 25%） | 48% | 44%（段） | 34%（段） | 47% |
| **折扣项（judge 准则 3）** | | | | | |
| `clear` 占命令 | 0.3% | 0.0% | 0.1% | **31.6%** | 25.2% |
| duration ≥30 s 命令 | 1.2% | 0.5% | 0.5% | **6.2%** | 6.2% |
| 纯等待 turn（空 keystrokes） | **12.4%** | 10.0% | 6.2% | 2.2% | 3.4% |
| 声明等待 + sleep 分钟/样本 | 0.8 | 1.1 | 0.3 | 0.8 | **3.6** |
| **折扣后有效指标** | | | | | |
| 有效 turn 占比 | 85.8% | **88.1%** | 91.3% | **72.0%** | 73.9% |
| 有效命令 / turn | **2.74** | 2.33 | 2.02 | **1.62** | 1.53 |
| 有效命令 / 有效 turn | **3.17** | 2.64 | 2.23 | 2.15 | 2.03 |
| 有效 turn 的 think p50 | 4,088 | **5,995** | 920 | 2,648 | 1,259 |
| think mean（全 turn） | 6,716 | **11,234** | 2,609 | 5,690 | 3,652 |
| **结构 / 准则 1** | | | | | |
| 以 task_complete 结尾 | **100%** | 93% | **30%** | **20%** | 87% |
| 段首为 handoff 摘要 | 0 | 0 | ~75% | ~75% | — |
| <5 turns 样本 | 0% | 0.1% | **9.3%** | 1.0% | 0% |
| 机械循环 turn（≥3 同批） | 8.3% | **17.1%** | 9.9% | 3.7% | 3.6% |
| 短 think <20 字符 | 0.2% | 0.9% | **3.9%** | 0.4% | 0.3% |
| 不可解析动作轮 | 1.1% | 1.1% | 0.7% | 2.1% | 1.5% |
| 错误观察后原样重发同批（准则 2） | 2.2% | 1.7% | 1.8% | 1.4% | 0.8% |

† th3k_mask 是 2852 里装进 262k 的 2441 条（丢的 411 条恰是最长），所以 turns 比论文 52.8 偏短；命令重复率 13.3% 与论文 Table 3 的 13.2% 精确吻合——度量口径对上了。

读表：
- **表面指标会选错 teacher**：qwen38 的 chain 78、exact repeats 4.8% 在三家里最优；折扣后有效命令/turn 1.53 最差。原因是它不用空 keystrokes 等（4.6%，最低），而是发 `clear; sleep 300; cat status` 这样的实命令去等，每次轮询被计成"有命令的 turn"，链不断、重复率低（sleep 参数每次不同）。th3k 等待 turn 最多（12.4%）但折扣后无损——它用空命令等，不进命令统计。
- **准则 2（环境接地）区分不了三家教师**（0.8–2.2%），大 teacher 都会看反馈改命令；这解释了 Opus judge 对 dsv41 KEEP 97.9%——准则本来就是为抓废轨迹/作弊设计的。
- **收尾率是整轨迹 vs 段的分水岭**：整轨迹三家 ≥87%，切段后 dsv41 30%、flash 20%。th3k 100% 一半是教师功劳、一半是没切段。
- **dsv41 超出论文画像的只有两项**：think mean 11.2k（th3k 的 1.7 倍，重尾）、机械循环 17.1%（th3k 8.3%）。

## D2. 学生评测行为（同一折扣口径）

| | th3k → c312 | dsv41 armA → c224 | dsv41_final → （未评测） | qwen38 → qf_solved c105 | qwen38 → qf_const c129 |
|---|---|---|---|---|---|
| pass@all | **0.429** | 0.341 | — | 0.172 | 0.138 |
| TTL | 24% | 43% | — | 69% | 72% |
| 宣告率 / 宣告后通过率 | **74% / 0.58** | 55% / 0.59 | — | 31% / 0.44 | 23% / 0.50 |
| `clear` 占命令 | 0.6% | 0.2% | — | 25.8% | **40.3%** |
| duration ≥30 s 命令 | 1.2% | 0.3% | — | 8.3% | 8.0% |
| 纯等待 turn | 10.2% | 8.6% | — | 7.6% | 5.4% |
| 有效 turn 占比 | 89.0% | **90.7%** | — | 70.2% | **57.3%** |
| 有效命令 / turn | **2.29** | 2.19 | — | 1.47 | **0.93** |
| 有效 turn 的 think p50 | 6,580 | 5,058 | — | 1,789 | 1,281 |
| 机械循环 turn | 1.1% | 1.2% | — | 0.2% | 2.2% |
| PASS 轨迹步数 p50 / 全部 p50 | **32 / 67** | 29 / 169 | — | 37 / 92 | 130 / 228 |

## D3. 规律：哪些指标从数据传递到学生

**A. 高保真复制（学生 ≈ 教师）——teacher 选取该看的：**

| 指标 | th3k 数据 → 学生 | dsv41 数据 → 学生 | qwen38 数据 → 学生 |
|---|---|---|---|
| `clear` 占比 | 0.3% → 0.6% | 0.0% → 0.2% | 31.6% → 25.8–40.3% |
| ≥30 s 命令占比 | 1.2% → 1.2% | 0.5% → 0.3% | 6.2% → 8.0–8.3% |
| 纯等待 turn | 12.4% → 10.2% | 10.0% → 8.6% | 2.2% → 5.4–7.6% |
| 有效 turn 占比 | 85.8% → 89.0% | 88.1% → 90.7% | 72.0% → 57–70% |
| 有效命令/turn | 2.74 → 2.29 | 2.33 → 2.19 | 1.62 → 0.93–1.47 |

命令构成几乎逐字复制，4B 学生首先学到的是"每步发什么形状的命令"。**这组指标的三家排序（th3k > dsv41 > qwen38）与分数排序一致，而表面指标（chain、repeats）的排序是反的。**

**B. 结构项复制但被放大——分数的第二解释变量：**
- 收尾率：th3k 100% → 宣告 74%；dsv41 93% → 55%；qwen38 段 20% → 23–31%。数据收尾率约束学生宣告率，而**宣告后通过率三家几乎一样（0.44–0.59）**——分差来自"交不交卷"。
- think：数据 p50 3.5k / 5.8k / 2.5k（段）→ 学生 5.7k / 4.4k / 1.1k。dsv41 学生均值更短但 TTL 轨迹里爆到 50k+，复制的是重尾分布；qwen38 学生 think 坍缩到 1.1k，复制的是整轨迹口径的 1.26k（段首 handoff 后首步 think 偏长，拉高了段统计）。

**C. 不传递——论文 Table 3 的两项正落在这里：**
- feedback chain：数据 24.7 / 40.1 / 78.1 → 学生 PASS 步数 32 / 29 / 37–130，无单调关系。
- exact repeats：数据 13.3% / 11.3% / 4.8% → 学生 19.6% / 15.1% / 13.8–14.3%，三家学生都比教师重复得多，排序与数据无关。
- 机械循环：dsv41 数据 17.1%（最高）→ 学生 1.2%（与 th3k 持平）。

所以 Table 3 四个指标里只有 pass rate 勉强有信息量；能预测学生分数的是**折扣后命令构成 + 收尾率 + think 重尾**。

## D4. dsv41_final 的预测（P1 臂，训完未评）

段指纹：有效 turn 91%（最高）、有效命令/turn 2.02（低于 th3k 2.74）、think p50 920（三家最低，tool-mode 采集压短了 think）、收尾 30%、9.3% 样本 <5 turns、短 think 3.9%。按 D3 规律：命令构成干净 → 不会 qf 式 TTL；收尾 30% + think 920 → 预期落在 dsvh const（think 515、收尾 25%、0.264）与 arm A（think 5.8k、收尾 93%、0.326）之间、更靠 dsvh，即 **0.26–0.30，大概率不超过 arm A**。若实测如此，证明 judge 过滤（真实性）对分数的边际贡献小于结构项（收尾 + think 密度），与 Terminal-Universe verifier-filter 消融（±噪声）一致。

## D5. 对 teacher / 数据选取的修正结论

1. **论文 Table 3 指标必须折扣后再用**：有效命令/turn、有效 turn 占比、clear/sleep/长 duration 占比三项计算成本为零，应是准入筛第一层。
2. **整轨迹 vs 段的口径差比 teacher 差还大**：同一 qwen38 教师整轨迹收尾 87%、段 20%；同一 dsv41 arm A 93% vs dsv41_final 段 30%。任何分段数据至少做"非终止段末轮 loss=0"或只保留含 task_complete 的段链。
3. **dsv41 armA 是目前唯一能与 th3k 并列过筛的数据**，输的 8.5 pt 中可归因于数据指纹的是 think 重尾（mean 11.2k vs 6.7k）和机械循环（17% vs 8%），两项都能用轨迹级规则处理，不需要换 teacher。
4. **qwen38 折扣后没有可用的"好部分"**：有效命令/turn 1.53 四套最低、有效 turn 的 think 1.26k 也最低——不是"轮询掩盖了好教师"，去掉轮询后剩下的动作本身也稀疏。维持弃用。

## D6. 复现

```bash
# 训练数据：表面 + judge 准则量化（整轨迹 cap=0 全量，段文件默认抽 700）
python3 coding_eval/tools/sft_paper_criteria.py              # 三套默认集
# 折扣指标：在 python 里调用 sft_paper_criteria.discount_summary(name, path, manifest)
# 评测轨迹：行为分布 + compaction 分解；--ext 输出 handoff 忠实度
python3 coding_eval/tools/traj_compaction_behavior.py
python3 coding_eval/tools/traj_compaction_behavior.py --ext
# 产物：/tmp/opencode/traj_compaction_behavior.json（逐轨迹 ncomp / score / steps）
```
