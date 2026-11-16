# 失败轨迹分析：th3k 模型自己输在哪、非 th3k 数据为什么输给 th3k

> §8 分析旧 verifier 的 110 条失败；mirror 复判为 105 条，含 42 条 DECLARED_PARTIAL 和 8 条 NONDONE。[mirror 逐例复判](c312_failure_cases.md)

日期：2026-11-04 → 11-07。对象：Qwen3.5-4B 在 TB2.1 上的失败轨迹（4 h TTL、撞 512 步、宣告完成但测试不过）。分数本身在 [sft_results](sft_results.md)；本文只回答"为什么"：

- **§1–§7**（11-07）：非 th3k 数据训出的模型为什么输给 th3k——逐条对比同题轨迹，发现差距全部来自"卡住后不收敛"，并把每个数据系的死法一一对应到训练数据指纹和训练流程的放大器。
- **§8**（11-04 / 11-06）：th3k 最佳模型（mask_const c312）自己的 110 条失败分六类；"宣告完成但部分过"的根因不是没跑测试，而是需求覆盖不全；宣告时刻 think 越长通过率越低，这条规律在教师数据里也成立。
- **§9**：针对"部分过"的零训练 / 数据侧 / RL 实验设计 E1–E5（截至 11-09 未执行）。
- **§10**：可直接打开的典型轨迹索引。
- **§11**（11-10）：compaction 行为归因——三家学生都学会了"处理" compaction（忠实执行 handoff、重建状态），分数差里可归咎于 compact 机制本身的 ≤1–2 pt；崩塌只发生在 ncomp≥3，而那是"已卡死 200+ 步"的下游症状。
- **§12**（11-10）：按 TH3K 论文（arXiv 2610.13166）Table 3 + judge 三准则量化三家教师数据——表面指标会把 qwen38 flash 排第一，必须加命令构成折扣（clear/sleep/长 duration）才能筛掉它；th3k 是论文准则的标准正样本，dsv41 过准则但带 think 重尾。
- **§13**（11-10）：dsv41_final 上的尾部处理模拟——字面"剪尾部失败轮"不可行（连续卡死 run 几乎不存在、会删掉 94% solved 的收尾）；"切尾部 >20k think 轮"是反向优化（71% 是宣告自审、一半来自 solved）；对症手段 = 失败轨迹宣告轮 mask + think 相对增长 mask。th3k 自己长想更多（36.8% think 字符在 >20k 轮）但长在中段、短在收尾。

工具：`coding_eval/tools/traj_failure_deep.py`（每任务 30 项行为指标）、`classify_tb21_errors.py`（六类失败分类，读 `tbench_test.log`）、`sft_data_fingerprint.py`（训练数据指纹）、`traj_compaction_behavior.py`（§11，压缩边界前后窗口行为 + handoff 忠实度，`--ext`）、`sft_paper_criteria.py`（§12，论文 Table 3 / judge 准则量化）、`pland_tail_trim_sim.py` / `pland_stuck_trim_sim.py`（§13，尾部裁剪模拟）。全文版本：`/storage/openpsi/users/chen/output/tb21_{failure_trajectory_analysis,error_taxonomy,why_th3k_wins}.md`。

**结论先行**

1. **顺利的任务上，th3k 模型和 dsv41 模型行为几乎重合**（PASS 轨迹步数 29–32、think 3.6–4.3k、13–14 分钟）；差距全部来自 PASS→TTL 的转移。不是能力差，是卡住后的策略差。
2. **卡住后**，th3k 模型装环境、等它装完、写 `/app` 产物、跑测试、修；dsv41 模型打断安装、改 `sources.list`、在 `/tmp` 写几十个试验文件、think 爆到 50k+ 字符/步直到 4 小时。这个差异在训练数据指纹里一一对应：dsv41 chat 教师末段 think 增长 7.5×（th3k 3.7×）、测试命令 0.4×、改环境命令 6×。
3. **每个数据系各有一种"不收敛"**：tool-mode 段 → think 坍缩到 500 字符 + 77% 段无交卷尾 → 撞 512 步；dsv41 JSON 段 → think 重尾被放大 → 每步 54 秒 → TTL；qwen flash 段 → 学会长 sleep → TTL 75%；tree 训练（历史 think 不可见）→ 过早交卷。
4. **训练流程放大了数据缺陷**：token 均值 loss 让超长 think 轮主导梯度；overlong 整条丢弃删掉了"长时间仍在收敛"的唯一教材（dsv41 chat 集丢 43%）；以 compaction 段为样本让模型把"被截断"学成正常结局。
5. **th3k 模型自己的失败**：TTL 42 条集中在 24 个无人能解的 env（不丢分，丢算力）；DECLARED_PARTIAL 39 条里一半只差 1 个测试，根因是需求覆盖不全（benchmark 基本不给可运行的验收脚本，0–9%）；宣告时 think ≤3k 的通过率 72%，>20k 只有 8%。用"两条采样取宣告 think 更短者"做事后选择，免费 +6–7 pt。
6. **可操作**：筛数据看收敛行为（末段 think 增长 ≤4×、单步 ≤20k、含测试命令、无 envfix）而不看 solved；think 相对增长 mask 而不是固定 cap；overlong 用 tail 窗；非终止段末轮 loss=0；harness 侧在宣告时强制"需求清单 + 证据"（E1/E2）。

---

> **数据**：本地轨迹 `output/<job>/tasks/*/s*/{trajectory.jsonl,task.json,tbench_test.log}`，覆盖 mirror 口径 5 个 job（c312 / arm A c224、c96、c64 / rand c336）
> 与旧口径 10 个 job（c312 旧、tree c156、dsvh_const c126/c336、dsvh_solved c152、lin_solved c51、treef c168、qf_const c129、4B/9B base）。
> 工具：`coding_eval/tools/traj_failure_deep.py`（每任务 30 项行为指标，输出 `<job>/traj_deep.json`），
> 临时脚本 `/tmp/opencode/{obs_err,ttl_anatomy,patience}.py`。训练数据指纹来自 `data/terminal_horizon_{th3k_mask,dsv41_json262k,dsv41_history}/*.jsonl` 各抽 500 条。
> 失败类别：TTL = 4h 被杀；CAP512 = 撞 512 步上限（仅 1.2.96 harness 有）；DECLARED_PARTIAL/ALLFAIL/NOTEST = 宣告完成后测试部分过/全不过/没跑起来；NO_DECLARE = 会话中途死。

## 0. 一句话结论

**所有非 th3k 模型输给 th3k 的方式只有一种：卡住后不会"收敛"。** th3k 模型和 dsv41 模型在"顺利的任务"上表现几乎一样（PASS 轨迹步数 29–32、think 3.6–4.3k、13–14 分钟），差距全部来自 **PASS→TTL 的转移**（arm A ep7 比 c312 多 29 条 TTL，占全部差距）。
卡住后 th3k 模型做的事是：装环境、等它装完、写一个 `/app` 产物、跑测试、修；dsv41 模型做的事是：打断正在跑的安装（C-c）、改 `sources.list`、在 `/tmp` 写几十个试验文件、把 think 拉到 50k+ 字符/步、每步 30–40 秒、直到 4 小时。这个差异**可以直接在训练数据里找到来源**（§4）。

## 1. 差距的结构：全是 TTL

mirror 口径，同 harness 1.0.0，同采样 index 逐条对比 c312（th3k ep8）与 arm A c224（dsv41 chat ep7）：

| 转移 c312 → c224 | 条数 |
|---|---|
| PASS → PASS | 45 |
| TTL → TTL | 33 |
| PARTIAL → PARTIAL | 19 |
| **PASS → TTL** | **14** |
| **PARTIAL → TTL** | **12** |
| PASS → PARTIAL | 9 |
| PARTIAL → PASS | 7 |
| TTL → PASS | 3 |

- c312 pass 73 / TTL 41；c224 pass 58 / TTL 70。**TTL 多了 29 条，pass 少了 15 条**。PARTIAL（39→31）、ALLFAIL（8→8）、NOTEST 没有恶化。
- 逐 env：th3k 净赢 25 题，其中 **19 题是 c224 跑成 TTL**（crack-7z-hash、rstan-to-pystan、compile-compcert、caffe、build-pov-ray、train-fasttext、polyglot-c-py、qemu-startup、regex-log、mailman…）；dsv41 净赢 10 题，全是 c312 自己 TTL/PARTIAL 的题。
- 这不是"dsv41 模型更笨"：**PASS 轨迹的所有行为指标两者几乎重合**（见下表第一行），说明基础能力相当，差在"不顺时怎么办"。

| | c312 PASS | c224 PASS | c312 TTL | c224 TTL |
|---|---|---|---|---|
| 步数 / 分钟 | 32 / 14 | 29 / 13 | 441 / 240 | 374 / 240 |
| think p50（字符） | 4257 | 3637 | 6952 | 4518 |
| think>50k 的步数（均值） | 0.3 | 0.5 | 13.2 | 8.4 |
| 测试命令/任务 | 7.3 | 5.2 | **53** | **25** |
| 等待步（空命令） | 10.5 | 7.1 | 29 | 36 |
| C-c / 任务 | 1.3 | 2.2 | 15.6 | 13.1 |
| 生成速度 tok/s | 104 | 83 | — | — |

同口径的 arm A ep2/ep3 更差（TTL 83/82 条），但模式完全一样。

## 2. 卡住之后两种模型做了什么（TTL 轨迹解剖）

对 c312 PASS 而 c224 TTL 的 14 条同题轨迹逐条统计（`/tmp` 写文件数、`/app` 写文件数、包管理命令数、环境改造命令数、think>20k 的步数）：

| env | c312：步/分 tmp app pkg envfix | c224：步 tmp app pkg envfix think>20k步 |
|---|---|---|
| polyglot-c-py s1 | 6 / 19，tmp 0，app 2，pkg 0 | 185，**tmp 103**，app 1，envfix 35，**64 步 >20k** |
| regex-log s0 | 15 / 20，app 1 | 312，tmp 21，app 1，66 步 >20k |
| rstan-to-pystan s0 | 77 / 49，pkg 3，envfix 0 | 670，**pkg 111，envfix 24** |
| fix-ocaml-gc s1 | 87 / 25，pkg 1 | 540，pkg 100，envfix 46 |
| build-pov-ray s1 | 167 / 132，pkg 97，envfix 30 | 330，**pkg 330**，envfix 48 |
| qemu-startup s0 | 57 / 33，pkg 0 | 214，tmp 0，35 步 >20k |
| compile-compcert s0 | 194 / 162，pkg 99 | 301，pkg 162 |

三种具体死法（都有原始轨迹可查）：

**(a) 打断安装 → 环境被自己搞坏 → 修环境到死（rstan-to-pystan s0）。**
两个模型前 3 步完全一样（ls、读 R 脚本、`pip install pystan`，撞 `externally-managed-environment`）。c312 第 2 步直接 `pip install --break-system-packages`，第 29 步 `apt-get update && apt-get install -y build-essential`，然后**连续 5 步空命令等待**（think 90–160 字符："apt-get is still installing… continue waiting"），第 35 步开始写 `/app/pystan_analysis.py`，77 步通过。
c224 第 2 步建 venv，第 3 步 `apt-get update -qq` 还没跑完**第 7 步 `C-c` 打断**去装 pip；第 44 步再装 build-essential 时 apt 列表是坏的（"Unable to locate package gcc"），于是开始用 python 改 `/etc/apt/sources.list.d/ubuntu.sources`（写出 `debs-src` 拼写错误），之后 600 步在 apt 缓存、dpkg 手工解包、cc1plus wrapper、clang 替代之间打转，`pkg` 命令 111 条、envfix 24 条，4 小时。

**(b) 把任务变成研究课题（polyglot-c-py s1 / regex-log s0）。**
c312 polyglot：第 0 步 think 116k 字符一次想清楚，直接写 `/app/polyglot/main.py.c` + 测试，6 步通过。c224：第 0–3 步 think 77k/209k/128k/97k，但**写的是 `/tmp/testpoly.c`、`/tmp/test_cross1..8.c`**——103 个 `/tmp` 试验文件，185 步里只碰过 `/app` 一次（第 6 步），其余时间在用 `gcc -E` 探索预处理器行为、`#pragma` 技巧、宏定义 `main`。regex-log 同理：c224 对同一个 `regex.txt` 重写 25 次，每次写新的 python 诊断脚本，第 311 步还在修诊断脚本自己的 SyntaxError。

**(c) 不先交产物（qemu-startup s0）。**
c312 第 36 步 QEMU 已后台跑起来、telnet 口已开，之后 20 步都在验证登录提示符。c224 从第 23 步开始解析 ISO 的 gzip 偏移、分区表、qcow2 头、写 MBR，214 步从未让 QEMU 在后台稳定跑起来。

**共同特征**：卡住时 think 爆涨（c224 TTL 轨迹平均 8.4 步 >50k 字符，单步最大 70k），每步 30–40 秒，而 th3k 模型卡住时主要靠"空命令等待"（think 100 字符）耗时。注意 c312 TTL 轨迹也有 13 步 >50k——**两者卡住后都会 think 爆炸，差别是 dsv41 模型卡住的频率是 th3k 的 1.7 倍**（TTL 70 vs 41）。

## 3. 其他失败家族：各有一种"不收敛"

| 家族 | 主失败类 | 轨迹特征 | 根因 |
|---|---|---|---|
| **tool-mode dsv41**（dsvh_const/rand/solved、lin_solved，harness 1.2.96） | **CAP512 61–93 条**（撞 512 步，仅 85–110 分钟） | think 500–1200 字符/步，每步 10–13 秒；等待步 42–45/任务、C-c 8–11/任务、命令重复率 0.25；compaction 6–10 次 | 教师轨迹 think 坍缩到 ~500 字符（训练集 think50=515，PASS 轨迹 think 986）；"短想→快发命令→不看结果→再发"，512 步跑完还没写完产物。dsvh_const c336 CAP512 轨迹 **no-tool 步 7.6/任务**（输出格式都开始崩） |
| **tree_flash c168**（qwen3-8-flash 96k 段，nothink 评测） | **TTL 73 条，只有 119 步** | **54 秒/步，completion 5.4k tok/步，12 步 >50k think**，几乎不等待（wait 0.9 分） | 纯 think 肿胀：教师 think 重尾（16% 步 >20k）被学生放大，TTL 不是卡住而是**每步都慢** |
| **qf_const c129**（qwen3-8-flash 段） | **TTL 122 条** | **sleep/duration 参数累计 81 分钟/任务、≥60s 的命令 39 条/任务**，pkg 命令多但测试 0 | 学会了教师的"长 sleep 等待"，4 小时里 1/3 在睁眼等；段样本只 23–36% 以 task_complete 结尾 |
| **tree_th3k c156**（同 th3k 数据、think 不可见训练） | **DECLARED_PARTIAL 59 + ALLFAIL 31**（不是 TTL） | PASS 只用 21 步 / 5 分；宣告时 think 2.4–3k；宣告后通过率 0.31（c312 0.58）；宣告前跑过测试的比例 0.16 | 过早交卷：think 不可见训练让模型学不到"上一步我想了什么"，一次写完就宣告 |
| **4B/9B base** | **DECLARED_NOTEST 151/127** | think 312–394 字符，宣告前跑测试 0.17，28 步就宣告 | 根本没做出产物就宣告；SFT 的首要收益就是把这一类消掉（th3k 后 NOTEST 1–13） |

## 4. 为什么：训练数据指纹对照

| 指纹（每轨迹，抽 500 条） | **th3k_mask**（2441） | **dsv41_json262k**（arm A，2037） | dsv41_history（tool-mode，11146） |
|---|---|---|---|
| 步数 p50 | 36 | 35 | 28 |
| think p50 / 第一步 / **最后一步** | 2992 / 304 / **1952** | 3474 / 329 / **9355** | 515 / 614 / 973 |
| 末 1/4 think ÷ 首 1/4 think | **3.7** | **7.5** | 1.8 |
| think>20k 的步占比 | 6.9% | **14.3%** | 1.9% |
| 测试命令/轨迹；含测试的轨迹比例 | **2.5；30%** | 1.0；22% | 1.2；20% |
| 命令/步 | 2.78 | 2.47 | 1.94 |
| 包管理命令/轨迹 | 4.1 | 3.2 | 1.1 |
| 环境改造命令（sources.list / alternatives / CC=）| 0.01 | **0.06** | 0.01 |
| /app 写入 ÷ /tmp 写入 | 1.1 / 0.9 | 1.9 / 1.0 | 0.4 / 0.7 |
| 以 task_complete 结尾 | 99% | 94% | **25%** |

对应关系是一一的：

1. **末段 think 7.5 倍增长 + 14% 超长步** → 学生学到"越往后越要想得多"，一卡住就 50k+ 字符/步 → 每步 30–40 秒 → TTL。th3k 教师末段只涨 3.7 倍（而且末步 1952 字符 = 简短收尾）。
2. **测试命令是 th3k 的 40%** → 学生卡住时自验证少一半（TTL 轨迹测试命令 25 vs 53），更依赖"再想一遍"而不是"跑一下看"。
3. **环境改造命令 6 倍** → 装不上就改源、改 alternatives，这是教师（DeepSeek-V4.1）在自己采集时的习惯，4B 学生复制了动作但复制不了判断。
4. **tool-mode 集 25% 以 task_complete 结尾、think 515** → dsvh/lin 系全体 CAP512：既不收尾也不思考，纯发命令。
5. th3k 的 **99% 收尾 + 高测试率 + 平稳 think** 不是因为它"全解对"（重放只有 24.9% pass），而是风格一致：每一步 2–3k think、2.8 条命令、该等就等、写完就测、测完就交。4B 模型能学会的是这个流程，学不会的是教师的推理深度。

## 5. 排除掉的解释

- **不是 verifier**：mirror 口径下所有 job pytest 未启动 = 0；c312 的 +11 题里只有 +5 可归因 verifier。
- **不是 serving 速度**：c312 104 tok/s vs arm A 83 tok/s，每步秒数相同（28s）；arm A TTL 轨迹的墙钟里只有 13–14 分钟在等待命令执行。
- **不是 compaction**：TTL 轨迹两者都 6–9 次 compaction；compaction 后探索比例 arm A 反而更低（0.36 vs 0.51）。
- **不是 nothink 评测条件**（tree 系 +2/−7/+6 方向不一致）。
- **不是过早交卷**（对 arm A 而言）：c224 宣告后通过率 0.59 与 c312 0.58 相同；过早交卷是 tree_th3k 和 base 的问题，不是 dsv41 的。
- **PARTIAL 的"差一个测试"两家都是 50%**，与数据无关，是 4B 能力上限。

## 6. 可操作的结论

1. **筛数据看"收敛行为"，不看 solved**：对 dsv41 chat 轨迹按（末段 think 增长 ≤4×、单步 think ≤20k、含 ≥1 条测试命令、无 sources.list/alternatives 改造）过滤或降权；solved 过滤已证明无效（−5~8 题），因为失败来源不在正确性。
2. **think-cap 不够，要 cap 的是增长**：arm B（think>20k 的步 loss=0）方向对，但 20k 阈值只覆盖 14% 的步；按"相对首段 ≤4×"做 mask 更贴近 th3k 分布。
3. **补"等待"与"先交产物"样本**：th3k 卡住时用空命令等待（think 100 字符），dsv41 用 C-c + 重来。可以从 dsv41 轨迹里单独保留"apt/pip 装包 → 等待 → 成功"的段落上采样。
4. **harness 侧止血（不改模型）**：单步 think 超过 N 字符 / 任务超过 M 分钟时注入"先把当前最佳产物写到 /app 并跑测试"的提示，专门打 PASS→TTL 的 26 条转移；这是 §9 E2 的变体，对 th3k 模型同样有效（它自己也有 41 条 TTL）。
5. 下一次评测前先跑 `traj_failure_deep.py` 看 PASS 轨迹是否重合、TTL 是否来自"卡住后 think 爆炸"——这比总分更早告诉你数据有没有学到流程。

## 7. 各家族的"不收敛" ↔ 训练数据 / 训练流程（11-07 补）

训练流程对所有家族是同一套：Qwen3.5-4B-text、262k 窗口、**整段一次展开、历史 think 可见**（tree 系除外）、bad action loss=0、**overlong 整条丢弃**、每步 64 条轨迹、`global_sort`（一步 = 64 条长度相近的轨迹）、constant lr 1e-5 + 10% warmup、8 epoch、token 均值 loss。所以家族之间的差别几乎全部来自数据，流程的几个固定选择决定了数据缺陷被放大的方式。

### 7.1 每个家族一张"症状 → 数据 → 流程放大器"表

| 家族（评测症状） | 训练数据的直接对应 | 流程放大器 |
|---|---|---|
| **arm A / dsv41 chat 整轨迹**（PASS→TTL：卡住后 think 爆到 50k+/步、C-c 打断安装、改 sources.list、/tmp 试验而不交 /app） | 教师 DeepSeek-V4.1 的习惯：末 1/4 think = 首 1/4 的 **7.5×**（th3k 3.7×）；**14.3%** 的步 think>20k（th3k 6.9%）；envfix 命令 6×；测试命令 0.4×。监督字符里 **86% 是 think，52% 落在 think>20k 的轮次**（th3k 79% / 34%） | ① token 均值 loss → 梯度被超长 think 轮主导，模型学到的首先是"卡住就长想"；② overlong 整条丢弃 **1556/3593 = 43%**（th3k 14%），丢掉的正是 p50 99 步的长轨迹——"长时间仍在收敛"的样本在训练集里不存在，评测里一旦超过 60 步就进了无监督区；③ 历史 think 可见训练 + 评测 harness 1.0.0 可见，条件一致，这一项不是问题 |
| **tool-mode dsvh_const / rand / solved、lin_solved**（CAP512：think 500/步、10 秒/步、不收尾、格式崩） | 采集用 tool-call 模式：think 坍缩到 **515 字符/步**（PASS 轨迹要 ~1000）；**75% 的段以 compaction 交接摘要开头、77% 的段不以 task_complete 结尾**（被 compaction 切断）；4.7% 静默工具失败；cmd/步 1.94 | ① 以"段"为样本：模型学到的终止方式是"对话被截断"而不是"宣告完成"→ 评测里跑到 512 步也不交卷；② 8 epoch 下 loss 降到 **0.39**（th3k 0.95、arm A 0.77）——短 think 被整段背下来，泛化最差；③ lin_solved 把段线性化、92% 收尾，**但 think 仍 ~850 且 solved 过滤掉了"失败后修正"的样本** → 0 收益，证明问题在教师风格不在切段 |
| **tree_flash c168**（TTL 但只 119 步：54 秒/步，每步 completion 5.4k tok） | qwen3-8-flash 教师 think 重尾（28% 监督字符在 think>20k 轮）且 nothink 采集 | ① tree 训练把每步当独立分支监督，**没有任何信号约束"相对前几步该想多长"**，学生把重尾当常态；② 评测 1.3.96 nothink：模型看不到自己上一步想了什么，每步从头想 → 肿胀叠加；③ 只训了 0.5 epoch（168/336 步）就评，欠拟合阶段风格最漂 |
| **qf_const / qf_solved**（TTL：sleep/duration 累计 81 分钟/任务、测试 0） | qwen3-8-flash 轨迹里 **duration p90 = 30 秒、5.5% ≥60 秒、每条轨迹声明等待 7.3 分钟**（th3k 1.6、dsv41 0.5–1.0）；段只有 21–26% 收尾 | ① duration 是被监督的 JSON 字段，模型逐字学会长等待；② 同样是"段"样本 → 不收尾；③ 96k harness 下 compaction 更频繁，每次交接后又回到"摘要开头"的训练分布，循环 |
| **tree_th3k**（DECLARED_PARTIAL/ALLFAIL：21 步 5 分钟就交卷，宣告前跑测试只有 16%） | 数据与 mask_const 完全相同 | **纯流程项**：tree = 历史 think 不可见训练。th3k 的 think 里大量是"上一步我验证了 X，还差 Y"的接力，去掉可见历史后，每一步只能从剥过 think 的主干推断进度 → 低估剩余工作 → 早交。mask_const 同 step 恒高 4–15 题是这个机制的代价，nothink 评测也救不回来（+2/−7/+6） |
| **mask_const（th3k）自己的 41 条 TTL** | th3k 卡住时也会 think 爆炸（TTL 轨迹 13 步 >50k），只是频率低；数据里 overlong 丢了 411 条（14%） | 和 arm A 同一机制，程度轻；这是 harness 侧止血（§9 E2）的目标 |

### 7.2 流程层面三个共同放大器

1. **token 均值 loss + 不裁 think**：谁的 think 长谁主导梯度。dsv41 chat 集 52% 的监督字符来自 think>20k 的轮次，而这些轮次恰恰是"卡住"时刻——等于专门强化了卡住的行为。arm B（think>20k 轮 loss=0）是对的方向，但 cap 应该作用在"相对首段的增长"上，固定 20k 只覆盖 14% 的轮。
2. **overlong 整条丢弃**：按轨迹总 token 过滤，系统性删掉长轨迹，而长轨迹正是"卡住后仍然收敛"的唯一教材。dsv41 chat 集丢 43%。替代：`mask_tail`（任务 prompt + 尾窗）已在 th3k 上实现，应用到 dsv41 上才能保住"长→收敛"的样本；或者按 think 裁短后再判 overlong。
3. **以 compaction 段为样本 + 不区分段尾**：tool-mode / qf 系 77% 的样本在非终止处结束，模型把"被截断"学成了正常结局。至少要给非终止段的最后一轮 loss=0，或只保留含 task_complete 的末段 + 它之前的段作为上下文。

### 7.3 两个反证，说明"不是能力、是风格"

- arm A 与 c312 的 PASS 轨迹行为重合（§1 表），训练 loss 曲线也几乎重合（ep1→ep8：1.06→0.92 vs 1.20→0.95），模型学到的"能力量"相当，差在卡住时的策略。
- dsvh_const ep3/5/7/8 = 0.242/0.258/0.213/0.219，训练 loss 从 0.83 降到 0.39 却不涨分——拟合得越好越像教师的 tool-mode 风格，而那个风格本身不收敛。

### 7.4 对下一轮训练的直接要求

- 数据：dsv41 chat 轨迹按（末段 think 增长 ≤4×、单步 ≤20k、含测试命令、无 envfix）过滤；tool-mode 段只保留以 task_complete 结尾的段链；qf 系弃用。
- 流程：think 相对增长 mask；overlong 用 tail 窗而不是丢；非终止段末轮 loss=0；评测前用 `traj_failure_deep.py` 看 PASS 轨迹是否与 th3k 重合、TTL 是否来自 think 爆炸。

## 8. th3k 最佳模型（mask_const c312）自己的失败：六类归类（11-04，旧口径 110 条）

> 工具 `classify_tb21_errors.py`，读 trajectory.jsonl + task.json + 官方 verifier 的 `tbench_test.log`（110 条 test log 全部拉齐）。对照 job：tree_th3k c156 @1.0.0、dsvh_const c210 @1.2.96、tree_flash c168 @1.3.96、qf_const c129 @1.2.96、4B/9B base。本节是旧口径（0.348，110 个失败）；mirror 口径重测后 TTL 41、pass 73，结构不变。

### 8.1 六类

| 类别 | 条 | 占失败 | 定义 | steps p50 | think/步 p50 | 时长 p50 |
|---|---|---|---|---|---|---|
| **TTL** | **42** | 38% | 4 h 被杀，从未交卷 | 382 | 6.8k | 240 min |
| **DECLARED_PARTIAL** | **39** | 35% | 宣告完成，测试部分通过 | 41 | 5.7k | 28 min |
| DECLARED_NOTEST | 13 | 12% | 宣告完成，verifier 没跑起来 | 29 | 4.7k | 14 min |
| DECLARED_ALLFAIL | 11 | 10% | 宣告完成，测试全挂 | 77 | 10.6k | 100 min |
| NO_DECLARE | 4 | 4% | 没交卷也没超时 | 108 | 5.8k | 64 min |
| CAP512 | 1 | 1% | 撞 512 步 | 554 | 6.5k | 232 min |

**13 条 NOTEST 全是 verifier 基础设施**（11 条 uv/pypi 超时、2 条 `prove-plus-comm` 的 "No working directory set"）；把 `tbench_test.log` 全部拉齐后实际是 17 个 task 的 pytest 从没启动（另 4 个是 TTL/CAP512 顺带挂的）。mirror 口径已修（[tb21_verifier_audit](../methods/tb21_verifier_audit.md) §7），对 c312 的实测贡献是 +5 题。模型侧真失败 = 97 条，三个主因：

**(a) TTL（42 条 / 24 个 env）——"想得多、干得多、不收敛"。** 不是空转：uniq-cmd 0.56–0.97、sleep ≥30 s 的命令只占 1.6%、无 apt/pip 挂死；每条 382 步、1.4M completion token、压缩 7.8 次。`video-processing` s1 806 步没写出最终版 `/app/jump_analyzer.py`，第 500 步 think 里自己写 "We've spent lots of turns exploring but not yet written final script"；`llm-inference-batching-scheduler` s0 415 步每步只改一点 cost model；`build-pov-ray` / `compile-compcert` 1173 / 888 条命令里 430 / 339 条是 `echo '--- xxx ---'; cat …` 带标签读文件，apt 源里没有源码，绕了 4 小时。TTL 轨迹的 think（6.8k）比 PASS（3.4k）长一倍。**这不是想太少，是没有止损机制**——"这条路走不通，先交一个最简版"的行为在 th3k 数据里不存在。

**反证——同样这 24 个 env，所有模型**：mask c312 46 条 1 pass / 42 TTL；tree_th3k 42 条 0 pass / 0 TTL；dsvh_const 28 条 0 / 0；treef 45 条 0 / 31 TTL；4B base 39 条 2 / 0；9B base 28 条 1 / 0。**这 24 题谁都做不出**，mask 耗 4 h 和 tree/dsvh 40 min 放弃，分数同为 0。TTL 不损失分数，损失的是 168 GPU·h 墙钟（和 RL 阶段的 rollout 预算）。整个 benchmark 有 30 题没有任何 4B/9B 模型解出过（含 mirror 重测；教师 dsv41 解出其中 22 题），这是天花板的硬核。

**(b) DECLARED_PARTIAL（39 条）——"做对 80%，差一个 edge case 就交卷"。** `cancel-async-tasks` s0 12 步 10 分钟 5/6 过，自测了"running 中的任务被取消"，没测"还在排队的任务"；`sparql-university` s0 发现 rdflib 不支持 HAVING 子查询，think 47k 后结论 "the query itself is intended standard SPARQL 1.1 syntax" → 交卷，用"规范上合法"替代了"在这个环境里跑通"；`path-tracing` ×2 挂 `test_no_deps`，根本没想到去检查；`torch-tensor-parallelism` s1 宣告前自测只跑了 world_size=1。**39 条里 19 条（49%）只差 1 个测试**，12 条差 2 个，8 条差 ≥3 个。

**(c) DECLARED_ALLFAIL（11 条）——"理解错题意，自信交卷"。** `polyglot-c-py` s0 3 步：step 0 think 199k 字符写了 `main.py.c` + 单独的 `c_code.c`，step 1 think 里自己写 "Our solution is not a single file… It works but not single file"，step 2 think 58k 后 "Probably okay?" → 交卷。**它看到了问题、否决了自己、还是交了。** `break-filter-js-from-html` s0 自己造了 `/tests/filter.py` 软链跑测试得 `EXIT:0` 就宣告，把"脚本没报错"当成"alert 触发了"。这一类 think 反而最长（10.6k），自我怀疑在 think 里出现过但没有转化成行动。

### 8.2 五个数据系的失败结构（同一分类器）

| 模型 | DONE | pass | TTL | CAP512 | 未交卷 | 交卷-部分过 | 交卷-全挂 | 交卷-verifier 挂 | **交卷→通过率** | think PASS / FAIL | steps PASS / FAIL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **th3k mask c312** | 172 | **62** | 42 | 1 | 4 | 39 | 11 | 13 | 50% | 3.4k / 6.5k | 30 / 130 |
| th3k tree c156 | 164 | 51 | **0** | 2 | 1 | **59** | **31** | 20 | **32%** | 2.2k / 2.3k | 21 / 62 |
| dsv41 tool-mode (dsvh c210) | 134 | 43 | 0 | **28** | 8 | 34 | 11 | 10 | 44% | 1.3k / 1.1k | 61 / 195 |
| dsv41-flash JSON (treef c168) | 163 | 35 | **73** | 24 | 1 | 17 | 9 | 4 | **54%** | 4.0k / 8.4k | 26 / 178 |
| qwen3-8-flash (qf c129) | 174 | 24 | **122** | 10 | 0 | 10 | 6 | 2 | 57% | 1.1k / 1.2k | 130 / 242 |

- **th3k mask 的"交卷质量"（50%）不是最高**——treef 54%、qf 57% 更高。它赢在**交卷数量**：125 条宣告 vs treef 65、qf 42，后两者把任务耗在 TTL 里。
- **tree_th3k 是另一个极端**：0 TTL、交卷后通过率只有 32%，59 条部分过 + 31 条全挂。学会了"快交"没学会"交之前验"——tree 训练（历史 think 不可见）丢掉了 mask "看着自己之前的推理接着验"的能力。
- **dsvh 的 28 条 CAP512 对应 mask 的 42 条 TTL**：同一批难题，一个用步数耗尽、一个用时间耗尽，两者都是 0，区别只在成本。

### 8.3 命令动词指纹（每 job 前 120 条 DONE 轨迹）

| 模型 | 步/条 | 命令/条 | `echo '---'` 标签 | `clear` | 读文件 | `python3` | sleep≥30s | 观察 p50 字符 |
|---|---|---|---|---|---|---|---|---|
| th3k mask c312 | 104 | 265 | **12.1%** | 0.8% | 30.6% | 15.2% | 1.6% | 2565 |
| th3k tree c156 | 71 | 118 | 1.7% | 0 | 27.3% | **22.0%** | 3.6% | 3489 |
| dsv41 dsvh c210 | 147 | 237 | 1.9% | 0 | **36.2%** | 21.7% | 0.7% | **1308** |
| dsv41-flash treef c168 | 144 | 318 | 3.4% | 0.2% | 24.6% | 13.6% | 0.4% | 1987 |
| qwen3-8-flash qf c129 | 155 | 266 | 9.0% | **35.2%** | 21.4% | 9.6% | **14.5%** | 1613 |
| 4B base | 42 | 89 | 10.1% | 0 | 25.2% | 10.8% | 8.2% | 3619 |

教师指纹在学生身上清晰可见：th3k 的 `echo '--- label ---'; cat f1; echo '--- ---'; cat f2` 一步看多个文件（12% 命令是标签），观察 2.5k 字符，**信息密度最高**，是它 30 步解题的来源；dsv41 tool-mode 观察只有 1.3k 字符、读文件 36%，一步看一小块，所以 147 步；qwen3-8-flash 35% 的命令是 `clear`、14.5% 带 ≥30 s sleep——它在清屏和睡觉。

### 8.4 "部分过"的根因：不是没跑测试，是没覆盖需求（11-06，按 test log + 轨迹重测）

| 指标 | PASS (62) | PARTIAL (39) | ALLFAIL (11) |
|---|---|---|---|
| **题目自己带可运行测试**（task 文本里出现 `test_outputs.py` / `/tests/`） | **2%** | **0%** | 9% |
| agent 全程跑过任务相关测试（pytest / test_outputs / check.py） | 5% | 8% | 9% |
| 交卷前 5 步跑过任务相关测试 | 5% | 3% | 9% |
| 交卷前 5 步只做了"自检"（py_compile / `--version` / import） | 16% | 21% | 9% |
| 宣告时刻的 think 长度 p50 | **2313** | **5513** | 4704 |

**结论 1：这个 benchmark 基本不给 agent 可运行的验收脚本**（0–9%），所以"让模型交卷前跑一遍测试"这条路在 TB2.1 上不存在。"部分过"的本质是需求覆盖不全：对题目声明的 N 条要求，自己写出了 N−1 条的证据就交卷了。

**结论 2：宣告时刻的 think 长度与结果强相关**：

| 宣告时刻 think | 条数 | pass |
|---|---|---|
| ≤3k 字符 | 50 | **72%** |
| >3k | 62 | 42% |
| >6k | 38 | 37% |
| >10k | 28 | 29% |
| **>20k** | 12 | **8%** |

同一规律在**教师数据**里也存在（`dsv41_json262k` 2037 条整轨迹）：宣告行 think >20k 的 623 条 solved 50.6% vs ≤20k 的 1325 条 67.8%；失败轨迹的 think@declare p50 15.5k vs 成功 7.2k。**"想得越久才敢交卷"本身是负相关**——长的、犹豫的收尾 = 它没把握哪条要求算满足。

**结论 3：这些题对本模型是可达的**：39 条里 11 条（28%）同 env 的另一条采样 PASS；30 个 env 里教师（dsv41-max 0.831 口径）20 个 2/2 通过、4 个 1/2、只有 6 个 0/2（cancel-async-tasks、gcode-to-text、install-windows-3.11、raman-fitting、torch-pipeline-parallelism、video-processing）才是"比教师还难"的硬题。

**结论 4：教师数据里"交卷后反悔"不预示成功**：2037 条里第一次 `task_complete` 后又回去干活的只有 261 条（12.8%），反悔者 solved 率 52% 反而低于直接确认者 64%；本模型 c312 也一样（PARTIAL 里 21% 反悔，PASS 只有 6%）。反悔是"没把握"的标志，不是纠错能力的标志。

### 8.5 一个零训练的收益点

用"**两条采样里取宣告 think 更短的那条**"当选择器（纯事后聚合，不改模型、不改 harness）：

| job | env 平均 pass | pass@any（oracle 2 选 1） | 选"declare-think 短的" | 选"长的" |
|---|---|---|---|---|
| mask c312（83 个双采样 env） | 0.367 | 0.482 | **0.434** | 0.301 |
| dsvh_const c336（83） | 0.229 | 0.337 | **0.289** | 0.157 |
| arm A c224（25，早期只有 25 对） | 0.560 | 0.680 | 0.560 | 0.560 |

**+6–7 pt，免费**；剩下的 4.8 pt 才需要 oracle。这个信号可以当纯聚合规则，也可以当触发"自我审计"的开关（§9 E2）。

## 9. 针对"部分过"的实验设计（E1–E5，截至 11-09 未执行）

统一前提：全部在 `cth-tb21-full89-mirror@1.0.1` 上做。**判定门槛**：178 个样本、p≈0.35 时单个分数的 95% 半宽约 ±0.07，两个 run 之差要 >0.10 才算有区别；要分辨 0.05 的效应需要 ×6 采样（534 个 task，两 run 之差 >0.057 可信）。

| | 做什么 | 为什么 | 怎么测 / 判据 |
|---|---|---|---|
| **E1**（零训练，最便宜） | 只改 harness 的确认提示 `_get_completion_confirmation_message()`（现在是 "Are you sure…?"），改成"列出题目里的每一条显式要求，对每条给出你刚刚看到的证据；有一条没证据就先去补，不要确认"。其余不动，基于 1.0.0 发新 harness 版本 | §8.4 结论 1（题目不给可跑测试 → 只能自审）+ 结论 2（宣告时 think 越长越可能错 → 在这一步把"犹豫"转成"核对"） | mask c312 和 arm A c224 各跑 {基线, E1}，×6 采样。主指标 pass@all；副指标 declare→pass、确认轮是否引用了具体命令输出。PARTIAL 里只差一个测试的 19 条能转化一半 → +0.05；两 run 差 <0.057 视为无效 |
| **E2**（零训练，与 E1 同时） | 宣告时 think >12k 字符才触发 E1 的清单，否则直接确认 | c312 上 >10k 的宣告通过率 29%，≤3k 的 72% | 同 E1 的 4 组再加一组，看是否以更少的额外步数拿到 E1 的大部分收益。§8.5 的聚合规则作为零成本对照一并报 |
| **E3**（数据侧，1 个训练 run） | 从 arm A 数据（2037 条）筛"最后 10 步内有任务特定核验命令（pytest / test_outputs / check.py / diff / md5sum）且最后一轮 analysis 逐条提到了题目要求"的轨迹，预计 900–1100 条 | 教师数据里末 5 步有核验命令的约 50–57% | 8 epoch，对照 = arm A 同 epoch；declare→pass 提升 ≥0.10 且 pass@all 不降，否则结论为"这个行为筛数据筛不出来" |
| **E4**（数据侧，对照臂） | 离线按题目要点切出要求列表，把每条轨迹最后一轮的 `analysis` 前缀替换成逐条 "requirement → evidence"，只在最后一轮加监督 | 不是教师产的文本，只作 E3 的对照（E3 有效而 E4 无效 ⇒ 必须用真轨迹） | 同 E3 |
| **E5**（RL，最后做） | KTO on 自己的 rollout：现有 mirror 口径 job 里已交卷的 rollout 按最终分标正负（每 job ≈ 90–120 条），只取最后 5–10 步做训练信号 | DPO 需要同 env 的正负配对，c312 目前只有 11 对，不够 | 最大不确定性是 reward 噪声（×2 采样、89 题）；先把 ×6 的基线跑出来再决定 |

顺序：E1/E2 只需发一个 harness 版本 + 5 个评测 job，不占训练卡，先做；E3 的筛选统计可以先出，训练是否提交看 E1 的方向。

## 10. 典型轨迹索引（可直接打开）

```
/storage/openpsi/users/chen/output/cth-tb21-stream-abl-mask-const-c312-2610260920-2610260120/tasks/
  tb2-video-processing/s1                   TTL 806 步，探索不收敛
  tb2-llm-inference-batching-scheduler/s0   TTL 415 步，优化问题微调不收敛
  tb2-build-pov-ray/s0                      TTL 388 步，echo-label 读文件 430 次，源不存在
  tb2-cancel-async-tasks/s0                 PARTIAL 12 步，5/6，自测漏排队任务
  tb2-sparql-university/s0                  PARTIAL 22 步，"规范合法"代替"环境跑通"
  tb2-polyglot-c-py/s0                      ALLFAIL 3 步，step0 think 199k，自己否决后仍交卷
  tb2-break-filter-js-from-html/s0          ALLFAIL 7 步，假 verifier EXIT:0 当通过
  tb2-mteb-retrieve/s0                      NOTEST，verifier pypi 超时（非模型错）
/storage/openpsi/users/chen/output/cth-tb21m-dsv41j-a-c224-2611051554/tasks/
  tb2-rstan-to-pystan/s0                    TTL 670 步，C-c 打断 apt → 改 sources.list → 修环境到死（§2 a）
  tb2-polyglot-c-py/s1                      TTL 185 步，103 个 /tmp 试验文件，/app 只碰过一次（§2 b）
  tb2-qemu-startup/s0                       TTL 214 步，解析 ISO/qcow2 头，QEMU 从未稳定跑起来（§2 c）
/storage/openpsi/users/chen/output/cth-tb21-96-dsvh-const-c210-2611020941-2611020141/tasks/
  tb2-mailman/s1                            CAP512，think 从 310 字符衰减到 64 字符，一步一个 sed -n
/storage/openpsi/users/chen/output/cth-tb21-96-abl-qf-const-c129-2611010046-2610311646/tasks/
  tb2-mailman/s1                            TTL 500 步，step0 think 43 字符 "Let me start by exploring"
/storage/openpsi/users/chen/output/cth-tb100-terminus-treef-c168-chen-ais-2611021620/tasks/
  tb2-overfull-hbox/s0                      TTL 426 步，单步 think 最长 110k 字符（zlib 比 0.28，不是复读）
```

每个 job 目录下 `error_classes.json`（六类分类）和 `traj_deep.json`（30 项行为指标）有逐条数据。

---

## 11. compaction 行为归因：模型学会了"处理"，输在"落入"（11-10）

> 工具 `coding_eval/tools/traj_compaction_behavior.py`（逐条解析 trajectory.jsonl 的 compaction 边界事件 + agent 步；默认输出行为分布与按 ncomp 分桶的分数，`--ext` 输出 handoff 忠实度）。中间量 `/tmp/opencode/traj_compaction_behavior.json`。
> 对象：`cth-tb21m-mask-const-c312`（th3k 最佳）、`cth-tb21m-dsv41j-a-c224`（dsv41 最佳，同 harness 1.0.0 同采样 index，严格可比）、`cth-tb21m-dsvh-rand-c336`、`qf-solved-c105` / `qf-const-c129`（96k harness，只作风格参照）。

### 11.1 压缩边界的"操作"三家都会

对每个 compaction 边界取前后 8/15 步窗口（th3k 452 个、dsv41 597、dsvh 886、qf 404–721）：

| 边界行为 | th3k c312 | dsv41 c224 | dsvh c336 | qf c105 | qf c129 |
|---|---|---|---|---|---|
| 压缩后第一步命令与 handoff 计划重合度 | **0.58** | 0.46 | 0.34 | 0.20 | **0.17** |
| 第一步与 handoff 零重合 | 19% | 25% | 43% | 58% | **70%** |
| 压缩后 15 步内重写"压缩前已写过的 /app 文件" | 4% | 4% | 5% | 10% | 10% |
| think：前 8 步 → 后 8 步 | 18.9k → 9.2k | 16.0k → 6.0k | 6.6k → 4.6k | 16.2k → 9.6k | 8.9k → 5.9k |
| 命令/步：前 → 后 | 2.08 → 3.25 | 2.01 → 3.08 | 1.43 → 2.11 | 1.99 → 2.66 | 1.52 → 2.15 |

三家都没有"compaction 后失忆乱跑"的硬故障：重写已完成文件 4–10%、think 边界后回落、命令密度提升（批量读状态）。处理质量有梯度（th3k 最忠实于 handoff，qf 70% 第一步无视自己的交接计划），但 PASS 与 TTL 轨迹的边界后行为几乎无差（重复命令率 20% vs 24%）——**压缩边界不是成败分水岭**。讽刺的是 qf 训练数据 compaction 暴露最多（78% 轨迹、75% 段以 handoff 开头）却处理最差，th3k 数据只有 12% 含 compaction 却处理最好：compaction 处理能力来自整体推理质量，不来自 compaction 样本量。

### 11.2 分数差里 compact 占多少

关键事实：**ncomp=0 的任务上三家 pass 完全相同（0.63 / 0.63 / 0.63）**，qf 即使不压缩也只有 0.42（sleep/clear 风格税与 compaction 无关）。

th3k vs dsv41（164 对同题，净差 −14 题 ≈ −8.5 pt）分解 `P(落入压缩区) × P(pass|压缩区)`：

- 反事实 1：给 dsv41 换 th3k 的落入频率 → 挽回 5.2 pt（59%）；
- 反事实 2：压缩区内通过率差（0.162 vs 0.233）→ 3.6 pt（41%）。

但逐条看 26 条 th3k 赢的配对：6 条 dsv41 没压缩就输了；可精确对比的 12 条（th3k 零压缩解出、dsv41 压缩后失败）里，**dsv41 在第一次压缩之前的段落就已用掉超过 th3k 整条解题轨迹的步数（10/12）和写文件次数（10/12）**，压缩前段 obs 错误率 0.40（th3k 全程 0.22）、C-c 4.2 次（th3k 0.3）。失败在第一次压缩前已成定局，compaction 是果不是因。

**定量结论：8.5 pt 差距里可归咎于 compaction 机制/处理能力的 ≤1–2 pt**（每次压缩 ~8–15 步重建税 + 压缩区内 handoff 忠实度差）；≥6.5 pt 是"不收敛→步数膨胀→被迫反复压缩→TTL"链的下游症状。反向证据：ncomp=1–2 时 pass 仍有 0.50–0.52、宣告率 89–96%——压缩一两次不致命；崩塌只在 ncomp≥3（pass ≤0.12、宣告率 9–11%），而 ncomp≥3 的充分条件是已卡 200+ 步。

**可操作**：①不需要补"compaction 处理"数据；②§6.4/E2 止血注入的触发点可用 **ncomp≥3** 代替单步 think 阈值（更干净的失败预警，覆盖几乎全部 TTL）。

## 12. 论文行为准则下的三家教师数据对照（11-10）

> 背景：精读 5 篇论文（TH3K 2610.13166、Tmax 2606.23321、Data Recipes 2606.24855、FACET 2608.18580、Terminal-Universe 2609.04148）后，把 TH3K Table 3 的 teacher 选取指标 + judge 三准则量化到我们三套训练数据上。工具 `coding_eval/tools/sft_paper_criteria.py`（th3k 2441 / dsv41 armA 3593 / qwen38 flash 2732 条整轨迹全量扫描）。

### 12.1 跨论文共识（与我们实验互证）

1. **最强模型 ≠ 最好 teacher**：Data Recipes 实测 GPT-5.3-Codex（最强）是 5 个 teacher 里最差（TB2.0 差 ~5pp）；我们是第三个独立复现（qf 教师 0.771 最高 → 学生 0.174 最差）。
2. **风格一致 > 数量，混源有害**：Terminal-Universe 直接 SFT 异构源轨迹掉到 base 以下（40.3 < 46.2），同任务单 teacher 重解 52.9。
3. **失败轨迹该留**（TH3K/Tmax/Data Recipes 都不做成功过滤；我们 solved-only 三次消融全负）；FACET 是唯一反例，但它靠题均 22.77 个 pytest 断言的密集 verifier，TH 不具备。
4. **轨迹过滤的有效维度是多轮交互密度**（Data Recipes min-turns≥5，compute 对照 +3.5pp），不是对错。
5. **扩数据扩任务，不扩采样**（同任务多 rollout 在 31.6k→100k 平台化；2× env > 2× query ≈ 2× 采样）。

### 12.2 Table 3 指标的反转与修正

| Table 3 指标 | 论文 DSV4-Pro（选中） | th3k（我们 2441） | dsv41 armA | qwen38 flash |
|---|---|---|---|---|
| mean turns | 53.5 | 40.2 | 84.8 | **147.2** |
| mean feedback chain | 30.3 | 24.7 | 40.1 | **78.1** |
| exact repeats %（命令级） | 13.2 | **13.3**（精确吻合） | 11.3 | **4.8** |
| teacher pass | 50% | ~47% | 48% | 47% |

**机械套用 Table 3 会把 qwen38 flash 排第一**（链最长、重复最低）。假象来源：flash 不用空 keystrokes 等待（4.6%，三家最低），而是发 `clear; sleep 300; cat status` 的实命令轮询——每次轮询都算一个"有命令的 turn"，链不断、重复率低（sleep 参数每次不同）。真正筛掉它的是 judge 准则 3 的折扣项量化：

| 命令构成 | th3k | dsv41 | qwen38 |
|---|---|---|---|
| `clear` 前缀命令 | 0.3% | 0.1% | **32.7%** |
| duration ≥30s 命令 | 1.1% | 0.4% | **9.3%** |
| 等待轮（空 keystrokes，论文认可的等法） | **13.2%** | 11.2% | 4.6% |

折扣后 flash 每 turn 有效动作密度 ~1.2 < th3k 2.81。judge 准则 1/2 则区分不了三家（收尾：整轨迹口径 100%/93%/87%；错误后原样重发 2.2%/1.7%/0.8%）——这解释了 Opus judge 对 dsv41 KEEP 97.9%：准则为抓废轨迹/作弊设计，三家教师都不废。**teacher 准入筛必须包含命令构成指标，只抄 Table 3 会选错。**

判定：th3k = 论文准则的标准正样本（命令重复率与论文自述精确对齐）；dsv41 过准则但带两项论文未度量的超标（think mean 11.2k = th3k 1.7×、机械循环 17.1% vs 8.3%）；qwen38 flash 表面全绿、实质违反准则 3。

### 12.3 借鉴落点（按可执行性）

- **A. teacher 行为准入筛**：100 题子集采集 → 指纹（feedback chain、repeat 率、末 1/4÷首 1/4 think 增长 ≤4×、测试命令率、**clear/sleep/长 duration 占比**、收尾率）。qf 当场出局，成本比一个 SFT run 低一个量级。
- **B. 坏数据重解不洗**（Terminal-Universe intent-recovery，+12.6 pt）：qf 覆盖的题用好 teacher 重打，redo pipeline 现成。
- **C. judge 之上叠行为指纹过滤**：judge 管"真不真"，指纹管"像不像好 teacher"（dsv41_final 风格不变是实证——Part C 流水线从设计上不处理风格分布）。
- **F. 9B 风险对冲**（Tmax：SFT 伤强学生 −6~−10）：dsv41_judged_9b 评测以"低于 9B base 0.267"为止损线，不涨则转 RL（E5 同向）。

## 13. dsv41_final 尾部处理模拟：剪尾与切长想都不对症（11-10）

> 工具 `coding_eval/tools/pland_tail_trim_sim.py`（最后有效动作定位）、`pland_stuck_trim_sim.py`（连续卡死 run 定位）。对象 `dsv41_final` 2,494 条 / 223,704 轮全量。

### 13.1 字面"剪尾部失败轮"（Terminal-Universe §6.5）不可行

- **连续卡死 run 几乎不存在**：尾部 ≥20 连续卡死轮（重复批/报错/空命令/envfix）只有 4/1,263 条失败轨迹——dsv41 的卡死是"交错"的（25% 卡死轮与干活轮交替），且 §C3.7 已折叠掉成片的；
- **"切到最后一个写文件/测试"会删掉 94% solved 的收尾**（最终验证 + 宣告，平均 4.5 轮）——重新引入 qf 系"干到一半就停"的病；
- "只留含验证通过轮的 session"在单轮 pass/fail 设定下退化为 success-only 过滤，已被三次消融否决。

### 13.2 "切尾部 think>20k 的 648 轮"是反向优化

| 尾部 >20k 轮（648 个）画像 | 数值 |
|---|---|
| 占 think 字符 | 20.5M / 572M = **3.6%** |
| 占全部 >20k 轮 | 14.3%（**另 85.7% / 141M 字符在轨迹中段，位置 p50 0.42，一个没碰到**） |
| 是宣告轮 / 在 "Are you sure" 确认轮 | **71% / 41%** |
| 成败 | failed 327 / **solved 321** |

人读 8 条：7 条是"确认提示下逐条复核需求再答 yes"的**宣告前自审**——正是 E1 想要的行为，一半来自 solved。切掉等于用数据手段反向实现 E1。预期对 pass@all 的效应在评测噪声（±0.03–0.04）以下。

### 13.3 th3k 对照：长想不是病，长错地方才是

同一扫描器跑 th3k 2441（98,038 轮）：

| | th3k | dsv41_final |
|---|---|---|
| think>20k 轮占比 / 占 think 字符 | **6.7% / 36.8%** | 2.0% / 28.2% |
| >20k 轮在中段（位置 p50） | 95%（0.49） | 86%（0.42） |
| 确认轮 think p50 / 确认后再宣告 | **2,922 / 89%** | 4,933 / 71% |
| 宣告轮 think p50 / >20k 占比 | **3,519 / 4.7%** | 4,266 / **9.7%**（失败 4,722 vs 成功 3,934） |

th3k 的长想总量**更多**（36.8% 的监督字符），但长在中段诊断（EVPN 调试 28k、审计表设计 29k）、短在确认与收尾；教师 DSV4-Pro 在确认提示下也会 20k+ 自审（347 个尾部长想轮照样进了最好的数据）。**dsv41 偏离 th3k 的不是长想总量，是宣告时刻的犹豫**（确认轮 p50 高 70%、宣告轮 >20k 占比高一倍、29% 确认后又反悔回去干活——对应 §8.4 结论 4"反悔不预示成功"）。

### 13.4 对症手术（更新 §6/§7.4 的数据侧清单）

| 手段 | 命中 | 误伤 |
|---|---|---|
| ~~切尾部 >20k 轮~~ | 3.6% think，50% 是好样本 | 高，弃用 |
| **think 相对增长 mask**（>4× 首段均值 loss=0，不删） | 中段 141M+ 字符（24.7%+ think） | 低（首段就长的任务不受罚；固定 20k 阈值会削掉 th3k 式中段深度推理） |
| **失败轨迹宣告轮 mask**（score=0 且以 task_complete 结尾的 1,497 段末轮 loss=0 或删） | 21% 监督字符的"结尾动作"，教"不干完别说干完" | 零（只碰失败轨迹）；需 E1/E2 对冲宣告率下降的 TTL 风险 |

两者正交：相对增长 mask 管"过程里越卡越想"，宣告 mask 管"长想后错误确认"。方案 D 的真正用武之地是未清洗的原始数据（qf 的成片轮询可切 20–40%），不是已过 judge + §C3.7 的 final。
