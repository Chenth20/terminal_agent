# mask_const c312 能力图谱与部署定位

> 分析对象：`abl-mask-const-c312-chen-ais`（Qwen3.5-4B-text 全参 SFT，th3k 整轨迹数据，8 epoch = 312 步）
> 评测 run：`/storage/openpsi/users/chen/output/cth-tb21m-mask-const-c312-2611051616`
> 口径：benchmark `cth-tb21-full89-mirror@1.0.1`（修复 verifier 后的正式口径）+ reward `cth-tb2-reward-mirror@1.0.1` + harness `cth-terminus-2-stream@1.0.0`
> 89 题 × 2 采样，4 h TTL。归档的 1.0.0 配置为 T=1、top_p=0.95、top_k=50、reasoning_effort=max，未设置 max_turns。早期报告另记 top_k=20/xhigh，各次请求的实际生效参数需查原始请求或服务日志。[归档配置](../arena_assets/harness/configs/1.0.0.env)
> 分数：**0.410（73/178）**。随文脚本：`../../coding_eval/tools/c312_domain_stats.py`（领域聚合）、`../../coding_eval/tools/c312_peek.py`（轨迹阅读器）。

---

## 1. 一句话结论

**这不是一个"会写代码的聊天模型"，是一个"在 shell 里照着现成规范把环境改对"的执行器。**
它的能力边界几乎完全由两件事决定：**(a) 目标是否可被本地命令验证；(b) 验证失败的信号是否足够具体。**
178 次采样中 41 次用满 4 小时，约占总计 259 agent-hour 的 63%。优先用新任务测验证据是否能帮助模型按时完成交付。

---

## 2. 分数结构：73 个通过是怎么来的，105 个不通过是怎么没的

| 类别 | 样本 | 占 178 | 含义 |
|---|---|---|---|
| **PASS** | **73** | 41% | 交卷且 verifier 全绿 |
| DECLARED_PARTIAL | 42 | 24% | 主动交卷，测试部分过（**24/42 只差 1 个测试**） |
| TTL | 41 | 23% | 4 h 被杀，从未交卷（27 个 env） |
| DECLARED_ALLFAIL | 8 | 4% | 自信交卷，测试全挂 |
| NO_DECLARE | 4 | 2% | 没交卷也没超时 |
| CAP512（旧分类标签） | 1 | 1% | 141 分钟后结束，1.0.0 无 max_turns，触发原因未核实 |
| DECLARED_NOTEST | 1 | 1% | 已宣告，验收未运行 |
| NONDONE（harness 失败） | 8 | 4% | 无正常 reward，计入总分母；涉及 harness 与模型对终端的操作，根因见逐例复判 |

换算成三个不同问题：

- **单次采样成功率** = 73/178 = **0.410**
- **给两次机会能不能解出来**（pass@any，对部署最有意义）= 48/89 = **0.539**
- **"我说我做完了"的可信度** = 73/124 = **59%**（170 个完成 episode 里 124 个主动宣告，其中 73 个真过）

env 级分布：**25 题 2/2 稳过、23 题 1/2 时过时不过、41 题 0/2 从来没解出过**。

---

## 3. 能力图谱：按领域（89 题全量，pass@all）

### 3.1 它做得好的（≥0.6）

| 领域 | env | 2/2 | 1/2 | 0/2 | pass@all |
|---|---|---|---|---|---|
| 代码迁移/现代化（py2→py3、COBOL、R 脚本→PyStan） | 2 | 2 | 0 | 0 | **1.00** |
| 软件构建/打包（Debian source 构建 pMARS、Cython 扩展、SQLite+gcov、pypi server） | 7 | 4 | 3 | 0 | **0.79** |
| 服务器配置/网络（nginx 限流、自签证书、git-over-SSH、mailman3、gRPC KV、HF 推理服务） | 7 | 4 | 3 | 0 | **0.79** |
| 实现给定接口/规格（async 并发池、会议排期约束、MuJoCo 调参） | 4 | 1 | 3 | 0 | 0.63 |
| 安全/vuln 修复（有 pytest 套件时：bottle 头注入、secret 提取、XSS 过滤器） | 5 | 3 | 0 | 2 | 0.60 |
| git/vcs 操作（找回丢失 commit、bundle 合并、历史泄露恢复） | 5 | 2 | 2 | 1 | 0.60 |

这六类合起来 **30 题 / 60 样本，pass@all ≈ 0.72**。共同特征：**规范完整、有现成工具、每步都能立刻看到结果**。这就是它的"舒适区"。

### 3.2 半能力（0.2–0.5）

| 领域 | env | pass@all | 典型形态 |
|---|---|---|---|
| 算法实现（拒绝采样、KL 约束分布搜索、特征值、投资组合、并行调度器） | 7 | 0.50 | **有明确数值判据的能解**（largest-eigenval 2/2：自己想到用 `scipy.linalg.lapack.dgeev` 绕开 numpy 开销；distribution-search 2/2：构造四组分布解两个 KL 方程）；**需要自研系统设计的全灭** |
| 数据/科学计算 | 10 | 0.35 | 计数/合并/reshard 能过；依赖 HF/wandb/leaderboard 真实快照、Stan 后验精度、Raman 峰拟合的全挂 |
| 虚拟化/QEMU | 3 | 0.33 | qemu-startup 2/2（telnet 串口通了）；qemu-alpine-ssh、win3.11 0/2 |
| 取证/恢复 | 6 | 0.25 | 7z 爆破 2/2（170 min）、sqlite 截断 1/2；ELF 提取、WAL 解密、已删文件恢复全挂 |
| C/C++ 调试 | 2 | 0.25 | OCaml GC 1/2、堆崩溃 0/2（4/6 测试过） |
| 分布式/GPU 训练 | 4 | 0.25 | 只有"反推 state_dict"过；tensor/pipeline parallel **13 个测试挂 12 个** |
| 领域文件解析 | 5 | 0.20 | 唯一得分靠"暴力搜哈希前缀"绕过了看图，不是真读图 |
| 查询/KG 语言 | 5 | 0.20 | SPARQL/正则/SQL 优化：**5 个测试过 4 个，卡在最后一条**（1.30 s vs 目标 1.02 s） |

### 3.3 它做不了的（=0，41 题）

按根因分五族，这比按领域分更有用：

| 失败族 | 题（0/2） | 机制 |
|---|---|---|
| **① 没有原生视觉，只能靠 shell 间接看图** | chess-best-move、gcode-to-text、extract-moves-from-video、video-processing、sam-cell-seg、financial-document-processor、protein-assembly | terminus-2 只喂文本；模型只能用 PIL/tesseract/ffmpeg 把图转文字，OCR 出来是 `FF wt oer Pee OUMPlUCUmhDD` 这种噪声。**这类任务的瓶颈不是代码能力，是感知通道** |
| **② 自研非平凡系统/算法，局部反馈太弱** | write-compressor（要逆推 range-coder 写编码器）、schemelike-metacircular-eval（422 步）、regex-chess、circuit-fibsqrt、winning-avg-corewars、path-tracing、make-doom-for-mips、make-mips-interpreter、gpt2-codegolf | 没有"跑一下就知道对不对"的中间判据，只能靠推理撑。它会 200 步反复 `objdump`/`sed -n` 猜同一个常量 |
| **③ 跨进程/分布式语义** | torch-tensor-parallelism、torch-pipeline-parallelism、pytorch-model-cli | `mp.spawn` 的初始化契约它读不懂，写出的代码在测试里进程直接崩（12/13 挂）。这类错误的反馈是"子进程异常"，没有可读的中间态 |
| **④ 领域知识密集（需要外部事实，不是编程）** | dna-assembly、dna-insert、feal-differential-cryptanalysis、feal-linear-cryptanalysis、raman-fitting、mcmc-sampling-stan、mteb-retrieve、mteb-leaderboard、query-optimize、sparql-university | 要么需要领域先验（引物设计规则、FEAL 轮函数结构、Raman 峰位），要么需要"和远端真实数据/榜单对齐"。它的失败都很接近：`Accuracy 0.618 vs 0.62`、`Alpha 2.926 vs [2.84,2.91]`、`1.30 s vs 1.02 s` |
| **⑤ 长程开放式工程，无止损** | video-processing、large-scale-text-editing、llm-inference-batching-scheduler、install-windows-3.11、qemu-alpine-ssh、custom-memory-heap-crash、db-wal-recovery、extract-elf、password-recovery、sanitize-git-repo、filter-js-from-html | 不是不会，是**不肯降级交付**。它在 TTL 里 441 步、908 条命令、520 万字符 think，从没写过"先交一个最简可用版" |

> 注：`filter-js-from-html`（0/2）与 `break-filter-js-from-html`（2/2）是同一能力的两面——**有现成 XSS 向量测试集的能过，要自己枚举攻击面的过不了**。这一对是整份分析里信息量最大的对照。

---

## 4. 它"为什么能解"：三种可复用的成功模式

从 73 条 PASS 轨迹里归纳，只有三种，没有第四种。

**模式 A：规范 → 实现 → 自验（最主流，约 6 成）**
`modernize-scientific-stack` 6 步：`ls` + `find -exec cat` 一次读完所有输入 → 写脚本 → `py_compile` → 跑 → 报错 `ConfigParser(encoding=)` → `sed` 修 → 跑出 `Station 101 mean temperature: -15.5°C` → 交卷。**32 步、14 分钟是它的中位数**。
关键：任务给的是"要什么产物"，不是"怎么做"；且产物能自己跑起来看输出。

**模式 B：把别人的测试套件当 oracle（交卷质量最高的一档）**
`fix-code-vulnerability` 13 步过 367 个测试：`pytest -rA` 看到 `test_environ.py` 挂了一个 → 读测试 → 定位 `HeaderDict` 不校验控制字符 → 补校验 → 全绿 → 交 `report.jsonl`。
这就是为什么安全修复类 0.60、而同类无测试的 `sanitize-git-repo` 0/2。**给它一个能红的测试，它就能变绿；让它自己定义"什么是干净"，它就交半成品。**

**模式 C：工具中介的感知 + 廉价验证器**
`code-from-image` 通过的那条根本没用 OCR：它装了 tesseract，发现输出是垃圾，转而**利用题目给的 `bee26a` 前缀提示暴力搜哈希构造**，搜到即停。
`largest-eigenval` 同理：不硬啃算法，直接 `scipy.linalg.lapack.dgeev` 省掉左特征向量计算，比 ref 快 → 过。
**它极擅长"把开放问题压成一个能快速判真假的小搜索"。** 这是 4B 模型最被低估的能力。

---

## 5. 它"为什么解不了"：机制，而不是标签

### 5.1 TTL：不是乱转，是没有止损线

| | PASS | PARTIAL | TTL |
|---|---|---|---|
| 步数中位 | 32 | 47 | **441** |
| 时长中位 | 14 min | 39 min | **240 min（打满）** |
| 命令数中位 | 82 | 96 | **908** |
| 累计 think 中位 | 20.6 万字符 | 46.6 万 | **522 万** |
| compaction 中位 | 0 | 0 | **9** |
| 重复率 | 0.18 | 0.22 | 0.18 |

重复率没升高、sleep 也不多（不是空转），**它是在真干活、真探索、就是不收敛**。`write-compressor` 546 步里在同一份反汇编上绕；`schemelike-metacircular-eval` 422 步里在 `interp.py` 的语义细节上打转；`chess-best-move` 445 步里做了 3 轮图像分割 + 2 轮 OCR，全失败，仍不写一个"用启发式猜着交"的版本。

**这一档的成本必须单独看**：259 agent-hour 里 164 小时（63%）烧在 41 个 0 分样本上。**换算成钱，它 63% 的推理预算买的是零产出。**

### 5.2 PARTIAL：需求覆盖不全，不是没测

- 42 条 PARTIAL 里 **24 条只差 1 个测试**。
- 典型：`cancel-async-tasks` 测了"运行中被取消"，没测"还在排队的"；`constraints-scheduling` 4 步就交，漏了"冲突检测"这个维度；`log-summary-date-ranges` 把 `today` 算成 2266 行（应为 370）。
- **宣告时的 think 长度是关键信号**：PASS 中位 2205 字符，PARTIAL 3753 字符。**它越长篇大论地解释"为什么我算做完了"，越可能是没底。**
- 这直接给出一个免费的部署技巧：同一 prompt 采样 2 次，**选宣告 think 短的那条**，0.410 → **0.438**；再放宽到 pass@any 就是 0.539。

### 5.3 视觉与跨进程：结构性缺口

- **视觉**：模型本身是 4B 文本 SFT，评测链路只给文本。任何"看懂图"的任务都要靠外部 OCR，而 OCR 在小字号/棋盘/视频帧上不可用。**7 个视觉相关 env 合计 pass@all 0.143。**
- **跨进程**：`torch.multiprocessing.spawn` 那类"初始化顺序 + 全局状态"的契约，4B 参数里装不下。它写出的代码语法正确、逻辑自洽，但一 spawn 就 `ProcessRaisedException`，而它看到的只是"子进程挂了"，无法反推。

### 5.4 长上下文能用，但不是自由用

73 条 PASS 里 **20 条发生过 compaction**（即单条轨迹超过 96k 触发压缩），说明 262k 窗口是真被用上的、也是它少数相对 base 模型的结构性优势。但 TTL 轨迹平均压缩 9 次后明显丢状态——`schemelike` 在压缩后重新 `nl -ba eval.scm` 读了 3 遍同一个文件。

---

## 6. 现实场景：它该被放在哪

判断标准只有一条：**这个场景里，"对不对"能不能在 15 分钟内被一条本地命令判出来？** 能，就用它；不能，别用。

以下部署设想来自这一批 TB2.1 轨迹；应用端效果、成本和延迟仍需按真实工作负载实测。

### 6.1 优先试验的场景（由 TB2.1 题型推断，待应用评测）

| 场景 | 为什么 4B 反而合适 | 预期 |
|---|---|---|
| **CI/构建修复机器人**：apt/pip/CMake/Cython 版本冲突，编译报错，依赖漂移 | 大模型单次调用贵、且这类任务 90% 是查报错 + 改配置的机械循环；它在这档 0.75–0.79 | 一次修复 pass ≈0.75，单次 10–40 min，可离线 |
| **环境配置/运维 runbook 执行**：装 nginx+限流、签证书、起 git-over-SSH、起 gRPC 服务、配 mailman | 规范是死的（端口、路径、参数都写在需求里），它照着做且会自验（`ss`/`curl`/`openssl verify`） | 0.79；失败也多是"漏一条参数"，人一眼能补 |
| **代码迁移/现代化批量任务**：py2→py3、老框架 API 替换、Python 包接口重写 | 有旧代码当规范、有新解释器当 oracle，**这是它唯一 1.00 的档** | 单文件/单模块级，分钟级 |
| **CVE 修复 + 回归**：仓库自带测试，CWE 明确 | 它会把 `pytest -rA` 当路标，13 步修完 367 个测试 | 0.60，且失败模式是"没修对"而非"改坏一片" |
| **git 事故恢复**：丢 commit、bundle 合并、历史泄露清理 | reflog/bundle/`git cat-file` 全是可验证的确定性操作 | 0.60 |
| **数据管道 ETL 修复**：字段映射、多源 merge、格式重排、大文件 reshard | 输入输出都能 diff，`multi-source-data-merger` 7–8 步就过 | 0.83（data-sci 里"纯数据搬运"那部分） |

### 6.2 加护栏可用（要把"止损"和"验收"从模型挪到 harness）

| 场景 | 需要的护栏 |
|---|---|
| **爬虫/采集脚本的 selector 维护与翻页逻辑修复** | 必须给"样例 HTML + 期望 JSON"当 oracle；不给就会像 `filter-js-from-html` 那样自己造攻击样本、自己判自己过。**这类活它擅长，但必须有 ground truth** |
| **SRE 故障处置的第一步（信息收集 + 初判）** | 限定步数（≤60）和时长（≤20 min），到点强制交"当前结论 + 已排除项"。它 43/73 的 PASS 在 40 步内完成，砍掉长尾损失很小 |
| **单元测试补写 / 覆盖率补齐** | 用"测试必须先在旧代码上红、在新代码上绿"当判据，否则它会写出永远为真的测试 |
| **大模型 agent 的廉价执行子节点（executor）** | 规划/分解交给大模型，它只负责"按这 5 条命令改完并跑通测试"。这是它最经济的用法：**它是 inner loop，不是 planner** |
| **RSI（递归自我改进）闭环里的 auto-repair 环节** | 见 §6.4，需要严格限定 |

### 6.3 不该用（省下这个钱）

- **需要看图的任何环节**：UI 截图定位、图表读数、视频事件抽取、扫描件分类。它没有感知通道，OCR 兜不住。
- **算法/系统设计本身**：从零写压缩器、解释器、模拟器、并行训练框架。不是"效果差"，是 0，且要烧 4 小时。
- **开放式的"把它做好"**（无验收标准）：会得到一份自信的半成品。
- **需要外部实时事实**：榜单名次、远端数据集内容、最新 API 行为 —— `mteb-*` 两题全挂就是例证（它编了一个看起来合理的模型名 `codefuse-ai/F2LLM-v2-14B`）。
- **多天、多会话、需要人类中途决策的长任务**：TB2.1 是 4 小时单会话，这个 benchmark 根本没测到那档。

### 6.4 关于 RSI 的正确期待

这个模型的 RSI 价值**不在"提出改进"，在"把改进落地并验证"**。

成立的形态：给定一个可跑的仓库 + 一套测试 + 一个明确指标，它能在 loop 里反复"改 → 跑 → 看 diff → 再改"，把指标从 A 推到 B。`largest-eigenval`（换 LAPACK 调用拿 1.3× 加速）、`portfolio-optimization`（C 重写拿加速）、`tune-mjcf`（改 solver 拿 2.0× 加速）都是这个形态，而且都是 2/2。
不成立的形态：让它自己发现"这里该做什么优化"。41 个 0/2 里，凡是需要"先想出一个非平凡方案"的，无一例外失败。

**所以它在 RSI 里的位置是 verifier-driven 的执行臂**：大模型/人类产出候选方案，它负责实现、跑测试、修回归、报结果。这个位置上 4B 的成本优势是碾压性的（单任务 ~15 min、可 8 路并发在单机上）。

### 6.5 关于端侧/离线

成立：
- **企业内网 / air-gapped CI runner**（它本来就要 shell + 沙箱，和 CI 同构）
- **单机 8 卡自建推理**（评测就是这么跑的：1 节点 × 8 × tp1）
- **数据不出域的代码修复/迁移**（安全修复档 0.60 已经够用）

不成立：
- **手机/嵌入式端侧**：这个"模型能力"高度依赖 262k 上下文 + 完整 shell 工具链 + 4 小时预算，不是"塞进 App"的形态。真正端侧可用的是它的 0.6 档子能力（配置生成、单文件迁移），那需要单独蒸馏，不是这个 checkpoint。
- **低延迟交互**：中位 14 min/任务，p90 126 min。它是**批处理成本优势**，不是延迟优势。

---

## 7. 历史分数（按 verifier 与 harness 分别读数）

| 模型 | verifier 与 harness | 全样本通过率 |
|---|---|---:|
| DeepSeek V4.1 Flash（教师） | mirror / 1.6.96 | 0.831 |
| 4B mask_const c312 | mirror / 1.0.0 | 0.410 |
| 4B mask_const c312 | 旧 verifier / 1.0.0 | 0.348 |
| Qwen3.5-4B base | 旧 verifier / 1.0.0 | 0.187 |
| Qwen3.5-9B base | 旧 verifier / 1.0.0 | 0.247 |
| qwen38_flash 数据系 4B 学生 | 旧 verifier / 1.2.96 | 0.174 |

4B 与教师的镜像评测采用不同 harness 配置；部署成本需要另测端到端时长、推理 token 和成功交付数。

---

## 8. 应用试验

在授权的网页抓取、结构化数据转换等任务上，先定义输出格式、边界输入与正确性检查，留出新站点或新文件做验收。分别记录整题通过率、部分通过率和耗时；交付前逐项核对与独立进程运行可作为两个单变量试验。上文的二次采样选择和时间预算分析来自同批轨迹的离线估算，实际收益要在留出任务测量。

## 9. 复现

```bash
cd /storage/openpsi/users/chen/output/cth-tb21m-mask-const-c312-2611051616
python3 /home/chentinghong/llama-factory/coding_eval/tools/c312_domain_stats.py traj_deep.json
# 示例，换成待复查的任务与采样编号
python3 /home/chentinghong/llama-factory/coding_eval/tools/c312_peek.py tasks/tb2-raman-fitting/s1/trajectory.jsonl all 300
```

单题完整证据链在 `tasks/<env>/s<k>/`：`task.json`（口径与模型参数）、`trajectory.jsonl`（每步 think/analysis/commands/observation）、`tbench_test.log`（verifier 原始输出）、`reward.log`。