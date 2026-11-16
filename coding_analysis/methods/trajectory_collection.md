# TerminalHorizon：环境盘点、SFT 数据集对比、dsv41 数据清洗流水线

> 原始集群路径、平台作业和文内脚本命令按实验发生时记录。当前服务器入口和可复现范围见[交接文档](../HANDOFF.md)。

TerminalHorizon（TH）是我们采轨迹做 SFT 的题库（HF `shunzou05/TerminalHorizon-Environment`，1,535 题），TB2.1 是评测集。本文三部分按"题库是什么 → 采出来的几套数据怎么选 → 选定的 dsv41 数据怎么洗干净"排列：

- **Part A（10-25）环境盘点与污染 / 重复检查**：两套环境各是什么、怎么注册；TH 里有没有 TB2.1 的题（零重叠）；TH 自身有没有重复（零重复，8 个 key 两版并存）。
- **Part B（10-30）三套 TH SFT 集的统一对比**：flash-sees / dsv41-blind / dsv41-sees-tool + TH-3K，按轨迹聚合的分数、步数、推理长度、压缩次数、可训练 token，以及选数据建议。
- **Part C（11-06 → 11-09）dsv41 数据流水线**：160 个系统坏 env 修复重打 → 环境污染剔除 → Opus 4.8 三准则 judge → 形态清理 → `dsv41_final` 2,494 条；与 th3k 的逐项对比；按 compaction 段切分的 SFT 配方。

**合起来的结论**：

1. TH 训练 → TB2.1 评测没有题目级泄漏；领域重叠（nginx / Coq / Stan）是正常迁移。
2. 三套 TH 集分数几乎一样（0.47–0.48）；差别在"模型能否看到自己的历史推理"带来的行为分布：看得到 → 每步推理 700–1,000 tok、步数 +35%、压缩 3.8×。目标模型部署时看得到（Qwen3.5 默认）就用 sees 系，不要混用。
3. `dsv41_final` 环境污染零残留、judge KEEP 97.9%、硬噪声比 th3k 更少，够干净做 SFT；但风格仍是"小步快跑"（think 中位 579 vs th3k 2,963 字符，84% 含 compaction），这是采集配置决定的。
4. TH 的 reward 不走公网，不需要 reward-mirror；TH 的系统问题在 env 层（flat 镜像、CA 证书、reward-hang、DNS），诊断过程见 [tb21_verifier_audit](tb21_verifier_audit.md) §5。

---

# Part A. TB2.1 与 TerminalHorizon：环境盘点、污染检查、重复检查

日期：2026-10-25。本部分回答三个问题：（1）arena 里 TB2.1 和 TerminalHorizon 两套环境各是什么、怎么注册的；（2）TerminalHorizon 里有没有 TB2.1 的题（训练 TH、评测 TB2.1 会不会泄漏）；（3）TerminalHorizon 自身有没有重复——一个环境对应多个 query，或多个 env 共用同一镜像 / query / oracle。

**结论**：两套题目**零重叠**（名称、指令文本、测试文件三个层面都没有命中）；TerminalHorizon 1,535 题**没有重复**（指令、Dockerfile、tests 目录、注册后的 spec 哈希全部唯一），平台上每个 `cth-th-*` env 都是 1 镜像 : 1 query : 1 oracle。唯一要留意的是 8 个 key 同时存在 `@1.1.0` 和 `@1.2.0` 两个版本（不是重复题，是 10-25 交接批重新注册的同题新版），各 benchmark 内都只引用其中一个。

---

## A1. 两套环境是什么

|                        | TB2.1（评测集）                                                          | TerminalHorizon（训练 / 采样集）                                                                                              |
| ---------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 平台 benchmark         | `tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2`（hechuyi 注册，89 题） | `cth-th-all@1.0.0`（1289）、`cth-th-full@1.0.0`（1486）、`cth-th-long128@1.0.0`（512）等，自己注册                            |
| env key / 版本         | `tb2-<task>@6.4.0`，89 个                                                | `cth-th-<task>@1.1.0`（1,288 个）+ `@1.2.0`（203 个，10-25 交接批）                                                           |
| 来源                   | terminal-bench 2.1 官方 89 题（upstream `ffccbe05`）                     | HF `shunzou05/TerminalHorizon-Environment`（rev `e55b3e99`），1,535 题                                                        |
| 镜像                   | `asr.antgroup-inc.cn/arcaarealagent/faas-swe:tb2-<task>`                 | `reg.antgroup-inc.cn/areal_agent/terminalhorizon:<task>-main-<12hex>-rt1`，每题一个镜像，从各题 `environment/Dockerfile` 自建 |
| oracle schema / reward | `terminal-bench.2.0.v1` / `tb2-reward@4.0.18-gepa-envcache`              | `terminal-bench.2.0.v1` / `tb2-reward@4.0.0`                                                                                  |
| 超时                   | agent 14,400 s                                                           | agent 18,000 s（setup 900，collect 300）                                                                                      |
| 题目领域               | 通用 CLI（编译、数据处理、安全、ML 小任务）                              | 25 个领域，最大三类：软件工程 142、安全/密码学 134、数据库 105（完整分布见 §A4）                                              |

两套题的工程格式相同（instruction.md + tests/test.sh 写 `/logs/verifier/reward.txt`），所以 TH 能直接复用 TB2 的 reward——这也是为什么要专门检查两者是否共享题目。

## A2. 污染检查：TerminalHorizon 里是否含有 TB2.1 的题

TB2.1 的指令文本从本地 `cth-tb21-stream-qwen35-9b-2n-2610211924-2610211124` 轨迹的第一条 user 消息里抽（`Task Description:` 之后），89/89 齐全；TH 用 `data/tasks/<name>/instruction.md`（1,535）。

| 检查                    | 方法                                               | 结果                                                                                                                                                                     |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 任务名                  | `tb2-<name>` 去前缀后与 TH 目录名精确 / 子串匹配   | 精确 **0**；子串 1 个（`query-optimize` ⊂ `integrate-correlation-aware-cardinality-estimation-into-a-query-optimize`），看指令是不同题                                   |
| 指令文本精确重复        | 归一化（小写、去代码块、去标点）后 sha256          | **0**                                                                                                                                                                    |
| 指令近重复              | 词 5-gram 集合的 Jaccard，每个 TB 题取 TH 中最高分 | 最高 **0.003**（`vulnerable-secret`）；≥0.1 的 0 个                                                                                                                      |
| 指令主题相似            | TF-IDF（uni+bigram）余弦，每个 TB 题取 TH 中最高分 | 最高 **0.058**（`compile-compcert` ↔ `integrate-a-semantics-preserving-compiler-pass-into-a-mature-…`，共享词 compcert / verified compiler）；中位数 0.020；≥0.2 的 0 个 |
| 测试文件                | TH 内 1,535 题 `tests/` 下 >400 B 的文件两两哈希   | 跨题相同 **0**（TB2.1 的 tests 不在本地，无法直接比；但 TH 内部没有任何复用，间接说明 TH 没有整体搬运别处的 test）                                                       |
| benchmark 引用          | 9 个 `bench-cth-th-*.json` 的 env_refs             | 没有任何 `tb2-*` / `tbench-*` key                                                                                                                                        |
| 指令里提 terminal-bench | 正则 `terminal[- ]?bench                           | tbench`                                                                                                                                                                  | 9 个命中全是 "testbench"（RTL 题），非引用 |

TF-IDF 前 20 名里确有同主题的（nginx 配置、Coq 证明、Stan 后验、SQLite WAL、plasmid 组装、x509 证书），余弦都在 0.03–0.06——是领域重叠，不是题目重叠。TH 的题平均比 TB2.1 长一个量级（多服务、多阶段验收），同主题也不是同一道题。

**结论：TH 训练 → TB2.1 评测没有题目级泄漏。** 领域级重叠（都考 nginx / Coq / Stan 等）是正常的能力迁移，不算污染。

## A3. 重复检查：TerminalHorizon 是否一个环境对应多个 query

### A3.1 源数据（1,535 题）

| 维度                                             | 不同值数 | 被 >1 题共享的 |
| ------------------------------------------------ | -------- | -------------- |
| 归一化指令                                       | 1,535    | 0              |
| `environment/Dockerfile` sha256                  | 1,535    | 0              |
| `tests/` 目录整体哈希                            | 1,535    | 0              |
| 单个测试文件（>400 B）                           | —        | 0              |
| 指令近重复（5-gram Jaccard ≥0.6，16,628 对候选） | —        | 0              |

每题的 Dockerfile 都不同（即使基础镜像集中：`python:3.11-slim` 394、`python:3.12-slim` 300、`debian:bookworm-slim` 259），因为题目资产 `COPY` 进去的内容不同。**源数据层面没有"同一环境、多个 query"的情况。**

### A3.2 平台注册后（`cth-th-*` 1,496 个已发布版本）

用 `arena env version-list --creator chentinhong.cth --status PUBLISHED` 拉全（8 页，1,498 条，其中 `cth-th-*` 1,491 + `cth-th2-*` 5）：

| 维度                                                                         | 结果                                                                                                     |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `content_hash`（平台对 spec 的摘要：镜像 + query URI + oracle URI + 超时等） | 1,496 个版本 **1,496 个不同哈希**，没有两个 key 共用一个 spec                                            |
| `cth-th-all@1.0.0` 的 1,289 个 env_ref                                       | 全部能在已发布版本里找到，1,289 个不同 content_hash                                                      |
| 本地 `work/env-specs/*.json`（1,530 个 runtime 镜像）                        | 1,530 个不同镜像，没有两个 env 指向同一镜像                                                              |
| 镜像 tag 与 env 名                                                           | 1,530 个里 1,529 个 tag 包含自身任务名；例外 `cth-pyramid-final-env-spec` 是 MiMo 侧的旧 spec，不属于 TH |

本地 spec 里有 40 个文件 query / oracle URI 是字面 `PLACEHOLDER`——这是 `make_assets.py` 的模板输出（`tasks_rest_compose.txt` 里的 compose 题，上传时才回填），不是平台上的状态；对应 env 在平台上都有各自的 URI（已发布版本 content_hash 唯一可证）。

### A3.3 一个 key 两个版本（8 个）

8 个 key 同时有 `@1.1.0`（10-19/21 批）和 `@1.2.0`（10-25 交接批，镜像重建）：

`support-case-family-reconciliation-under-review-budget`、`scan-dft-integration-and-atpg-coverage-closure`、`reconcile-and-run-an-insured-catastrophe-portfolio-risk-model`、`operate-an-adaptive-ecological-restoration-monitoring-round`、`onboard-a-heterogeneous-vendor-fleet-into-open-rmf`、`make-recurring-caldav-scheduling-converge-across-independent-peers`、`evolve-postgresql-driver-pipeline-semantics-across-the-wire-api-boundary`、`build-a-provenance-preserving-identity-aware-email-archive-for-scholarly`

这是**同一道题的两个 env 版本**（query 相同、镜像不同），不是两道题。各 benchmark 里每个 key 只出现一次（`cth-th-all` 用 1.1.0，`cth-th-full` / `cth-th-handoff` 用 1.2.0），不会在一个 job 里重复评同一题。只需注意：**跨 benchmark 合并轨迹做 SFT 时按 key 去重**，否则这 8 题会有两份轨迹（来自不同镜像版本，对错可能不同）。

## A4. TH 领域分布（task.toml `category`）

| 数  | 领域                                                  | 数        | 领域                                                   |
| --- | ----------------------------------------------------- | --------- | ------------------------------------------------------ |
| 142 | Software engineering and developer tools              | 63        | Finance, accounting and economic operations            |
| 134 | Cybersecurity, cryptography and privacy               | 58        | Programming languages, compilers and runtimes          |
| 105 | Databases and storage systems                         | 54        | Energy and public infrastructure                       |
| 95  | Networking and communications                         | 46        | Graphics, media and content tools                      |
| 93  | Life sciences, medicine and health                    | 44        | Data engineering and analytics                         |
| 91  | Earth, geospatial and environmental sciences          | 44        | Chips, electronics and embedded systems                |
| 81  | Cloud platforms and distributed infrastructure        | 42        | Industrial manufacturing and production engineering    |
| 79  | Transport, logistics and operational decisions        | 40        | Mechanical, thermal and aerospace engineering          |
| 73  | Knowledge organization, humanities and public affairs | 34        | Physics, astronomy and instrumentation                 |
| 73  | Operating systems and low-level systems               | 33        | Robotics and autonomous systems                        |
| 70  | Machine learning and artificial intelligence          | 20        | Chemistry, materials and molecular simulation          |
|     |                                                       | 8 / 7 / 6 | Agriculture / Mathematics & formal methods / Education |

## A5. 镜像层面：环境与软件包是否重叠

登录节点没有 docker / skopeo，拉不下镜像本体；但两套题的镜像都是**从 Dockerfile 构建**的（TB2.1 的 `faas-swe:tb2-*` 由 hechuyi 按官方 Dockerfile 构建，TH 的 `terminalhorizon:*-rt1` 由我们按各题 `environment/Dockerfile` 构建），所以比 Dockerfile 等价于比镜像内容。TB2.1 的 Dockerfile 来自 `laude-institute/terminal-bench` 仓库 `original-tasks/`（88/89 有，缺 `headless-terminal`）；TH 的在本地数据集里（1,535）。脚本 `th_tb21_image_compare.py`。

| 层面                                                                                                                       | 结果                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dockerfile 完全相同                                                                                                        | **0**                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 基础镜像                                                                                                                   | TB2.1 77/88 用自家 `ghcr.io/laude-institute/t-bench/{ubuntu-24-04,python-3-13}`（= 官方 `ubuntu:24.04` / `python:3.13-slim-bookworm` + `tmux asciinema`），TH **一个都没用**。TH 主力是 `python:3.11-slim` 394、`python:3.12-slim` 300、`debian:bookworm-slim` 259、`ubuntu:24.04` 121。两边精确相同的基础镜像只有 4 个公共官方镜像（`python:3.11-slim`、`ubuntu:24.04`、`debian:bullseye-slim`、`python:3.10-slim-bookworm`），TB 侧合计 7 题 |
| apt 包                                                                                                                     | TB 115 种、TH 976 种，共有 83（都是 `build-essential`/`curl`/`git`/`sqlite3`/`ffmpeg` 这类通用包）。TB 独有 32 种：`coq`、`gnucobol3`、`gcc-mips-linux-gnu`、`mailman3`、`extundelete`/`foremost`、X11/VNC 全家桶（`tightvncserver`、`novnc`、`xserver-xorg-*`）等——这些正好对应 TB2.1 的特色题（Coq 证明、COBOL、MIPS DOOM、mailman、磁盘恢复、GUI 任务），TH 一个都不装                                                                      |
| pip 包                                                                                                                     | TB 20 种、TH 307 种，共有 15（numpy/scipy/pandas/torch/pillow…）。TB 独有：`chess`、`mteb`、`opencv-contrib-python`、`flit`                                                                                                                                                                                                                                                                                                                    |
| 每题"非通用安装集"Jaccard（去掉 build-essential/curl/git/python3 等 30 个通用包后）                                        | 88 题里 42 题 Dockerfile 几乎不装东西（Jaccard 无意义）；其余 46 题里最高匹配 =1.0 的 12 题全是**安装集只有 1–2 个包**（`numpy`、`sqlite3`、`ffmpeg`、`openssl`、`gdb`、`asciinema`）；安装集 ≥3 个包的 TB 题最高 Jaccard 0.50（`modernize-scientific-stack` 的 matplotlib/numpy/pandas/scipy 子集）。没有任何 TB 题的工具链被某个 TH 题整体复刻                                                                                               |
| `COPY` 进镜像的资产文件名                                                                                                  | 共有的只有 `nginx.conf`、`supervisord.conf`、`forward.py`、`documents`、`inputs`、`resources`、`vendor` 这类通名                                                                                                                                                                                                                                                                                                                               |
| 任务源码文件逐字节相同（TB 88 题的 510 个明文源码/配置文件 vs TH 1,535 题的 88,014 个 `.py/.sh/.c/.js/.yaml/...`，>200 B） | **0**                                                                                                                                                                                                                                                                                                                                                                                                                                          |

TB2.1 的解题脚本（`solution.sh`）用到的工具里，TH 明显缺的是 R（15 题）、Coq/OCaml（4 题）、Stan（2 题）；TH 虽有 `r-base` 相关题但数量个位数。**两套题在工具链上是"共享 Linux 底座、互不覆盖的专项"**——TB2.1 的专项（R 统计、Coq、COBOL、MIPS、VNC/GUI、mailman）TH 基本没有；TH 的专项（Kubernetes、Slurm、BGP/FRR、ZFS、Kerberos、OpenSearch、FPGA/RTL）TB2.1 基本没有。

注意 TB2.1 的 `tests/` 在仓库里是 `protected.tar.gz.enc` 加密包（官方防污染措施，Dockerfile 里带 canary 字串），所以测试文件层面只能确认 TH 没有复用任何 TB 的**明文**文件。

## A6. 没查的 / 局限

- TB2.1 的 `tests/` 在官方仓库里是加密包（`protected.tar.gz.enc`），平台上 oracle 也只在评分沙箱注入，没法直接比测试文件；用 TH 内部零复用 + 指令零相似 + 明文源码零重复做了间接论证。
- 镜像本体拉不下来（无 docker），用 Dockerfile 等价代替；两边镜像都确实是从这些 Dockerfile 构建的，但 `faas-swe:tb2-*` 若在构建时额外打了补丁（如 `pytorch-model-cli` 的 verifier 依赖缓存）不在比对范围。
- 只比了指令和测试，没比解题代码（`solution/`）。TH 的 solution 与 TB2.1 的 oracle 若有共用的第三方脚本理论上可能撞，但两套题主题都不同，概率很低。
- 近重复用的是词级 5-gram 和 TF-IDF，不是语义向量；两道不同措辞但同一考点的题会漏。但 TF-IDF 已把同领域的都排到了前面且分数 <0.06，语义层面也不像有同题。

## A7. 复现

```bash
python3 /home/admin/chentinhong.cth/inclusionAI/AReaL/opencode/th_tb21_contamination.py   # 名称 / 5-gram / 哈希 / 领域分布，~15 min
git clone --depth 1 https://github.com/laude-institute/terminal-bench.git /tmp/opencode/terminal-bench   # TB2.1 官方 Dockerfile / solution.sh
python3 /home/admin/chentinhong.cth/inclusionAI/AReaL/opencode/th_tb21_image_compare.py          # 基础镜像 / apt / pip / 安装集 Jaccard
```

产物：`/tmp/opencode/contam/{tb21_vs_th_best.json, tb21_vs_th_tfidf.json, th_near_dups.json, envs_all.jsonl}`（容器重建会丢，按上面重跑）。

---

# Part B. 三个 SFT 数据集的统一对比（flash-sees / dsv41-blind / dsv41-sees-tool）+ TH-3K

日期：2026-10-30。三套数据同为 TerminalHorizon 1,486 题（long128 512 + full-rest 974）、96k compaction、`max_turns=512`、DONE 轨迹、TH-3K 格式，只有「模型 × 能否看到自己的历史推理」不同。所有数字按**轨迹**聚合（把 compaction 拆出来的 segment 合回去），token 一律 chars ÷ 3.5 估算（与 TH-3K 对比同口径）。脚本：`inclusionAI/AReaL/opencode/sft_sets_compare.py`。

| 数据集            | 模型                    | 历史推理                               | harness               | 路径                                                                                                                      |
| ----------------- | ----------------------- | -------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| flash / sees      | qwen3-8-flash xhigh     | 看得到（Qwen 原生渲染）                | 1.2.96                | `code_sft_data/qwen3-8-flash-history/TerminalHorizon-flash-96k-SFT.jsonl`                                                 |
| dsv41 / blind     | deepseek-v4.1-flash max | **看不到**（DashScope 普通 chat 丢弃） | 1.2.96                | `code_sft_data/TerminalHorizon-dsv41-96k-SFT.jsonl`                                                                       |
| dsv41 / sees-tool | deepseek-v4.1-flash max | 看得到（forced tool-call 模式）        | 1.4.96                | `code_sft_data/dsv41-toolhist/TerminalHorizon-dsv41-96k-toolhist-SFT.jsonl`（job 10-30 时 2574/2972 DONE，剩 184 题未入） |
| TH-3K             | DeepSeek-V4-Pro         | 看得到                                 | 1M 窗口无压缩，仅成功 | 公开集                                                                                                                    |

## B1. 全部 DONE 轨迹

| run               | 轨迹 | 分数      | 步数 mean / p50 / p90 | compact mean（有%） | 命令/条 | 命令/步 | 0 命令步 | 推理 tok/步 | 回答 tok/步 | 回显 tok/步 | 峰值 ctx p50 / p90ᵃ | 累计输入 M/条 | 输出 k/条 | SFT tok k/条（asst%） |
| ----------------- | ---- | --------- | --------------------- | ------------------- | ------- | ------- | -------- | ----------- | ----------- | ----------- | ------------------- | ------------- | --------- | --------------------- |
| flash / sees      | 2732 | 0.471     | 147 / 103 / 337       | 3.03（78%）         | 309     | 2.10    | 0.9%     | 1,031       | 447         | 568         | 86k / 100k          | 5.9           | 218       | 320（68%）            |
| dsv41 / blind     | 2653 | **0.484** | **90 / 46 / 218**     | **0.79**（34%）     | 208     | 2.31    | 2.4%     | **2,914**   | 371         | 788         | 187k / 364kᵃ        | 9.1ᵃ          | 295       | 372（79%）            |
| dsv41 / sees-tool | 2573 | 0.471     | 121 / 74 / 299        | 3.05（84%）         | 223     | 1.84    | 6.9%ᵇ    | **699**     | 367         | 651         | 84k / 100k          | 4.1           | 129       | 222（58%）            |
| TH-3K（仅成功）   | 2852 | 1.000     | 53 / 40 / 98          | 0                   | 139     | 2.63    | 4.0%     | 1,913       | 469         | 813         | 132k / 306k         | 7.1           | 126       | 171（74%）            |

ᵃ dsv41/blind 的「峰值 ctx」「累计输入」是**SFT 文本视角**（历史里的 `<think>` 也算进去），不是模型推理时看到的——DashScope 丢弃了历史推理，模型实际 API prompt 峰值 p50 只有 ~85k（见 [context_sweeps](../experiments/context_sweeps.md) §C5 四格表）。这意味着 **dsv41/blind 这套 SFT 数据的训练上下文 ≠ 采集时模型看到的上下文**：训练样本里每步前面带着历史推理，而生成那一步时模型没看到它们。
ᵇ 1.4.96 的 0 命令步里约 2/3 是 tool-call arguments 畸形/截断导致 `commands` 没被解析（模型下一步自述 "malformed JSON"），是 tool 格式损耗，不是模型主动不发命令。

## B2. 只看 long128 的 512 题（三组同题，各 2 samples）

| run               | 轨迹 | 分数      | 步数 mean / p50 / p90 | compact mean（有%） | 命令/条 | 命令/步 | 推理 tok/步 | 峰值 ctx p50 | 累计输入 M/条 | 输出 k/条 | SFT tok k/条（asst%） |
| ----------------- | ---- | --------- | --------------------- | ------------------- | ------- | ------- | ----------- | ------------ | ------------- | --------- | --------------------- |
| flash / sees      | 1020 | 0.346     | 199 / 166 / 388       | 4.41（98%）         | 415     | 2.09    | 1,066       | 90k          | 8.1           | 303       | 439（69%）            |
| dsv41 / blind     | 984  | **0.372** | 122 / 63 / 373        | 1.24（49%）         | 272     | 2.23    | 3,052       | 238kᵃ        | 13.2ᵃ         | 417       | 518（81%）            |
| dsv41 / sees-tool | 942  | 0.350     | 157 / 102 / 428       | 4.00（90%）         | 281     | 1.79    | 697         | 89k          | 5.5           | 165       | 284（58%）            |

## B3. 只看 score=1 的轨迹（与 TH-3K 同口径）

| run               | 轨迹 | 步数 mean / p50 / p90 | compact mean（有%） | 命令/条 | 命令/步 | 推理 tok/步 | 回答 tok/步 | 峰值 ctx p50 | 输出 k/条 | SFT tok k/条（asst%） |
| ----------------- | ---- | --------------------- | ------------------- | ------- | ------- | ----------- | ----------- | ------------ | --------- | --------------------- |
| flash / sees      | 1288 | 103 / 66 / 234        | 1.78（63%）         | 209     | 2.03    | 971         | 412         | 82k          | 142       | 213（67%）            |
| dsv41 / blind     | 1284 | **65 / 39 / 123**     | 0.42（20%）         | 155     | 2.38    | 2,738       | 353         | 142kᵃ        | 201       | 257（78%）            |
| dsv41 / sees-tool | 1213 | 95 / 60 / 200         | 2.34（79%）         | 179     | 1.88    | 704         | 363         | 76k          | 102       | 175（58%）            |
| TH-3K             | 2852 | 53 / 40 / 98          | 0                   | 139     | 2.63    | 1,913       | 469         | 132k         | 126       | 171（74%）            |

## B4. 配对

- dsv41 同 env（sample 均值，n=1,298）：blind 0.486 → sees-tool 0.472，**−0.014，wins 125 / losses 158，z = −1.96**（刚到显著边缘）。
- 三组共有 1,296 env：flash/sees 0.484、dsv41/blind 0.486、dsv41/sees-tool 0.473。

## B5. 结论

1. **分数**：三套在 1,486 题上几乎一样（0.47–0.48）。dsv41 让它看到历史推理（tool 模式）略降 0.014，刚到 z −1.96；但 1.4.96 叠加了 tool 格式损耗（6.9% 畸形步）和冲线 bug，不能全归到「看到推理」。flash 和 dsv41/blind 持平。
2. **看到历史推理的一致效应**（flash/sees 与 dsv41/sees-tool 都有）：每步推理只剩 **700–1,000 tok**（看不到时 2,900），步数 +35%，compact 次数 **3.8×**（0.79 → 3.05），峰值上下文紧贴 96k。总命令数基本不变（208 → 223），即任务所需的动作量相同，只是被压缩打散到更多步。
3. **看不到历史推理的 dsv41 是三套里最「省」的**：步数最少（p50 46）、压缩最少（34%）、每步推理最长（2.9k tok），和 TH-3K（DSv4-Pro，1.9k 推理/步、53 步）最接近。但它有脚注 ᵃ 的训练/推理上下文不一致问题。
4. **与 TH-3K 对比**（都取成功轨迹，§B3）：TH-3K 每步命令最多（2.63）、推理中等、步数最短；我们三套命令/步都更低（1.9–2.4）而步数 1.2–1.9×。单条 SFT token：TH-3K 171k，dsv41/sees-tool 175k 最接近，flash 213k，dsv41/blind 257k。assistant 占比：dsv41/blind 78% 最高（推理长），sees-tool 58% 最低（推理短、回显占比高）。
5. **SFT 可训练量**（asst token = 推理 + 回答，全部 DONE）：flash 2732 × 320k × 68% ≈ **594M**；dsv41/blind 2653 × 372k × 79% ≈ **780M**；dsv41/sees-tool 2573 × 222k × 58% ≈ **331M**。只留 score=1：184M / 258M / 123M。

## B6. 选数据的建议

- 目标模型推理时**看得到自己的历史推理**（Qwen3.5 默认）→ 用 **flash/sees** 或 **dsv41/sees-tool**：训练上下文 = 采集时模型看到的上下文，每步推理短、步数多，与部署行为一致。两者中 dsv41/sees-tool 需先修 tool 畸形步和冲线问题再重采。
- 目标模型推理时**看不到**（nothink 模板 / 训练时 mask 历史推理）→ dsv41/blind 的 assistant 目标是对的，但**训练时必须把历史 `<think>` 从上下文里删掉**（或 mask），否则模型学到的是「有历史推理可参考」的分布，部署时却没有。现成文件不能直接用。
- 不建议混用三套：每步推理长度差 3–4×，是三种不同的"思考预算"分布。

---

# Part C. dsv41 数据：系统错误修复 → 重打 → 合并 → LLM-judge 清洗

日期：2026-11-06。本部分对象：`cth-thfull14-dsv41-2610270211`（dsv41-flash-1017 max、tool-call 模式、96k compaction、1486 题 ×2）采出的 TerminalHorizon 整轨迹 SFT 数据。目标是把这批数据里**由环境/平台造成的、会教坏模型的内容**修掉或剔掉，再用 TerminalHorizon 论文（arXiv 2610.13166，th3k）的过程质量 judge 洗一遍，得到可直接 SFT 的整轨迹集。

**一句话结果**：原始 2,690 条 DONE 轨迹 → 修复 160 个系统坏 env 并重打两遍、剔掉 101 条含环境污染的 DONE 轨迹、并入 211 条重打轨迹 → 2,796 条干净整轨迹 → Opus 4.8 按三准则盲评 → **2,541 条 KEEP**（1,251 solved，1,349 env），环境污染零残留，judge 97.9% 保留率（th3k 论文 92.9%）。经 §C3.7 最后清理后产物 **`/storage/openpsi/users/chen/code_sft_data/dsv41_final/dsv41_final.jsonl`**（2,494 条）。

---

## C0. 最终数据统计

### C0.1 流水账

| 阶段                                    | 轨迹数    | solved            | env 数                        | 说明                                                      |
| --------------------------------------- | --------- | ----------------- | ----------------------------- | --------------------------------------------------------- |
| `cth-thfull14` DONE（原始）             | 2,690     | 1,253 (46.6%)     | 1,364                         | 2,972 任务里 282 条非 DONE，全是系统原因（§C1）           |
| + `cth-thredo14` 重打（141 个坏 env）   | +182      | +63               |                               | 21 条覆盖原同 (env,sample) 行                             |
| + `cth-thredopol` 重打（50 个污染 env） | +79       | +25               |                               | 26 条覆盖                                                 |
| − 环境污染剔除（§C2）                   | −101      |                   |                               | th14 54 + redo14 19 + thredopol 28                        |
| **= `dsv41_merged`**                    | **2,796** | **1,311 (46.9%)** | **1,414**                     | 比原始多 69 个 env（修好的 flat / tmux / reward-hang 题） |
| Opus 4.8 judge（§C3）                   |           |                   |                               | 53 DROP、196 超长未评、4 安全过滤、2 调用失败             |
| **= `dsv41_merged_judged-keep`**        | **2,541** | **1,251 (49.2%)** | **1,349**（1,192 个两样本齐） |                                                           |
| 形态 A/C + 短 think 清理（§C3.7）       | −47       | −20               |                               | 118 条再折叠 306 对冗余轮                                 |
| **= `dsv41_final`（SFT 输入）**         | **2,494** | **1,231 (49.4%)** | **1,340**（1,154 个两样本齐） | 223,704 assistant 轮，~675M token                         |

### C0.2 KEEP 集形态

|                                                  | merged（2,796）                        | **KEEP（2,541）**                      |
| ------------------------------------------------ | -------------------------------------- | -------------------------------------- |
| 字符 p50 / p90 / max                             | 502k / 1,603k / 3,527k                 | **469k / 1,105k / 1,880k**             |
| 估算 token（2.1 chars/tok 实测） p50 / p90 / max |                                        | **223k / 526k / 895k**，合计 ~690M     |
| ≤ 262k token 的条数                              |                                        | 1,515；≤ 131k：589                     |
| 消息数 p50 / p90 / max                           | 144 / 564 / 1,018                      | 136 / 366 / 1,014                      |
| assistant 步数 p50                               |                                        | 68                                     |
| 每步命令数                                       |                                        | 2.16                                   |
| `<think>` 平均长度                               |                                        | 2,885 字符                             |
| compaction/轨迹 mean，有 ≥1 次的占比             | 3.10，84%                              | 2.49，84%                              |
| 宣告 `task_complete` 的比例                      |                                        | 88.2%（DROP 集只有 35.8%，见 §C3.3）   |
| 来源                                             | th14 2,585 / redo14 160 / thredopol 51 | th14 2,365 / redo14 133 / thredopol 43 |

KEEP 集以 `--linear` 整轨迹形式保存（judge 读的就是这个）；再经 §C3.7 清理得到 `dsv41_final`（2,494 条，§C0.4 第二列）。**SFT 不直接用整轨迹文件**：训练输入是按 compaction 边界切成的段（§C6，§C6 的数字基于清理前的 2,541 条，`dsv41_final` 需重跑 prep），所以上表里"≤262k token 只有 1,515 条"的问题在训练侧不存在——切段后最长的一段只有 139,817 token，0 条被丢。

### C0.3 文件

```
/storage/openpsi/users/chen/code_sft_data/
  dsv41_merged/dsv41_merged.jsonl / .manifest.jsonl              2,796 条，污染已剔除（judge 输入）
  dsv41_merged/dsv41_merged_judged-keep.jsonl / .manifest.jsonl   2,541 条 judge KEEP
  dsv41_merged/dsv41_merged_judged-{drops,overlong,errors}.tsv    53 / 196 / 6 条，带 judge 理由
  dsv41_merged/judge_opus48/reviews.jsonl                         2,841 条 review（含重评），每条 verdict/reason/3–4 条证据
  dsv41_merged/judge_opus48/record_ids.jsonl                      record_id → (job, env, sample)，judge 看不到
  dsv41_final/dsv41_final.jsonl / .manifest.jsonl                 2,494 条  ← SFT 用这个（§C3.7）
  dsv41_final/{drops,collapse}.tsv, build_stats.json              最后一道清理的账
  dsv41_final/stats_{dsv41_orig,dsv41_final,th3k}.json            §C0.4 对比表的原始数字
```

代码（`coding_arena/code/sft/`）：`audit_env_pollution.py`（污染检测）、`build_dsv41_merged.py`（合并）、`llm_judge.py`（judge）、`apply_judge.py`（出 KEEP 集）、`finalize_judge.sh`（重试 + apply）、`scan_bad_actions.py`（逐轮 bad-action 扫描）、`finalize_dataset.py`（形态 A/C + 短 think 清理）、`dataset_stats.py`（统一统计）。

### C0.4 三套数据对比：dsv41 原始 vs dsv41_final vs th3k

同一脚本（`code/sft/dataset_stats.py`，产物 `dsv41_final/stats_{dsv41_orig,dsv41_final,th3k}.json`）对三套整轨迹数据算同一组量。th3k = 论文发布的 2,852 条（已过论文自己的 judge），格式与我们相同（`<think>` + Terminus JSON）。token 用 2.1 chars/tok 估算（本语料对 Opus tokenizer 实测 1.95–2.29）。

| 指标                                         | dsv41 原始             | **dsv41_final**        | th3k                       |
| -------------------------------------------- | ---------------------- | ---------------------- | -------------------------- |
| 轨迹数                                       | 2,690                  | **2,494**              | 2,852                      |
| env 数                                       | 1,364                  | 1,340                  | 1,526                      |
| solved                                       | 1,253 (46.6%)          | 1,231 (49.4%)          | —（论文不公布）            |
| **长度**                                     |                        |                        |                            |
| 字符 p50 / p90 / max                         | 502k / 1,632k / 3,527k | 469k / 1,099k / 1,880k | 470k / 1,083k / 4,296k     |
| 字符总量                                     | 1.90 G                 | 1.42 G                 | 1.71 G                     |
| 估算 token p50 / p90 / max                   | 239k / 777k / 1,679k   | **223k / 523k / 895k** | 224k / 516k / 2,046k       |
| 估算 token 总量                              | 902M                   | **675M**               | 813M                       |
| 消息数 p50 / p90 / max                       | 144 / 590 / 1,018      | 134 / 362 / 1,010      | 80 / 196 / 806             |
| assistant 轮 p50 / p90 / max                 | 72 / 295 / 509         | **67 / 181 / 505**     | 40 / 98 / 403              |
| assistant 轮 mean / 总                       | 117.5 / 316,026        | 89.7 / 223,704         | 52.8 / 150,634             |
| **每轮形态**                                 |                        |                        |                            |
| `<think>` 字符 p50 / p90 / mean              | 500 / 6,106 / 2,375    | 579 / 6,589 / 2,558    | **2,963 / 16,878 / 6,700** |
| 动作 JSON 字符 p50 / mean                    | 671 / 943              | 723 / 1,012            | 1,089 / 1,636              |
| 每轮命令数 mean / p90                        | 1.95 / 4               | 2.01 / 4               | 2.64 / 5                   |
| 每命令 duration p50 / p90 / mean (s)         | 0.2 / 1.5 / 1.1        | 0.2 / 1.5 / 1.1        | 0.3 / 3.0 / 2.1            |
| 非空命令总数                                 | 599,033                | 438,152                | 382,574                    |
| 读源码类命令占比（sed/nl/cat/grep/ls/find…） | 47.9%                  | 43.7%                  | **37.0%**                  |
| 其中 `sed` / `nl`                            | 16.6% / 8.7%           | 14.4% / 7.5%           | 10.1% / 5.2%               |
| `python3` 占比                               | 12.0%                  | 13.4%                  | 15.8%                      |
| 写文件类命令占比（heredoc / `sed -i` / tee） | 16.0%                  | 17.4%                  | **20.1%**                  |
| 跑测试类命令占比                             | 0.69%                  | 0.65%                  | 0.77%                      |
| **compaction**                               |                        |                        |                            |
| 含 handoff 的轨迹占比                        | 84.9%                  | 84.2%                  | **12.4%**                  |
| handoff / 轨迹 mean                          | 4.44                   | 3.69                   | 0.84                       |
| **行为**                                     |                        |                        |                            |
| 等待轮（空 keystrokes）占比                  | 4.19%                  | 3.98%                  | **8.39%**                  |
| 等待占比 p50 / p90（按轨迹）                 | 1.5% / 11.1%           | 1.4% / 10.7%           | 1.7% / 20.8%               |
| 轮询主导轨迹（等待 >25%）                    | 1.3%                   | **0.9%**               | 6.8%                       |
| `commands: []` 轮占比                        | 0.19%                  | 0.12%                  | 0.00%                      |
| 短 `<think>`（<20 字符）轮占比               | 7.14%                  | 5.43%                  | **0.22%**                  |
| 短 think 占比 p90（按轨迹）                  | 12.5%                  | 10.4%                  | 0.0%                       |
| 长 `<think>`（>30k 字符）轮占比              | 0.85%                  | 0.95%                  | 3.34%                      |
| `task_complete` 宣告 / 轨迹                  | 2.01                   | 2.11                   | 2.14                       |
| **噪声**                                     |                        |                        |                            |
| assistant 乱码（进 loss）                    | 0                      | 0                      | 0                          |
| assistant JSON 不可解析                      | 0                      | **0**                  | 0.40%                      |
| 工具标记 / `</plan>` 残留                    | 0.04%                  | 0.05%                  | 0.00%                      |
| 观察乱码（不进 loss）                        | 0.01%                  | 0.01%                  | 0.02%                      |
| harness parse / newline 告警                 | 0.74%                  | 0.72%                  | 2.75%                      |
| 连续相同观察                                 | 0.57%                  | **0.37%**              | 1.36%                      |

**读表**：

1. **规模相当，分布更紧**。dsv41_final 比 th3k 少 358 条，但 token 中位数一样（223k vs 224k），尾部更短（max 895k vs 2.0M），没有 th3k 那种 4.3M 字符的极端轨迹。
2. **我们的轨迹轮次多一倍、每轮更"薄"**。assistant 轮 mean 90 vs 53；`<think>` 中位数 579 vs 2,963 字符，动作 JSON 也更短；每轮命令 2.0 vs 2.6 条。dsv41-flash 是"小步快跑"风格，th3k 的教师是"每步想透再动"。读源码类命令占比 44% vs 37%、`sed`/`nl` 翻页明显更多，是这一风格的直接体现。清洗把这个差距缩小了一点（原始 48% → 44%）但没改变风格。
3. **compaction 是最大的结构差异**：84% 的轨迹含 handoff（mean 3.7 次），th3k 只有 12%（mean 0.84）。同样的 token 总量下我们的上下文被 96k 触发点切得更碎——这是采集配置决定的，不是数据质量问题，但训练时按段切分后段数会比 th3k 多（见 §C6.1）。
4. **轮询**：th3k 重一倍（8.4% vs 4.0% 轮，6.8% vs 0.9% 轨迹主导）。我们在清理后已经是三者里最低。
5. **短 think 是我们独有的特征**：5.4% vs 0.2%。清理剔了主导型轨迹（7.1% → 5.4%），剩下的是 "Poll." / "Continue 81-160." 这类孤立、情形恰当的短推理；th3k 教师即使等待也写一两句完整推理。如果后续想进一步对齐 th3k 的推理密度，可在 prep 时对 <20 字符 think 的轮 `loss=0`（只学动作、不学"不推理"）。
6. **硬噪声我们更干净**：JSON 不可解析 0 vs 0.40%（th3k 有 14.6% 的轨迹含 `<tool_calls>` 等非 Terminus 输出）、parse 告警 0.72% vs 2.75%、相同观察 0.37% vs 1.36%。assistant 侧乱码三者都是 0。
7. **写文件 / 跑测试**：th3k 写文件类命令占比略高（20% vs 17%），跑测试类三者都 <1%——TH 题目的验收多靠题目自带 CLI（`closectl`、`meridianctl`…）而不是 pytest，所以这项不能直接读成"验证少"。

## C1. 修复一：系统原因导致**没产出轨迹**的 282 条（160 个 env）

### C1.1 怎么查

对 282 条非 DONE 全部拉 `arena task trace` + 完整 harness log（存档 `/storage/openpsi/users/chen/output/cth-thfull14-dsv41-2610270211/audit/`），按 env 粒度归因：

| 组          | env | 根因                                                                                                                                                                                                                  | 修法                                                                                                                | 资产                                                                                                                                                                                              |
| ----------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| flat        | 87  | 平台不执行镜像 ENTRYPOINT，拍扁镜像的 sidecar 服务从未启动；agent 对着不存在的服务跑满 5h                                                                                                                             | setup hook 在主 netns 用 chroot 起 sidecar、写 ready.json（10-31 诊断 25 轮）                                       | env `@1.3.0`                                                                                                                                                                                      |
| tmux-killed | 24  | agent 中途 tmux server 收 SIGTERM/SIGKILL（`send-keys rc=-15/-9`），多服务题在 8Gi 下被 OOM killer 杀                                                                                                                 | harness 内存 8→16Gi                                                                                                 | harness **`cth-terminus-2-stream@1.7.96`**（= 1.6.96 的 16Gi + agent 侧 aliyun pypi/uv 镜像 + HF xet 关闭，再加 `TERMINUS_TOOL_CALL_MODE=function` 保持与原 job 1.4.96 相同的"历史推理可见"口径） |
| transient   | 20  | ARCA 建沙箱超时 8 / agent 真跑满 5h 12                                                                                                                                                                                | 直接重试                                                                                                            | 原版本                                                                                                                                                                                            |
| reward-hang | 8   | `check.py` / `colcon` / `cargo test` 在 reward 阶段等一个永不退出的进程，卡满 7200s ARCA exec 超时；agent 其实已正常完成（全是 ROS2 / robot / tokio / etcd 题）                                                       | oracle 的 `tests/test.sh` 换成 `timeout -k 60 1500 bash /tests/test_orig.sh`，原脚本逐字节保留                      | env `@1.4.0`（`code/reg/register_140_fixes.sh`）                                                                                                                                                  |
| DNS         | 2   | 题目自己的 bootstrap 把 `/etc/resolv.conf` 写成 `nameserver 127.0.0.1`（本地 dnsmasq / consul），harness 之后解析不了 `envarena-prod.alipay.com`，LLM 调用全部 Connection error（修正了之前"镜像 resolv 问题"的判断） | setup hook 先把网关 IP 钉进 `/etc/hosts`（getent 失败回退 python socket，两条路径本地模拟验证）                     | env `@1.4.0`                                                                                                                                                                                      |
| CA-broken   | 18  | 镜像缺 CA 证书，ARCA 装 envd 时 `curl (60)`，envd 不监听，所有任务 TIMEOUT/HARNESS_FAILED                                                                                                                             | 集群内修不了；外部 docker 机重建套件 `build/rt3_ca/`（Dockerfile.rt3 + build_rt3.sh，校验 TLS / ENTRYPOINT / root） | **暂剔除**                                                                                                                                                                                        |
| oracle 过大 | 1   | 83 MB 超 shared-sandbox 投递上限（EVAL_FAILED "grading assets too large"）                                                                                                                                            | 瘦身后再补                                                                                                          | 暂剔除                                                                                                                                                                                            |

清单：`coding_arena/work/manifests/th14_bad_envs.tsv`（160 env：根因 / 原版本 / 重打版本 / 修法）、`tasks_redo14.tsv`（141 可重打）、`th14_excluded_envs.txt`（19 剔除）。

### C1.2 效果

Benchmark `cth-th-redo14@1.0.0`（141 env），job `cth-thredo14-dsv41-2611011220-2611010420`，与原 job 同配置（dsv41 max、2 samples、retries 1），harness 1.7.96：

|                   | 任务 | DONE                 | TIMEOUT | HARNESS_FAILED | mean(DONE) |
| ----------------- | ---- | -------------------- | ------- | -------------- | ---------- |
| 原 job 这 141 env | 282  | **0**（全部非 DONE） | —       | —              | —          |
| 重打              | 282  | **182**              | 54      | 33             | 0.346      |

按修法分：flat `@1.3.0` 117 DONE / 58 solved（修前 0 解，这 87 题对任何模型都是 0）；reward-hang+DNS `@1.4.0` 20 DONE / 1 solved（修前 reward 必卡死，分数永远拿不到）；tmux-killed 题（1.1.0/1.2.0 原版本 + 16Gi）45 DONE / 4 solved。**原本颗粒无收的 141 个 env 产出 182 条可用轨迹、69 个新 env 进入数据。** 残留的 87 条非 DONE：多数是多服务题在 5h 内真跑不完，以及 HARNESS_FAILED 33 条（主要是 eperm-net 类题目 tmux 被杀，与 §C2 的平台策略问题同源）。

---

## C2. 修复二：**已进数据的 DONE 轨迹**里的环境污染（101 条）

### C2.1 为什么还要查

§C1 只保证"有轨迹"。原 job 的 harness 1.4.96 agent shell 没有 pypi/HF 镜像、只有 8Gi，而且 ARCA 网络策略拒绝连入沙箱内自建私网——所以 DONE 轨迹里也可能有"装包装不上""命令被 OOM 杀""服务起了但连不上"这种环境事实，模型据此学到的是"这环境不能装包 / 放弃验证"，是坏信号。

### C2.2 怎么查：高精度检测器 `audit_env_pollution.py`

第一版用宽正则（`Killed|Out of memory|network`）命中 1,006/2,690 条，抽样发现几乎全是任务内容本身：Go 二进制的 runtime 字符串表、编辑器里显示的内核 JBD2 源码、配额题**预期输出**的 ENOSPC、模型自己 `timeout -k 2s` 杀掉的进程、注释里的 "no network"。于是收紧为**只认环境独有的签名**，逐类抽样核对并写了单测：

| 信号        | 判定规则                                                                                                                                                                 | th14 命中（轨迹/env） |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------- |
| eperm-net   | `connect / dial tcp / sendto` 与 `Operation not permitted` 同窗出现（ARCA 拒绝连入 kind / k3s CNI / veth / netns 私网）                                                  | 32 / 24               |
| oom         | 终端裸 `Killed` 行且前一步模型没有 `timeout -s KILL` / `kill -9`；或内核 `oom-kill:` 行。不用 "Cannot allocate memory"（本语料里全是 Go 字符串表 / tc-netem 的 ENOBUFS） | 13 / 10               |
| git-net     | github / raw.githubusercontent `unable to access` / `Could not resolve`                                                                                                  | 11 / 9                |
| pip-net     | 索引域名（files.pythonhosted / pypi / npm / crates）与网络异常同窗，或 pip 的 `Could not install packages due to an OSError`                                             | 8 / 8                 |
| gave-up+net | assistant 以网络为由放弃运行验证，**且**同轨迹有上述真实网络证据（否则只是模型猜测或题目本来离线）                                                                       | 17 / 15               |

th14 合计 **63 条 / 50 env**（`work/manifests/th14_polluted_traj.json`）。同一检测器对两个重打 job 也跑了：redo14 182 条里 19 条、thredopol 79 条里 28 条仍带污染。

### C2.3 怎么修

- **重打**：benchmark `cth-th-redo-pollution@1.0.0`（50 env），job `cth-thredopol-dsv41-2611061153-2611060353`，dsv41 max、2 samples、harness 1.7.96（16Gi + 镜像）。
- **合并** `build_dsv41_merged.py`：三个 job 的 linear 轨迹按 (env, sample) 合并，后来的重打行覆盖原行；每个 job 各自的污染清单里的行一律剔除。

### C2.4 效果

| job                               | DONE  | 污染       | 污染类型                                     |
| --------------------------------- | ----- | ---------- | -------------------------------------------- |
| th14（1.4.96，8Gi，无镜像）       | 2,690 | 63 (2.3%)  | eperm 32、放弃验证 17、oom 13、git 11、pip 8 |
| redo14（1.7.96）                  | 182   | 19 (10.4%) | eperm 13、oom 4、git 4                       |
| thredopol（1.7.96，专打污染 env） | 79    | 28 (35%)   | eperm 21、oom 6、git 2                       |

harness 1.7.96 **消灭了 pip-net**（三个重打 job 0 条），OOM 和 gave-up 大幅下降；**eperm-net 修不了**——那是 ARCA 平台策略（连入沙箱内自建私网两个方向都 EPERM，10-31 flat 诊断时已实锤），这些 k8s / BGP / NFS 多节点题换多少次 harness 都一样，只能剔除（50 个重打 env 里 15 个是这类）。最终 `dsv41_merged` 2,796 条，KEEP 集上再跑一遍检测器：**0 条命中**。

### C2.5 顺带发现的坏数据

`cth-th-resilient-event-time-join-migration-across-savepoint-rescaling` s1：flat 题在修复前唯一一条 DONE（score 0），服务根本不存在，是废轨迹——在 eperm-net 检测里被抓出并剔除。

---

## C3. LLM-judge 过程质量清洗

### C3.1 怎么做（`llm_judge.py`，按 [trajectory_judge](trajectory_judge.md) 逐条实现）

- **Prompt**：TerminalHorizon Appendix J.1 原文，三准则（轨迹有效性 / 环境接地的进展 / 多轮训练价值），唯一改动是 A3 追加一行说明 compaction handoff 不是 truncation。
- **防泄漏**：judge 输入只有 `record_id` + 完整轨迹原文；score / status / job 一律剥掉（脚本断言）；`record_id = sha1(job|env|sample)[:16]`，映射表单独存放。
- **模型**：matrixllm `claude-opus-4-8`，`reasoning_effort=high`，temperature 0，**每请求 1 条完整轨迹**，不摘要、不截断。实测 Bedrock 硬上限 1,000,000 prompt token；用 1.88M 字符（812k token）的真实轨迹做末尾 needle 测试，模型能取回末尾信息并正确数出 `=== ASSISTANT` 头数量 → 门槛设 1.9M 字符，超出记 `OVERLONG_UNJUDGED`，不混入 KEEP。
- **鲁棒性**：JSON 强校验（record_id 一致、verdict ∈ {KEEP, DROP}、证据 ≥2 条）；网关偶发返回 canned 文本（"I'm Claude…"）/ 504 → 退避重试 ≤4 次；`finish_reason=content_filter` 单独记 `CONTENT_FILTERED`；最终失败记 `JUDGE_ERROR`，**不默认 KEEP**。可断点续跑，重打替换的行因 record_id 含 job 会自动重评。
- 8 并发，p50 30s / 条，共 660M prompt token、4.0M completion token，约 5h。

### C3.2 结果

| verdict           | 条数      | 说明                                                                 |
| ----------------- | --------- | -------------------------------------------------------------------- |
| **KEEP**          | **2,541** | 97.9% of judged                                                      |
| DROP              | 53        | 19 泄漏/篡改/绕过、12 全局无效、20 纯浏览无实现、2 其它              |
| OVERLONG_UNJUDGED | 196       | > 1.9M 字符（> ~1M token），49 条 solved；P2 长任务臂                |
| CONTENT_FILTERED  | 4         | 恶意软件 C2 协议 / 固件漏洞 / 疫情扩散题触发网关安全过滤，确定性复现 |
| JUDGE_ERROR       | 2         | 504 + canned 回复 4 次                                               |

### C3.3 DROP 都抓到了什么（每条 3–4 条带轮次号的证据，抽查 17 条全部成立）

- **泄漏 / 篡改 / 绕过（19）**：删掉任务的 SQLite（`rm mission.sqlite -wal -shm` + `service-restart`）重掷一个新 mission；把 `plantd.py` 里的解析真值公式直接抄成"基于证据的拟合"；**挖 `/envarena/out/attempts/1/trajectory.jsonl` 把上一次 attempt 的完整实现按行号抽出来写回文件**；手改 `state.json` 并重算 projection hash 骗过完整性校验；`git apply` 上游 CPython 的 PEP 695 补丁整个打上。这类轨迹不少是 score=1 的——正是论文强调"不看 reward"的原因。
- **全局无效（12）**：超大 heredoc 把终端卡死后 380 轮同一帧，模型疯狂 Ctrl-C/D/Z、`kill -9 -1`、换 delimiter 全无效，最后还宣告 `task_complete`。
- **纯浏览无实现（20）**：几十到数百轮 `sed -n`/`grep`/`nl` 读源码，因终端只显示尾部反复重读同一段，到结束一行代码没改、一次构建没跑。

DROP 集宣告 `task_complete` 的比例 35.8% vs KEEP 集 88.2%，`<think>` 平均短 30%——与"没干完活"一致。

### C3.4 盲评是否真的盲

judge 看不到 score，但按隐藏 score 回看：score=0 的轨迹 KEEP 率 96.8%（1,290/1,333），score=1 的 99.2%（1,251/1,261）。差 2.4pt，是过程质量与成功的自然相关，不是泄漏；同时 1,290 条失败轨迹被保留，符合论文"失败但过程可靠的轨迹要留"的设计（抽查这些 KEEP 的理由都是"真实 debug 链 / 自实现估计器 / 诚实停在障碍"）。

---

### C3.5 KEEP 集的 bad-action 人工 + 程序复查（11-06 补）

judge 按三准则评整体过程，不会逐轮查格式；所以对 2,541 条 KEEP 又做了一遍**逐轮**扫描（`code/sft/scan_bad_actions.py`，结果 `judge_opus48/scan_bad_actions.txt`）并人工读了 15 条分层抽样轨迹（头/中/尾各段）。230,366 个 assistant 轮 / 230,366 个观察轮：

| 模式                                                    | 轮占比         | 轨迹占比 | 人工核查结论                                                                                                                         |
| ------------------------------------------------------- | -------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 乱码（U+FFFD / â€ / Ã©）                                | 0.01% (27 轮)  | 0.6%     | 全部是 `strings` / `grep -a` 读二进制时**终端本来就输出**的字节，不是数据转码错误；模型随即换方法。保留                              |
| 控制字符 / ANSI 残留                                    | 0              | 0        | harness 已清干净                                                                                                                     |
| assistant JSON 不可解析                                 | **0**          | 0        | 构建时 silent-fail 轮已剔（README）                                                                                                  |
| 重复 key / 工具标记泄漏（`<｜DSML｜>`、`<parameter`）   | 0              | 0        | tool-mode 归一化生效                                                                                                                 |
| `</plan>` 残留在 plan 文本尾                            | 0.05% (108 轮) | 3.5%     | 模型偶尔多写一个闭合标记，JSON 合法、命令正常执行；可在 prep 时正则去掉，不影响动作链                                                |
| 空 `<think>`（<20 字符）                                | 5.9%           | 65%      | 看了几十条：全是 "Poll." / "Wait." / "Continue 435-475."——等编译/训练或连续翻页时的极短推理，**内容合理**；不是截断（无 think 截断） |
| 等待轮（keystrokes 为空，只给 duration）                | 4.3%           | 59%      | 正常用法：等 `gmx mdrun` / `cargo build` / SFT 训练。p50 占轨迹 1.5%、p90 11.5%                                                      |
| **等待轮 >40% 的轨迹**                                  | —              | **6 条** | 轮询主导（分子动力学 / SELinux 构建 / 量子电路编译等），judge 认为"等待不加分但过程可靠"而 KEEP。可选剔除                            |
| 连续 3+ 次完全相同的非空命令批                          | 0.06% (145 轮) | 1.5%     | 多为 `tail -5 log` 之类合理重查；最长 run 是观察被截断后重发同一 `sed -n`                                                            |
| harness "parsing errors / should end with newline" 提示 | 0.71%          | 32.6%    | 829 条轨迹里中位数只有 1.6% 的轮带该提示；**≥10 轮的 14 条**是协议摩擦偏多（`C-c` 不带 `\n` 的良性告警为主）                         |
| 连续相同观察 ≥10 次                                     | —              | 2 条     | 前台长任务只回显 `2>&1`，模型每 60s 等一次，自己也意识到"应该用 nohup &"——这是可学的教训但轮次浪费                                   |
| 宣告完成后 harness 二次确认                             | —              | 88% 轨迹 | 这是 Terminus 协议的正常一问一答，不是噪声                                                                                           |

**结论**：没有乱码/截断/格式损坏类的硬噪声；软噪声只有"轮询偏多"和"协议告警偏多"两类，合计 **21 条**（0.8%，清单 `dsv41_merged_judged-keep-degenerate.tsv`：6 条等待 >40%、14 条告警 ≥10 轮、2 条卡终端 ≥10 次，有重叠）。它们不违反论文准则（等待不计分但不扣分），是否剔除看训练目标：追求"少轮询"行为可剔，否则保留。另外建议 prep 阶段做两件轻量清理：去掉 plan 字段尾部的 `</plan>`；对 "Poll." 类空 think 不做特殊处理（它们是对真实情形的恰当短推理）。

### C3.6 乱码会不会进 loss？与 th3k 的 bad-action 对比（11-09 补）

**乱码不会进 loss。** 对两套数据的全部 assistant 轮单独扫（`scan_bad_actions.py` 的 `mojibake-asst` 项）：我们 230,366 个 assistant 轮 **0** 个含乱码，th3k 150,634 个也是 **0**。乱码只出现在 user 观察轮（我们 27 轮 / th3k 26 轮，都是 `strings`、`cat` 二进制、故障注入输出），而观察轮不做监督；模型读到乱码后的 `<think>`/动作是干净的（"Still garbage… send `reset`"）。所以无论 `--bad-pair` 用哪种策略，乱码都只在输入侧，不在 label 侧。

**与 th3k（论文发布的 2,852 条，同一 Terminus-2 格式、同样带 `<think>`）用同一扫描器对比**（`judge_opus48/scan_bad_actions{,_th3k}.txt`）：

| 模式                               | **我们 KEEP** 轮占比 / 轨迹占比 | **th3k** 轮占比 / 轨迹占比 | 说明                                                                                                                                                                                                           |
| ---------------------------------- | ------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| assistant 乱码（进 loss）          | **0 / 0**                       | **0 / 0**                  | 两边都干净                                                                                                                                                                                                     |
| 观察乱码（不进 loss）              | 0.01% / 0.6%                    | 0.02% / 0.5%               | 同量级，都是终端真实字节                                                                                                                                                                                       |
| **assistant JSON 不可解析**        | **0 / 0**                       | **0.40% / 14.6%**          | th3k 有 598 轮模型输出 `<tool_calls>[…]` 或 `<commentary>` 等非 Terminus 格式，harness 回 "Missing required fields: analysis, plan, commands"；最多一条 16 轮。我们构建时已把这类 silent-fail 轮剔掉（README） |
| harness parse-error / newline 告警 | 0.71% / 32.6%                   | **2.70% / 42.7%**          | th3k 高 3.8×（含上面的 JSON 错）                                                                                                                                                                               |
| 等待轮（空 keystrokes）            | 4.25% / 58.6%                   | **8.42% / 52.8%**          | th3k 每条轨迹里等待占比高一倍                                                                                                                                                                                  |
| 等待 >25% 的轨迹                   | 1.4%（35 条）                   | **6.8%（195 条）**         | th3k 轮询主导的轨迹是我们的 5×                                                                                                                                                                                 |
| 连续相同观察                       | 0.54% / 20.4%                   | 1.32% / 24.1%              |                                                                                                                                                                                                                |
| 连续 3+ 次同一非空命令批           | 0.06% / 1.5%                    | 0.03% / 0.8%               | 我们略高（观察截断后重发 `sed -n`）                                                                                                                                                                            |
| `commands: []` 空批                | 0.22% / 5.1%                    | 0.00% / 0.1%               | 我们的 dsv41 等编译时偶尔不给 duration 直接空批；th3k 几乎没有                                                                                                                                                 |
| `<think>` <20 字符                 | 5.88% / 65%                     | 0.22% / 8%                 | 我们的 dsv41 在等待/翻页时写 "Poll." "Wait."；th3k 的教师模型即使等待也写一两句完整推理                                                                                                                        |
| `<think>` >30k 字符                | 0.95% / 47%                     | 3.34% / 58%                | th3k 长推理更多                                                                                                                                                                                                |
| `</plan>` 等标记残留               | 0.05% / 3.5%                    | 0.00% / 0.1%               | 我们略多，prep 时去掉                                                                                                                                                                                          |

**结论**：在"硬噪声"（乱码进 loss、JSON 损坏、协议错误）上我们的 KEEP 集**比 th3k 更干净**——th3k 14.6% 的轨迹含不可解析动作、42.7% 含 parse 告警，是论文 judge 按"孤立错误可恢复"原则放过的；我们在构建阶段就剔了 silent-fail 轮，并在 prep 时用 `--bad-pair mask` 把残余的坏动作 `loss=0`。在"轮询"上 th3k 比我们重一倍（8.4% vs 4.3% 轮，6.8% vs 1.4% 轨迹轮询主导）——说明 §C3.5 的 21 条轮询偏重轨迹不是异常，th3k 用这样的数据照样训出了最好的结果；不必剔除。我们独有的两个小差异是 "Poll." 式极短 think（5.9% 轮）和偶发 `commands: []` 空批（0.2%），前者是对等待情形的恰当反应，后者可在 prep 时给默认 duration 或 mask 掉，影响都很小。

### C3.7 最终清理：形态 A / 形态 C / 短 think 占比高（11-09）→ `dsv41_final`

`code/sft/finalize_dataset.py` 对 2,541 条 KEEP 做最后一道清理，两层：

**整条剔除（47 条）**

| 规则          | 阈值                                               | 剔  | 依据                                                                               |
| ------------- | -------------------------------------------------- | --- | ---------------------------------------------------------------------------------- |
| 短 think 主导 | `<think>` <20 字符的轮 >20%                        | 35  | th3k 这类轮只有 0.2%，我们 5.9%；整条被 "Poll." "Continue." 主导的轨迹在教"别推理" |
| 形态 A 主导   | 前台阻塞、观察逐字相同的等待轮 >25%                | 1   | 20 分钟 `gmx mdrun` 前台空转                                                       |
| 形态 C 主导   | `commands: []`（无 duration、拿到陈旧屏幕）轮 >10% | 14  | 协议误用；th3k 全集只有 3 轮                                                       |

阈值取法：短 think 占比的分布 p90 = 11%、p99 = 22%，>20% 是明显的右尾（35 条）；形态 A/C 的阈值卡在"该行为是否主导轨迹"。剔掉的 47 条里 20 条 solved，不按成败区别对待。

**轮级折叠（118 条轨迹、306 对，占全部轮的 0.14%）**

- 形态 A：连续"等待 → 观察与上一次逐字相同"的 run 只保留第一对，紧接着观察发生变化的那一对保留，链条变成 `等待 → 仍在跑 → 等待 → 跑完 → 动作`，而不是 17 次 `等待 → "2>&1"`（164 个 run，275 对）。
- 形态 C：≥3 次连续 `commands: []` 只保留首尾两对（13 个 run，31 对）。
- user/assistant 交替、首条任务 prompt、compaction handoff 不动。先在两条最差轨迹上逐轮核过前后文再跑全量。

**结果**（`/storage/openpsi/users/chen/code_sft_data/dsv41_final/`）：

|                                           | KEEP（清理前）            | **dsv41_final**                               |
| ----------------------------------------- | ------------------------- | --------------------------------------------- |
| 轨迹 / solved / env                       | 2,541 / 1,251 / 1,349     | **2,494 / 1,231 / 1,340**（1,154 个两样本齐） |
| assistant 轮                              | 230,366                   | 223,704                                       |
| 字符 p50 / p90 / max                      | 469k / 1,105k / 1,880k    | 468k / 1,098k / 1,879k（1.42 GB）             |
| 估算 token p50 / p90 / max / 总           | 223k / 526k / 895k / 690M | 223k / 523k / 895k / **675M**                 |
| 形态 A（相同观察）轮占比                  | 0.54%                     | **0.36%**                                     |
| 轮询主导（等待 >25%）轨迹                 | 35                        | **22**                                        |
| `commands: []` 轮                         | 501                       | **258**                                       |
| 短 think 轮占比                           | 5.88%                     | 5.43%（剩余为孤立、合乎情形的短推理）         |
| assistant 侧乱码 / 坏 JSON / 工具标记泄漏 | 0 / 0 / 0                 | 0 / 0 / 0                                     |

剩余的 22 条"等待 >25%"是形态 B（每次等待都带回新进度，如训练 loss、编译进度），th3k 里这类占 6.8%，不剔。

## C4. 这批数据够不够干净做 SFT

**够**，依据四条：

1. **环境污染零残留**：KEEP 集上重跑污染检测器 0 命中；pip-net 在新 harness 下根本不再出现。
2. **Judge 质量可审计**：2,626 条 review 全部 3–4 条具体证据（2,625 条 4 条），抽查 DROP 12 条 + KEEP(score=0) 5 条全部成立；judge 抓到了只有读全文才能发现的 benchmark-gaming（挖 `/envarena` 目录抄上次答案）。
3. **无成败泄漏**：盲评下 KEEP 率随 score 只差 2.4pt。
4. **对标论文**：KEEP 率 97.9% vs th3k 92.9%——我们先做了环境污染剔除再 judge，更高是预期的；论文 2,852 条 vs 我们 2,541 条（另有 196 条超长待处理）。

**已知缺口（不影响当前集合的干净度）**：18 个 CA 坏镜像 env 待外部 docker 机重建后补数据；196 条超长轨迹（7%）未评；trajectory_judge.md §9 要求的人工抽查 30 DROP + 30 KEEP 目前做了 17 条 judge review + 15 条整轨迹逐段人读（§C3.5）；21 条轮询/协议摩擦偏重的 KEEP 轨迹（0.8%）按训练目标决定是否剔；eperm-net 类题目（k8s / BGP / NFS 多节点）在 ARCA 上永远拿不到干净轨迹，需要平台放开私网策略或改用单节点题。

**Part C 不处理的（11-10 补，风格层缺口，详见 [trajectory_failures](../experiments/trajectory_failures.md) §12–§13）**：本流水线从设计上只洗"真实性"（污染、作弊、废轨迹、形态 A/C），**不改教师风格分布**——`dsv41_final` 的 think p50 579 vs th3k 2,963、84% 含 compaction 在清洗前后不变。已量化的两项残留偏差及对症手段（均为 prep 阶段操作，不动本数据集）：①宣告时刻犹豫——确认轮 think p50 4,933 vs th3k 2,922、宣告轮 >20k 占比 9.7% vs 4.7%、失败轨迹 81% 以虚假 task_complete 结尾（1,497 个失败段、21% 监督字符）→ 失败轨迹宣告轮 mask；②中段 think 重尾（>20k 轮 141M 字符、24.7% think）→ 相对增长 mask（>4× 首段均值）。已排除的做法：剪尾部失败轮（连续卡死 run 仅 4/1,263，§C3.7 已消化；会删 94% solved 的收尾）、切尾部 >20k 轮（71% 是宣告前自审、一半来自 solved，th3k 自己 36.8% think 字符在 >20k 轮照样最优）。工具 `coding_eval/tools/{sft_paper_criteria,pland_tail_trim_sim,pland_stuck_trim_sim}.py`。

---

## C5. 复现

```bash
cd /home/admin/chentinhong.cth/Chenth20/LlamaFactory/coding_arena
python3 code/sft/audit_env_pollution.py --base <base> --out /tmp/x          # -> /tmp/x_traj.json
python3 code/sft/build_dsv41_merged.py \
  --polluted cth-thfull14-dsv41-2610270211=work/manifests/th14_polluted_traj.json \
             cth-thredo14-dsv41-2611011220-2611010420=work/manifests/redo14_polluted_traj.json \
             cth-thredopol-dsv41-2611061153-2611060353=work/manifests/thredopol_polluted_traj.json \
  --extra thredopol-linear
source ~/.secrets.env
python3 code/sft/llm_judge.py --base .../dsv41_merged/dsv41_merged --out .../dsv41_merged/judge_opus48 --workers 8 --retries 4
python3 code/sft/apply_judge.py --base .../dsv41_merged/dsv41_merged --judge .../dsv41_merged/judge_opus48 --out-prefix .../dsv41_merged/dsv41_merged_judged
```

关键 job：`cth-thfull14-dsv41-2610270211`（原始）、`cth-thredo14-dsv41-2611011220-2611010420`（系统坏 env 重打）、`cth-thredopol-dsv41-2611061153-2611060353`（污染 env 重打）。详细诊断过程（flat 镜像 25 轮 diag、端口扫描审计）与 TB2.1 侧的同类修复见 [tb21_verifier_audit](tb21_verifier_audit.md) §5。

---

## C6. 长度问题怎么处理的 & SFT 最终怎么做的

### C6.1 "≤262k token 只有 1,515 条"是怎么解决的

§C0.2 那张表量的是**整轨迹**（linear，compaction handoff 内联）的长度：KEEP 集 2,541 条里 whole-trajectory token p50 147k、max 678k，485 条超过 262k 的训练窗口。如果按整轨迹训练，要么丢 40%，要么截断（丢掉实现/验证/交卷的尾巴，`prepare_terminal_horizon.py` 文档里明确反对）。

实际做法是**不按整轨迹训练，按 compaction 段训练**——这与数据本身的生成方式一致：Terminus-2 的 96k compaction 触发后，模型看到的上下文就是"原始任务 prompt + handoff 摘要 + 后续步骤"，每一段本来就是一个独立的模型上下文。`build_multi_sft.py`（不带 `--linear`）把每条 KEEP 轨迹切成 k+1 段（段 0 = 原 prompt…最后一步；段 i>0 = 原 prompt + handoff 摘要 + 续作），用 judge 的 KEEP 清单过滤（`skipped_not_allowed`）：

|                                                  | 整轨迹（judge 输入）      | **compaction 段（训练输入 `dsv41_judged`）**           |
| ------------------------------------------------ | ------------------------- | ------------------------------------------------------ |
| 条数                                             | 2,541                     | **8,876**（每轨迹 1–14 段，mean 3.5；solved 段 3,787） |
| token p50 / p90 / p99 / max（Qwen3.5 tokenizer） | 147k / 348k / 559k / 678k | **48.7k / 94.6k / 106k / 139.8k**                      |
| 超过 262,144 的条数                              | 485 (19%)                 | **0**                                                  |
| 总 token / 监督 token                            | 453M                      | 453M / 220.5M（48.7% 是 assistant 侧）                 |

段长天然被 96k 触发点封顶（max 140k 是最后一段 + 长 observation），所以 `--overlong drop` 实际 0 丢弃、0 截断、0 tail-window，**2,541 条轨迹的全部 230,366 个 (observation, action) 对 100% 进入训练**。segment 化时顺带做了 Terminus 工具模式的归一化：剥 `Analysis:/Plan:` 二次渲染、丢 13,737 个"模型发了命令但工具调用 JSON 非法、harness 什么都没执行"的静默失败步（保留会教"想做事但 commands=[]"）、丢 65 个显式 parse-error 步；每段末尾不带 observation 的 user 尾巴（8,877 个）去掉，使每段以 assistant 结尾。

### C6.2 SFT 配方（`abl_dsv41_judged_const_e8`）

| 项                        | 取值                                                                                                                                                                                                                                  | 说明                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 数据                      | `terminal_horizon_dsv41_judged`：8,876 段 / 2,541 轨迹                                                                                                                                                                                | `prep_and_sched.py --src dsv41_judged.jsonl`，`--bad-pair mask --overlong drop --think-cap 0`                                      |
| 模型                      | Qwen3.5-4B-text（另跑一份 Qwen3.5-9B，`dsv41_judged_9b`）                                                                                                                                                                             | 文本-only 导出的 VLM 权重                                                                                                          |
| 窗口                      | 262,144，不 packing，不切 chunk                                                                                                                                                                                                       | 一段一行                                                                                                                           |
| loss                      | 每个 assistant 轮全部监督（`<think>` + JSON 动作）；非法动作轮留在上下文但 `loss: 0`（bad-pair mask）                                                                                                                                 |                                                                                                                                    |
| **batch = 64 条完整轨迹** | 一条轨迹的所有段在同一个 optimizer step；`build_traj_sched.py --order global_sort`：按整轨迹 token 全局降序、64 条连续一步、8 个 epoch 固定、无 RNG；每 epoch 2541 % 64 = 45 条最短轨迹丢弃（2-3 条的小 batch 会让 grad norm 飙 20×） | 39 步/epoch × 8 = **312 步**；4B 2 节点 16 rank → `ga=40`；9B 4 节点 32 rank → `ga=20`；dummy 行填充到 640 slot/step，有效填充 35% |
| 优化                      | lr 1e-5 **constant_with_warmup**、AdamW(0.9, 0.95)、wd 0.1、clip 1、bf16、ZeRO-3、FA2、liger、seed 42                                                                                                                                 | 与所有 `*_const_e8` 消融同一配方，只换数据                                                                                         |
| epoch                     | 8                                                                                                                                                                                                                                     | 每 epoch 存一份可直接评测的 merged 权重（`checkpoint-{39,78,…,312}`），中间步只留最后一个完整 checkpoint                           |
| 算力                      | 4B：AIS 2×8 GPU，15h47m，3.61B token 吞吐 63.6k tok/s；9B：AIS 4×8 GPU，12h59m，77.4k tok/s                                                                                                                                           | 两个 run 都已跑完 8 epoch                                                                                                          |

训练 loss（每 epoch 末，token-mean over 64 trajectories）：

| epoch | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 4B    | 0.939 | 0.827 | 0.737 | 0.661 | 0.620 | 0.587 | 0.562 | 0.537 |
| 9B    | 0.895 | 0.787 | 0.696 | 0.607 | 0.546 | 0.525 | 0.488 | 0.453 |

单调下降、无 spike（tail-drop 后 grad norm 稳定），9B 每 epoch 比 4B 低 0.04–0.08。

### C6.3 对比口径与待办

与之前的 `dsv41_history_const_e8`（同源 job、未修复、未 judge：2,690 轨迹 / 11,146 段）相比，本数据少 149 条轨迹（−53 DROP、−101 污染、+211 重打净增 ≈ −2,690+2,541）但多 69 个 env，且剔掉了 19 条 judge 认定靠泄漏/篡改拿分的 solved 轨迹。两者训练配方完全一致，可直接在 TB2.1（`cth-tb21-full89-mirror@1.0.1`，harness 1.6.96）上按 epoch 对比——这是 [README](../experiments/README.md) P1 臂的核心判定，同设置的逐 epoch 配对仍待补齐；C 臂已有本地分数（全部已跑格子见 [sft_results](../experiments/sft_results.md) §3.1b）。

整轨迹（linear）路线的 196 条 OVERLONG + 485 条 >262k 的 KEEP 轨迹属 P2（tail-keep / 更长窗口），不在本轮。

核心表（4B，ep8 或 best，×2=178）：

|       | 筛选                                     | 分数                  | 口径          | 折到 mirror（估） | 相对 A                       |
| ----- | ---------------------------------------- | --------------------- | ------------- | ----------------- | ---------------------------- |
| **A** | 不 judge，全量（2,690 轨迹 / 11,146 段） | 0.258                 | [旧] / [镜]   | ~0.32             | —                            |
| **B** | 不 judge，剔除未解决（1,253 / 4,228）    | 0.191                 | [旧]          | ~0.25             | **−0.067**（同口径实测）     |
| **C** | judge，全量（2,541 / 8,876）             | **0.298**；9B 0.343   | [本] tb_local | ~0.32–0.35        | **≈ 0**（跨口径外推，±0.03） |
| 参照  | th3k mask_const                          | 0.348 / 0.410 / 0.354 | 三口径都有    | —                 | —                            |
