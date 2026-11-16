# mask_const c312 @ mirror 口径（0.410）的 105 条失败：手动归因与数据补法（2026-11-09）

> **样本口径**　178 次采样中 73 次通过。105 次失败含 42 次主动交付且部分测试通过、8 次无 reward 的 NONDONE；完整细分见 §1。CAP512 是旧分类器标签，1.0.0 无 max_turns，本例在 141 分钟后结束，确切终止原因需核对运行日志。分类器结局和人工根因是对同批记录的两种计数。
>
> 归档的 1.0.0 配置为 T=1、top_p=0.95、top_k=50、reasoning_effort=max，未设置 max_turns。早期报告另记 top_k=20/xhigh，各次请求的实际生效参数需查原始请求或服务日志。[归档配置](../arena_assets/harness/configs/1.0.0.env)

> job：`cth-tb21m-mask-const-c312-2611051616`（benchmark `cth-t*@1.0.1`，harness `cth-terminus-2-stream@1.0.0`，178 task，73 pass，105 fail）。
> 本文是对 [trajectory_failures](trajectory_failures.md) §1（旧口径 110 条）的 mirror 口径重做，并把分类器六类再往下拆一层，拆到"模型到底哪个动作做错了"。
> 方法：`traj_deep.json` + 逐条 `trajectory.jsonl` / `tbench_test.log` / `runner.log`；每条失败都看了验收断言，TTL 和 PARTIAL 全部逐条读了轨迹首尾与关键段。
> 辅助脚本在 `/tmp/opencode/c312/{extract,timeline}.py`（一次性，不入库）。训练数据对照抽样自 `data/terminal_horizon_th3k_mask/th3k_mask.jsonl`（25%，603 条）。

## 0. 一页结论

105 条失败按"根因"而不是按"结局"归类，得到 7 个根因（条数 / 占失败 / 可挽回估计）：

| #   | 根因（每条失败只记一个主因）                   | 条  | 占  | 本质                                                                                                                                                                                                         | 数据能不能治                                                                     |
| --- | ---------------------------------------------- | --- | --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| R1  | **题目超出 4B 能力，4 h 在正确方向上磨不出来** | 41  | 39% | 29 TTL + 1 CAP512 + 8 PARTIAL（差最后一位精度）+ 3 ALLFAIL；密码学攻击、CoreWars、调度器优化、polyglot、逆向、生物序列等；这些 env 所有模型都 0                                                              | 不能。只能止损（交最简版、省墙钟）                                               |
| R2  | **需求漏项 / 没跑就交卷**                      | 26  | 25% | 做出 N−1 条要求就 `task_complete`；环境缺工具时用 py_compile + 读代码代替运行                                                                                                                                | **能**（交卷前"需求清单对照"+"先装先跑"数据）                                    |
| R3  | **终端被自己弄死**                             | 13  | 12% | 5 TTL + 4 NO_DECLARE + 4 HARNESS_FAILED。三种具体错误：heredoc 内反斜杠/引号让 bash 进续行吞输入；前台起 qemu / `bash -i` 子进程占住 harness 的 tmux pane；交互 shell 里写 `set -e` / 裸 `exit` 把 bash 杀掉 | **能**（"终端卡死 → 标准恢复 / 交互程序隔离到独立 tmux window / 禁 set -e"数据） |
| R4  | **环境外部依赖不可达（网/源/包），绕 4 h**     | 4   | 4%  | povray 源不存在、opam 等 55 min、R 包装 1 h、postfix 配置；另有 5 条（nginx、caffe、mcmc ×2、protein ×2 的前半段）外部依赖耗掉大半步数后以 R1/R2 收场                                                        | 部分能（"外部不可达 → 换路径或本地实现"的数据）                                  |
| R5  | **走了题意禁止的捷径 / 取了字面解**            | 12  | 11% | `image.c` 直接 `system("./orig")`；`steal.py` 直接读 `forward.A1`；leaderboard 过滤口径、编码方式、"所有变量"等隐含正确做法没取到                                                                            | 部分能（"题意里的隐含禁止"负例，适合 DPO）                                       |
| R6  | **自写探针反复出错，debug 自己而不是解题**     | 3   | 3%  | 独立记 3 条 TTL；但叠加在 R1 的 29 条 TTL 上：一条轨迹里 25–38% 的步在看自己 python 脚本的 Traceback，输出被截断就再写一个脚本                                                                               | **能**（"长输出重定向到文件 + sed -n 分段读"、"探针一次写对"数据）               |
| R7  | **基础设施 / verifier 口径**                   | 6   | 6%  | 4 条 serving 断流 `MidStreamFallbackError`（伴随 compaction 摘要撞 max_tokens）；2 条 sanitize-git-repo（第三个污染文件是 `.json` 数据文件里形态不同的 token，两条采样都看到了并判为非真实）                 | 不能/不该由数据治                                                                |

对用户的三个具体问题：

1. **能不能归到"某个工具不会用"？** 能归到 3 个，但不是"工具"，是"终端纪律"：
   - **heredoc 内含 `\u`/`\x`/反斜杠** → bash 把 `'` 配对吃错，shell 进入续行模式，之后所有输入被当作字符串吞掉，屏幕只剩 `n'`、`>`。`large-scale-text-editing` 两条（927 步合计 300+ 步在恢复）、`headless-terminal`（301/304 步无提示符）、`raman-fitting s0`（50 步连续无提示符）都是这个。训练集里 `C-*` 只占 0.39% 的命令，teacher 几乎没遇到过这种状态，模型没有"恢复模板"。
   - **在 tmux 主 pane 里起交互程序**（`python3 -` 子进程里 spawn `bash -i`、`vim -Es` 卡住、qemu `-nographic` 前台）→ pane 被占住，harness 的 `tmux send-keys` 超时（return_code=-15）或 pane 消失，直接 HARNESS_FAILED（4 条：install-windows ×2、make-mips-interpreter s0、qemu-alpine-ssh s1）。`qemu-alpine-ssh s1` 甚至自己 `tmux new-session -d -s alpine`，然后把命令发到 `terminus-2.0`——就是 harness 自己的 pane。
   - **`set -e` / 裸 `exit` 写进交互 shell** → bash 退出，harness 后续 send-keys 全部落空，轨迹以 `Current Terminal Screen:` 空屏结束（NO_DECLARE 4 条里 3 条：compile-compcert s1、git-multibranch s0、make-mips-interpreter s1）。
   这三类合计 **13 条（12%）**，是本轮最"便宜"的可修项，因为它们不是能力问题。
2. **多少是 TTL？** 41 条 TTL + 1 CAP512 = 42。但 TTL 不是根因是结局：其中 R1 硬题 30（含 CAP512）、R3 终端死 5、R4 外部依赖 4、R6 debug 自己 3。**TTL 中 71% 是不可解题**（24 个 env 里 9B 也只中 1，和旧口径结论一致），这部分数据治不了，只能让它 40 分钟就停。
3. **应该加什么数据？** §4 给了 6 类，按优先级：(a) 交卷前需求清单对照（打 R2，26 条，最大的可挽回块）；(b) 终端卡死恢复 + 交互程序隔离到独立 tmux window（打 R3，13 条）；(c) 长输出走文件 + 探针一次写对（打 R6）；(d) 外部不可达时的降级路径（打 R4）；(e) "题意隐含禁止"的负例（打 R5）；(f) 止损：N 分钟没有 /app 产物就先写最简版再继续（打 R1 的墙钟，不涨分但省 3/4 算力）。

## 1. 分类器六类 → 根因七类的映射

先给分类器的结局分布（`traj_deep.json`，mirror 口径）：

| 结局                     | 条  | 说明                                                         |
| ------------------------ | --- | ------------------------------------------------------------ |
| TTL                      | 41  | 4 h 被杀                                                     |
| DECLARED_PARTIAL         | 42  | 宣告后部分测试过                                             |
| DECLARED_ALLFAIL         | 8   | 宣告后全挂                                                   |
| NONDONE (HARNESS_FAILED) | 8   | harness 崩，无 reward                                        |
| NO_DECLARE               | 4   | 没超时也没交卷，shell 死了                                   |
| DECLARED_NOTEST          | 1   | 验收没跑起来                                                 |
| CAP512                   | 1   | 旧分类标签；141 分钟后结束，配置无步数上限 |

每条失败的根因判定（env/s 级别）：

### 1.1 TTL 41 + CAP512 1

| 根因          | env/s                                                                                                                                                                                                                                                                                                                                                                                                                  | 条                      | 证据                                                                                                                                                                                                                                                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| R1 硬题       | circuit-fibsqrt ×2、feal-differential s1、feal-linear ×2、llm-inference-batching ×2、winning-avg-corewars ×2、write-compressor ×2、schemelike-metacircular ×2、regex-chess ×2、polyglot-rust-c ×2、path-tracing-reverse s0、make-doom-for-mips ×2、extract-moves-from-video ×2、video-processing ×2（含 CAP512）、db-wal-recovery s0、dna-assembly s1、chess-best-move ×2、custom-memory-heap-crash s0、extract-elf s0 | 30（29 TTL + 1 CAP512） | 每条 300–990 步、think 均值 4–30k、`/app` 写入 0–88 次，产物要么没写（NO-DELIVERABLE 17 条）要么写了但错（12 条）。跨模型：这些 env 在 arm A / dsvh / 9B base 全部为 0。典型：`video-processing s0` 990 步 13 次 compaction，取 takeoff frame 27 vs 期望 50–54，方向对但精度不够；`llm-inference-batching` 两条 cost 超阈值 10–30 倍 |
| R3 终端死     | headless-terminal s0、large-scale-text-editing ×2、raman-fitting s0、protein-assembly s0（158/466 步无提示符）                                                                                                                                                                                                                                                                                                         | 5                       | 见 §2.3                                                                                                                                                                                                                                                                                                                              |
| R4 外部依赖   | build-pov-ray s0（apt 源无 povray，18 次 pkg、130 次 apt-cache）、fix-ocaml-gc s0（55 min 在等 opam）、adaptive-rejection-sampler s1（Rscript 装包 + 自测 R 脚本超时）、mailman s1（postfix 配置，602 步）                                                                                                                                                                                                             | 4                       | 见 §2.4                                                                                                                                                                                                                                                                                                                              |
| R6 debug 自己 | 独立记 3 条，它们的 selferr 步占比 20–31% 且产物从未落到 /app，debug 自己是比题难更直接的死因：protein-assembly s1（103/503 步 selferr）、dna-assembly s0（104/335）、gpt2-codegolf s1（105/398）。chess-best-move ×2（163/166 步）、llm-inference ×2（183/141）、video-processing s0（380/990）的 selferr 更高，但题本身所有模型都 0，仍记 R1                                                                         | 3                       | §2.6                                                                                                                                                                                                                                                                                                                                 |

TTL 的几个量化事实（41 条）：
- **requested wait（duration 字段之和）中位数 19 分钟 / 240**，即 92% 的墙钟是模型在生成，不是在等命令。和 qf 系（81 分钟 sleep）完全不同，th3k 模型不睡觉。
- **completion token 中位数 1.74M / 条**，think>20k 字符的步占 think 字符总量的中位数 54%（`regex-chess` 96–98%、`polyglot-rust-c` 82–91%）。卡住后单步 think 爆炸是 TTL 的墙钟来源。
- **自写脚本报错的步占比中位数 12%**，最高 38%（video-processing s0 380/990 步在看自己脚本的 Traceback）。合计 42 条 = R1 30 + R3 5 + R4 4 + R6 3。
- `/tmp` 写入 vs `/app` 写入：R1 的 29 条里 17 条 `/app` 写入 ≤6 次，`/tmp` 写入 50–644 次。模型在 `/tmp` 做研究，没把阶段成果落到 `/app`。

### 1.2 DECLARED_PARTIAL 42

逐条看了验收断言和宣告前 5 步，分成：

| 根因                                   | env/s                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 条  | 具体漏了什么                                                                                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **R2a 需求显式列出但没核对**           | constraints-scheduling s0（选了 Tue 10–11，但 Alice 10–12 / Carol 11–12 当天有会，第 0 步 cat 过日历、think 39k 推错后不再 grep 验证）、cancel-async-tasks s1（6 测试过 4，漏"排队中任务也要 cleanup"和 "max_concurrent 限流"）、nginx-request-logging s1（`limit_req_zone` 没写 `rate=10r/s`，题目原文"10 requests per second"；32 次写这行都没带 rate，但 think 里 11 次提到 `rate=`）、caffe-cifar-10 s0（题目要 CPU_ONLY，CMakeCache 没开）、kv-store-grpc s1（题目要 `{class name}_pb2.py`，模型把 `kv_store_pb2.py` 改名成 `KVStore_pb2.py`，验收要的正是原名）、financial-document-processor s0（分类漏 1 个发票 + CSV 格式）、db-wal-recovery s1（11 条都在，但 WAL 的 update 没 apply：恢复了行数没恢复值）、log-summary-date-ranges s0（8 步就交，today 口径算错 370 vs 2266）、overfull-hbox s1（只能改 synonyms.txt 里的词，却把 `I've` 小写成 `i've`） | 9   | 宣告时 think 1.3k–5.4k，宣告前 5 步全是 `cat 文件` / `py_compile` / `ls -l`，没有一条是对照题目逐项打勾                                       |
| **R2b 无法本地验证就按"规范上合法"交** | sparql-university ×2（本地 rdflib 不支持 EXISTS/HAVING 子查询，两条都知道，一条写了 "standard SPARQL 1.1, should be valid on a standard engine" 就交；验收恰恰用 rdflib）、torch-tensor-parallelism ×2（环境没 torch，`find / -name torch*` 为空，于是 6 次重写 + py_compile 就交；验收报 `NameError: name 'rank' is not defined`——一个运行一次就会暴露的 bug）、torch-pipeline-parallelism ×2（同上，`RuntimeError: only Tensors of floating point dtype can require gradients`）、sam-cell-seg ×2（从未运行过 `python /app/convert_masks.py`，验收第一个测试就是跑它：`ValueError: cannot reshape array of size 25 into shape (2)`）、query-optimize ×2（原查询太慢没等完，用 LIMIT 10 对比就交；验收是 1.05× 原时间，模型版本反而慢 1.4×）                                                                                                                       | 10  | 共同点：**验收工具就在环境里或可以装**（sparql 的 rdflib 已装；torch 可以 pip；sam 的 convert_masks.py 可以直接跑），模型选择了"推理替代运行" |
| **R2c 多采样 / 多候选只交了一个**      | password-recovery s1（找到的是 14 字符 `8XDP5Q2RT9ZW54`，题目说 23 字符；模型在 think 里把 "PASSWORD=" 前缀算进去凑成 23 自圆其说）、gcode-to-text ×2（输出 "Embossed text" / "MK4S"，是 G-code 注释里的对象名，不是实际雕出来的字；题目允许"写多个候选"模型只写了一个）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 3   | 题面明确允许多猜 / 有明显自相矛盾时不复核                                                                                                     |
| **R5 捷径/理解偏**                     | path-tracing ×2（`image.c` 用 `system("./orig")` + rename；验收 `chroot /jail` 里没有 `orig`。两条都在 think 里写了 "might be questionable… but not reading image.ppm" 然后交）、mteb-leaderboard ×2（算出 F2LLM-v2-14B，期望 GritLM-7B：leaderboard 数据口径/"all tasks"过滤理解错）、mteb-retrieve ×2（第 5 相似取成 HumanEval 行，期望 MTEB 行：用 `T2Retrieval` 的 prompt/任务封装，与验收编码方式不同）、extract-elf s1（只抽了 18.5%，题目要 75%：只扫 LOAD 段的 int32，验收要的是 .data 里所有变量）、bn-fit-modify s1（DAG 多了一条边 U→R 少 Y→D，约束"U 无父节点"没用上）                                                                                                                                                                                                                                                                                  | 9   | 都是**题意里有一个隐含的"正确做法"**（算法性而非调用 orig；全任务覆盖；标准编码），模型选了字面满足                                           |
| **R1 硬题（数值精度）**                | raman-fitting s1（G 峰 1682 vs 1580，数据文件格式两列混淆）、mcmc-sampling-stan ×2（s0 alpha 1.99 vs [2.84,2.91]：模型用了错的先验实现；s1 2.926 vs 2.91 上限，只差 0.5%）、code-from-image s0（hash 前缀对、全文错：伪代码一处细节抄错）、pytorch-model-cli ×2（29/50 和 44/50 张预测错：没 torch，手写 CNN 推理，weights.json 导出正确但卷积实现有偏）、train-fasttext s0（0.618 vs 0.62，差 0.002）、custom-memory-heap-crash s1（debug 和 release 都崩：主程序 double free，模型只能改 user.cpp，用 LD 符号覆盖 free 失败）                                                                                                                                                                                                                                                                                                                                     | 8   | 方向全对、差最后一个精度 / 一个细节；mcmc s1 和 fasttext 两条是 1% 以内                                                                       |
| **R7 verifier**                        | sanitize-git-repo ×2（验收 CONTAMINATED_PATHS 含 `exp_data/datasets/tokenized/rw_v2_*.json`，里面还有第二个 HF token `hf_ocff…`；模型两条都 grep 到了那个 json 文件名但判断为"非真实 token"没改。按题意 "sanitize all API keys" 算模型漏项，但 json 里的 token 形态与另外两处不同，归 R2a/R7 边界，这里记 R7，因为两条独立采样做了同一个判断）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 2   | —                                                                                                                                             |
| **R3 终端**                            | filter-js-from-html ×2（24–28 步就交，1 个向量没过：`<meta content="…javascript:">` 变体；不是终端问题，是 XSS 向量覆盖——归 R2a）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 0   | 修正：filter-js ×2 → R2a                                                                                                                      |

计数：R2a 表里 kv-store-grpc 实为 NOTEST 不是 PARTIAL，所以 PARTIAL 42 = R2a 10（8 + filter-js 2）+ R2b 10 + R2c 3 + R5 9 + R1 8 + R7 2 = 42。

### 1.3 DECLARED_ALLFAIL 8

| env/s                           | 根因 | 一句话                                                                                                                                                                                                            |
| ------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model-extraction-relu-logits ×2 | R5   | 4 步交卷：`steal.py` 直接 `import forward; forward.A1`。题目说 "By making queries to forward()"，think 第 2 步写了 "task wants queries… However we did read forward.A1 directly" 仍交。验收把 A1 换掉了，所以全错 |
| polyglot-c-py s0                | R5   | C 和 Python 分别算 f(42) 得 433494437 vs 267914296：C 版 off-by-one（fib 下标），12 步 think 均值 40k，没跑过 `gcc && ./a.out 42` 对照                                                                            |
| dna-insert ×2                   | R1   | 引物不含插入片段：生物学约束理解不到位                                                                                                                                                                            |
| feal-differential s0            | R1   | 攻击恢复的 key 错                                                                                                                                                                                                 |
| regex-log s1                    | R2a  | 正则多匹配 3 个（`2021-12-01` 等），题目要求的边界条件没做负例测试                                                                                                                                                |
| reshard-c4-data s0              | R2a  | 输出目录 331 个文件，题目要求 ≤30：254 步 132 min 做完压缩，没看"文件数"这一条                                                                                                                                    |

### 1.4 NO_DECLARE 4 + NONDONE 8 + NOTEST 1

| env/s                                                                                  | 结局                         | 根因 | 怎么死的                                                                                                                                                                                  |
| -------------------------------------------------------------------------------------- | ---------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| compile-compcert s1                                                                    | NO_DECLARE（60 步 32 min）   | R3   | 第 58 步在交互 shell 里 `set -e`，下一条命令 `opam repo list` 非零退出 → bash 退出 → 之后 `Current Terminal Screen:` 空                                                                   |
| git-multibranch s0                                                                     | NO_DECLARE（67 步 40 min）   | R3   | 最后一步 `set -e` + for 循环里 `kill $!`                                                                                                                                                  |
| make-mips-interpreter s1                                                               | NO_DECLARE（20 步 11 min）   | R3   | `for …                                                                                                                                                                                    |  | exit 1; done` 里 gcc 失败 → `exit 1` 退出交互 shell |
| qemu-alpine-ssh s0                                                                     | NO_DECLARE（299 步 173 min） | R3   | expect 脚本里 `exit`，写成 heredoc 的 `PY` 结尾（复制粘贴错）                                                                                                                             |
| install-windows-3.11 ×2                                                                | HARNESS_FAILED               | R3   | `timeout 8 qemu-system-i386 … -vnc :1` 前台起 qemu，tmux send-keys 被 SIGTERM（return_code=-15）：pane 被 qemu 占住                                                                       |
| make-mips-interpreter s0                                                               | HARNESS_FAILED               | R3   | 同上，`qemu-system-mips -nographic` 前台                                                                                                                                                  |
| qemu-alpine-ssh s1                                                                     | HARNESS_FAILED               | R3   | 第 92 步 `tmux new-session -d -s alpine`，95 步把 `date; id\n` 发进 `terminus-2.0`（harness 自己的 pane）；下一步 `no server running on /tmp/tmux-0/default`                              |
| gpt2-codegolf s0、password-recovery s0、path-tracing-reverse s1、sqlite-db-truncate s0 | HARNESS_FAILED               | R7   | `litellm.MidStreamFallbackError: upstream stream request failed`，serving 侧断流；同时 `Error in proactively summarizing: hit max_tokens`——compaction 摘要本身撞 max_tokens（think 太长） |
| kv-store-grpc s1                                                                       | NOTEST                       | R2a  | 见 1.2：改名 `kv_store_pb2.py → KVStore_pb2.py`，验收找不到模块                                                                                                                           |

## 2. 七个根因各自长什么样（逐条可查的轨迹证据）

### 2.1 R1 硬题（29 TTL + 1 CAP512 + 8 PARTIAL + 3 ALLFAIL = 41，39%）

- 不是空转：uniq-cmd 高、不 sleep；是**在正确方向上精度不够**。`video-processing` 两条分别给 takeoff 27 / 48，期望 50–54；`train-fasttext` 0.618 vs 0.62；`mcmc s1` 2.926 vs 2.91。
- 特征是 **`/tmp` 研究、`/app` 不落地**：R1 TTL 里 17 条验收报 "File /app/xxx does not exist"。`dna-assembly ×2`、`protein-assembly ×2`、`write-compressor ×2`、`chess-best-move ×2`、`raman-fitting s0`、`extract-moves ×2`、`gpt2-codegolf s1`、`large-scale-text-editing ×2`、`path-tracing-reverse s0`、`polyglot-rust-c ×2`、`db-wal-recovery s0`——**全部 4 h 后一个字节都没交**。哪怕写一个 50% 正确的版本，`video-processing`、`extract-elf`、`write-compressor` 这类有阈值/相似度的题也可能拿分。
- 训练数据里没有"止损"：th3k 抽样 603 条里，`fall back / simplest / minimal / give up / workaround` 类措辞只出现在 2.5% 的步。teacher 的轨迹都是"一直往前做"。

### 2.2 R2 需求漏项（PARTIAL 10 + 10 + 3，ALLFAIL 2：regex-log / reshard，NOTEST 1：kv-store = 26，25%）

三个子型：
- **R2a 显式要求没打勾（13）**：`nginx` 的 `rate=10r/s` 是最干净的案例——题目原句 "allow only 10 requests per second per IP address with a burst capacity of 10 requests using a 10MB memory zone"，模型写了 zone、写了 burst、32 次重写从没写过 rate；think 里 11 次提到 `rate=` 都是在读 nginx 源码分析解析器。`constraints-scheduling` 第 0 步把三份日历全 cat 出来了，然后 39k 字符 think 心算，算错，没有用 grep 20240116 回头验证。
- **R2b 以"规范上合法"替代"在这个环境里跑通"（10）**：sparql ×2、torch ×4、sam ×2、query-optimize ×2。共同模式是环境缺一个工具（torch / rdflib 的某特性 / 慢查询），模型 **不装、不等、不跑，改成 py_compile + 读代码**。`torch-tensor-parallelism s1` 34 步里 6 次重写 5.8KB 的文件，每次 `python3 -m py_compile` 通过就满意；验收是 `NameError: 'rank'`，运行一次就能发现。
- **R2c 不复核自相矛盾的结果（3）**：`password-recovery s1` 找到 14 字符密码，题目说 23 字符，模型在 think 里把 `PASSWORD=` 9 个字符加进去凑数。

宣告时 think 长度：R2 的 26 条中位数 4.1k，PASS 的 73 条中位数 2.3k——和旧口径 §7.2 的结论一致（犹豫越久越可能错）。

### 2.3 R3 终端死（5 TTL + 4 NO_DECLARE + 4 HARNESS_FAILED = 13，12%）

三种具体死法，每种都有可复现的触发命令：

**(a) heredoc 里的反斜杠 / 引号配对错，shell 进入续行吞输入**
- `large-scale-text-editing s1` 第 189 步：`python3 - <<'PY' … run('a\n', ':s/a/\uA/\n:wq\n') … PY`，`\u` 在 Python 源里是 unicode escape 报 SyntaxError；第 192 步 `cat > x.vim <<'EOF' call setreg('a', "\C-u\C-S-a") …`，之后屏幕永远只有 `n'`（bash 在等一个 `'` 配对）。**第 192 → 417 步（225 步，54% 的轨迹）全部在恢复**：`C-c` ×108、`C-d`、`C-z`、`C-\`、`:q!`、`q`、`\u0003`、`\u001bc`、`reset`，没有一个有效。
- `large-scale-text-editing s0` 第 28 步 `grep -n "\u" …`，同款；之后 97 步无提示符。
- `headless-terminal s0` 第 3 步在 `python3 - <<'PY'` 里 `subprocess.Popen(['bash','-i'])` 并 `cp.stdout.read(500)` 阻塞；之后 **301/304 步没有提示符**，模型发了 183 次控制键，第 152 步甚至推断出 "C-e 是 ESC"（其实 harness 只认 tmux 键名），第 153 步发 `C-]C-e] [C-\`。
- `raman-fitting s0` 第 22–24 步连续三个 Python 脚本 SyntaxError / UnboundLocalError，第 24 步后 50 步无提示符。

**(b) 交互程序占住 harness 的 tmux pane**
- `install-windows-3.11 ×2`、`make-mips-interpreter s0`：前台 `timeout 8 qemu-system-i386 -vnc :1` / `qemu-system-mips -nographic`，harness `tmux send-keys` 等不到 pane 空闲，SIGTERM（return_code=-15）→ HARNESS_FAILED。
- `qemu-alpine-ssh s1`：模型正确地想到了"用 tmux 开独立会话跑 qemu"，但 (i) 用了错的 `capture-pane -X` 语法，(ii) 发现 `tmux ls` 只显示 `terminus-2` 就把命令发进 `terminus-2.0`——这是 harness 自己的 pane，(iii) 下一步 tmux server 没了。

**(c) `set -e` / 裸 `exit` 写进交互 shell**
- `compile-compcert s1`、`git-multibranch s0`、`make-mips-interpreter s1`、`qemu-alpine-ssh s0`：4 条 NO_DECLARE 全部是这个。th3k 训练集 603 条抽样里 `set -e` 在交互命令里出现 0 次，模型是从别处学来的脚本习惯。

训练数据对照：th3k 抽样 603 条 / 70k 条命令里 `C-*` 只有 273 条（0.39%），其中 `C-c` 260；"无提示符屏幕"的 streak 长度分布 1 步 1219 次、≥10 步只有 58 次；teacher 面对无提示符屏幕的动作是 `other_cmd` 3748 / `wait` 828 / `ctrl` 108。**即 teacher 几乎从未陷入过长时间卡死，所以数据里没有"卡死 → 系统化恢复"这个技能。** 模型学到的只是"发 C-c"，而 C-c 对 heredoc 续行无效（需要 `C-c` 后再发一个 `'` 或 `C-d`；对 bash -i 子进程需要 `C-d`；对 qemu 需要 `C-a x`）。

### 2.4 R4 外部依赖不可达（4 TTL 为主因；另 5 条为前半段诱因，计在 R1/R2）

- `build-pov-ray s0`：apt 里没有 povray，130 次 `apt-cache`、38 次 `curl`、72 次写 `/tmp`，0 次写 `/app`。题目要从源码编译 2.2 版，源在 github；沙箱外网不稳，模型 4 h 在镜像列表里绕。
- `protein-assembly ×2`：需要 fpbase / PDB API，`urllib` 全 404；s0 第 45–225 步在扫端口、找本地服务、猜 endpoint。
- `fix-ocaml-gc s0`：55 分钟 requested wait 全在等 opam。
- `adaptive-rejection-sampler s1`：R 包装 + 自写 R 测试脚本本身 >30 s 超时；验收也是 Rscript 超时。
- `nginx s1`：apt 源被 Debian 12 的 `.sources` 新格式绊住（第 43 步 `apt-get source` 失败后改 `/etc/apt/sources.list`），之后下载 nginx 源码、改 `ngx_http_limit_req_module.c`、重编译——**81 步后才 `nginx -t` 通过，然后忘了 `rate=`**。这是 R4 把 R2 挤出来的典型：工程量把注意力耗尽。
- `mcmc-sampling-stan ×2`：装 rstan 2.32.7 用了 60–90 分钟（69/126 和 39/73 步是等待）。

### 2.5 R5 作弊捷径 / 题意隐含禁止（9 PARTIAL + 3 ALLFAIL = 12，11%）
 
- `path-tracing ×2`：两条独立采样走了同一条路——`image.c` 里 `system("./orig")`。两条的 think 都写了 "algorithmic rather than embedding data… might be questionable… but it doesn't read image.ppm" 然后交。**模型看到了禁止、给自己找了理由、还是交了。** s1 前 40 步其实在正经逆向 `orig` 的 objdump（已经拿到 sphere/plane/trace 的反汇编和 `.rodata` 常量），第 41 步放弃转捷径。
- `model-extraction ×2`：4 步交卷，`forward.A1` 直接读。think：*"task wants queries… However we did read forward.A1 directly… Probably okay?"*。
- `polyglot-c-py s0`（旧口径是"两个文件"，本轮是 C 版 fib off-by-one）、`mteb-leaderboard ×2`、`mteb-retrieve ×2`、`extract-elf s1`、`bn-fit-modify s1`：题意有一个"正确口径"（all tasks 过滤 / 标准编码 / 所有变量 / U 无父）模型取了字面解。

### 2.6 R6 debug 自己的探针（3 独立 + 大量叠加在 R1 上）

- `video-processing s0` 990 步里 380 步（38%）观察含 Traceback/SyntaxError。
- `chess-best-move ×2` 163/166 步（37%/25%）。
- `llm-inference-batching ×2` 183/141 步（34%/32%）。
- `dna-assembly s0` 104/335（31%）、`gpt2-codegolf s1` 105/398、`protein-assembly s1` 103/503。
- 模式：(i) 写一个 python heredoc 探针 → 输出太长被终端截断 → 再写一个"只打印一部分"的探针 → 这个有 bug → 修 bug → 又太长。`path-tracing s0/s1` 前 20 步各有 6–8 次 "output was truncated, print smaller chunks"。(ii) f-string 里嵌反斜杠 / 引号（`parse_rodata.py` 连续 3 次 SyntaxError）。th3k teacher 的做法是 `cmd > /tmp/x.txt; sed -n '1,80p' /tmp/x.txt`，th3k 数据里有，但模型在压力下退化成"再写一个脚本"。

### 2.7 R7 基础设施（4 HARNESS_FAILED + 2 sanitize = 6，6%）

- 4 条 `MidStreamFallbackError`（serving 断流）。伴随 `Error in proactively summarizing: hit max_tokens`：**compaction 摘要本身被 think 撑爆**。这是 think 过长的第二个代价（第一个是墙钟）。
- sanitize-git-repo ×2：第三个被污染文件是 `exp_data/.../*.json`，两条采样都 grep 到了那个文件却判断为非真实 token。算 R2a 边界，这里保守记 R7。
- nginx 的 `limit_req_zone` 解析器在镜像里被故意改坏（模型最后通过改源码重编译绕过了，然后漏 rate），题目设计如此，不计 verifier 问题。

## 3. 与旧口径（[trajectory_failures](trajectory_failures.md) §1）的差异

| 项                          | 旧口径 110 失败 | mirror 105 失败 | 变化                                                                                                   |
| --------------------------- | --------------- | --------------- | ------------------------------------------------------------------------------------------------------ |
| TTL                         | 42              | 41              | 持平（都是那 24 个 env）                                                                               |
| PARTIAL                     | 39              | 42              | 持平                                                                                                   |
| NOTEST（verifier 装包失败） | 13              | 1               | **-12，verifier 修好了**；这 12 题里 6 题转成 PASS（0.348→0.410 的主要来源），其余转成 PARTIAL/ALLFAIL |
| HARNESS_FAILED              | 未统计          | 8               | 新看到：4 条 qemu/tmux 占 pane、4 条 serving 断流                                                      |
| NO_DECLARE                  | 4               | 4               | 旧口径说"忽略"，本轮确认 4 条全是 `set -e`/`exit` 杀 shell，可治                                       |
| "只差 1 个测试"             | 19/39           | 20/42           | 持平                                                                                                   |

新增的洞察只有一个但很重要：**旧口径把 TTL 归成"想得多不收敛"，本轮拆开后 TTL 里有 5 条（12%）+ NO_DECLARE 4 + HARNESS 4 = 13 条是终端被自己弄死**，这部分和"想多想少"无关，是可以用几百条数据直接教会的。

## 4. 配套数据方案（按性价比排序）

统一前提：所有新增样本保持 th3k 风格（整轨迹、历史 think 可见、以 `task_complete` 收尾、每步 2–3 条命令、think 2–5k），否则会稀释 th3k 的"流程一致性"优势（[trajectory_failures](trajectory_failures.md) §4 结论 5）。

### D1. 交卷前需求清单对照（打 R2，26 条，最大可挽回块）

**要加的行为**：在第一次 `task_complete: true` 之前的那一轮，`analysis` 里逐条列出题目的显式要求 + 每条对应刚看到的命令输出；有一条没证据就先去补命令。具体到本轮失败：
- "listen 8080 / 10 req/s / burst 10 / 10MB zone / log format / 404 page" → 一条一条 `grep` 配置文件。
- "earliest slot, no conflict with A/B/C calendars" → `grep 20240116 *.ics`。
- "{class name}_pb2.py" → `ls /app/*_pb2*.py` 对照题面文件名。
- "≤30 files in output dir" → `ls | wc -l`。

**数据来源**（三选一或并用）：
1. **从 dsv41 chat 整轨迹（2037 条）里筛**：最后 3 步内 `analysis` 含 ≥3 个 "requirement/verified/confirmed" 式枚举，且这 3 步里有 ≥2 条非 `cat`/`ls` 的核验命令（`grep -c`、`diff`、`curl -I`、`python -c 'import …'`、运行一次产物）。预计 400–600 条。这是 taxonomy §8 E3 的具体化。
2. **规则合成尾轮**：对 th3k 2441 条里 solved 的 660 条，用题面抽要求列表（编号/破折号/"must"/"should"），把最后一轮 `analysis` 前缀替换为 `Requirement check: (1) … — evidence: <上一步输出摘要>; (2) …`。只改最后一轮监督，不改命令。作为 E4 对照。
3. **自采集**：用 c312 自己在 mirror 口径上 PASS 的 73 条，在 `task_complete` 前插入一轮"清单 + 核验命令"（用 dsv41-max 做这一轮的 teacher），形成 on-policy 的收尾样本。**这条最贴近 KTO/DPO 的用法**：PARTIAL 的 42 条作为负例的最后一轮，PASS 的 73 条加清单作为正例。

**另加一个子集专打 R2b**：环境缺工具时 **先装/先跑再交**。样本：轨迹里出现 `ModuleNotFoundError: torch` / `rdflib 不支持` 后，下一步是 `pip install torch --index-url …` 或 `pip install rdflib==7.x` 并实际 `python /app/xxx.py` 跑一遍。dsv41 chat 里这类片段估计 150–300 条；th3k 里 `pip install` 命令 4.1 条/轨迹，本来就有，但要筛"装完后立刻跑产物"的。

### D2. 终端卡死恢复 + 交互程序隔离（打 R3，14 条，最便宜）

**要加的行为**（三个固定模板，每个 100–200 条就够，因为是机械技能）：
1. **heredoc / 引号续行恢复**：屏幕只有 `>` 或奇怪残片且无提示符 → 发 `C-c` 然后**紧接着独立一条 `\n`**（harness 警告过 108 次 "C-c should end with newline"），仍无效 → `C-d`，再无效 → 发一个 `'` 或 `"` 配对 + `\n`。配套：**写 heredoc 时反斜杠一律用 `<<'EOF'` 且避免 `\u`、`\x`，或先 `cat > /tmp/x.py` 再 `python3 /tmp/x.py`**。
2. **交互程序隔离**：任何 qemu / `bash -i` / `vim`（非 `-Es`）/ 服务进程 → `tmux new-window -d -n qemu 'cmd …'` 或 `nohup cmd > /tmp/x.log 2>&1 &`，然后用 `tmux capture-pane -p -t qemu` / `tail /tmp/x.log` 观察。**绝不**向 `terminus-2` 发键。th3k 数据里 `nohup/setsid/&` 有 396 条命令，但没有"起 qemu 前台"的负例对照。
3. **禁 `set -e` / 裸 `exit`**：交互 shell 里用 `cmd || echo FAIL` 代替；脚本逻辑写进文件再 `bash /tmp/x.sh`。th3k 抽样 0 次 `set -e`，模型从预训练带来的，需要负例压制。

**数据来源**：这三类在 teacher 轨迹里很少（teacher 不常把终端弄死），**需要合成**：从 th3k 任意轨迹中间截一段，注入一条"错误命令"（heredoc 含 `\u`；前台 qemu；`set -e`）+ 对应的真实 shell 反馈（可以在沙箱里实录），再接 teacher 风格的恢复轮 + 回到原轨迹。每类 150 条，总 ~450 条，占 th3k 的 18%。

**harness 侧对照（零训练）**：在 system prompt 里加一句 "If the screen shows no prompt for 3 consecutive turns, send `C-c` then `C-d` then an empty line as three separate commands"——先测这句能不能把 headless-terminal / large-scale-text-editing 的 300 步恢复缩到 5 步，再决定 D2 的规模。

### D3. 长输出走文件 + 探针一次写对（打 R6，叠加在 R1 上 ~15 条的 30% 步数）

**要加的行为**：
- 任何预期 >40 行的输出 → `cmd > /tmp/out.txt 2>&1; wc -l /tmp/out.txt; sed -n '1,60p' /tmp/out.txt`。th3k teacher 本来就这么做（`echo '--- label ---'; cat` 占 12% 命令），但模型在 python heredoc 探针上退化。
- python 探针模板：**不用 f-string 嵌引号**、先 `python3 -c 'import ast; ast.parse(open("/tmp/x.py").read())'`。
- 同一个探针报错 2 次 → 改用 `grep`/`awk`/`od` 等不需要写代码的命令。

**数据来源**：从 th3k 里筛"连续 ≥2 步 Traceback 后，第 3 步换了工具或改用文件重定向"的片段（估计 th3k 里有 200–400 条），上采样 ×2。不需要合成。

### D4. 外部不可达时的降级路径（打 R4，9 条）

**要加的行为**：apt/pip/curl 连续失败 2 次 → 判断"这个依赖是否是验收必需"：
- 必需（povray 二进制、rstan）→ 换源一次（`--index-url` 镜像 / `apt-get -o Acquire::http::Proxy`）→ 仍失败就**用本地已有工具写最简替代**（povray 任务里验收 `test_povray_version` 只看 `/usr/local/bin/povray --version` 输出，可以先放一个 stub 拿 1/3 分）。
- 非必需（fpbase API 查荧光蛋白光谱）→ 用 think 里的领域知识直接给值并在 analysis 里注明。

**数据来源**：dsv41 chat 里 `pip install … ERROR` / `Could not resolve host` 后 5 步内成功继续的片段，估计 200 条。注意要排除 [trajectory_failures](trajectory_failures.md) §2(a) 那种 "改 sources.list 把环境搞坏" 的负面样本（envfix 命令 0.06/轨迹的那批）。

### D5. 题意隐含禁止的负例（打 R5，12 条）

**要加的行为**：出现 "algorithmic / by making queries / single file / must use X package / do not read Y" 这类措辞时，think 里把它翻译成一条硬约束并在 analysis 里重复；选路线时排除违反它的捷径。
- 本轮两个最干净的样本对：`path-tracing`（s1 第 1–40 步正经逆向 vs 第 41 步 `system("./orig")`）、`model-extraction`（4 步读 A1 vs 应该做 ReLU 边界查询）。
- 这类适合 **DPO**：同 env 两条采样一条守约一条走捷径的对子，c312 + arm A 的 mirror 口径里目前能凑约 15 对；加上 th3k 重放里 "solved 但用了捷径被判 fail" 的样本（[trajectory_collection](../methods/trajectory_collection.md) 的 fail 集里筛）。

### D6. 止损：N 分钟 / N 步没有 `/app` 产物就先写最简版（打 R1 的墙钟，不直接涨分）

**要加的行为**：每 ~40 步或每次 compaction 后，`ls -la /app` 对照题目要的产物文件；缺的先写一个能跑的最简版（哪怕输出是常数），再继续优化。
- 直接收益：R1 的 17 条 NO-DELIVERABLE 里，`video-processing`（阈值 ±2 帧）、`extract-elf`（75% 覆盖）、`write-compressor`（≤2500 字节）、`winning-avg-corewars`（75% 胜率）这类**阈值型验收**，最简版有非零概率；`chess-best-move`、`regex-chess` 这类精确匹配型没有。保守估计 +1–2 题。
- 间接收益：TTL 从 240 min → 可以配合 harness 在 60 min 时注入提示（taxonomy §8 E2 变体），评测 3× 快，RL rollout 预算 3×。

**数据来源**：th3k 里"中途写了产物 v1 然后迭代到 v3"的轨迹本来就有（solved 的 660 条里 `/app` 写入 ≥3 次的），筛出来上采样；再合成 100 条 "step 40 ls /app 发现缺产物 → 写 stub → 继续" 的片段。

### 优先级与预期

| 方案        | 打击面   | 新增样本                  | 来源难度          | 预期 pass@all 增量（178 口径，±0.07 噪声）               |
| ----------- | -------- | ------------------------- | ----------------- | -------------------------------------------------------- |
| D1 需求清单 | R2 26 条 | 400–600 筛 + 660 合成尾轮 | 低（筛选 + 规则） | +0.03–0.05（26 条转一半）                                |
| D2 终端恢复 | R3 14 条 | ~450 合成                 | 中（要沙箱实录）  | +0.01–0.02 直接；更大的是 HARNESS_FAILED 8→4 和 TTL 墙钟 |
| D3 探针纪律 | R6 叠加  | 200–400 筛                | 低                | 间接（让 R1 的 4 h 里多 30% 有效步）                     |
| D4 外部降级 | R4 9 条  | ~200 筛                   | 低                | +0.01                                                    |
| D5 隐含禁止 | R5 12 条 | 15 对 DPO + 筛            | 高（对子少）      | +0.01–0.02，更适合 RL 阶段                               |
| D6 止损     | R1 墙钟  | 筛 + 100 合成             | 低                | +0.005–0.01 直接；算力 -60%                              |

D1 + D2 + D3 合计 ~1500 条，占 th3k 2441 的 60%，一个 8 epoch run 可以验证。**先做 D1 和 D2**：D1 是分数，D2 是把 13% 的失败从"不可解释"变成"不再发生"。D2 的 harness 侧对照（system prompt 一句话）今天就能发 job。

## 5. 轨迹索引（可直接打开）

```
/storage/openpsi/users/chen/output/cth-tb21m-mask-const-c312-2611051616/tasks/
R2a  tb2-nginx-request-logging/s1        112 步，32 次写 limit_req_zone 全无 rate=，think 提过 11 次
R2a  tb2-constraints-scheduling/s0       4 步，cat 过日历后心算错，不回查
R2a  tb2-kv-store-grpc/s1                20 步，第 15 步把 kv_store_pb2.py 改名成 KVStore_pb2.py
R2b  tb2-torch-tensor-parallelism/s1     34 步，6 次重写 + py_compile，验收 NameError: rank
R2b  tb2-sparql-university/s1            24 步，rdflib 报 HAVING 不支持，"standard SPARQL should be valid" 交卷
R2b  tb2-sam-cell-seg/s0                 51 步，从未运行 convert_masks.py
R3a  tb2-large-scale-text-editing/s1     第 192 步 heredoc 含 \C-u 后 225 步只见 n'
R3a  tb2-headless-terminal/s0            第 3 步 Popen(bash -i) 阻塞，301/304 步无提示符
R3b  tb2-qemu-alpine-ssh/s1              第 92–95 步 tmux new-session 后把键发进 terminus-2.0
R3b  tb2-install-windows-3.11/s0         第 23 步前台 qemu -vnc，harness send-keys -15
R3c  tb2-compile-compcert/s1             第 58 步 set -e，第 59 步屏幕为空
R3c  tb2-make-mips-interpreter/s1        第 19 步 for … || exit 1
R4   tb2-build-pov-ray/s0                130 次 apt-cache，0 次写 /app
R5   tb2-path-tracing/s1                 第 1–40 步逆向 orig，第 41 步 system("./orig")
R5   tb2-model-extraction-relu-logits/s0 4 步，forward.A1 直读，think 自己质疑后仍交
R6   tb2-video-processing/s0             990 步，380 步 Traceback，takeoff 27 vs 50–54
R1   tb2-llm-inference-batching-scheduler/s0  538 步，cost 超阈值 8×，183 步 selferr
R7   tb2-gpt2-codegolf/s0                MidStreamFallbackError + summarizing hit max_tokens
```
