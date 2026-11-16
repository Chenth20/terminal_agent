# Harness：Terminus 2 内部机制与版本演进，以及官方 DeepSeek Harness 对照

> 本文记录 Arena 版本演进，原平台目录是当时操作位置。仓库中可查的固定配置见[版本配置](../arena_assets/harness/configs/)及[选型记录](arena_verifier.md)。

两部分。**Part A** 讲我们评测和采轨迹用的 `cth-terminus-2-stream`：一次 attempt 的 workflow、episode 循环、压缩机制、历史推理可见性由 chat template 决定、当前默认版本 1.6.96 每个字段的来源。**Part B** 讲把官方 DeepSeek Harness（DSH sdk-minimal）接进平台的全过程、它与 Terminus 2 的逐项差异、三个模型上的对照结果，以及用抓包 + 模板 + 计数证实的"Qwen 到底看不看得到历史推理"。

- 该用哪个版本：[arena_verifier](arena_verifier.md) §2。
- 压缩触发点 / 历史推理可见性对分数的影响：[context_sweeps](../experiments/context_sweeps.md)。

---

# Part A. Terminus 2（`cth-terminus-2-stream`）

## A0. 基线版本：`cth-terminus-2-stream@1.0.0`

就是上游 Harbor `v0.20.0-58-g0348989a` 的 Terminus-2（agent 2.0.0、parser=json、litellm 1.93.0），只改了一处：LLM 请求 `stream=True` + `stream_chunk_builder` 拼回，避免沙箱出口代理在 ~900s 静默后 502。其余文件与 `terminus-2-tb21@1.0.12` 逐字节相同。

源码（本次结论全部基于实际阅读 + 轨迹 token 账目验证）：

- `envarena_terminus2/{supervisor,runner,settings,environment}.py`
- `harbor/agents/terminus_2/terminus_2.py`、`tmux_session.py`、`templates/terminus-json-plain.txt`
- `harbor/llms/{chat,lite_llm}.py`

路径：`/storage/openpsi/users/chen/AReaL/opencode/terminus_stream/orig/app_src/app/`（`pkg/app.tar.gz` 与之一致，`pkg_v2` 是未用于本批 job 的 1.2.x）。

---

## A1. 一次 attempt 的 workflow

```
arena 调度 → 任务镜像沙箱容器（4 CPU / 8Gi / 50Gi）
 └ supervisor.py   子进程组；deadline = ARENA_AGENT_TIMEOUT_SEC(1800) - 60 = 1740s
    └ runner.py    anyio.fail_after(1770s)，实际总是 supervisor 先 killpg(SIGTERM) → rc=-15
       └ Terminus2.setup()   tmux kill-session；新建 session "terminus-2"（继承镜像 PATH/PYTHON/LD_LIBRARY_PATH）
       └ Terminus2.run()     拼初始 prompt → episode 循环
 └ agent 退出/被杀 → 导出 trajectory.jsonl（含 partial）+ harness_result.json
 └ Reward 阶段：容器内跑任务自带 tests；tmux session 保留，服务类任务端口仍可达 → 0/1
```

关键事实：

- **无 system message**。Terminus 规范 + 任务描述全在第一条 user 消息里。
- **无 max_turns**（`_max_episodes = 1_000_000`），唯一硬约束是 30 分钟墙钟。
- `harness_result.json` 恒为 `status=OK`（超时也是），**分数只由 reward 决定**。只有 runner 抛异常才是 `ERROR` → 这是 `HARNESS_FAILED` 的唯一来源（网关 503 六次重试耗尽、tmux 失败等）。

## A2. episode 循环

1. `is_session_alive()`：模型发 `C-d` 退掉 shell → session 消失 → 直接 return（`no server running on /tmp/tmux-0/default` 就是这类）；接着 reward 跑在已无 shell 的容器里 → 失败。
2. **主动压缩检查**：`_count_total_tokens` 估算历史，若 `262144 - 估算 < 65536` → `_summarize`。
3. `chat.chat(prompt)`，一次 chat completion：T=1.0 / top_p .95 / top_k 50 / `separate_reasoning=true` / `reasoning_effort=max` / 6 次网关退避重试（5–30s 随机指数，litellm 内部重试关掉）。
4. JSON parser 解析 `analysis / plan / commands[] / task_complete`。
   - 解析失败 → 下一轮 prompt 是 `Previous response had parsing errors...`，**不执行命令**，也不消耗 episode 上限。
   - `finish_reason=length` → 截断内容塞进历史，下一轮是 `ERROR!! NONE of the actions ... exceeded max output`；XML parser 还能 salvage，json parser 不能。
5. 逐条 `tmux send-keys`，每条等 `min(duration, 60)` 秒，然后取**增量**屏幕输出，超 10000 字节则保留首尾各 5000（中间省略）。
6. 下一轮 prompt = 终端输出（有 warning 时前置 `WARNINGS:`）。
7. **完成需两次确认**：第一次 `task_complete:true` → 回 "Are you sure…include task_complete again"；第二次才结束。

模型每轮看到的 user 消息只有三类：初始任务、终端增量输出、错误/确认提示。没有别的东西。

## A3. 历史 reasoning_content 是否进入后续轮次 —— 由 chat template 决定

链路三段：

| 段            | 行为                                                                                                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| harness       | `interleaved_thinking=True`，assistant 消息写成 `{"role":"assistant","content":<JSON>,"reasoning_content":<思考>}` 追加进 `chat.messages`，下一轮整个 list 原样发送。**不裁剪** |
| sglang        | `_apply_jinja_template` 用 `msg.model_dump()`，`reasoning_content` 原样进模板                                                                                                   |
| chat template | **分叉点**                                                                                                                                                                      |

模板差异只在一行（第 100 行）：

| 模型                                          | 第 100 行                                     | 效果                                                                                                 |
| --------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Qwen3.5-4B / 9B 官方                          | `{%- if loop.index0 > ns.last_query_index %}` | 只有最后一条 user 之后的 assistant 才渲染 `<think>`；Terminus 每轮都是新 user → **历史推理全部丢弃** |
| 我们 SFT 的 serve 目录（取自训练 checkpoint） | `{%- if true %}`                              | **每轮历史 assistant 都带完整 `<think>推理</think>`**                                                |

真实轨迹验证（相邻两轮 `prompt_tokens` 增量 vs. 上一轮 reasoning/content/终端输出 token，用 Qwen tokenizer 精确计数）：

| job            | 配对数 | Σ prompt 增量 | Σ(答案+输出) | Σ(推理+答案+输出) | 结论 |
| -------------- | ------ | ------------- | ------------ | ----------------- | ---- |
| qwen35-4b base | 173    | 329,724       | **328,673**  | 461,829           | 丢弃 |
| qwen35-9b base | 40     | 53,408        | **53,165**   | 97,762            | 丢弃 |
| abl-base-final | 2,013  | 8,214,251     | 2,969,088    | **8,193,230**     | 保留 |
| abl-keep-final | 3,101  | 12,673,312    | 4,581,527    | **12,589,037**    | 保留 |

**结论：SFT 消融模型每轮都能看到自己此前全部思考；base 4B/9B 一份都看不到。** 这与训练渲染一致（SFT 数据带历史 think），不是 bug，但它是 base 与 SFT 之间的**系统性差异**，横向比分数必须记住。

### 连锁影响

1. **上下文增速差 ~2.7×**。SFT 每轮增量 ≈ 推理 + 答案 + 输出，base 只有后两项。max_prompt p50：abl-base-final 177k、abl-keep-final 205k，vs 4B base 84k、9B base 33k。SFT 的 KV 压力、单机真实并发因此明显更紧。
2. **压缩频率**：abl-base-final 176 条轨迹 308 次压缩，keep-final 418 次；4B base 266 条 107 次。
3. **1.0.0 的 token 估算对 base 是错的**。`_count_total_tokens` 用 `litellm.token_counter(messages=...)`，已实测它**会把 `reasoning_content` 计入**（5000 词推理 → +5001 tokens）。对 SFT 准；对 base，服务端根本没收到这些 token，harness 却按"收到"算 → **base 在真实 prompt 可能 ~100k 时就被判定逼近 196k 而提前压缩**，等效上下文被缩水。`pkg_v2`（1.2.x）的 `TERMINUS_TOKEN_COUNT_MODE=usage`（用服务端 `prompt_tokens+completion_tokens` 做锚点）就是修这个，**当前这批 job 用的是 1.0.0，没有修正**。
4. 因此 SFT-vs-base 的分数差不只是"训没训过"，还混着"历史推理可见性 / 何时压缩"两种待遇差异。
5. **（10-25 补）1.2.x（usage 计数）下 base 的触发点扫描**：9B 1.2.256 = 0.239，与 1.0.0 的 0.247 持平——说明 1.0.0 的"提前压缩"对 base **不是损失**；base 在 96k–128k 触发时反而最高（4B 0.188、9B 0.264），所以 1.0.0 的真实触发点（~100k）恰好落在对 base 有利的区间。SFT-vs-base 的差距不能用"base 被提前压缩"来解释。数据见 [context_sweeps](../experiments/context_sweeps.md) §B8。

## A4. 压缩机制（`_summarize`）

触发：主动（剩余 < 65536）或被动（服务端 context length exceeded → 先 `_unwind` 丢最近几轮直到空出 4000 token）。

1. **Summary 子代理**：history = 当前全部对话（SFT 含历史推理），要求总结已完成动作 / 学到的信息 / 难题 / 当前状态。
2. **Questions 子代理**：**空历史**，输入 = 原任务 + 总结 + 当前终端画面，提 ≥5 个问题。
3. **Answers 子代理**：history = 全部对话 + summary 问答，逐条回答。
4. 主对话重置为 `[初始任务消息, question_prompt(总结+画面), 模型提的问题]`，下一轮 prompt = "Here are the answers… Continue working… You can no longer ask questions."

回退链：短总结（原任务 + 屏幕末 1000 字符）→ 无 LLM 的纯屏幕回退。

**压缩后，压缩前的推理对 SFT 模型也全部消失**，只剩总结文字。子代理 token 计入 usage，不进主 trajectory。

看轨迹时 `step.extra.context_management = {type: compaction, boundary: replace}` 且 `observation.results[0].subagent_trajectory_ref` 存在的就是压缩边界；跨过它比较 `prompt_tokens` 增量会得到负数/异常值，做 token 账目时必须跳过。

## A5. 其他影响结果的硬参数

| 项           | 值                                                             |
| ------------ | -------------------------------------------------------------- |
| 墙钟         | 1800s（supervisor 1740s 杀进程组 → `rc=-15`，与 TIMEOUT 等价） |
| 单条命令等待 | `min(duration, 60)` s                                          |
| 终端输出     | 10000 字节，保首尾                                             |
| 网关重试     | 6 次，5–30s 随机指数；`num_retries=0`                          |
| 上下文注册   | 262144；主动压缩阈值 65536 剩余                                |
| 解析失败     | 不执行命令，重发，不占 episode                                 |
| 完成         | `task_complete` 需连续两次                                     |

## A6. 如何复核（可复现步骤）

- 渲染对比：取 `chat_template.jinja`，构造 `[user, assistant(content+reasoning_content), user, …]`，分别用 base 与 SFT 模板 `render(add_generation_prompt=True)`，直接看历史 assistant 是否带 `<think>`。
- token 账目：对 `output/<job>/tasks/*/s*/trajectory.jsonl` 中 `step_id` 连续、且后一步不带 `context_management` 的相邻 pair，比较 `prompt_tokens[i+1] - prompt_tokens[i]` 与 `len(tokenize(reasoning_i)) + completion_i + len(tokenize(observation_i))`；比值贴近后者说明历史推理保留。
- tokenizer 用 `/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B/tokenizer.json`（`tokenizers` 直接加载，无需 transformers）。


---

## A7. 当前 harness：`cth-terminus-2-stream@1.6.96`（10-31 起默认）完整记录

§A1–§A6 描述的 1.0.0 语义仍是基底；1.6.96 = 1.0.0 + 四层增量（`pkg`→`pkg_v2`→`pkg_v3`→`pkg_v4`→`pkg_v5`），每层只改 1–3 个文件。本节给出**现在这个版本每一个字段的来源和作用**，不必再翻中间版本。版本总表见 [arena_verifier](arena_verifier.md) §2；每版 spec 在 `coding_arena/work/harness-specs/terminus-stream-*.json`（五个关键版本的副本在 `arena_assets/legacy/harness/specs/`，对应的 `agent.env` 已抽成 `arena_assets/harness/configs/<ver>.env`）。

### A7.1 spec（平台上已发布、不可变）

```
runtime.type = runner
runtime.uri  = uploads/261030/b5776b45981e0f36/pkg/      # pkg_v5
resource_spec = 4 CPU / 16Gi / 50Gi                      # 1.0.0–1.4.96 是 8Gi
ttl_seconds   = 14400                                    # 平台 agent 上限 4 h（TB2.1）；env 可另设 agent_seconds（TH 是 18000）
```

`agent.env`（按作用分组）：

| 组 | 变量 | 值 | 引入版本 / 作用 |
|---|---|---|---|
| 采样 | `TERMINUS_REASONING_EFFORT` | `max` | 1.0.0。**覆盖 Model.parameters**，所以档位要在 harness 改（`1.2.96-xhigh` 就是只改这一项） |
| | `TERMINUS_TEMPERATURE` / `TOP_P` / `TOP_K` | 1.0 / 0.95 / 50 | 1.0.0 |
| | `TERMINUS_SEPARATE_REASONING` | true | 1.0.0，让上游把推理放 `reasoning_content` |
| 传输 | `TERMINUS_STREAM` | true | 1.0.0 的唯一补丁：流式 + `stream_chunk_builder`，绕过沙箱 egress ~900 s 非流式 502（§A5） |
| | `TERMINUS_GATEWAY_RETRY_ATTEMPTS` | 6 | 1.0.0，5–30 s 随机指数；litellm 内部 retry 关 |
| 上下文 | `TERMINUS_MODEL_CONTEXT_LIMIT` | 98304 | 1.2.96：**压缩触发点**，不是模型真实窗口。触发条件 `limit − 估算 < threshold` |
| | `TERMINUS_PROACTIVE_SUMMARIZATION_THRESHOLD` | 1024 | 1.2.96 → 实际触发 ≈ 97k |
| | `TERMINUS_TOKEN_COUNT_MODE` | usage | 1.2.x：以上游最近一次 `prompt_tokens+completion_tokens` 为锚只估增量，修 1.0.0 对 DeepSeek 的 2.7× 过计（§A3 第 3 条） |
| | `TERMINUS_MAX_TURNS` | 512 | 1.1.0：到顶正常返回、照常出 trajectory + reward |
| 任务 shell 注入（**1.5.96 新机制**） | `TERMINUS_SANDBOX_ENV_PIP_INDEX_URL` / `UV_DEFAULT_INDEX` / `UV_INDEX_URL` | `https://mirrors.aliyun.com/pypi/simple/` | 1.5.96：沙箱到 pypi.org / files.pythonhosted.org 常 15 s ReadTimeout，aliyun 0.3 s |
| | `…PIP_TIMEOUT` / `PIP_RETRIES` | 60 / 10 | 1.5.96 |
| | `…UV_HTTP_TIMEOUT` / `UV_CONCURRENT_DOWNLOADS` / `UV_HTTP_RETRIES` | 600 / 4 / 6 | 1.5.96 是 120；1.6.96 提到 600——300+ MB 的 nvidia wheel 120 s 拉不完 |
| | `…HF_HUB_DISABLE_XET` / `HF_HUB_DOWNLOAD_TIMEOUT` / `HF_HUB_ETAG_TIMEOUT` | 1 / 120 / 60 | 1.6.96：HF xet 的 `cas-server.xethub.hf.co` 对沙箱返 403，关掉走普通 HTTP |

没有的东西：`TERMINUS_INTERLEAVED_THINKING`（默认 true，回传历史推理；nothink/mask 模板要用 1.0.1/1.3.96）、`TERMINUS_TOOL_CALL_MODE`（默认 none；DeepSeek 要看到历史推理用 1.4.96）、`TERMINUS_FORCE_JSON_OBJECT`（默认 false；JSON mode 消融无收益已删）。pkg_v5 包含这三个开关的代码，需要时在 1.6.96 的 spec 上加一个 env 发新版本即可。

### A7.2 runtime `pkg_v5` 的文件

```
pkg/
├── runner                        bash 引导：解 python313 / site-packages / app 三个 tar，探测 tmux -e 支持否则用 fallback-bin，exec supervisor
├── python/python313.tar.gz       CPython 3.13.6（glibc 2.17）
├── site-packages/py313.tar.gz    harbor 0.20.0 依赖 + litellm 1.93.0
├── fallback-bin/tmux{,.bin}, lib/ tmux 3.4（glibc 2.28）
├── app/app.tar.gz                ← 唯一随版本变的文件
└── runtime-manifest.json         sha256 + patch 说明（pkg_v2..v4 的 site-packages 哈希记错了，v5 改为实际值；文件本身四版相同）
```

`app.tar.gz` 相对上游 Harbor `v0.20.0-58-g0348989a` 的全部改动（累计）：

| 文件 | 改动 | 版本 |
|---|---|---|
| `harbor/llms/lite_llm.py` | `_acompletion` 走 `stream=True` + `stream_chunk_builder` 拼回；tool-call 模式下 `tool_calls[0].function.arguments` 当 content | pkg / pkg_v4 |
| `harbor/llms/chat.py` | 每次 chat 后记 usage 锚点 `(prompt+completion, len(messages))`；`interleaved_thinking=false` 时锚点改为 `(prompt, len+1)` 不计被丢弃的推理 | pkg_v2 / pkg_v4 |
| `harbor/agents/terminus_2/terminus_2.py` | `_count_total_tokens` 在 `usage` 模式下用锚点 + 增量估算；`_main_llm_call_kwargs` 在 function 模式加强制 `terminus_action` tool | pkg_v2 / pkg_v4 |
| `envarena_terminus2/settings.py` | 读 `TERMINUS_TOKEN_COUNT_MODE` / `INTERLEAVED_THINKING` / `TOOL_CALL_MODE` / `FORCE_JSON_OBJECT` / `MAX_TURNS` 等 env | pkg_v2–v4 |
| `envarena_terminus2/runner.py` | 把上述 settings 传给 `Terminus2(...)`；**pkg_v5：`sandbox_extra_env()` 把所有 `TERMINUS_SANDBOX_ENV_<NAME>` 并进 `extra_env`**，经 `TmuxSession` 的 `tmux new-session -e NAME=value` 进入任务 shell；`harness_result.raw.sandbox_env_keys` 记录注入了哪些名字 | pkg_v5 |

`envarena_terminus2/{supervisor,environment,trajectory}.py` 与 1.0.0 相同。源码：`coding_arena/build/terminus_pkg_v5/src/app/`；1.0.0 原版：`/storage/openpsi/users/chen/AReaL/opencode/terminus_stream/orig/app_src/app/`。

### A7.3 注入机制的边界（为什么它不会污染评分）

- 只写进 **agent 的 tmux 会话环境**（`tmux new-session -e`），与 1.0.0 已有的 `PATH/LD_LIBRARY_PATH/PYTHONHOME/PYTHONPATH` 恢复走同一条路。Reward 阶段是平台另起的进程，`env` 里没有这些变量（用 `cth-tb2-reward-dump` 实测）。
- 命令行显式 `--index-url` 优先于 `PIP_INDEX_URL`（`tb2-pypi-server` 要装 `localhost:8080` 的包，实测不受影响）。
- `UV_NO_INDEX=1` 的场景（reward 的 pytorch-model-cli 离线分支）与 agent 无关。
- 模型在轨迹里能看到 `env | grep PIP_`，个别轨迹确实 `pip config list` 看过一眼，没有观察到因此改变行为。

### A7.4 本地验证方法

```bash
PATH=<pkg>/fallback-bin:$PATH PYTHONPATH=<src>/app:<sp>/py313 PYTHONHOME=<rt>/python313 \
  python3 -c 'from envarena_terminus2.runner import sandbox_extra_env; ...TmuxSession(extra_env=...).send_keys(["env | grep -E ^PIP_", "Enter"])'
grep -c "Looking in indexes: https://mirrors.aliyun.com" output/<job>/tasks/*/s*/trajectory.jsonl
```

### A7.5 与哪个 benchmark / reward 搭配

- **TB2.1**：`cth-tb21-full89-mirror@1.0.1`（reward `cth-tb2-reward-mirror@1.0.1`：uv 预装、pypi 镜像、workdir 回退、uv 600 s、HF xet 关）。harness 修的是 agent 侧，reward-mirror 修的是评分侧，**两者缺一不可**——只修 harness 的 `cth-tb21-88-dsv41-max-1596` 在 astral.sh 不可达那天掉到 0.51。dsv41-flash 在这套口径下 **0.831**（旧口径 0.753）。
- **TerminalHorizon**：**`cth-th-full@1.1.0`** + `tb2-reward@4.0.0`。不用换 reward（verifier 是 `python3 /tests/check.py`，不走公网）；harness 用 1.6.96（16Gi 对多服务题的 tmux 被 OOM 杀有帮助，25/2972 条）。1.1.0 相对 1.0.0 只把 87 个拍扁镜像 env 换成 `@1.3.0`——它们的 sidecar 服务由 env 的 `hooks.setup_script`（`coding_arena/code/reg/th_flat_setup_hook.sh`）在主 netns 用 chroot 拉起，因为平台不执行镜像 ENTRYPOINT；这是 env 层的修复，与 harness 无关。另有 18 个 env 镜像缺 CA 证书（`work/manifests/th_ca_broken_envs.txt`）待重建。见 [tb21_verifier_audit](tb21_verifier_audit.md) §5 与 [trajectory_collection](trajectory_collection.md) §C1。
- 要和 1.0.0 时代的 SFT 消融（[sft_results](../experiments/sft_results.md) §2）直接对齐的实验，仍用 `1.0.0` / `1.0.1`；它们的绝对分带 10–16% 的 reward 噪声，组内比较仍有效。

### A7.6 版本号规约与全表

第二位 = 行为族：`1.0.x` 旧窗口语义（262k/65536、本地计数）、`1.1` 加 max_turns、`1.2.N` 触发点 N k、`1.3.x` 推理回传/窗口特例（1.3.96 nothink，1.3.0 1M 不压缩）、`1.4.x` tool-call、**`1.5.x` 任务 shell 环境注入 + 16Gi、`1.6.x` 在 1.5 上放宽 uv/HF**；`-后缀` = 单参数变体（`1.2.96-xhigh`）。14 个版本全部 PUBLISHED，不可变，改配置 = 发新版本。

---

# Part B. DeepSeek Harness（DSH sdk-minimal）：注册、与 Terminus 2 的差别、评测结果、历史推理的真实渲染

本部分记录把官方 DeepSeek Harness（`deepseek-harness-sdk` 0.2.1rc2，`sdk-minimal` profile）接进 EnvArena 的全过程：harness 包与代理、1.0.0→1.2.4 修掉的六个缺陷、与 Terminus 2 逐项对比、dsv41-flash / Qwen3.5-4B / 9B 三个模型的 TB2.1 结果，以及用网关抓包 + 真实 chat template 验证出的「历史推理到底有没有进 prompt」。相关：Part A（Terminus 2 的机制与版本）、[context_sweeps](../experiments/context_sweeps.md) Part C（历史推理可见性）、[tb21_verifier_audit](tb21_verifier_audit.md)（TB2.1 评测的非模型因素）。

源码与 spec：`/storage/openpsi/users/chen/code_data/dsh-harness/`（`pkg/runner`、`pkg/app/dsh_runner.py`、`pkg/app/dsh_proxy.py`、`harness-spec-1.*.json`，`build_pkg.sh`；wheels 原件已丢，重建按 `rebuild_pkg.sh` 的做法：从一次完整提取的 runtime 目录重打 tarball）。平台实体 `cth-dsh-minimal@1.0.0 … 1.2.4`。

## B1. 一句话结论

- **TB2.1 上 DSH ≈ Terminus 2**，三个模型都不显著：dsv41-flash +0.006、Qwen3.5-4B −0.019、9B −0.026（配对 95% CI 全含 0）。dsv41 与官方 90.6 的差距换 harness 不会消失。
- DSH **更快**（单题墙钟 −20～55%）但 **prompt token 高 2–7 倍**（历史 thinking 全部回传）。
- Qwen3.5 上 DSH 是**唯一真正让模型看到历史推理的路径**；Terminus 1.4.96 的 tool-call 开关在 Qwen 自家 sglang 上无效（§B6）。
- 小模型（4B/9B）的 0 分 run 里只有 3–5% 跑过测试，不论 harness；harness 不是它们的瓶颈。

## B2. DSH sdk-minimal 是什么

两进程结构：Node 单文件运行时（316 MB，闭源但 bundled ESM 明文可读）跑 agent 循环、工具、压缩、重试；Python SDK 只是 JSON-RPC stdio 壳。`sdk-minimal` profile 的全部可观测行为（从二进制里的 `@deepseek-ai/dsh-*` 包源码 + 网关 wire 抓包确认）：

| 项 | 值 |
|---|---|
| system prompt | `You are a helpful software engineer assistant.`（一句） |
| 工具 | 3 个原生 function：`bash{command}`、`read_image{file_path}`、`working_directory{cd?}` |
| bash 工具 | 持久 bash，`eval` 包裹 + nonce marker 精确截取输出和 exit code（`[Command finished with exit code N]`）；**单命令 300 s 超时**（`tool-bash-persistent` `timeoutMs: 3e5`），超时后 shell 重置并告知模型；输出 **16,000 chars 只保头**，尾部换成 `<response clipped><NOTE>…grep -n…</NOTE>` |
| 循环结束 | 模型某一步**不发 tool_call** → `turn/end completed`。没有 mark_task_complete，没有确认环节，**没有 turn 上限** |
| 压缩 | `compaction-basic`：阈值 `min(0.8·ctx, ctx − maxTokens − headroom)`，保留最近 `0.16·(ctx − maxTokens)` tokens 原文，单次 LLM 调用输出 8 段 Markdown checkpoint 替换前缀 |
| 协议 | Anthropic Messages `/v1/messages`，流式 SSE，`x-api-key` 认证，`thinking:{type:enabled}`（不带 budget），`output_config.effort`，`max_tokens` 默认 256000；**不发 temperature/top_p/top_k** |
| effort 白名单 | `deepseek-official` 适配器只认 `off/low/high/max`，`xhigh` 在发请求前就抛 `UNSUPPORTED_REASONING_EFFORT` |
| 重试 | `llm-retry`：5 次，500 ms 起指数退避上限 10 s，可重试码 `EMPTY_RESPONSE/RATE_LIMIT/SERVER/TIMEOUT/TRANSPORT`；耗尽后整个 turn 以 `error` 结束 |
| 历史推理 | 每条历史 assistant 消息带完整 `thinking` block 随请求回传 |

## B3. 包装层：runner + loopback proxy

网关与 DSH 之间有四处不兼容，全部在 `dsh_proxy.py`（127.0.0.1 回环代理）和 `dsh_runner.py` 里解决，DSH 本体零改动：

| 问题 | 处理 |
|---|---|
| 网关只收 `Authorization: Bearer`，DSH 发 `x-api-key` | 代理改写头 |
| 网关要求 `thinking.budget_tokens`，DSH 不发 | 代理注入 `max_tokens − 1` |
| bundled Python 3.13 无 CA | certifi 兜底 |
| SSE 要边收边转 | `resp.read1()` + flush（`read()` 会缓冲整条流） |
| DSH 无 turn 上限 | runner 数 `assistant/message` 事件，到 `DSH_MAX_TURNS` 时 kill 运行时，`finish_reason=max-turns`、status OK 让 reward 照常打分 |
| `max_tokens` 不可配 | `DSH_MAX_TOKENS`（qwen 262k 窗口容不下 SDK 默认 256k；compaction 要求 ctx > maxTokens） |
| harness_result.json 不是平台产物 | runner.log 末尾写一行 `summary status= finish= steps= llm_retries= turn_end= failure=` |

env 约定与 Terminus 相同（`LLM_*/OPENAI_*/ANTHROPIC_*` + key），所以同一个版本化模型两边都能用。

### B3.1 1.0.0 → 1.2.4 修掉的缺陷（每个都曾造成 20%+ 的 0 分）

| 版本 | 缺陷 | 根因 | 修复 |
|---|---|---|---|
| 1.1.0 | 无 turn 上限 | DSH 设计如此 | `DSH_MAX_TURNS=512`；`killpg` 对 SDK 子进程无效（同进程组），改 `/proc` 树遍历 |
| 1.2.0 | `xhigh` 被拒 | DSH effort 白名单 | 改 `max`（qwen 后端第二高有效档）；加 `DSH_MAX_TOKENS` |
| 1.2.1/1.2.2 | 25–30% run `finish_reason=error`，TTFB 130/260/390 s | `envarena-prod.alipay.com` 解析出 8 个 A 记录，**4 个黑洞**；Python 顺序 connect 每个坏 IP 吃满一次超时（390 = 3×130）；DSH 5 次重试 ≤10 s 退避扛不住 | 代理并发探测全部地址（3 s），缓存可用 IP 300 s，TLS SNI/证书仍按主机名；只重试 connect 阶段失败。p50 10 s → 0.27 s |
| 1.2.3 | 剩余 error：网关 0.1 s 内返 in-band `api_error: upstream stream request failed` | DSH 在命令无输出时发 `tool_result content: []`，网关 Messages→chat 转换拒收。抓包二分：20 条消息 OK，加上空块那条就失败，补一个文本块就 200 | 代理把空 content 换成 `[{type:text,text:"(no output)"}]` |
| 1.2.4 | kv-store-grpc / configure-git-webserver / hf-model-inference / headless-terminal 全 0，文件类测试全过、连接类全拒 | reward 在同一沙箱里跑，要连模型起的 daemon；DSH graceful shutdown 会 dispose `tool-bash-persistent` 杀掉所有 shell 和作业；即使只 SIGKILL 运行时，PTY 关闭也会给 bash 发 SIGHUP，bash 再 fan-out 给普通 `&` 作业 | 跳过 `harness.close()`；先 SIGKILL 运行时的直接子进程（被 SIGKILL 的 bash 发不出 SIGHUP），再 kill 运行时；模型起的进程孤儿化存活，与 Terminus 留 tmux server 到 eval 阶段等价。验证：`nohup` 与裸 `&` 的 http.server 都在 runner 退出后存活；configure-git-webserver 0/0/0 → 1/1/0 |

修到 1.2.4 后两个 Qwen job 共 525 个 run 里 `error` 为 0。

**也不是缺陷但要知道**：网关对非 DashScope 路由（slurm/AIS 自起）同样有黑洞 IP 问题，只是 Terminus 走 LiteLLM 6 次 5–30 s 退避扛过去了；Terminus 的轨迹里那些"30–130 s 的慢请求"大概率就是它。

## B4. DSH vs Terminus 2 逐项对比

| 维度 | Terminus 2（pkg_v2 / 1.2.96 / 1.3.0） | DSH sdk-minimal |
|---|---|---|
| system prompt | 无 system role；54 行 JSON 协议说明作为第一条 user 消息 | 一句话 system |
| 工具机制 | 无 function calling；模型输出纯文本 JSON `{analysis,plan,commands,task_complete}`，harness 解析 | 原生 tools（bash / read_image / working_directory） |
| 执行 | `tmux send-keys` 敲键，`block=False`，sleep `duration`（默认 1 s、上限 60 s）；观察是屏幕快照/增量 scrollback，不知道命令结束没有 | 持久 bash `eval`，精确 stdout + exit code |
| 单命令超时 | 无；靶死是 supervisor `ARENA_AGENT_TIMEOUT_SEC − 60 s` | 300 s，超时重置 shell（丢 cwd/env，告知模型） |
| 输出截断 | 10,000 bytes，保头 5K + 尾 5K | 16,000 chars，只保头 |
| 完成信号 | `task_complete: true` **连续两次**（第一次回"Are you sure…"） | 模型不再调工具即结束，无确认 |
| turn 上限 | `TERMINUS_MAX_TURNS=512` | 无（runner 加 512） |
| 压缩 | 3 个子调用（总结→提问→回答），整体替换历史；nocomp = 把窗口挪到 1M 让它永不触发（`enable_summarize` 仍 hardcode True） | compaction-basic 单次总结替换前缀；qwen 262k/65k 下阈值 ≈196k，dsv41 1M/256k 下 ≈744k，实际等价 nocomp |
| 协议 | OpenAI chat via LiteLLM，`reasoning_effort=max`，显式 temp 1 / top_p .95 / top_k 50，不设 max_tokens | Anthropic Messages，`max_tokens` 显式，不发采样参数（需路由 `upstream_extra_params` 钉死） |
| 输出撞上限 | `finish_reason=length` → 截断内容塞回历史再问一轮 | turn 直接以 `max-tokens` 结束，**不续问**，整题 0 分 |
| 顶层 `exit` | 杀 tmux shell → session 死 → 静默终止（≤4/178，~2 分） | 返回 `[shell exited]` + 自动重建 |
| 重试 | 6×(5–30 s) × 外层 3 次 | 5 次 ≤10 s（代理前置连接重试补足） |
| 历史推理 | `reasoning_content` 挂在历史 assistant 上，但 DashScope 纯 chat 丢弃 | thinking block 随请求回传，**网关渲染**（§B6） |

## B5. 结果

### B5.1 dsv41-flash-1017（effort max，2 samples，178 run）

| job | harness | 分数 |
|---|---|---|
| `cth-tb21-dshmin-dsv41-max-10291834-2610291035` | cth-dsh-minimal@1.0.0（无 cap） | **0.747**（178/178 DONE） |
| `cth-tb21-nocomp-dsv41-max-2610290159` | cth-terminus-2-stream@1.3.0 nocomp | 0.730 |
| 配对 171 | | DSH 0.749 vs Term 0.743，**+0.006 ± 0.077**；任务级 21 赢 / 18 输 |

1.1.0（512 cap）的重跑 `…10292004-2610291204` 跑到 99/178 时被停掉让位，没有完整分数；1.0.0 轨迹里超过 512 步的极少，cap 影响 <1 分。

与官方 90.6（DSH Minimal，1M，N=3）的 ~15 分差距在换成同一个 harness 后依然在，剩余变量只能是模型供给（DashScope 路由 vs 官方 API）、网关转换（§B6.4 第 5 条：effort 被降成 high）、评测镜像/网络（见 [tb21_verifier_audit](tb21_verifier_audit.md) §2）。

### B5.2 Qwen3.5-4B / 9B（3 samples，267 run，slurm 1 节点 4+4 卡，MTP，fp32 SSM）

harness `cth-dsh-minimal@1.2.4`：`effort=max`、`ctx 262144`、`max_tokens 65536`、`512 turns`；路由 `upstream_extra_params` 钉 temp 1 / top_p .95 / top_k 50。对照是 10-21 的 Terminus 基线（`cth-tb21-stream-qwen35-4b-fp32-1n-…0908`、`cth-tb21-stream-qwen35-9b-2n-…1124`）。

| | DSH | Terminus 2 | 配对差（95% CI） | 任务 DSH 赢/输/平 | ≥1/3 解出 DSH / Term / 并集 |
|---|---|---|---|---|---|
| 4B | **0.169**（`…4b-1n-11011104-2611010304`） | 0.188 | −0.019 ± 0.053 | 10 / 15 / 63 | 24 / 29 / 35 |
| 9B | **0.223**（`…9b-1n-11011104-2611010304`） | 0.249 | −0.026 ± 0.057 | 13 / 16 / 60 | 34 / 37 / 43 |

效率：

| | 4B DSH / Term | 9B DSH / Term |
|---|---|---|
| 单题墙钟 p50 | 10 / 22 min | 9 / 12 min |
| 模型调用 p50 | 74 / 48 | 54 / 19 |
| prompt tokens/run p50 | 2.7M / 1.4M | 1.5M / 0.2M |
| completion tokens/run p50 | 35k / 26k | 27k / 10k |

失败模式：

| finish | 4B | 9B | 均分 |
|---|---|---|---|
| completed | 84% | 92% | 0.20 / 0.24 |
| **max-tokens**（单轮 65536 用完） | **13%**（20 题 34 run） | 5% | **0.00** |
| max-turns | 2% | 1% | 0.00 |

Terminus 在同样的 (task, sample) 上拿 0.09 / 0.07——它遇到 length 会续问。**剔除撞上限的 run 后配对：4B −0.005、9B −0.024**，4B 的全部落后来自这里。

两边 0 分 run 里跑过任何测试的只有 3–5%（DSH 4%/3%，Terminus 同量级）；自宣完成但测试没过：DSH 66–69%，Terminus 48–59%。

DSH 赢的典型：多步构建/修复（cobol-modernization 0.33→1.00、git-leak-recovery 0.33→1.00、mcmc-sampling-stan、nginx-request-logging、pypi-server）。输的典型：password-recovery 1.00→0.00、fix-git、qemu-startup、compile-compcert——长命令撞 300 s 超时丢 shell 状态，或交互式终端场景（tmux 天然适合，DSH 只有一次性 eval）。

### B5.3 如果还要用 DSH 跑小模型

唯一值得做的改动是 max-tokens 续问（像 Terminus 那样截断后再给一轮），能收回 4B 上 ~0.02。其余差异都在噪声里。成本上同分 Terminus 便宜 2–7 倍 prefill；DSH 快 1.3–2 倍墙钟。

## B6. 历史推理到底有没有进 prompt：抓包 + 模板 + 计数（11-02）

[context_sweeps](../experiments/context_sweeps.md) §C3 里"带 tools 且 tool_choice≠none 就渲染历史推理"是在 **DashScope dsv41 路由**上的探针结论。换到 Qwen3.5 自家 sglang 后结论不同。做法：建临时路由 `cth-capture-probe` 指到 slurm 计算节点上的抓包 HTTP 服务（登录节点网关不可达），拿到网关**转换后发给上游的 chat payload**；再用 `Qwen__Qwen3.5-9B/chat_template.jinja` 渲染；最后在 AIS 9B（record 353623139）上用 token 计数证实。

### B6.1 网关发给上游的是什么

DSH 的 Messages 请求被网关转成**标准 OpenAI tool-calling 对话**：system → user → `assistant{content, reasoning_content, tool_calls[]}` → `tool{tool_call_id, content}` → …；thinking block 变成 `reasoning_content` 字段；`output_config.effort=max` 被改写成 `reasoning_effort: "high"`（注意：**网关把 max 映射成了 high**）；`max_tokens` → `max_completion_tokens`。

Terminus 1.4.96 toolhist 的 chat 请求原样透传：user → `assistant{content(JSON 文本), reasoning_content}` → user（观察）→ …，外加 `terminus_action` wrapper tool + forced `tool_choice`，**历史里没有 `tool_calls`，观察仍是普通 user**。

### B6.2 Qwen3.5 模板的规则

```jinja
{# 从后往前找"最后一条真 user query"：不以 <tool_response> 包裹的 user 消息 #}
{%- for message in messages[::-1] %}
  {%- if ns.multi_step_tool and message.role == "user" %}
    {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
      {%- set ns.multi_step_tool = false %}{%- set ns.last_query_index = index %}
...
{%- elif message.role == "assistant" %}
  {%- if loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>assistant\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}   {# 只有这些才带 think #}
  {%- else %}
    {{- '<|im_start|>assistant\n' + content }}                                                 {# 之前的全部丢掉推理 #}
```

`tool` role 的消息被渲染成 `<|im_start|>user\n<tool_response>…</tool_response>`，**不算**真 user query。所以：

- **DSH 形态**：第一条任务消息之后全是 tool_response，`last_query_index = 0`，**每一条历史 assistant 都带 `<think>`** —— 渲染结果里 STEP1/STEP2 的推理全在。
- **Terminus 形态**（toolhist 或纯 chat 都一样）：观察是普通 user 消息，`last_query_index` 就是最后一条观察，它之前的所有 assistant **都不带 `<think>`** —— 历史推理全部丢弃。带不带 `terminus_action` tool 毫无影响。

### B6.3 真实计数（AIS 9B，塞 ~1400 token 的假推理到历史 assistant）

| 请求形态 | 不带历史推理 | 带 | 增量 |
|---|---|---|---|
| Terminus 纯 chat | 32 | 32 | **0** |
| Terminus + tools + `tool_choice:none` | 32 | 32 | **0** |
| Terminus + tools + `auto` | 297 | 297 | **0** |
| Terminus + tools + forced fn（= 1.4.96） | 297 | 297 | **0** |
| **DSH 形态**（Messages，thinking block + 原生 tool_use/tool_result） | 323 | 2425 | **+2102** |

### B6.4 含义

1. **Qwen3.5 上，Terminus 全系（1.0.0 / 1.2.96 / 1.3.0 / 1.4.96）都看不到历史推理**；1.4.96 的 tool 开关只对 DashScope 的 DeepSeek 模板有效，对 Qwen 模板无效。[context_sweeps](../experiments/context_sweeps.md) §C3 的"规则"要加限定：决定渲染的不是请求带不带 tools，而是**上游模板怎么找 last user query**——DeepSeek 模板看 tools，Qwen 模板看观察是不是 `<tool_response>`。
2. 想让 Qwen 看到历史推理，**观察必须以 `tool` role 回传**（即真 function calling 循环）。DSH 天然如此；Terminus 若要做，得把观察从 user 消息改成 tool 消息，这等于改掉 Terminus 的协议。
3. §B5.2 的 Qwen 对比因此混着两个变量（harness 结构 + 历史推理可见性），不能单独归因。但 [context_sweeps](../experiments/context_sweeps.md) §C5 四格里 flash 看得到 vs 看不到只差 +0.024（不显著），所以可见性本身对分数影响有限；差异主要还是协议/工具层。
4. 4B/9B 的 Terminus 基线用 `interleaved_thinking=true`（pkg_v2 hardcode）把 `reasoning_content` 发了出去，但模板不渲染——所以 usage 锚点（P+C）在 Qwen 上也是**虚高**的（没扣 R），和 [context_sweeps](../experiments/context_sweeps.md) §C4 "需要做的 1" 是同一个 bug，范围比那里写的更大：**所有 Qwen 评测都受影响**，不只 nothink 模板。
5. dsv41 走网关时 `effort=max` 被降成 `high`；DSH 的 dsv41 分数（0.747）是 high 档的分数。Terminus 发 `reasoning_effort=max` 走 chat 路径不经过这个映射。这是 §B5.1 比较里一个未控制的变量，方向上对 DSH 不利。

## B7. 运维备忘

- 4B/9B 服务用 `providers/slurm/serve_multi.sh`（一节点双模型，`profile.sh:N` 指定卡数），参数与 AIS profile 完全相同。节点空闲时 slurm 作业会被回收，**长时间评测请用 AIS**（`providers/ais/launch.sh`）。
- `qwen35-4b-chen-ais@1.0.2` 是按 `qwen35-9b-chen-ais@1.0.0` 的 spec 新注册的（旧的 1.0.x-dual 系列是 htnn 域名的废弃版本）。
- 路由 `upstream_extra_params` 钉采样参数时，DSH 和 Terminus 都会被覆盖（优先级高于客户端字段），A/B 期间别改。
- 看 DSH error run 的原因：`arena task download <tid> dsh-proxy.log` / `dsh-proxy-failures.jsonl`（1.2.2 起随 artifact 上传），或 `runner.log` 的 summary 行。
- 验证 daemon 存活的最小探针：让模型跑 `python3 -m http.server N &`，runner 退出后 `pgrep -f "http.server N"`。
