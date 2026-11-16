# arena_assets - Terminus-2 评测/采轨迹套件（可脱离平台部署）

本仓库提供 harness 源码、五个版本的配置、reward 脚本、任务清单、环境修复和轨迹处理工具。原部署使用的 harness/bin/ 与 reward/uv-dist/ 二进制包未随仓库分发。复用 §5 的运行命令需先从原部署恢复这两部分及对应任务环境。本机入口见[交接文档](../HANDOFF.md)，版本选择见[评测资产说明](../methods/arena_verifier.md)。

下图为完整部署布局，其中标为「外部」的目录由原部署提供。

```
arena_assets/
├── harness/
│   ├── bin/ 〔外部〕            可直接执行的运行时包（85 MB）：runner（bash 入口）+ python313.tar.gz
│   │                            + site-packages/py313.tar.gz + app/app.tar.gz + fallback-bin/tmux；
│   │                            runtime-manifest.json 记录 8 个文件的 sha256
│   ├── src/app/                 app.tar.gz 的原文：envarena_terminus2/（入口、配置、supervisor）
│   │                            + harbor/（Harbor v0.20.0-58-g0348989a，含流式/计数/tool-call 补丁）
│   ├── configs/<ver>.env        五套 agent 配置（1.0.0 / 1.2.96 / 1.6.96 / 1.6.262 / 1.7.96），KEY=VALUE
│   ├── requirements.lock.txt    site-packages 里 90 个包的精确版本（从 py313.tar.gz 读出）
│   └── selftest/                离线自检：mock 模型 + 一次完整 harness→reward 跑通（见 §5）
├── reward/
│   ├── tbench_reward_runner_mirror.sh   TB2 评分 runner（官方 4.0.18 + 5 处基建修复，test.sh 不动）
│   ├── official_runner_4.0.18.sh        官方 runner 原文，供 diff
│   └── uv-dist/uv-x86_64-unknown-linux-gnu.tar.gz 〔外部〕 runner 预装的 uv/uvx 0.9.5（原部署的 PROVENANCE.txt 未附）
├── env/
│   ├── hooks/th_flat_setup_hook.sh      87 个 flat 镜像 TerminalHorizon 题的容器内启动脚本
│   ├── fixes/reward_hang_test.sh        8 个 reward 卡死题的 tests/test.sh 替换件（timeout 包装）
│   ├── fixes/dns_hosts_pin.sh           2 个改写 resolv.conf 的题：预先把 LLM 端点钉进 /etc/hosts
│   ├── rt3_ca/                          18 个缺 CA 证书镜像的重建套件（Dockerfile + 校验脚本 + 清单）
│   └── manifests/                       坏 env 归因表、剔除清单、污染轨迹清单、交接批元数据
├── tasks/
│   ├── tb21_full89.tsv                  TB2.1 的 89 道题（task, 官方 env 版本）
│   └── th_full_1486.tsv                 TerminalHorizon 1486 题：image / workdir / cpu / mem / disk / 超时 / 修复标记
├── scripts/                             轨迹 → SFT 数据的全链路（build_multi_sft → audit → merge → judge → finalize）
└── legacy/                              原平台（arena CLI）专用件：spec JSON、注册脚本、提交/拉取脚本。新服务器不需要
```

---

## 1. 该用哪一套

| 用途 | 任务清单 | Reward | Harness 配置 | 备注 |
|---|---|---|---|---|
| TB2.1 评 API 模型 | `tasks/tb21_full89.tsv` | `reward/tbench_reward_runner_mirror.sh` | `configs/1.6.96.env` | 默认 |
| TB2.1 评 SFT 学生模型 - 262k 整窗口口径 | 同上 | 同上 | `configs/1.6.262.env` | 训练数据是整轨迹 262k 时用；与 1.6.96 分数并列报告，不混 |
| TB2.1 评 SFT 学生模型 - 96k compaction 口径 | 同上 | 同上 | `configs/1.6.96.env` | 训练数据按 compaction 段切时用；与 API 教师同口径 |
| TH 采 DeepSeek（DashScope）轨迹 | `tasks/th_full_1486.tsv` | 题目自带 `tests/test.sh`（见 §3.3） | `configs/1.7.96.env` | tool-call 模式：DashScope 只在带 tools 时渲染历史 reasoning |
| TH 采 Qwen flash 轨迹 | 同上 | 同上 | `configs/1.6.96.env` | chat 模式已可见历史推理 |

五套配置只差 `agent.env`，运行时包完全相同：

| 配置 | 窗口 / 压缩阈值 | 计数 | max_turns | tool-call | agent shell 注入 pip/uv/HF 镜像 | 推荐沙箱内存 |
|---|---|---|---|---|---|---|
| 1.0.0 | 262144 / 65536 | estimate（本地） | 无 | 否 | 否 | 8Gi |
| 1.2.96 | 98304 / 1024 | usage（按 provider 回报） | 512 | 否 | 否 | 8Gi |
| 1.6.96 | 98304 / 1024 | usage | 512 | 否 | 是 | 16Gi |
| 1.6.262 | 262144 / 65536 | estimate | 无 | 否 | 是 | 16Gi |
| 1.7.96 | 98304 / 1024 | usage | 512 | **是** | 是 | 16Gi |

---

## 2. Harness：运行时契约

### 2.1 它是什么

`harness/bin/runner` 是一个 bash 脚本：解包自带的 Python 3.13 + 依赖 + app，必要时换用自带 tmux 3.4，
然后 `python -m envarena_terminus2.supervisor` → 子进程跑 Terminus-2 agent（Harbor）。
agent 在**本容器**内开 tmux 会话执行命令（`envarena_terminus2/environment.py` 的 `LocalEnvironment` 就是本地 `bash -c`），
所以 runner 必须**在任务容器里**启动，与被测环境同一个文件系统。

### 2.2 输入（环境变量）

| 变量 | 必填 | 含义 |
|---|---|---|
| `ARENA_HARNESS_DIR` | 否（默认 `/envarena/harness`） | runner 在 `$ARENA_HARNESS_DIR/bin/` 下找 `python/ site-packages/ app/ fallback-bin/`。把本目录的 `harness/` 挂到容器里，令此变量指向它 |
| `ARENA_QUERY_PATH` | 是 | 任务指令 markdown（TH：题目的 `instruction.md`） |
| `ARENA_OUTPUT_DIR` | 是 | 输出目录 |
| `ARENA_AGENT_WORKDIR` | 否（默认 `/app`） | agent 起始目录 = 题目 workdir（`tasks/*.tsv` 的 `workdir` 列） |
| `ARENA_AGENT_TIMEOUT_SEC` | 否（默认 1800） | agent 总时限；supervisor 提前 60 s 杀子进程并导出部分轨迹。TB2.1 用 14400，TH 用 18000 |
| `ARENA_ATTEMPT_DIR` / `ARENA_TRAJECTORY_PATH` / `ARENA_TASK_ID` | 否 | 日志根目录 / 轨迹输出路径 / tmux 会话标识 |
| `LLM_API_KEY` `LLM_BASE_URL` `LLM_MODEL` | 是 | OpenAI 兼容端点。`LLM_MODEL` 不带 `provider/` 前缀时自动加 `openai/`；`ANTHROPIC_*` 三件套是备选入口 |
| `TERMINUS_*` | 否 | 即 `configs/<ver>.env` 的内容，`set -a; . configs/1.6.96.env; set +a` 或 `docker run --env-file` |

`TERMINUS_SANDBOX_ENV_<NAME>=<v>` 会以 `<NAME>=<v>` 注入 agent 的 tmux shell（只影响 agent，不影响 reward 阶段）。
1.6.x 配置用它给 agent 的 shell 设 `PIP_INDEX_URL` / `UV_*` / `HF_HUB_*`；**换服务器时把 aliyun 镜像改成你那边可达的源**，或整组删掉。

### 2.3 输出

- `$ARENA_OUTPUT_DIR/harness_result.json`：`status`（OK / ERROR）、`summary`、`raw`（token 数、完成确认次数、生效的配置）
- `$ARENA_OUTPUT_DIR/trajectory.jsonl`：每行一步，`source` ∈ user / agent / system，agent 行含 `message`、`reasoning_content`、`tool_calls`、`observation`。超时或出错时也会导出已有部分
- `$ARENA_ATTEMPT_DIR/terminus2/`（默认在输出目录下）：Harbor 原始日志与 `trajectory.json`
- 最后一个 tmux 会话**保留不杀**，题目里起的服务到 reward 阶段仍可达；reward 跑完后由外层销毁容器

### 2.4 对容器的要求

- x86_64，glibc >= 2.17（自带 Python）；镜像自带 tmux 不支持 `new-session -e` 时自动换自带 tmux 3.4（需要 glibc >= 2.28）
- 能写 `/logs/agent`、`/tmp`、输出目录；能出网到 `LLM_BASE_URL`
- 若题目会改 `/etc/resolv.conf`（TH 有 2 题，见 `tasks/th_full_1486.tsv` 的 `dns-pin`），先 `sh env/fixes/dns_hosts_pin.sh <LLM 端点主机名>`

### 2.5 从源码重打包

```bash
tar -czf harness/bin/app/app.tar.gz -C harness/src app      # 改了 src/ 之后；runner 只认这个 tar
sha256sum harness/bin/app/app.tar.gz                        # 同步更新 runtime-manifest.json
```

`python313.tar.gz` / `py313.tar.gz` 不必重打；需要从头构建时按 `requirements.lock.txt` 装进 Python 3.13.6。
补丁位置：`harbor/llms/lite_llm.py`（流式 + tool-call 内容提取）、`harbor/llms/chat.py`（usage 锚点）、
`harbor/agents/terminus_2/terminus_2.py`（`_count_total_tokens`、强制 `terminus_action` tool）。

---

## 3. Reward：评分契约

### 3.1 TB2.1：`reward/tbench_reward_runner_mirror.sh`

在 agent 跑完后、**同一个容器**里执行。官方 `tests/test.sh` / `test_outputs.py` 一字不动，只改它们的运行条件：

| 输入变量 | 含义 |
|---|---|
| `ARENA_ORACLE_PATH` | 题目的 oracle 目录，内含 `tests/test.sh`（和可选 `details.json`，`{"workdir": ...}`） |
| `ARENA_REWARD_DIR` | 本目录的 `reward/`（runner 从 `$ARENA_REWARD_DIR/uv-dist/` 预装 uv） |
| `ARENA_REWARD_OUTPUT_FILE` | 结果 JSON 路径 |
| `ARENA_ARTIFACTS_DIR` | 产物目录（默认结果文件旁的 `artifacts/`） |
| `ARENA_AGENT_WORKDIR` | 没有 `details.json` 时的工作目录回退（再回退 `/app`） |
| `ARENA_ENV_KEY` | 只有 `tb2-pytorch-model-cli` 这一题会看（走镜像内置的离线 wheel 缓存） |
| `TB2_REWARD_PYPI_MIRROR` | 可选，覆盖默认的 aliyun 源 |

输出：`{"status","score"(0/1),"pass","raw":{"exit_code","workdir","log_excerpt"}}`，
以及 `$ARENA_ARTIFACTS_DIR/tbench_test.log`（完整 verifier 输出，归因脚本靠它判断 pytest 是否真跑了）、`reward.txt`、`ctrf.json`。

做了什么（对照 `diff reward/official_runner_4.0.18.sh reward/tbench_reward_runner_mirror.sh`）：
① 从随包 tarball 预装 `uv`/`uvx` 并预建 `~/.local/bin/env`，`test.sh` 开头的 `curl astral.sh | sh` 失败也无所谓；
② `UV_DEFAULT_INDEX`/`UV_INDEX_URL`/`PIP_INDEX_URL` 指向镜像源（题目显式 `--index-url` 仍优先）；
③ WORKDIR 回退链 `details.json → $ARENA_AGENT_WORKDIR → /app`；
④ `UV_HTTP_TIMEOUT=600`、并发 4、重试 6；⑤ `HF_HUB_DISABLE_XET=1` + HF 超时放宽。
runner 会 `rm -rf /tests /logs/verifier` 再把 oracle 的 tests 链接到 `/tests`——只能在一次性任务容器里跑。

TB2.1 的题目包（env 镜像 + `tests/`）来自 terminal-bench 2.1 官方发布，不在本目录里；`tasks/tb21_full89.tsv` 给出题名与所用版本
（86 题 6.4.0，`feal-differential-cryptanalysis`、`qemu-startup` 6.4.1，`pytorch-model-cli` 6.4.2-gepa-cache = 依赖烘进镜像的离线评分版）。

### 3.2 TerminalHorizon：直接跑题目的 `tests/test.sh`

TH 的 oracle 是 `python3 /tests/check.py`，不走公网、不装包，不需要 mirror runner。约定：把题目的 `tests/` 放到 `/tests`，
`bash /tests/test.sh`，读 `/logs/verifier/reward.txt`（1 / 0）。官方 runner 的后半段（读 reward.txt → score）在
`official_runner_4.0.18.sh` 第 60 行起，可直接复用。

### 3.3 TH 题目数据

1486 题的 `instruction.md` / `tests/` / `task.toml` / `environment/Dockerfile` 在 TerminalHorizon 原始数据集里
（本集群：`/storage/openpsi/users/chen/code_data/terminal_horizon/data/tasks/<task>/`），随镜像一起迁。
`tasks/th_full_1486.tsv` 的 `image` 列是我们实际跑过的镜像 tag（`reg.antgroup-inc.cn/...`，需要搬到你的 registry）；
`notes` 列标出需要特殊处理的题，见 §4。

---

## 4. Env 修复（TerminalHorizon）

`env/manifests/th14_bad_envs.tsv` 是 160 个系统性失败 env 的归因表（原因 / 处置 / 修法），下面四类修法对应的文件都在：

| notes 标记 | 题数 | 问题 | 修法 / 文件 |
|---|---|---|---|
| `flat-sidecars` | 87 | compose 多服务题被拍成单镜像，sidecar 由镜像 ENTRYPOINT 的 `/opt/flatten/runtime.py` 启动；运行平台用自己的 wrapper 当 PID 1 时 ENTRYPOINT 不执行，服务没起 | 容器启动后、agent 之前执行 `env/hooks/th_flat_setup_hook.sh`：读 `/opt/flatten/plan.json`，chroot 到各服务 rootfs 在主 netns 起服务，`/etc/hosts` 把服务名指到 127.0.0.1，写 `ready.json`。**如果你的运行方式会执行镜像 ENTRYPOINT，不需要这个 hook** |
| `exec-prefix` | 4 | 交接方要求 agent 与 verifier 命令都加前缀 `python3 /opt/flatten/main_exec.py` | 未实现；这 4 题（也都是 flat）用上面的 hook 跑过但语义不完全对齐 |
| `reward-hang` | 8 | `check.py` 等一个永不退出的进程，评分卡死 | 把题目 `tests/test.sh` 改名 `test_orig.sh`，用 `env/fixes/reward_hang_test.sh` 做新的 `test.sh`（`timeout -k 60 1500`） |
| `dns-pin` | 2 | 题目自己把 `/etc/resolv.conf` 改成 127.0.0.1，之后 harness 解析不了 LLM 端点 | agent 之前 `sh env/fixes/dns_hosts_pin.sh <LLM 端点主机名>` |
| `needs-16Gi` | 24 | 8Gi 下 OOM killer 杀 tmux server | 给 16Gi（1.6.x 配置推荐值），env 不改 |
| `EXCLUDED-ca-broken` | 18 | 镜像缺 CA 证书，任何 `curl https://` 都失败 | 在有 docker 的机器上 `bash env/rt3_ca/build_rt3.sh env/rt3_ca/images_ca_broken.txt` 重建（Dockerfile 只加 `ca-certificates`，脚本校验 ENTRYPOINT/用户/TLS 不变）；**尚未执行** |
| `EXCLUDED-oracle-too-large` | 1 | oracle 83 MB | 待瘦身 |

修不了的一类：k8s / BGP / NFS 等在容器内自建私网的题，受运行平台网络策略限制（`connect: operation not permitted`）。
只能在数据层剔除：`scripts/audit_env_pollution.py` 的 `eperm-net` 检测器；换平台后这类限制可能不存在，先跑再说。

`env/manifests/th_handoff_meta.json` 是交接批 203 题的原始元数据（镜像 digest、flat / exec_prefix 标记、交接方的使用说明）。

---

## 5. 从 0 跑通

### 5.1 恢复原运行时包后做离线自检

```bash
sudo bash harness/selftest/run.sh            # 默认 1.6.96 1.7.96 1.6.262；也可以列出任意 configs/ 里的版本
```

起一个脚本化的 OpenAI 兼容 mock 端点，在私有 mount namespace + tmpfs chroot 里跑一次完整的 harness → reward：
断言 harness 正常结束、两次 task_complete 确认、轨迹 4 行无解析错误、请求是流式、历史带 `reasoning_content`、
tool-call 模式与配置一致、sandbox env 注入到了 agent shell、reward 从随包 tarball 装到 uv 并在 `/app` 下给出 score=1。
宿主机的 `/` 不被触碰。5 个配置在本集群全部 PASS。

### 5.2 一道题的执行顺序（你的调度器要做的事）

```
1. 起任务容器（题目镜像；TB2.1 推荐 4 cpu / 16Gi / 50Gi，TH 用 tasks/th_full_1486.tsv 的资源列）
   挂载：本目录 harness/ -> /harness（只读）、reward/ -> /reward（只读）、题目 instruction.md、oracle(tests/)、输出目录
2. [TH flat 题]  sh /path/th_flat_setup_hook.sh
   [TH dns 题]   sh /path/dns_hosts_pin.sh <LLM 端点主机名>
3. env -i PATH=... HOME=/root $(cat configs/<ver>.env | grep -v '^#') \
       ARENA_HARNESS_DIR=/harness ARENA_QUERY_PATH=/env/instruction.md ARENA_OUTPUT_DIR=/out \
       ARENA_AGENT_WORKDIR=<workdir> ARENA_AGENT_TIMEOUT_SEC=14400 \
       LLM_API_KEY=... LLM_BASE_URL=https://.../v1 LLM_MODEL=<name> \
       /harness/bin/runner
4. [TB2.1]  ARENA_ORACLE_PATH=/oracle ARENA_REWARD_DIR=/reward ARENA_REWARD_OUTPUT_FILE=/out/reward.json \
            ARENA_ARTIFACTS_DIR=/out/artifacts ARENA_AGENT_WORKDIR=<workdir> bash /reward/tbench_reward_runner_mirror.sh
   [TH]     ln -s /oracle/tests /tests; mkdir -p /logs/verifier; (cd <workdir> && bash /tests/test.sh); cat /logs/verifier/reward.txt
5. 收走 /out/{harness_result.json,trajectory.jsonl,reward.json,artifacts/tbench_test.log}，销毁容器
```

`harness/selftest/run.sh` 的 inner 部分就是这五步的一个最小实现，可以直接照抄。

### 5.3 落盘布局（下游脚本依赖）

```
<out_root>/<job_id>/
  summary.tsv                       env_key  sample  status  score  agent_ms  eval_ms
  job.json                          可选；若存在，build_multi_sft 优先从 data.tasks[] 读 status/score
  tasks/<env_key>/s<i>/
    task.json                       {"status": "DONE"|..., "score": 0|1, ...}
    trajectory.jsonl  harness.log  reward.log  tbench_test.log
```

---

## 6. 轨迹 → SFT 数据

```bash
# 1. 整轨迹 jsonl（--linear 不切 compaction 段；去掉 --linear 则一段一条）
python3 scripts/build_multi_sft.py --out-dir <out> --dataset-name <name> --done-only --linear <out_root>/<job_id> [...]
# 2. 环境污染审计（pip-net / hf-net / git-net / oom / eperm-net / gave-up）-> <prefix>_traj.json
python3 scripts/audit_env_pollution.py --base <out>/<name> --out <out>/<name>_pollution
# 3. 多来源合并：后面的 --src 覆盖同 (env, sample)，剔除 --polluted 里的轨迹
python3 scripts/build_merged.py --out <merged>/<m> --src <out>/<name> [--src <rerun>] --polluted <job>=<..._traj.json>
# 4. LLM 三准则 judge（断点续跑；1 条轨迹 1 请求，上限 1.9M 字符）+ 应用 -> <m>_judged-keep
python3 scripts/llm_judge.py --base <merged>/<m> --out <merged>/judge --workers 8 --retries 4
python3 scripts/apply_judge.py --base <merged>/<m> --judge <merged>/judge --out-prefix <merged>/<m>_judged
# 5. 终处理（form A/C、short-think 过滤）+ 统计 + 坏动作扫描 + 污染复审
python3 scripts/finalize_dataset.py --base <merged>/<m>_judged-keep --out <final> --name <f>
python3 scripts/dataset_stats.py --base <final>/<f> --name <f> --out <final>/stats.json
python3 scripts/scan_bad_actions.py --base <final>/<f>
# 3-5 一条龙（等 judge 跑完、重试 JUDGE_ERROR 两轮、apply、finalize、stats、复审）：
bash scripts/run_clean_chain.sh <merged_dir> <m> <final_dir> <f>
```

`llm_judge.py` 顶部的 `API` / `MODEL` 常量指向 matrixllm 的 `claude-opus-4-8`，key 读 `MATRIXLLM_API_KEY`；
换判官改这两行和 key 变量名。prompt 是 TerminalHorizon App J.1 原文 + 一行 compaction 说明，不要改准则。
`failure_audit.py <job_id> --out-root <out_root>` 把每条 0 分轨迹归到唯一主因（reward 没跑 pytest / 超时 / 512 步 / OOM / 网络放弃 / 模型错）。
脚本里残留的 `/storage/openpsi/...` 默认路径都可以用参数覆盖。

---

## 7. 迁移后的烟测（上量前）

1. `sudo bash harness/selftest/run.sh` 全 PASS。
2. TB2.1 挑 4 道评分阶段装 torch 的题（`mteb-retrieve`、`pytorch-model-recovery`、`reshard-c4-data`、`torch-tensor-parallelism`）：
   `tbench_test.log` 里 4/4 出现 `test session starts` → reward 的 uv 预装 + 镜像源生效。
3. TB2.1 挑 4 道 agent 要 pip 装包的题（`dna-insert`、`raman-fitting`、`sam-cell-seg`、`train-fasttext`）：
   轨迹里出现 `Looking in indexes: <你的镜像源>`、0 次 `ReadTimeoutError` → sandbox env 注入生效。
4. TH flat 题任一道（`tasks/th_full_1486.tsv` 里 `flat-sidecars`）：容器日志有 `[th-flat] started <svc>` 且 `ready=` 非 `NO`。
5. DeepSeek + 1.7.96 任一题：相邻两步 `prompt_tokens` 增量 ≈ 上一步 reasoning + content + 终端输出（而不是只有后两项）→ 历史推理可见。
6. `failure_audit.py` 的 `reward bootstrap fail` 列为 0。

---

## 8. legacy/

原平台（arena CLI）才用得上的东西，原样保留，新服务器不需要：
`harness/specs/*.json`（平台 harness spec，`configs/*.env` 由其 `agent.env` 生成）、`reward/reward-spec-1.0.1.json`、
`env/reg/`（env 注册脚本）、`env/specs-1.4.0/`、`env/spec-flat-example.json`、`env/hooks/`（含平台守护进程重启段的 hook 原文）、
`benchmarks/*.json`（`tasks/*.tsv` 由其生成）、`env/manifests/tasks_redo14.tsv`、
`scripts/{common.sh,submit.sh,pull_trajectories.sh,finalize_judge.sh,build_dsv41_merged.py}`、[平台发布说明](legacy/README.arena.md)（原 README）。
