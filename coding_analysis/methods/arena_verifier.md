# Reward / Harness / Env 最终版本总表（takeaway）

> 原始集群路径、平台作业和文内脚本命令按实验发生时记录。当前服务器入口和可复现范围见[交接文档](../HANDOFF.md)。

整理日期：2026-11-07。回答一个问题：**现在到底该用哪个 reward、哪个 harness、哪个 env 版本，它们各修了什么。** 过程性细节在 [tb21_verifier_audit](tb21_verifier_audit.md)（TB2.1 reward / harness 修复）、[harness_comparison](harness_comparison.md)（Terminus 2 机制与 1.6.96 字段来源）、[trajectory_collection](trajectory_collection.md)（TH env 修复与数据清洗）；本文只给结论和修复逻辑。

## 0. 一页速查

| 用途 | Benchmark | Reward | Harness | Env 版本 |
|---|---|---|---|---|
| **TB2.1 评 API 模型**（dsv41 / qwen flash） | `cth-tb21-full89-mirror@1.0.1` | `cth-tb2-reward-mirror@1.0.1` | `cth-terminus-2-stream@1.6.96` | 官方 `tb2-*@6.4.0`（不动） |
| **TB2.1 评 SFT 学生模型 — 262k 整窗口口径** | `cth-tb21-full89-mirror@1.0.1` | 同上 | `cth-terminus-2-stream@1.6.262`（1.0.0 窗口语义 + 16Gi + 镜像） | 同上 |
| **TB2.1 评 SFT 学生模型 — 96k compaction 口径** | `cth-tb21-full89-mirror@1.0.1` | 同上 | `cth-terminus-2-stream@1.6.96`（与教师同口径） | 同上 |
| **TH 采 dsv41 轨迹**（历史推理可见，tool-call 模式） | `cth-th-full@1.1.0` 为底；重打用 `cth-th-redo14@1.0.0` / `cth-th-redo-pollution@1.0.0` | `tb2-reward@4.0.0`（官方，TH 的 `check.py` 不走公网，不需要 mirror） | **`cth-terminus-2-stream@1.7.96`** | flat `@1.3.0`、reward-hang/DNS `@1.4.0`、其余 `@1.1.0`/`@1.2.0` |
| **TH 采 qwen3.8-flash 轨迹** | 同上 | 同上 | **`cth-terminus-2-stream@1.6.96`**（flash 走 chat 模式即渲染历史推理，不需要 tool-call） | 同上 |

三条硬规则：
1. TB2.1 分数**只认 mirror 口径**；旧 benchmark `tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2` 的分数每 job 有 7–16% 任务 pytest 没跑就记 0，不可比。
2. 采轨迹的 harness 必须带 16Gi + agent 侧镜像（1.6.96 / 1.7.96），否则 DONE 轨迹里会混进 OOM / pip 超时 / 放弃验证。
3. 评学生模型的 harness 要和它训练数据的采集口径一致（历史推理可见性、窗口语义），否则比的不是模型：整轨迹 262k 训练 → 1.6.262；按 compaction 段训练或要与教师并列 → 1.6.96；同一 checkpoint 两个口径都跑时并列报告，不混。

---

## 1. Reward

### 1.1 `cth-tb2-reward-mirror@1.0.1`（TB2.1 用）

**问题**：官方 `tb2-reward@4.0.18` 直接跑题目的 `tests/test.sh`，而 89 道题里 88 道的 `test.sh` 开头都是

```
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh     # 装 uv
uvx -p 3.13 -w pytest==8.4.1 ... pytest /tests/test_outputs.py
```

从 reward 沙箱出口 `astral.sh` 不可达（20s 超时/502）、`pypi.org` / `files.pythonhosted.org` 频繁超时 → `uvx: command not found` → reward.txt 写 0。按 job 统计 10–16% 任务是这样记 0 的；10-31 那天 astral.sh 整段不可达，同模型分数从 0.75 掉到 0.51。另一个变体：runner 把 WORKDIR 写死 `/app`，`tb2-prove-plus-comm`（`workdir=/workspace`、无 `details.json`）永远 exit 1（全平台 112 次 0 解）。

**修法**（源码 `coding_arena/build/reward_mirror/`，官方 4.0.18 runner 原文也在那里可 diff）——**只改 runner 给 `test.sh` 的运行条件，`test.sh` / `test_outputs.py` 一字不动**：

| # | 改动 | 对应故障 |
|---|---|---|
| 1 | 评分前从随包 uv tarball 预装 `uv`/`uvx` 到 `~/.local/bin`，预建 `env` 文件 → `test.sh` 的 `curl astral.sh \| sh` 失败也无所谓 | astral.sh 不可达 |
| 2 | `UV_DEFAULT_INDEX` / `UV_INDEX_URL` / `PIP_INDEX_URL` → `mirrors.aliyun.com/pypi`（沙箱 0.15–2s 稳定可达）；题目显式 `--index-url` 仍优先 | pypi 超时 |
| 3 | WORKDIR：`details.json` → `$ARENA_AGENT_WORKDIR` → `/app` 回退链 | prove-plus-comm |
| 4 (1.0.1) | `UV_HTTP_TIMEOUT` 120→600、`UV_CONCURRENT_DOWNLOADS=4`、`UV_HTTP_RETRIES=6` | 300MB+ nvidia wheel 超时（`mteb-retrieve`） |
| 5 (1.0.1) | `HF_HUB_DISABLE_XET=1` + HF 超时放宽 | `cas-server.xethub.hf.co` 对沙箱 403（`reshard-c4-data`） |
| — | `pytorch-model-cli` 的 env-cache 分支逐字节保留 | 官方唯一已修的题 |

**效果**：pytest 没跑就记 0 的任务 20/178 → 1/178（1.0.0）→ **0**（1.0.1，smoke 8/8）。dsv41-flash 0.753 → **0.831**（同轨迹按条件口径 ≈0.84）。代价：reward 阶段耗时变长（装 torch 的题 7–22 min），仍在 ttl 7200s 内。

### 1.2 `tb2-reward@4.0.0`（TerminalHorizon 用，官方，不改）

TH 的 oracle 是 `python3 /tests/check.py`，400 题抽样 `test.sh` 里 0 处 astral.sh / uvx / pip / apt，`curl` 只打 127.0.0.1；200 条 `log_excerpt` 抽样 bootstrap 失败 0 条。**不需要 mirror**。TH 侧 reward 的问题是另一类——`check.py` 等一个永不退出的进程卡满 7200s——这在 env 层修（§3.2）。

---

## 2. Harness：`cth-terminus-2-stream`

### 2.1 版本谱系（只列与最终使用相关的）

所有版本 = Harbor v0.20.0 的 Terminus-2 + 流式 litellm 补丁，差异只在 `agent.env` 与两处计数代码。spec 存档 `coding_arena/work/harness-specs/terminus-stream-*.json`，运行时包 `coding_arena/build/terminus_pkg_v5/`。

| 版本 | 窗口 / 阈值 | 计数 | max_turns | 历史推理 | 内存 | agent 侧镜像 | 相对上一版的改动 | 用在哪 |
|---|---|---|---|---|---|---|---|---|
| 1.0.0 | 262144 / 65536 | 本地 tokenizer | 无 | 可见 | 8Gi | 无 | = 官方 1.0.12 + `stream=True`（避免 ~900s 静默后 502） | 10-25 前全部评测；**SFT 学生模型评测基线** |
| 1.2.96 | 98304 / 1024 | **usage** | 512 | 可见 | 8Gi | 无 | compaction 触发点扫描选定 96k；usage 计数修正 1.0.0 把历史 reasoning 计入导致的提前压缩 | flash TH 轨迹（`cth-thfull96-flash`、`thl128-96s1-flash`、`csweep-c96k-flash`） |
| 1.4.96 | 同上 | usage | 512 | 可见（**tool-call**） | 8Gi | 无 | `TERMINUS_TOOL_CALL_MODE=function`：主循环带强制 `terminus_action` tool，DashScope DeepSeek 只有这样才渲染历史 `reasoning_content` | dsv41 TH 原始轨迹（`cth-thfull14-dsv41`） |
| 1.5.96 | 同上 | usage | 512 | 可见 | **16Gi** | pypi → aliyun（`PIP_TIMEOUT=60`、`PIP_RETRIES=10`、`UV_HTTP_TIMEOUT=120`） | `TERMINUS_SANDBOX_ENV_<NAME>` 透传机制（pkg_v5）；8→16Gi | TB2.1 0.831 那次 |
| **1.6.96** | 同上 | usage | 512 | 可见 | 16Gi | + `UV_HTTP_TIMEOUT=600`、`UV_CONCURRENT_DOWNLOADS=4`、`UV_HTTP_RETRIES=6`、`HF_HUB_DISABLE_XET=1` + HF 超时 | 大 wheel / HF xet | **API 模型 TB2.1 评测默认；flash TH 采轨迹用这个** |
| **1.7.96** | 同上 | usage | 512 | 可见（tool-call） | 16Gi | 同 1.6.96 | 1.6.96 + `TERMINUS_TOOL_CALL_MODE=function`（spec-only，pkg_v5 本就带该 patch） | **dsv41 TH 重打**（`cth-thredo14`、`cth-thredopol`）；数据语义 = 1.4.96 |
| **1.6.262** | 262144 / 65536 | estimate | 无 | 可见 | 16Gi | 同 1.6.96 | 1.6.96 运行时 + 1.0.0 窗口语义 | **SFT 学生模型 262k 口径评测**（11-06 发布；11-07 终表的 1.0.0 分数是 8Gi 无镜像，新基线统一换到 1.6.262 后再比） |
| 1.8.0-nocomp-{4b,api} | 262144 / 1M，阈值 1024 | usage | 1024 | 可见 | 16Gi | 同 1.6.96 | 无压缩消融 | TB4 试跑，4B 顶满 262k → 引擎 400，无结论 |

版本号规约：第二位 = 行为族（0 旧窗口语义 / 2 触发点扫描 / 4 tool-call / 5-6 沙箱环境透传 + 16Gi / 7 = 6 + tool-call），第三位 = 触发点 k。

### 2.2 两个关键机制

**沙箱环境透传（1.5.96+）**：`agent.env` 里任何 `TERMINUS_SANDBOX_ENV_<NAME>=<v>` 以 `<NAME>=<v>` 进入 Terminus 的 tmux 任务 shell（`tmux new-session -e`）。只影响 agent 的 shell，**reward 阶段不受影响**；命令行显式 `--index-url` 仍优先（`tb2-pypi-server` 用 `localhost:8080` 不受干扰）。为什么要加：沙箱到 `files.pythonhosted.org` 经常 15s ReadTimeout 反复 retry（baseline dsv41 178 条里 24 条），pip 慢会让模型放弃运行时验证直接交卷。

**tool-call 模式（1.4.96 / 1.7.96）**：DashScope 上的 DeepSeek 只有请求带 `tools` 且 `tool_choice != none` 时才把历史 assistant 消息的 `reasoning_content` 渲染进上下文；chat 模式下 1.2.96 的 dsv41 轨迹实际上历史推理被 API 静默丢弃。qwen flash 走 chat 模式即可见历史推理，所以 flash 不需要 tool-call。摘要 subagent 不带 tools。

### 2.3 哪个模型用哪个 harness

| 场景 | harness | 理由 |
|---|---|---|
| dsv41 采 TH 轨迹 | **1.7.96** | 历史推理可见（tool-call）+ 16Gi + 镜像；与原始 1.4.96 数据语义相同，重打数据可直接并入 |
| qwen3.8-flash 采 TH 轨迹 | **1.6.96** | chat 模式已可见历史推理；现有 flash 数据是 1.2.96（8Gi 无镜像）采的，若重采/补采应用 1.6.96 |
| API 模型评 TB2.1 | **1.6.96** | 与 `cth-tb21-full89-mirror@1.0.1` 配对；`submit.sh` 默认 |
| SFT 学生模型 TB2.1 — 262k 口径 | **1.6.262** | 1.0.0 的窗口语义（262144 / 65536 / 无 max_turns / 本地计数——SFT 模型的 template 每轮渲染历史 `<think>`，本地计数对它是准的）+ 16Gi + 镜像。11-07 终表的分数是 1.0.0（8Gi 无镜像），与 1.6.262 不可直接比，新基线需重跑 |
| SFT 学生模型 TB2.1 — 96k compaction 口径 | **1.6.96** | 训练数据按 compaction 段切（`dsv41_history` 段数据、`dsvh_*`、`tree_*`）的模型；或要与 API 教师（也是 1.6.96）同口径并列 |

两类重要的"不能横比"：(a) base 模型的官方 template 丢历史推理、SFT 模型的 template 保留——同一 harness 下两者上下文增速差 2.7×；(b) 1.0.0 vs 1.2.96 压缩触发点不同，对 flash/dsv41 越晚越好，对 base 4B/9B 96k–128k 最优。

---

## 3. Env（TerminalHorizon `cth-th-*`）

1,486 题 ×（原 `@1.1.0` 1294 + 交接批 `@1.2.0` 203）。`cth-thfull14` 的 282 条非 DONE 归因出 160 个坏 env，按根因分四类修法；全部修复逻辑如下（注册脚本 `coding_arena/code/reg/`，状态 `work/state/<ver>/`）。

### 3.1 `@1.3.0`：87 个拍扁镜像（flat）env

**问题**：compose 多服务题被拍成单镜像，sidecar 由镜像 ENTRYPOINT 里的 `/opt/flatten/runtime.py` 启动；但平台沙箱 PID 1 是自己的 wrapper，**镜像 ENTRYPOINT 从不执行** → sidecar 没起 → setup hook 等 `ready.json` 10 min 放弃 → agent 对着不存在的 `http://api:8080` 跑满 5h。三个 job、两模型、三 harness 全是 99% TIMEOUT。

**修法**（`code/reg/th_flat_setup_hook.sh`，25 轮 diag 验证）：不走 `runtime.py` 的私网方案（`/proc/sys/net/ipv4/ip_forward` 在 ARCA 下只读；绕过后建的 veth/netns 私网 `connect()` 两个方向 EPERM；`unshare --fork` 子进程继承 hook stdout 让平台等不到 hook 结束）。改为 setup hook 直接 `chroot /opt/svc/<name>` 在**主 netns** 起各服务 argv（`Popen(preexec_fn=chroot+setuid, stdout=file, close_fds=True)`），`/etc/hosts` 把服务名指到 `127.0.0.1`，自己写 `ready.json`。

**效果**：重打 117 DONE / 58 solved（修前任何模型 0 解）。副作用：失去 compose 网络隔离；同端口服务会冲突（87 题按 plan.json 核过无冲突）；4 题要求 `exec_prefix`（命令经 `main_exec.py`）本 hook 没做。

### 3.2 `@1.4.0`：8 个 reward-hang env + 2 个 DNS env

**reward-hang**（ROS2 / robot / tokio / etcd 题）：agent 正常 complete，但 `check.py` / `colcon build` / `cargo test` 等一个永不退出的进程，reward 卡满 7200s ARCA exec 超时 → 任务 TIMEOUT，分数拿不到。
修法：oracle 的 `tests/test.sh` 换成 `timeout -k 60 1500 bash /tests/test_orig.sh`，原脚本逐字节保留为 `test_orig.sh`（资产 `/storage/.../terminal_horizon/env_assets_v140/`，脚本 `code/reg/register_140_fixes.sh`）。

**DNS**（sip-proxy、consul 题）：题目自己的 bootstrap 把 `/etc/resolv.conf` 写成 `nameserver 127.0.0.1`（本地 dnsmasq / consul），harness 之后解析不了 `envarena-prod.alipay.com`，LLM 调用全部 Connection error（纠正了之前"镜像 resolv 问题"的判断）。
修法：setup hook 先 `getent ahostsv4 envarena-prod.alipay.com`（失败回退 python `socket.getaddrinfo`）把网关 IP 钉进 `/etc/hosts`，再跑原 envd hook。

效果：10 env 重打 20 DONE / 1 solved（修前 reward 必卡死或 LLM 不可达）。

### 3.3 不改 env、靠 harness 修：24 个 tmux-killed env

多服务题（k3s、minio、ROS、wireguard、slurm）在 8Gi 下 OOM killer 杀 tmux server（`send-keys rc=-15/-9`）。harness 1.7.96 的 16Gi 解决；原版本 env 直接重打，45 DONE / 4 solved。

### 3.4 暂剔除：18 CA-broken + 1 oracle 过大

- 18 个镜像缺 CA 证书，ARCA 装 envd 时 `curl (60)`，envd 不监听，所有任务必失败。集群无 docker / registry 访问，修不了。外部 docker 机套件 `coding_arena/build/rt3_ca/`：`Dockerfile.rt3`（加 `ca-certificates`）+ `build_rt3.sh`（校验 TLS / ENTRYPOINT 不变 / root 后 push `-rt3`）+ 18 镜像清单（取自平台实际 spec）。重建后注册 `@1.4.0` 并补 benchmark。
- `recovering-a-partially-written-audiovisual-container` oracle 83 MB 超 shared-sandbox 投递上限，待瘦身。
- 清单 `work/manifests/th14_excluded_envs.txt`。

### 3.5 修不了的：eperm-net 类题

k8s / BGP / NFS / 多节点网络题在沙箱内自建私网（kind、veth、netns），ARCA 策略拒绝连入（`connect: operation not permitted`）。这不是 env 或 harness 能修的——1.7.96 重打这类 env 后污染率仍 35%（21/60 条 eperm）。只能在数据层剔除（`audit_env_pollution.py` 的 eperm-net 检测器），或等平台放开私网策略。

### 3.6 通用 setup hook（所有版本都带）

Debian 13 镜像上 ARCA 的 installer 在 `arca_envd` 二进制写完前就 nohup 启动，`:11452` 不监听，`shared_env` reward 502。hook 检查 `/proc/net/tcp` 没监听就重启 envd，是 `make_assets.py` 给每个 env 的默认 `hooks.setup_script`；1.3.0 / 1.4.0 的修复都是在它前面加内容。

### 3.7 Benchmark 对照

| key | env | 说明 |
|---|---|---|
| `cth-th-full@1.0.0` | 1486 | 原始（87 flat 用坏的 1.2.0） |
| **`cth-th-full@1.1.0`** | 1486 | 87 flat 换 `@1.3.0`，其余不变；**TH 全量采轨迹用这个**（reward-hang/DNS 的 `@1.4.0` 尚未并入，下次发 1.2.0 时一并换） |
| `cth-th-redo14@1.0.0` | 141 | 160 坏 env − 19 剔除；flat 1.3.0、reward-hang/DNS 1.4.0、其余原版本 |
| `cth-th-redo-pollution@1.0.0` | 50 | DONE 轨迹含环境污染的 env |
| `cth-tb21-full89-mirror@1.0.1` | 89 | TB2.1 唯一口径 |

---

## 4. 这套配置下的数据产出（TH dsv41）

原始 2,690 DONE → 修 env + 换 harness 重打两次（+182 +79）→ 剔 101 条环境污染 → 2,796 → Opus 4.8 三准则 judge → **2,541 条 KEEP**（1,251 solved，1,349 env），环境污染零残留。详见 [trajectory_collection](trajectory_collection.md) Part C。

qwen3.8-flash 的 TH 数据（`qwen3-8-flash-history/`，2,732 条，harness 1.2.96，8Gi 无镜像）**没有**做过同样的污染审计和 judge；若要用，先跑 `audit_env_pollution.py` + `llm_judge.py`，缺口部分用 1.6.96 补采。

---

## 5. 本仓库内的资产

- 五个版本的参数与平台 spec 可在[配置目录](../arena_assets/harness/configs/)和[原始 spec](../arena_assets/legacy/harness/specs/)逐项核对。
- [TB2.1 reward](../arena_assets/reward/tbench_reward_runner_mirror.sh)、[环境修复](../arena_assets/env/fixes/)、[历史注册脚本](../arena_assets/legacy/env/reg/)和[任务清单](../arena_assets/tasks/)保留了评测条件。
- 原集群的 runtime 二进制、镜像与训练数据的可用范围见[资产说明](../arena_assets/README.md)。原 Arena 任务提交使用[平台脚本](../arena_assets/legacy/scripts/)。

