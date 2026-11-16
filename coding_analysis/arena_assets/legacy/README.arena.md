# arena_assets — Reward / Harness / Env 最终版本的可迁移副本

> 157 节点的提交和拉取命令见[评测指南](../../evaluation.md)。下文的 `submit.sh` 和 `finalize_judge.sh` 是原平台脚本，分别依赖原目录的 `lib/common.sh` 和旧集群的 judge 路径。

> 此页保留原 Arena 平台的资产布局。当前仓库的资源范围及运行时包依赖见[资产说明](../README.md)。

原平台发布流程依赖 arena CLI、平台权限与原始运行时二进制。下述命令与目录布局按原发布包记录。
配置选型的理由见同级 [arena_verifier](../../methods/arena_verifier.md)；本 README 只讲"怎么用"。

```
arena_assets/
├── harness/
│   ├── specs/terminus-stream-{1.0.0,1.2.96,1.6.96,1.6.262,1.7.96}.json   harness spec（发布时 --file）
│   └── runtime_src/            pkg_v5 运行时：runner（bash 入口）、runtime-manifest.json、app/（打补丁的 Harbor 源码）
├── reward/
│   ├── reward-spec-1.0.1.json              cth-tb2-reward-mirror@1.0.1 spec
│   ├── tbench_reward_runner_mirror.sh      修复版 runner（spec 的 files[0]）
│   └── official_runner_4.0.18.sh           官方 runner 原文，供 diff
├── env/
│   ├── hooks/th_flat_setup_hook.sh         flat env @1.3.0 的 setup hook
│   ├── reg/{make_assets.py, register_batch.sh, register_flat_1.3.0.sh, register_140_fixes.sh}   注册脚本
│   ├── specs-1.4.0/*.json                  10 个 @1.4.0 已发布 spec（reward-hang 8 + DNS 2）
│   ├── spec-flat-example.json              flat 题注册前的 spec 模板（1.2.0 版 hook，对照用）
│   ├── rt3_ca/                             18 个 CA 坏镜像的外部重建套件
│   └── manifests/                          坏 env / 剔除 / 污染 env 清单
├── benchmarks/*.json                       cth-tb21-full89-mirror@1.0.1、cth-th-full@1.1.0、redo14、redo-pollution
└── scripts/                                submit.sh、pull_trajectories.sh、failure_audit.py、
                                            audit_env_pollution.py、build_multi_sft.py、build_dsv41_merged.py、
                                            llm_judge.py、apply_judge.py、finalize_judge.sh、common.sh
```

---

## 1. 该用哪一套

| 用途 | Benchmark | Reward | Harness | 备注 |
|---|---|---|---|---|
| TB2.1 评 API 模型（dsv41 / qwen flash） | `cth-tb21-full89-mirror@1.0.1` | `cth-tb2-reward-mirror@1.0.1` | `cth-terminus-2-stream@1.6.96` | `submit.sh` 默认 |
| **TB2.1 评 SFT 学生模型 — 262k 整窗口口径** | 同上 | 同上 | **`1.6.262`** | 1.0.0 的窗口语义（262144 / 65536 阈值 / 无 max_turns / 本地计数）+ 16Gi + agent 侧镜像。与训练数据（整轨迹 262k）口径一致。**已发布的 1.0.0 分数（11-07 终表）是 8Gi 无镜像版本，与 1.6.262 不可直接比；新基线统一用 1.6.262** |
| **TB2.1 评 SFT 学生模型 — 96k compaction 口径** | 同上 | 同上 | **`1.6.96`** | 96k 触发 / 1024 阈值 / 512 步 / usage 计数 + 16Gi + 镜像。用于：训练数据按 compaction 段切分的模型（`dsv41_history` 段数据、`dsvh_*`、`tree_*` 系列），以及与 API 教师（也是 1.6.96）同口径对比 |
| TH 采 dsv41 轨迹（历史推理可见） | `cth-th-full@1.1.0` / `cth-th-redo14@1.0.0` | `tb2-reward@4.0.0`（官方） | **`1.7.96`** | = 1.6.96 + `TERMINUS_TOOL_CALL_MODE=function`；DashScope DeepSeek 只在 tool-call 下渲染历史 reasoning |
| TH 采 qwen3.8-flash 轨迹 | 同上 | 同上 | **`1.6.96`** | chat 模式已可见历史推理 |

学生模型评测选哪个口径的规则：**看训练数据怎么切的**。整轨迹 262k 训练（arm A `dsv41_json262k`、`dsv41_merged_judged-keep`）→ 1.6.262；按 compaction 段训练或要与教师同口径 → 1.6.96。两套都跑时同一 checkpoint 两个分数并列报告，不混。

---

## 2. 重新发布（迁移到新账号 / 新平台实例）

前置：`arena` CLI 可用、`~/.envarena/config.toml` 有 token；所有命令都是 `create → version-create → publish` 三步，已发布版本不可变。

### 2.1 Harness 运行时上传

运行时 = `runtime_src/runner` + `runtime-manifest.json` + 四个目录 `app/ python/ site-packages/ fallback-bin/`。本目录只带了 **源码可读的部分**（`app/`、runner、manifest，6 MB）；`python/python313.tar.gz`（31 MB）、`site-packages/py313.tar.gz`（52 MB）、`fallback-bin/`（1.5 MB tmux）是平台上已有的上传物，sha256 在 `runtime-manifest.json` 里。完整包在本集群 `coding_arena/build/terminus_pkg_v5/pkg/`；迁移时整目录 `arena upload <dir> --unpack` 后把返回 uri 填进 spec 的 `runtime.uri`。

如需从源码重打包：`app/` 打成 `app/app.tar.gz`（spec 里 `metadata.app_archive_sha256` 要同步更新）。pkg_v5 是**超集运行时**，五个 harness 版本共用它，差异只在 spec 的 `agent.env`：

| spec | 行为由这些 env 决定 |
|---|---|
| 1.0.0 | `CONTEXT_LIMIT=262144 THRESHOLD=65536`，无 `MAX_TURNS`，无 `TOKEN_COUNT_MODE`（= estimate）；原 uri 是 10-18 的旧包，用 pkg_v5 + 这组 env 行为等价 |
| 1.2.96 | `CONTEXT_LIMIT=98304 THRESHOLD=1024 MAX_TURNS=512 TOKEN_COUNT_MODE=usage` |
| 1.6.96 | 1.2.96 + `TERMINUS_SANDBOX_ENV_{PIP_INDEX_URL,PIP_TIMEOUT,PIP_RETRIES,UV_*,HF_HUB_*}` + `resource_spec.memory=16Gi` |
| 1.6.262 | 1.0.0 的 env + 1.6.96 的镜像/16Gi + `TOKEN_COUNT_MODE=estimate` |
| 1.7.96 | 1.6.96 + `TERMINUS_TOOL_CALL_MODE=function` |

```bash
A="env -u ANT_COOKIE arena"
$A harness create --key cth-terminus-2-stream --name "Terminus 2 (streaming, cth)" --json
for v in 1.0.0 1.2.96 1.6.96 1.6.262 1.7.96; do
  $A harness version-create cth-terminus-2-stream $v --file harness/specs/terminus-stream-$v.json --json
  $A harness publish cth-terminus-2-stream $v --yes --json
done
```

### 2.2 Reward

```bash
$A upload reward/tbench_reward_runner_mirror.sh --json      # -> uri 填进 reward-spec-1.0.1.json files[0].uri
# files[1] uv tarball、files[2] python313.tar.gz 复用平台已有 uri（spec 里已写）；换平台时需重新上传同名文件
$A reward create --key cth-tb2-reward-mirror --name "TB2 reward, mirror runner (cth)" --json
$A reward version-create cth-tb2-reward-mirror 1.0.1 --file reward/reward-spec-1.0.1.json --json
$A reward publish cth-tb2-reward-mirror 1.0.1 --yes --json
```

runner 做了什么（官方 `test.sh` 不动）：预装 uv/uvx → `UV_*_INDEX`/`PIP_INDEX_URL` 指 aliyun → `UV_HTTP_TIMEOUT=600`、并发 4、重试 6 → `HF_HUB_DISABLE_XET=1` → WORKDIR 回退链 `details.json → $ARENA_AGENT_WORKDIR → /app` → `bash /tests/test.sh`。`diff reward/official_runner_4.0.18.sh reward/tbench_reward_runner_mirror.sh` 可见全部改动。

### 2.3 Env（TerminalHorizon）

通用流程（`env/reg/make_assets.py` 生成 spec + assets → `register_batch.sh` 上传/发布，幂等，`state/<ver>/<task>.ok` 为标记）见 `coding_arena/docs/guide/env/terminal-horizon.md`。四类修复版本：

| 版本 | env 数 | 怎么生成 |
|---|---|---|
| `@1.3.0` flat | 87 | `env/reg/register_flat_1.3.0.sh`（`DRY_RUN=1` 先看补丁后的 spec）：把 `make_assets.py` 产出 spec 的 `hooks.setup_script` 整个替换成 `env/hooks/th_flat_setup_hook.sh`（原 hook 等 `ready.json` 10 min 的逻辑废弃），其余字段不变；清单 = `handoff_meta.json` 里 `flat: true` 的 87 题 |
| `@1.4.0` reward-hang | 8 | `env/reg/register_140_fixes.sh`：oracle 目录里 `tests/test.sh` → `test_orig.sh`，新 `test.sh` = `timeout -k 60 1500 bash /tests/test_orig.sh`，重新上传 oracle、改 spec `oracle.uri` |
| `@1.4.0` DNS | 2 | 同脚本：spec 的 `hooks.setup_script` 前面拼上 `/etc/hosts` 钉网关 IP 的片段（getent → python socket 回退） |
| `@1.4.0` CA-broken | 18 | 外部 docker 机 `bash env/rt3_ca/build_rt3.sh env/rt3_ca/images_ca_broken.txt` → `images_rt3.json` → `IMAGES_FILE=... make_assets.py` → `register_batch.sh 1.4.0 cth-th-ca 1.0.0 <tasks>`；**尚未执行** |

`env/specs-1.4.0/` 是 10 个已发布 spec 的原样副本（`oracle.uri` / `query.uri` 指向本平台上传物，换平台需重传后改 uri）。

### 2.4 Benchmark

```bash
for b in benchmarks/*.json; do echo $b; done
$A benchmark create --key cth-tb21-full89-mirror --name "TB2.1 89 tasks, mirror reward (cth)" --json
$A benchmark version-create cth-tb21-full89-mirror 1.0.1 --file benchmarks/bench-cth-tb21-full89-mirror-1.0.1.json --json
$A benchmark publish cth-tb21-full89-mirror 1.0.1 --yes --json
# cth-th-full@1.1.0 / cth-th-redo14@1.0.0 / cth-th-redo-pollution@1.0.0 同理
```

benchmark 只是 env_refs + reward_ref 的列表；env/reward 必须先存在。

---

## 3. 跑

```bash
source scripts/common.sh     # arena 封装、token、日志
# TB2.1 评 API 模型
bash scripts/submit.sh --model dsv41-flash-1017-max-chen@1.0.0 --benchmark cth-tb21-full89-mirror@1.0.1 --samples 2 --parallel 32
# TB2.1 评 SFT 学生 — 两个口径
bash scripts/submit.sh --model <student@ver> --benchmark cth-tb21-full89-mirror@1.0.1 --harness cth-terminus-2-stream@1.6.262 --samples 2 --job-id cth-tb21m-<name>-262k
bash scripts/submit.sh --model <student@ver> --benchmark cth-tb21-full89-mirror@1.0.1 --harness cth-terminus-2-stream@1.6.96  --samples 2 --job-id cth-tb21m-<name>-96k
# TH 采轨迹
bash scripts/submit.sh --model dsv41-flash-1017-max-chen@1.0.0 --benchmark cth-th-full@1.1.0 --harness cth-terminus-2-stream@1.7.96 --samples 2 --parallel 128
bash scripts/submit.sh --model qwen3-8-flash-xhigh-chen@1.0.0  --benchmark cth-th-full@1.1.0 --harness cth-terminus-2-stream@1.6.96 --samples 2 --parallel 128
```

`submit.sh` 会顺带起 `pull_trajectories.sh` 把轨迹落盘到 `$OUT_ROOT/<job>/tasks/<env>/s<i>/`。注意 dsv41 的 `--model-ref` 手敲会 404（CLI 转义问题），从已有 `job.json` 读出来放变量再传。

学生模型的 serving 配置（与 11-07 终表一致）：sglang、tp1、SSM fp32、EAGLE MTP、T=1 / top_p 0.95 / top_k 20 / xhigh、ctx 262144。

---

## 4. 轨迹 → SFT 数据

```bash
# 1. 整轨迹 jsonl（--linear 不切 compaction 段；去掉 --linear 则一段一条）
python3 scripts/build_multi_sft.py --out-dir <out> --dataset-name <name> --done-only --linear <job_dir> [...]
# 2. 环境污染审计（eperm-net / oom / git-net / pip-net / gave-up+net）
python3 scripts/audit_env_pollution.py --base <out>/<name> --out /tmp/<name>_pollution   # -> _traj.json
# 3. 多 job 合并，剔污染，后来的重打行覆盖同 (env, sample)
python3 scripts/build_dsv41_merged.py --polluted <job>=<...>_traj.json [...] --extra <name2>
# 4. Opus 4.8 三准则 judge（需 MATRIXLLM_API_KEY；断点续跑；1 条轨迹 1 请求，上限 1.9M 字符）
python3 scripts/llm_judge.py --base <merged> --out <merged>/judge_opus48 --workers 8 --retries 4
bash   scripts/finalize_judge.sh      # 重试 JUDGE_ERROR 两轮 + apply_judge.py 出 -keep.jsonl
```

`llm_judge.py` 里 `API` / `MODEL` 常量指向 matrixllm 的 `claude-opus-4-8`；换判官改这两行。prompt 是 TerminalHorizon App J.1 原文 + 一行 compaction 说明，不要改准则。

---

## 5. 版本一致性校验

迁移后跑一遍 smoke 再上量：

1. `cth-tb21-rewardsmoke2`（4 题，reward 阶段装 torch 的题）：`tbench_test.log` 里 4/4 有 `test session starts` → reward-mirror 生效。
2. `cth-tb21-pipsmoke`（4 题）：轨迹里 `Looking in indexes: https://mirrors.aliyun.com/pypi/simple/`、0 次 `ReadTimeoutError` → harness 镜像透传生效。
3. flat 题 `cth-th-close-the-loop-on-a-serialized-warranty-repair@1.3.0` 单题：harness.log 有 `[th-flat] started api` 且 `ready=started` → hook 生效。
4. dsv41 + 1.7.96 任一题：相邻两步 `prompt_tokens` 增量 ≈ 上一步 reasoning + content + 终端输出（而不是只有后两项）→ tool-call 历史推理可见。
5. `failure_audit.py <job>` 的 `reward bootstrap fail` 列为 0。
