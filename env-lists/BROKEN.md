# 坏掉的 env（TerminalHorizon，2026-10-21 16:05）

数据来源：`/storage/openpsi/users/chen/output/<job>/tasks/*/s0/task.json` + `th500_stats/attempts.json` + `arena job log/task explain`。
`broken_envs.txt` 只收 A 类（环境本身坏）；B 类是模型/执行行为，env 本身可用。

## A. 环境本身坏（4 个，重试/换模型都救不回来）

| # | env | 现象 | 根因 | 证据 | 建议 |
|---|---|---|---|---|---|
| 1 | `cth-th-multi-model-gpu-inference-capacity-admission-tuning` | `HARNESS_FAILED`，agent 阶段都没开始 | 平台创建沙箱超时：`lastManagerStatus=Pending`（镜像要的资源/设备拿不到） | 3 模型 × 2 attempts = **6/6 失败**，`cls=sandbox_pending` | 剔除；或找平台确认 GPU/设备配额 |
| 2 | `cth-th-onboard-a-heterogeneous-vendor-fleet-into-open-rmf` | `TIMEOUT`，agent 正常结束但评分卡住 | reward 阶段 `tests/test.sh` 不返回，`POST /terminal/exec_command` 2 h `context deadline exceeded` | **6/6 TIMEOUT**，`cls=timeout_reward_exec_2h` | 改 `tests/test.sh`（去掉无限等待/加超时）后发新版本；否则剔除 |
| 3 | `cth-th-scan-dft-integration-and-atpg-coverage-closure` | `TIMEOUT` 跑满 5 h，harness 日志几乎为空 | 镜像缺 `/etc/ssl/certs/ca-certificates.crt`，`curl: (77)` 装不上 `arca_envd` → 平台读不到日志也不判死 | `cls=timeout_curl77_ca_envd0`，3 模型 5/6 次 TIMEOUT | 修镜像（装 `ca-certificates curl`）；否则剔除 |
| 4 | `cth-th-move-a-saas-tenant-to-a-compliant-region` | `HARNESS_FAILED`（27b 已复现 2/2） | 题目工作区有 `/app/platform.py`，harness runner 在 `/app` 启动时 `import platform` 命中它 → 它 `import psycopg2` 失败 → runner 崩溃（stdlib shadowing） | harness.log：`File "/app/platform.py", line 22, in <module> import psycopg2 / ModuleNotFoundError: No module named 'psycopg2'` | 改 harness 启动（`PYTHONSAFEPATH=1`、换 cwd、或把 site-packages 置于 sys.path 首位）；否则该题只能剔除 |

## B. 环境没问题，是模型/执行行为导致（3 个，别的模型能过）

| # | env | 现象 | 根因 | 证据 |
|---|---|---|---|---|
| 5 | `cth-th-adaptive-relief-allocation-and-dispatch-through-uncertain-corridors` | flash `HARNESS_FAILED` ×2 | 镜像默认用户非 root（`USER` 非空），ARCA 偶发以 uid=1000 启动 → `install arca egress gateway ca requires root` → envd 起不来 | `cls=uid1000_nonroot`；同为 27b/dsv41 时 **DONE** |
| 6 | `cth-th-cross-backend-inference-release-qualification` | dsv41 `HARNESS_FAILED` ×2 | 模型在交互 shell 顶层敲 `set -e`，后续非零返回让 bash 退出、tmux server 消失，`send-keys` 失败 | `cls=tmux_send_keys`；flash/27b **DONE** |
| 7 | `cth-th-repair-cross-hop-http-2-streaming-backpressure-in-a-reverse-proxy` | dsv41 `HARNESS_FAILED` ×2 | 同上 | `cls=tmux_send_keys`；flash/27b **DONE** |

B 类对应两个已知系统性风险：
- **非 root 镜像**（500 批次 `-rt1` 里 30 题、本批 69 题的 Dockerfile 以非 root USER 结尾）：约 1/4 attempt 以 uid=1000 启动，`max_retries=1` 大多能兜住，全挂就是 A 类 #5 这种。
- **dsv41 的 `set -e` 习惯**：500 批次 218 条轨迹里 11 条触发了 tmux 死亡；根治要改 harness（过滤顶层 `set -e` 或 `remain-on-exit on`）。

## 统计口径

- 已注册 env 1294（500 旧 + 794 新）
- `usable`（有 ≥1 次 DONE）：526
- `pending`（已注册但还没跑出结果）：764
- `broken_known`（A 类）：4
- 另有 59 题 `needs_privileges`（privileged/cap_add/devices…）、6 题 `single_service`（compose 只有非 main 服务，当前用主镜像占位），均**未**判为坏：pilot 里 `behavioral-detection-engineering-from-controlled-executable-analysis` 两跑都是 1 分，说明 ARCA 能满足这类需求。