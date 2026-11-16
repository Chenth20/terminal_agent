# tb_local：在 slurm 池节点上离线复刻 arena 的 TB2.1 评测（构建 → 运行 → 调试全流程）

> 原始集群路径、平台作业和文内脚本命令按实验发生时记录。当前服务器入口和可复现范围见[交接文档](../HANDOFF.md)。

整理日期：2026-11-09。所有脚本在 `/storage/openpsi/users/chen/tb_local/`（下文 `$T`）。
目标是**不经过 EnvArena 平台**，在自己的 slurm 节点上用 singularity 跑出和 arena 同口径的 Terminal-Bench 2.1 分数：
同一份 harness（`cth-terminus-2-stream@1.6.96` 的 `app.tar.gz`，sha256 `beb1801d…`）、同一份 reward runner（`cth-tb2-reward-mirror@1.0.1`）、
同一批 89 个官方 Dockerfile 建出来的任务镜像、同样的资源规格（4 核 / 16Gi / ttl 14400 s / 2 样本）。
结果分析与本地-arena 对比见同目录 [tb21_verifier_audit.md §8](tb21_verifier_audit.md)。

```
tb_local/
├── tasks/<task>/            89 题官方 task 目录（environment/Dockerfile、instruction.md、tests/、solution/、task.toml）
├── cache/                   离线物料：base rootfs tar、python-build-standalone、debian OCI 层、无镜像可用的大文件（doom1.wad、win311.img）
├── sif/tb2-<task>.sif       89 个任务镜像（49 GB，最大 mteb-retrieve 6.4 GB）
├── harness/                 arena 的 harness 原件：bin/{runner,app/app.tar.gz,python/python313.tar.gz,site-packages/py313.tar.gz,fallback-bin}
├── reward/                  tbench_reward_runner_mirror.sh + uv-dist/ + python313.tar.gz + mnist/ + files/{mobile_sam.pt,bin/curl}（本地新增，§3.4）
├── build_base.sh            §1.2  基础 rootfs tar
├── build_task.py            §1.3  Dockerfile → sandbox → sif（单题）
├── build_all.sh / run_one.sh / build_chess.sh / verify_sifs.sh / build_report.tsv
├── egress_proxy.py          §2.2  宿主侧 allow-list 代理 + 容器侧桥
├── run_task.sh              §2.3  跑一条 (task, sample)：隔离、harness、oracle 投递、reward、result.json
├── run_batch.sh / batch_step.sh   §2.4  一个 slurm 作业上的多槽批跑（可多节点并发、可断点续跑）
├── progress.sh / summarize.py     §2.5  进度与汇总
├── sweep_until_done.sh      §2.4  循环调用 run_batch.sh 直到 178/178（作废的样本自动补跑）
├── hold.sh / hold_pool.sh   占节点的 sbatch（build 节点 32c/128G 3h；pool 节点 sleep infinity 24h）
├── probe.sh netprobe*.sh isoprobe*.sh cgprobe.sh smoke_*.sh   §3 调试时用的探针，保留作记录
└── runs/<run>/<task>/s<k>/  每条评测的产物（§2.6）
```

---

## 0. 一页流程

```bash
# 0) 占节点（build 节点要 32c/128G/本地 /tmp ≥ 200G；pool 节点要多核，CPU-only）
sbatch hold.sh            # -> JOBID_BUILD
sbatch hold_pool.sh       # -> JOBID_POOL（可以多个；写进 pool_jobs.txt）

# 1) 基础 rootfs（每种 FROM 一次）
for b in ubuntu2404 bookworm trixie bullseye py313 py311 py311full py310; do srun --mpi=pmi2 --jobid=$JOBID_BUILD --overlap bash build_base.sh $b; done

# 2) 89 个任务镜像（并行 4，断点续建：已有 sif 的跳过）
JOBID=$JOBID_BUILD PAR=4 bash build_all.sh      # 失败的看 logs/build-<task>.log 与 build_report.tsv 的 note
bash build_chess.sh                              # chess-best-move 字体源不可达，单独手搓
srun --jobid=$JOBID_BUILD --overlap bash verify_sifs.sh

# 3) 冒烟（先 plumbing：假模型 → harness 快速失败、reward 仍要跑 pytest 给 0 分；再真模型一题）
bash smoke_plumb.sh && bash smoke_real.sh

# 4) 正式批跑（每个 pool 作业起一个 detached batch；同一 run 目录可被多个节点同时消费）
bash batch_step.sh $JOBID_POOL dsv41-local deepseek-v4.1-flash-1017 https://dashscope.aliyuncs.com 2 auto
#    API 模型需要 API_KEY=... LLM_PATH=/compatible-mode/v1；本地 vLLM 直接给 host:port，LLM_PATH 默认 /v1
bash watch.sh   # = watch -n 60 bash progress.sh

# 5) 汇总
python3 summarize.py runs/dsv41-local      # -> summary.tsv + pass_at_all
```

---

## 1. 构建任务镜像

### 1.1 约束

- 节点上只有 apptainer 1.3.2（无 docker/podman），root 运行，`allow setuid = yes`，cgroup v1。
- 公网只有部分可达：`mirrors.aliyun.com`（apt / pypi / debian / ubuntu）、`mirrors.tuna.tsinghua.edu.cn`（alpine ISO）、
  `registry.npmmirror.com`（python-build-standalone）、`hf-mirror.com`、`github.com` / `raw.githubusercontent.com` /
  `objects.githubusercontent.com`、`download.pytorch.org`、`storage.googleapis.com`。
  **不可达**：`deb.debian.org`、`pypi.org` / `files.pythonhosted.org`、`astral.sh`、`archive.org`、`distro.ibiblio.org`、
  `openaipublic.blob.core.windows.net`、`crates.io`、`proxy.golang.org`、`notofonts.github.io`、`ossci-datasets.s3.amazonaws.com`。
  探针：`netprobe2.sh`（一行一个 URL 的 `curl -skI`）。
- 构建节点的 `/tmp` 是本地盘（沙箱在本地解包；共享存储上 `tar -x` 几万个小文件慢 10 倍以上）。`build_task.py` 要求 `/tmp` 空闲 ≥ 40 GB。

### 1.2 基础 rootfs（`build_base.sh <name>` → `cache/base-<name>.tar`）

| FROM（Dockerfile） | base 名 | 来源 |
|---|---|---|
| `ubuntu:24.04` | `ubuntu2404` | `cache/ubuntu-base-24.04.3-base-amd64.tar.gz`（Ubuntu 官方 base tarball），apt 源改 aliyun |
| `debian:bookworm-slim` / `debian:13.0-slim` | `bookworm` / `trixie` | `cache/deb-art/<name>/slim/oci/blobs/rootfs.tar.gz`（debuerreotype 的 OCI 层） |
| `debian:bullseye-slim` | `bullseye` | 没有离线层 → 在 bookworm 沙箱里 `debootstrap --variant=minbase`。apptainer 禁止 `mknod`，把 debootstrap 里的 `mknod` 全部 `true mknod` 掉（运行时 singularity 会绑 `/dev`） |
| `python:3.13-slim-bookworm` 等 | `py313` / `py311` / `py310` | bookworm + python-build-standalone `install_only` 包解到 `/usr/local`，布局与官方镜像一致（`/usr/local/bin/python3.13`、pip 在位）。`python:3.11`（非 slim）= `py311full`，额外装 build-essential 等 |

每个 base 之后统一 `apt-get install ca-certificates netbase tzdata`、设 `LANG=C.UTF-8`（写进 `.singularity.d/env/91-tb2base.sh`，
singularity build 会保留）。bullseye 的 `-security` 源在 aliyun 上索引过期（pool 404）→ 省略。

**已知与官方镜像的差异**：python-build-standalone 是 clang 编的，`sysconfig` 里 `CC=clang -pthread`；官方 `python:*-slim` 是 gcc。
这会让 `setup.py build_ext` 在没有 clang 的镜像里失败（`feal-differential-cryptanalysis`、`portfolio-optimization` 等都撞过）。
**不在 base 里修**（避免重建 42 个 sif），而是在运行时 `inside.sh` 开头把 `_sysconfigdata__*.py` 里的 `clang`/`clang++` sed 成 `gcc`/`g++`（§2.3）。

### 1.3 单题构建（`build_task.py <task>`，由 `run_one.sh` 在 build 节点上调用）

Dockerfile → singularity 的翻译规则（只用 python3.6 stdlib，因为 build 节点 python 很老）：

1. 解析 `FROM / RUN / COPY / WORKDIR / ENV / ARG / CMD`，保留反斜杠续行（bash 和 docker 的 `/bin/sh -c` 语义一致），丢注释行。
   支持 TB2 里出现的两种多阶段：`COPY --from=ghcr.io/astral-sh/uv:0.8.x /uv /uvx`（替换成从 aliyun pypi 装 `uv` wheel）和 `COPY --from=<stage>`。
2. `FROM` 查 `BASE_MAP` → 解 `cache/base-<name>.tar` 到 `/tmp/tb2-sbx-<task>`；`COPY` 直接拷进沙箱；`WORKDIR` 记录并 `mkdir -p`；`ENV/ARG` 展开 `$VAR`。
3. 所有 `RUN` 拼成**一个** driver 脚本，一次 `singularity exec --writable --no-home --containall -B /etc/resolv.conf` 执行；
   每步保留 docker 的"命令串退出码"语义，非 0 中止并打 `##### RUN[i] FAILED rc=`。RUN 的环境固定为 `RUN_ENV`：
   `DEBIAN_FRONTEND=noninteractive`、pip/uv 索引 → aliyun、`UV_PYTHON_INSTALL_MIRROR` → npmmirror、`HF_ENDPOINT` → hf-mirror、`HF_HUB_DISABLE_XET=1`。
4. RUN 文本做字面替换（`URL_SUBS`，每条都写进 `build_report.tsv` 的 note）：`deb.debian.org` → aliyun；`dl-cdn.alpinelinux.org` → tuna
   （aliyun 的 ISO 会 302 到集群内不可达的 `iso-osm.mirrors.aliyuncs.com`）；`huggingface.co` → hf-mirror；
   `distro.ibiblio.org/.../doom1.wad` → `file:///.tb2-files/doom1.wad`（从 debian `doom-wad-shareware` deb 里抠出的同一 v1.9 shareware wad，md5 `f0cefca4…`）；
   `archive.org/.../win311.img` → `cp /.tb2-files/win311.img`（github `ChenChiShui/tb2-assets` 4×64 MiB 分片重组，sha256 与题目钉死的一致）；
   `TheMikeMerrill/personal-site`（404）→ `wangsikuan/personal-site`（同一 commit `d7d3e4ba`）。`cache/files/` 以 ro 绑到 `/.tb2-files`。
   **坑（11-09 20:00 发现）**：单文件 `COPY interp.py /app` 且 `/app` 已由 `WORKDIR` 建好时，第一版把 staged 文件（名字带 `N_` 前缀）直接
   `cp -a` 到 `/app`，得到 `/app/0_interp.py`。docker 语义是"目标是已存在目录 → 以原名拷进去"。已改为 `if [ -d dst ]; then cp src dst/<原名>; else … fi`；
   受影响的只有 `schemelike-metacircular-eval`、`sqlite-db-truncate`（grep 全部 build log 的 `cp -a /.tb2-copy/N/N_x /<dir>` 形态确认），已重建。
   **复查新镜像的方法**：`singularity exec sif/tb2-<t>.sif ls -la <WORKDIR>` 对照 Dockerfile 的 COPY 列表，文件名不能带 `N_` 前缀。
5. `singularity build -F sif/tb2-<task>.sif <sandbox>` → `verify`（exec 进去 `cd WORKDIR && ls`）→ 追加一行到 `build_report.tsv`
   （`task status base_image sif_size_MB seconds note`，带文件锁）。
6. 失败分类：RUN 输出里出现 `UNREACHABLE_HOSTS` 之一、wget rc 4/5/8、github 认证提示 → `FAILED-NET`；其余 `FAILED-BUILD`。
   `--keep` 保留沙箱供手工进去看。

`build_all.sh`：对 `tasks/` 下没有 sif 的题（`FORCE=1` 全量重建）起 `xargs -P $PAR` 个 `srun --overlap` 到 build 作业。
89 题累计 3541 s，p50 14 s，p90 103 s，最长 348 s（`mteb-retrieve` 装 torch）。两次 `FAILED-NET`（fix-git 404、install-windows archive.org）
在补了 `URL_SUBS` 后重建 OK。`chess-best-move` 的 Dockerfile 从 `notofonts.github.io` 拉字体 → `build_chess.sh` 手工用 aliyun 的 `fonts-noto-core` 替代并预跑 `make.py`。

`verify_sifs.sh`：每个 sif `cd WORKDIR && ls`；python 镜像再查 `python3 --version`、`pip`、`import ssl`、`readlink -f $(command -v python3) == /usr/local/bin/python3.X`。

### 1.4 和 arena 镜像的已知差异（无法/不打算修的）

- `tb2-pytorch-model-cli` 用的是官方 `6.4.0` Dockerfile，没有 arena `6.4.2-gepa-cache` 烘焙的 `/opt/gepa-verifier-cache`（torch wheel + MNIST）。
  reward 侧按 §3.4 退回官方 `test.sh` + 预置 MNIST。
- 镜像里没有 `tmux`？——harness 自带 tmux 3.4 fallback（`harness/bin/fallback-bin`），runner 会探测 `new-session -e` 支持后自动切换，无需处理。

---

## 2. 运行

### 2.1 资源与隔离模型（= arena harness 1.6.96 的 `resource_spec`：cpu 4 / memory 16Gi / ttl 14400）

每条 (task, sample) 一个 `singularity exec`：

| 手段 | 作用 | 验证探针 |
|---|---|---|
| `--containall --no-home --writable-tmpfs --pid` | 独立 mount/pid namespace，看不到 `/storage` `/home`，rootfs 改动随容器丢弃 | `isoprobe.sh` |
| `--net --network none` | 容器**完全没有网络**，连 lo 之外都没有 | `isoprobe.sh` |
| `--no-mount tmp` | `/tmp` `/var/tmp` 留在 rootfs 上（docker 语义）；apptainer 默认另挂 tmpfs 会让 `os.rename(/app/x, /tmp/x)` 报 EXDEV | §3.3 |
| `--uts --hostname sandbox-<hash>` | 提示符里是 `root@sandbox-xxx:/app#` 而不是池节点名 | §3.3 |
| memory cgroup v1 `tb2/<task>-s<k>-<pid>` 16G，OOM killer 开 | 与 arena 同样的 16Gi；`memory.max_usage_in_bytes` 记进 `result.json` 的 `peak_mem_mb` | `cgprobe.sh` |
| `taskset -c <4 cores>` | slurm 在这个集群不钉 cpuset，由 `run_batch.sh` 按槽分配不重叠的核段。**必须 4 核/槽**（= arena `resource_spec.cpu`），`run_batch.sh` 对 <4 拒绝启动 | `cgprobe.sh` |

探针结论（`isoprobe.sh`）：`unshare -n` 可用；`--fakeroot` 不需要（本来就是 root）；`--memory/--cpus` flag 被接受但在 cgroup v1 下不生效 → 自己建 cgroup；
默认 `--containall` 下宿主进程不可见、`/storage` 未挂载。

### 2.2 网络：两条 loopback 端口通过 unix socket 桥到宿主代理（`egress_proxy.py`）

容器内没有网络，但 harness 要调 LLM、任务要装包。做法：

- 宿主侧（每条评测一个进程）：`egress_proxy.py --host-dir $NET --router <host:port | https://host>` 在 `$NET/` 下监听两个 unix socket：
  `llm.sock`（原样 TCP 转发到 LLM 路由；`https://` 时在宿主侧做 TLS 并重写 `Host:` 头，容器内仍是明文 http）和
  `egress.sock`（HTTP 代理，支持 CONNECT + 明文 GET）。
- 容器侧：`$NET` 绑到 `/sandbox-net`，`inside.sh` 用 harness 自带的 py3.13 在一个 `while :` 循环里起 `egress_proxy.py --bridge-dir /sandbox-net`（被 agent 杀掉 1 s 内重生，见 §3.3 #13），
  把 `127.0.0.1:30100` → `llm.sock`、`127.0.0.1:3128` → `egress.sock`。
- 策略：**允许任何公网主机**（arena 上任务也是 `allow_internet=true`），但解析结果落在 `10./33./127./169.254./172.16-31./192.168./200./0.`
  的一律 403 并记 `[egress] DENY`。安全目标是"看不到集群"，不是"看不到公网"。
- harness 进程**不**走代理（`LLM_BASE_URL=http://127.0.0.1:30100/...` 直连）；任务 shell 通过 `TERMINUS_SANDBOX_ENV_http_proxy=...` 透传拿到代理；
  reward 阶段由 `inside.sh` 显式 export 代理变量。

验证（`isoprobe4.sh`）：LLM api 200、aliyun pypi 经代理 200、github 200、集群 IP 403、裸 `/dev/tcp` 到邻居节点不通、`pip download six` 成功。
中途试过 socat 桥（`isoprobe3.sh`），因为镜像里没有 socat 且动态库不配，改成纯 stdlib python。

### 2.3 单条评测（`run_task.sh <task> <run_dir> <model> <router> [sample]`）

1. 从 Dockerfile 取 `WORKDIR`（默认 `/app`）；按 `HARNESS_SPEC` 选窗口语义：`1.6.96`（默认）= 98304 / 1024 / max_turns 512 / `usage`；
   `1.0.0|1.6.262` = 262144 / 65536 / 无上限 / `estimate`。
2. 写 `env.list`：与 arena spec `terminus-stream-1.6.96.json` 一致的 `TERMINUS_*` + `TERMINUS_SANDBOX_ENV_*`（aliyun pip/uv、超时、HF 镜像、代理），
   `ARENA_*` 路径，`LLM_*`。`ARENA_REWARD_DIR=/sandbox-net/reward`（reward 物料 ro 挂在桥目录下，agent 阶段看得到脚本但看不到 oracle）。
3. 写 `inside.sh`（容器内 PID 1 之下的驱动）：
   `unset PROMPT_COMMAND` → sysconfig `clang→gcc` → `chmod 1777 /tmp /var/tmp` → 起桥 → 等 30100 可连 → `cd $WORKDIR` →
   `bash /envarena/harness/bin/runner`（agent 阶段，产出 `harness_result.json`、`trajectory.jsonl`）→ 写 `/envarena/out/.agent_done` →
   **等宿主投递 oracle**（轮询 `/sandbox-net/oracle/.ready`，最多 300 s）→ `cp -a` 到可写的 `/envarena/oracle/tests` →
   export 代理变量 → `bash /sandbox-net/reward/tbench_reward_runner_mirror.sh`（产出 `reward_result.json`）→ 写 `stages.txt`（两个退出码 + 两段耗时）。
4. 宿主：起代理、建 cgroup、`taskset`，后台起 `singularity exec`，循环 `kill -0` 等待；看到 `out/.agent_done` 就把 `tasks/<task>/tests` 拷进 `$NET/oracle/` 并打 `.ready`。
   **oracle 在 agent 阶段不存在于容器内**，这是 arena 的契约（arena 轨迹里 `ls /envarena` 只有 `bin env harness out workspace`）。
5. 退出后汇总 `result.json`：`task sample harness_status harness_summary score reward_exit agent_sec reward_sec peak_mem_mb contract`。
   `contract: "v2-oracle-after-agent"` 标识新契约；没有该字段的是 11-09 之前的旧口径产物。

### 2.4 批跑（`run_batch.sh <run> <model> <router> [samples=2] [slots=auto] [tasklist]`，`batch_step.sh` 把它 detached 到某个 pool 作业）

- 核分配：`CORE_RANGE="a-b"` 显式给（同一节点上并发多个 batch **必须**不重叠），否则读本步的 `Cpus_allowed_list`；`slots = cores / CPUS_PER_SLOT`（默认 4）。
  每个槽是一个锁文件目录 `.slots.<pid>/<k>`，槽内核列表固定。
- 任务领取：队列 = 所有 (task, sample) 里既没有 `result.json` 也没有 `.lock/<task>.s<k>` 的；领取 = `mkdir .lock/<task>.s<k>`（共享文件系统上原子），
  所以**多个节点可以同时消费同一个 run 目录**而不重复。跑完没有 `result.json`（崩了）就删锁让下一轮重试。
- `config.txt` 记录 `model router samples spec ttl effort tool_call_mode`；全部跑完自动 `summarize.py`。
- 日志：`runs/<run>/batch.log`（每条的 `[run_task] ...` 两行）、`logs/batch_<run>.j<job>.log`。
- **用 `sweep_until_done.sh <jobid> <cores> <run> <model> <router>` 而不是裸 `run_batch.sh`**：batch 的队列在启动时固定，
  跑完就退出；之后作废（`mv` 到 `runs/invalidated-*`）的样本要新一轮才会被领取，而且 batch 退出时会把仍在跑的长题一起杀掉
  （`result.json` 没来得及写，锁被回收，表现为"未领取"）。driver 每 120 s 起一轮直到 `result.json` 数满 178。
- 一个 pool 节点（192 核）最多 48 个 4 核槽；11-09 曾为了多开槽用 2 核（#15，已禁止）。dsv41 的 agent 阶段 p50 320 s、p90 2193 s、最长 5655 s；reward p50 35 s、最长 1446 s（装 torch 的题）。
  峰值内存撞到 16384 MB 的题会被 cgroup OOM 掉，与 arena 行为一致。

### 2.5 看进度 / 汇总

- `progress.sh [run ...]`：每个 run 一行 `done/178  passed (pct)  in-flight` + 最近 3 条结果；`in-flight = locks − results`。`watch.sh` 每 60 s 刷一次。
- `summarize.py <run_dir>`：`summary.tsv`（每题 n / passed / agent_sec_mean / statuses）+ `pass_at_all`（failed_as_zero，与 arena `mean(fail=0)` 同口径）。
- 与 arena 对比：arena job 落盘在 `/storage/openpsi/users/chen/output/<job>/summary.tsv`（`env_key sample status score agent_ms eval_ms`）和
  `tasks/<env>/s<k>/{trajectory.jsonl,tbench_test.log,...}`；按 `(task, sample)` 对齐即可。

### 2.6 一条评测的产物

```
runs/<run>/<task>/s<k>/
├── result.json            汇总（§2.3 第 5 步）
├── env.list  inside.sh  Target.md     复现这条评测需要的全部输入
├── exec.log               singularity 本身的 stderr（挂载 / 启动错误在这里）
├── egress.log             宿主代理日志：DENY 行 = 任务试图访问的被拒主机
├── logs/verifier/         reward.txt / ctrf.json
└── out/
    ├── harness.log  harness_result.json  trajectory.jsonl  terminus2/   agent 阶段
    ├── reward.log  reward_result.json  artifacts/tbench_test.log        reward 阶段（pytest 全文在 tbench_test.log）
    └── stages.txt  .agent_done
```

---

## 3. 调试流程与踩过的坑（按时间）

### 3.1 构建阶段

| 现象 | 根因 | 处置 |
|---|---|---|
| `debootstrap` 在沙箱里失败 | apptainer 禁 `mknod` | sed 掉 debootstrap 的 mknod（§1.2） |
| `pip install` 全部超时 | `pypi.org` 不可达 | `RUN_ENV` 固定 aliyun 索引；`COPY --from=ghcr.io/astral-sh/uv` 改为 pip 装 uv wheel |
| alpine ISO 302 后连接拒绝 | aliyun 把 ISO 跳到集群内不可达的域名 | `URL_SUBS` → tuna |
| `doom1.wad` / `win311.img` / `personal-site` 下不到 | 源站不可达或仓库已删 | 找到 hash 一致的替代来源放 `cache/files/`，`file://` 或 `cp` 替换（§1.3 第 4 条） |
| `chess-best-move` 字体 404 | `notofonts.github.io` 不可达 | `fonts-noto-core` 替代，手工脚本 |
| 共享盘上解 tar 极慢 | 小文件 IO | 沙箱放本地 `/tmp`，只有最终 sif 写回 `/storage` |
| `/app/0_interp.py`、`/app/0_trunc.db`（文件名带 staged 前缀） | `COPY file dir` 单文件分支没按 docker 语义以原名落到目录内 | 修 `_stage_copy`，重建 2 个 sif（§1.3） |

### 3.2 网络/隔离阶段

| 现象 | 根因 | 处置 |
|---|---|---|
| 容器能直连集群其他节点的 30100 | 默认共享宿主网络 | `--net --network none` + unix socket 桥（§2.2） |
| 桥用 socat 起不来 | 镜像无 socat、宿主 socat 动态库不匹配 | 改为 harness 自带 py3.13 跑 stdlib 代理 |
| harness 调 DashScope 报 TLS / vhost 错 | 容器内只能明文 | 宿主侧 TLS，重写 `Host:`，`LLM_PATH=/compatible-mode/v1` |
| 非流式响应被代理静默切断 | 代理 600 s 空闲超时 | harness 本身是 stream 版（1.6.96），无需处理；代理 `select` 超时 600 s 只在双向静默时触发 |
| `--memory` 不生效 | cgroup v1 | 自建 memory cgroup（`cgprobe.sh` 验证 `/sys/fs/cgroup/memory` 可写） |
| 两个 batch 互相抢核 | slurm 不钉 cpuset | `CORE_RANGE` 显式分段 |

### 3.3 评测口径阶段（11-09，详见 [tb21_verifier_audit.md §8](tb21_verifier_audit.md)）

首轮 dsv41-local 跑到 155/178 时 0.78，与 arena 同题 0.835 差 5 pt。逐条对比 `reward_result.json.raw.log_excerpt`、`trajectory.jsonl` 和 arena 的
`tbench_test.log`，定位到 12 处非模型问题（6 处当天下午，#7–#9 当晚复查 pytest 全文补出，#10–#12 是运行期/外部）——完整台账见 [tb21_verifier_audit.md](tb21_verifier_audit.md)：

| # | 偏差 | 症状 | 方向 | 修法 |
|---|---|---|---|---|
| 1 | oracle 从容器启动就挂在 `/envarena/oracle`（arena 里 agent 阶段没有） | 模型 `sed /envarena/oracle/tests/test_outputs.py`，14/15 条得分；`extract-moves-from-video` 直接从测试里抠 SOLUTION | 本地虚高 | oracle 在 `.agent_done` 之后才由宿主投递 |
| 2 | `/tests` → ro 挂载的 oracle | `compile-compcert` `ld: cannot open /tests/positive_probe: Read-only file system`；`feal` `build_ext --inplace` 写 `/tests` 失败 | 本地压低 | oracle 拷进可写目录 |
| 3 | `/tmp` 独立 tmpfs | `path-tracing-reverse` `os.rename("mystery","/tmp/mystery")` → EXDEV | 压低 | `--no-mount tmp` |
| 4 | apptainer 注入 `PROMPT_COMMAND=PS1="Apptainer> "; unset PROMPT_COMMAND` | `headless-terminal` 用 `--rcfile` 设 PS1 哨兵被覆盖 → pexpect 超时；所有轨迹提示符 `Apptainer> ` 无 cwd | 压低 + 观测失真 | `unset PROMPT_COMMAND`、`--uts --hostname sandbox-*` |
| 5 | `pytorch-model-cli` 无 gepa 缓存 → reward `exit 96` | score=null ×2 | 压低 | runner 无缓存时跑官方 `test.sh` + 预置 MNIST（§3.4） |
| 6 | PBS python `CC=clang` | 修完 #2 后 feal 仍 `command 'clang' failed` | 压低 | `inside.sh` sed sysconfig |
| 7 | 镜像 COPY 翻译错（`/app/0_interp.py`） | `schemelike-metacircular-eval` 6/6 全 0，测试找不到 `interp.py` | 压低 | 修 `build_task.py`，重建 sif，作废重跑 |
| 8 | `/tests` 只读（第二批受害者） | `train-fasttext` 的 `tar -C /tests` 失败；第一轮作废的正则漏掉了它（报错在 `log_excerpt` 3500 字节截断之前） | 压低 | 同 #2；作废重跑。教训：扫 `artifacts/tbench_test.log` 全文而不是 `log_excerpt` |
| 9 | `python:3.11` 全量镜像缺运行库 | `sam-cell-seg` reward `import cv2` → `libgthread-2.0.so.0` 缺失 | 压低 | `build_base.sh` 的 `py311full` 补 buildpack-deps 同款运行库，重建 base + sif |
| 10 | 热替换 reward 脚本 | 正在跑的容器读到 `Stale file handle` | 压低 | 作废重跑；改脚本前先停 batch |
| 11 | 模型端点中途换地址 | 学生模型 19:18 后所有新样本 `Connection error`（agent ~150 s） | 压低 | 作废 37 条，`config.txt` 换 router，重跑；启动前核对 `endpoints/<model>/router.txt` |
| 12 | batch 收尾杀掉仍在跑的长题 | 目录/`stages.txt` 在，`result.json` 缺，锁被回收 | 缺样本 | `sweep_until_done.sh` 循环补跑到 178/178 |
| 13 | agent `kill`/`pkill` 掉容器内 loopback 桥 | 之后 LLM 调用 `Connect call failed 127.0.0.1:30100` | 压低 | `inside.sh` 桥进程 `while :; do …; sleep 1; done` 自动重生 |
| 14 | `sam-cell-seg` reward 从 `raw.githubusercontent.com` 拉 40 MB 权重失败 | `FileNotFoundError: /app/mobile_sam.pt`，`SSL_ERROR_SYSCALL` | 压低 | `reward/files/bin/curl` shim 用随包副本（hf-mirror 同哈希）应答这一个 URL |

验证顺序：先用探针确认每项（`singularity exec ... env | grep PROMPT`、`stat -c %d / /tmp /app`、`sysconfig.get_config_var("CC")`），
再用新 `run_task.sh` 对 dsv41 跑 `runs/fixcheck/`（headless 1、feal 1、compcert 1、path-tracing 0——这次是模型没写 `/app/mystery.c`、pytorch-model-cli 0——和 arena s1 挂同一个断言但 reward 跑完了）；
然后把三个 run 里受影响的 89 条样本移到 `runs/invalidated-20261109/`（`list.tsv` 带原因），删 `.lock`，重新起 batch 补跑。
补跑的 18 条 dsv41 结果：16/18 通过，`compile-compcert`、`feal`、`path-tracing-reverse` 均 2/2（与 arena 一致），`extract-moves-from-video` 仍 2/2——
这次不是靠 oracle，而是 `git clone harbor-framework/terminal-bench-1` 读原始任务的 `tests/test_outputs.py`（§3.5）。

### 3.4 reward runner 的本地改动（`reward/tbench_reward_runner_mirror.sh`，旧版 `.bak-20261109`）

只改了 `tb2-pytorch-model-cli` 分支：`/opt/gepa-verifier-cache/bin/uv` 不存在时**不再 exit 96**，而是把 `reward/mnist/` 下的 4 个 MNIST raw 文件
（`storage.googleapis.com/cvdf-datasets/mnist/`，可达；与 gepa 镜像烘焙的是同一批文件）拷到 `$WORKDIR/data/MNIST/raw/`，然后走官方 `test.sh`
（torch/torchvision 从 `download.pytorch.org`，其余从 aliyun 镜像；`torchvision.datasets.MNIST(download=True)` 发现文件已在就不去 `ossci-datasets.s3.amazonaws.com`）。
reward 耗时 ~28 min（下 torch + opencv）。arena 上有缓存的路径逐字节不变。

### 3.5 仍开放的问题

- **模型从公网拿到 TB 原始仓库的测试文件**：`extract-moves-from-video` 补跑两条都 `git clone --depth 1 github.com/harbor-framework/terminal-bench-1`，
  从 `original-tasks/<task>/tests/test_outputs.py` 抠出 SOLUTION。arena 上 `allow_internet=true` 理论上也能这么干，但 arena 该题 0/2（沙箱出口
  `github.com` 慢/不稳）。这是"本地网络比 arena 好"带来的差异，不是沙箱 bug；是否在 egress 代理里封 `github.com/harbor-framework/*` /
  `laude-institute/*` 需要决定——封了会偏离 arena 契约，不封则地板题可能被刷掉。目前**没有封**，`egress.log` 能看到所有外联主机，可事后审计。
- 旧口径（无 `contract` 字段）与新口径混在同一 run 目录里；`summarize.py` 不区分。补跑结束前不要拿 `summary.tsv` 和 arena 比。
- `/etc/hosts` 只有 localhost（比 arena 干净，arena 会泄漏宿主 IP 导致模型扫端口，见 tb21_verifier_audit.md §5.2）。

---

## 4. 常用命令速查

```bash
# 看某条为什么 0 分（pytest 全文）
less runs/<run>/<task>/s0/out/artifacts/tbench_test.log
# 这条评测模型访问过哪些外部主机、哪些被拒
grep -c . runs/<run>/<task>/s0/egress.log; grep DENY runs/<run>/<task>/s0/egress.log
# 模型有没有读到不该读的东西
grep -l "/envarena/oracle\|/sandbox-net" runs/<run>/*/s*/out/trajectory.jsonl
# 本地 vs arena 同题对齐
python3 - <<'EOF'
import json,glob,csv
A={}
for r in csv.DictReader(open('/storage/openpsi/users/chen/output/<arena_job>/summary.tsv'),delimiter='\t'): A.setdefault(r['env_key'][4:],{})[int(r['sample'])]=int(r['score'])
L=[json.load(open(p)) for p in glob.glob('runs/<run>/*/s*/result.json')]
print(sum(r['score']==1 for r in L)/len(L), sum(A[r['task']][r['sample']] for r in L)/len(L))
EOF
# 作废一条并让 batch 重跑
mv runs/<run>/<task>/s0 runs/invalidated-<date>/<run>/<task>/s0 && rmdir runs/<run>/.lock/<task>.s0
# 在已占的 pool 作业上再起一个 batch（核段不能和同节点其它 batch 重叠）
setsid nohup srun --mpi=pmi2 --jobid=$J --overlap --job-name=bash env CORE_RANGE=128-151 CPUS_PER_SLOT=2 API_KEY=$KEY LLM_PATH=/compatible-mode/v1 \
  bash run_batch.sh dsv41-local deepseek-v4.1-flash-1017 https://dashscope.aliyuncs.com 2 auto > logs/batch_dsv41-local.j$J.redo.log 2>&1 < /dev/null & disown
# 进容器手工复现某题环境（无网络、同样的隔离）
srun --mpi=pmi2 --jobid=$J --overlap singularity shell --containall --no-home --writable-tmpfs --pid --net --network none --no-mount tmp sif/tb2-<task>.sif
```
