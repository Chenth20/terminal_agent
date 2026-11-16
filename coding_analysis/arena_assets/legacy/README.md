# legacy/ - 原平台（arena CLI）专用件

> 157 节点的提交和拉取命令见[评测指南](../../evaluation.md)。下文的 `submit.sh` 和 `finalize_judge.sh` 是原平台脚本，分别依赖原目录的 `lib/common.sh` 和旧集群的 judge 路径。

> 此页保留原 Arena 平台的资产布局。当前仓库的资源范围及运行时包依赖见[资产说明](../README.md)。

这些文件只在原评测平台上有意义，迁到自己的服务器后不需要。保留原样供对照；上层各目录的可用件都是从这里派生或与之对应的：

| 这里 | 上层对应 |
|---|---|
| `harness/specs/terminus-stream-<ver>.json`（平台 harness spec：运行时 uri、`agent.env`、资源、元数据） | `harness/configs/<ver>.env` = spec 的 `agent.env`（1.0.0 额外补了 `TERMINUS_TOKEN_COUNT_MODE=estimate`，原因见文件头） |
| `reward/reward-spec-1.0.1.json`（reward spec：入口命令、随包文件 uri） | `reward/tbench_reward_runner_mirror.sh` + `reward/uv-dist/`；spec 里的 `python313.tar.gz` 官方 runner 并不使用 |
| `benchmarks/bench-*.json`（env_refs + reward_ref 列表） | `tasks/tb21_full89.tsv`、`tasks/th_full_1486.tsv`；redo14 / redo-pollution 两个清单 = `env/manifests/th14_bad_envs.tsv`（action=rerun）与 `th14_polluted_envs.tsv` |
| `env/reg/{make_assets.py,register_batch.sh,register_flat_1.3.0.sh,register_140_fixes.sh}`（生成 env spec、上传、发布） | 修复逻辑本身在 `env/hooks/`、`env/fixes/`、`env/rt3_ca/` |
| `env/specs-1.4.0/*.json`、`env/spec-flat-example.json`（已发布的 env spec 原样副本） | `tasks/th_full_1486.tsv` 的 image / workdir / 资源 / 超时列 |
| `env/hooks/th_flat_setup_hook.sh` | 上层同名文件去掉了末尾重启平台守护进程（`arca_envd`）的 5 行，其余逐字相同 |
| `env/manifests/tasks_redo14.tsv` | 与 `th14_bad_envs.tsv` 中 action=rerun 的 141 行一一对应 |
| `scripts/common.sh`、`submit.sh`、`pull_trajectories.sh`（arena CLI 封装、提交 job、拉轨迹） | 由你的调度器替代；落盘布局见上层 README §5.3 |
| `scripts/finalize_judge.sh`、`build_dsv41_merged.py`（写死本集群路径的旧版） | `scripts/run_clean_chain.sh`、`scripts/build_merged.py` |
| [README.arena](README.arena.md) | 原 README（含 `arena harness/reward/env/benchmark create → version-create → publish` 的发布流程） |
