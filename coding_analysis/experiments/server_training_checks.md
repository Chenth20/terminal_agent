# 当前服务器的训练复现记录

核对日期为 2026-11-16。本机项目在 `/home/chentinghong/llama-factory`，`storage` 链接到 `/cephfs_ssd_zhangjz/chentinghong`。原集群的历史 job 快照已整理入[离线作业表](evidence/all_jobs_offline_2026-11-07.tsv)。当前共享存储中可读的是数据、基座、缓存和以下工程实验。

|产物|本机位置（相对仓库根）|说明|
|---|---|---|
|训练环境 / 模型| `storage/conda/envs/lf`、`storage/models/Qwen__Qwen3.5-4B{,-text}`|环境锁与冒烟日志见 `storage/llamafactory-demo/`。加法 LoRA 仅验证安装、转换、训练与推理|
|TH3K mask| `storage/data/tb_agent/th3k/{dataset_info.json,th3k_mask.jsonl,th3k_mask.manifest.jsonl,th3k_mask_overlong.jsonl}`|注册名 `th3k_mask`。调度 manifest 与实际训练 2441 条一致|
|其他教师数据| `storage/data/tb_agent/{th_dsv41,th_dsv41_no_hs,th_qwen38-flash,facet_no_hs}/`|各批次以自己的 manifest / build_stats 为准。如 th_dsv41 有 2494 条、FACET 此版本有 5832 条；历史实验用的版本需逐一确认|
|整轨迹 flat cache| `storage/runs/abl_mask_rand_const_e8_ais2n-repro/flat`|长度分组 c312 复现应从 flat cache 取样；使用原配方有效 batch 和 3→5→8 轮续训|
|本次工程 run| `storage/runs/abl_mask_rand_const_e8_ais2n-repro/sqz8/`|8 ranks，GA8，seed 42，64 条/步，每轮随机重排、丢尾 9 条，38 步/轮，8 轮共 304 步。对照 `sched8_rand/manifest.json`、`full.yaml`|

**工程实测。** 每个目录的 `checkpoints/trainer_log.jsonl` 记录实际 optimizer step。A100 的 `a100_131072` 至少完成 6 步，`selective16_262k` 至少完成 3 步。H100 的 `h100_offload28_10step` 已完成 262144 窗口、28 层卸载、GA8 的 10 步校验，`driver.log` 末行是 `check10 passed`。两组卸载短测进一步测得 24 层能完成最长批次的一次更新，20 层在同一批次的反向传播显存不足。[H100 运行命令、配对结果与原始日志](h100_offload.md)

从项目根查看实际进度及有效训练配置：

```bash
R=storage/runs/abl_mask_rand_const_e8_ais2n-repro/sqz8
for d in a100_131072 selective16_262k h100_offload28_10step; do
  echo "$d"
  tail -1 "$R/$d/checkpoints/trainer_log.jsonl"
done
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print({k:m[k] for k in ("n_trajs","world","G","order","steps","steps_per_epoch","tail_dropped_per_epoch")})' "$R/sched8_rand/manifest.json"
```

原成绩复现使用[训练指南](../training.md)的 c312 配方。本机未找到历史 `checkpoint-312`、逐样本 Arena job 目录或原集群的 `tb_local` 镜像；离线表可直接核分，端到端重测依赖取回权重或按原配方训练，以及可用的模型服务和评测平台。
