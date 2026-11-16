# 8×H100 训练复现与激活卸载对照（2026-11-16）

从 `/home/chentinghong/llama-factory` 运行。本次复现是 Qwen3.5-4B-text 的 TH3K mask 随机调度实验，使用 8 张 H100、262144 上下文、全参 BF16、FA2、Liger、ZeRO-3 和梯度检查点。每卡微批 1、梯度累积 8，每次更新处理 64 条轨迹。学习率按原实验的 304 步计划计算，回调在第 10 步保存并结束。

已完成 28 层卸载的 10 步校验。训练日志记录 81,642,520 个 token、最后一步 loss 1.28295064，耗时 1:58:00。第 10 步 checkpoint 保存在共享目录 `storage/runs/abl_mask_rand_const_e8_ais2n-repro/sqz8/h100_offload28_10step/checkpoints/checkpoint-10`。这次运行验证训练配置与内存开销，尚无对应 TB2.1 得分。[本机训练记录](server_training_checks.md) · [实验账本](README.md)

## 训练入口

```bash
cd /home/chentinghong/llama-factory
ST=/cephfs_ssd_zhangjz/chentinghong
"$ST/conda/envs/lf/bin/python" coding_sft/h100/launch.py \
  --output "$ST/runs/h100-th3k-$(date +%Y%m%d-%H%M%S)"
```

默认卸载 28 个解码层的输入激活。每个输出目录独立保存有效配置、各 rank 的验证结果、日志和 checkpoint。完成标志为 `driver.log` 最后一行 `check10 passed`。校验逐步比较有效 token、学习率、loss、实际卸载层，并检查最终 trainer state 和八份优化器分片。每次启动使用新目录。

## 文件与数据

|路径|内容|
|---|---|
|`coding_sft/h100/reference/`|从成功的 H100 运行逐字节保存的 driver、训练回调、激活卸载实现和 YAML。直接运行原始 driver 会引用历史目录，请使用上面的入口。|
|`coding_sft/h100/launch.py`|创建独立运行目录和有效 YAML，默认复用原始三份 Python 文件。可显式选择 `--layers 24` 或 `--layers 20`。|
|`coding_sft/prep/`、`coding_sft/checks/`、`src/llamafactory/`|节点的预处理、检查工具和训练框架源码。历史 Slurm/AIS 启动器保留在 `coding_sft/launch/`。|
|`$ST/models/Qwen__Qwen3.5-4B-text`、`$ST/data/tb_agent/th3k`|训练基座、tokenizer 和原始数据。|
|`$ST/runs/abl_mask_rand_const_e8_ais2n-repro/`|普通缓存 `flat`、随机调度 `sqz8/sched8_rand`、通过验证的 `sqz8/h100_offload28_10step`。|

## 复测卸载开销

```bash
bash coding_sft/h100/bench.sh "$ST/runs/h100-offload-ab-$(date +%Y%m%d-%H%M%S)"
```

短测复用同一个模型进程，以零学习率保持权重固定。前九步让每个 rank 使用同一条 65563 token 的真实样本。前三步分别给 28、0、24 层预热，第 4 至 9 步交错计时，每种配置两次。耗时覆盖前向、反向、梯度通信和优化器更新，取最慢 rank。最后一步使用八条最长轨迹测试 24 层的显存峰值。结果写入 `metrics.jsonl`，不保存模型。长序列探测的 GA 为 1，正式复现的 GA 为 8。

针对 20 层与 24 层的短测命令如下。先分别预热一次，然后使用相同输入按 24、20、20、24 的顺序计时，最后单测 20 层的最长八条轨迹。

```bash
bash coding_sft/h100/bench.sh "$ST/runs/h100-20-vs-24-$(date +%Y%m%d-%H%M%S)" 20-vs-24
```

## 实验一　0、24、28 层的吞吐与显存

8 × H100 80GB HBM3，同一模型、真实样本、进程组和软件栈。计时区域内所有 rank 同步，记录最慢 rank 的用时，显存取各 rank 的最大值。ZeRO-3 的模型与优化器策略保持一致。改变的是梯度检查点的输入激活在前向后是否同步复制到普通 CPU 内存，并在反向时复制回 GPU。

|卸载层数|每批平均秒数|集群 tokens/s|峰值已分配显存 GiB|
|---:|---:|---:|---:|
|0|9.96|52,641|32.24|
|24|17.17|30,558|25.05|
|28|17.87|29,353|23.79|

每卡输入 65,563 个 token，补齐后全机每批 524,544 个。28 层的吞吐较 0 层下降 44.2%，单批用时增加 79.3%。从 28 层降到 24 层，吞吐提高 4.1%。每种设置预热一次、计时两次。学习率为零，GA 为 1；不同长度下，内核、激活传输与通信的耗时占比会变化。

24 层额外完成一次最长批次的前向、反向和参数更新。八条轨迹为 260,211 至 261,646 个 token，合计补齐后 2,085,864 个 token。耗时 189.59 秒，峰值已分配 67.17 GiB，峰值预留 69.30 GiB。正式训练有八次梯度累积且权重会更新。28 层已通过完整 10 步的八个 rank 校验；24 层通过本次长样本一步，尚未跑这一组完整 10 步。长序列 0 层的用时和显存亦未测量。

实测目录为 `/cephfs_ssd_zhangjz/chentinghong/runs/h100-offload-ab-20261116-131504`，内含输入编号、各步耗时、有效配置、硬件信息和日志。

```bash
cd /home/chentinghong/llama-factory
python3 coding_sft/h100/summarize.py /cephfs_ssd_zhangjz/chentinghong/runs/h100-offload-ab-20261116-131504
```

## 实验二　20 与 24 层的配对对照

同一台机器的 8 张 H100 80GB。每张卡都使用同一条 65,563 token 的真实轨迹，单批合计 524,544 token。学习率为零，模型权重固定。两种配置各预热一次，然后交错计时两次。

|卸载层数|平均耗时 (秒)|集群吞吐 (tokens/s)|峰值已分配显存 (GiB)|
|---:|---:|---:|---:|
|24|15.98|32,829|25.05|
|20|15.05|34,851|26.30|

从 24 降到 20 层，65K 对照中每秒处理的 token 数变化 +6.2%。两次计时均保存在原始 JSONL。

最长八条轨迹为 260,211 至 261,646 个 token。两次探测使用完全相同的样本编号、补齐长度和卡数，均尝试完成前向、反向和优化器更新。24 层通过，峰值已分配 67.17 GiB，预留 69.30 GiB。20 层在最长批次的 Gated DeltaNet 反向传播出现 CUDA OOM。rank 5 请求再分配 1.99 GiB 时仅剩 806.25 MiB。该批次没有完整步耗时。

短测的 GA 为 1。原 262144 上下文、GA8、真实学习率的 10 步复现，已通过验证的设置为 28 层。

配对测速原始记录 `/cephfs_ssd_zhangjz/chentinghong/runs/h100-offload20-vs24-20261116-132527`。24 层最长批次记录 `/cephfs_ssd_zhangjz/chentinghong/runs/h100-offload-ab-20261116-131504`。
