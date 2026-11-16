# 训练与复现

从仓库根目录执行命令。本机项目位于 `/home/chentinghong/llama-factory`，`storage` 指向 `/cephfs_ssd_zhangjz/chentinghong`。历史 Slurm/AIS 作业的 `/storage/openpsi/users/chen` 路径属于原集群。各检查点的 TB2.1 结果、口径和任务级证据见[实验索引](experiments/README.md)。

## 在 157 上运行单节点 H100 检查

本机入口采用 8 张 H100 和随机调度缓存 `sqz8/sched8_rand`。它沿用 304 步学习率计划，完成第 10 步后保存并核验 checkpoint。使用新的 Ceph 输出目录，`driver.log` 中的 `check10 passed` 表示逐 rank 数据、配置和优化器分片通过检查。[H100 实验记录](experiments/h100_offload.md)

```bash
cd /home/chentinghong/llama-factory
ST=$(readlink -f storage)
"$ST/conda/envs/lf/bin/python" coding_sft/h100/launch.py \
  --output "$ST/runs/h100-th3k-$(date +%Y%m%d-%H%M%S)"
```

## 重建 TH-3K 参照 c312

R1 用 Qwen3.5-4B-text 全参训练。原始 2,852 条轨迹中，411 条超过 262,144 tokens；剩余 2,441 条各占一行。非法动作轮保留在上下文中，标签置为 -100。其余 assistant 轮的推理和动作都参与监督。原记录使用 HF `group_by_length`、微批 1、16 卡、梯度累积 4、有效 batch 64、每 epoch 39 步、seed 42。不开 packing。AdamW β=(0.9, 0.95)，weight decay 0.1，clip 1.0，恒定学习率 1e-5。BF16、FA2、ZeRO-3、Liger 和梯度检查点见[基础 YAML](../coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml)。

原始 Slurm 作业 985346、985544、985679 分别训练到 117、195、312 步，后两段接续上一段的完整 checkpoint。首段 warmup 比例 10%，即 ceil(117 × 0.1) = 12 步。直接以 8 epoch 冷启动且只设比例会得到 32 步 warmup。本机已有普通缓存 `flat/train/dataset_info.json` 记录 2,441 行；预处理日志记录 319.8M tokens、202.8M 监督 tokens。旁边的 `sqz8/sched8_rand` 是 304 步随机调度实验的输入。

原始 c312 使用两台各 8 张 GPU 的节点，共享同一份普通缓存及输出目录。在每台机器的仓库根目录运行[双节点训练入口](../coding_sft/launch/train_th3k_c312.sh)，首节点取 `NODE_RANK=0`，第二台取 1。`MASTER_ADDR` 填两台节点均可访问的首节点地址。

```bash
MASTER_ADDR="首节点地址" NODE_RANK=0 bash coding_sft/launch/train_th3k_c312.sh
MASTER_ADDR="首节点地址" NODE_RANK=1 bash coding_sft/launch/train_th3k_c312.sh
```

157 默认使用仓库的 `storage/` 和其中已有的 `flat` 缓存。迁到其他集群时，可指定 `SQZ_ROOT` 为共享存储根，或用 `MODEL_DIR`、`DATA_DIR`、`FLAT_CACHE`、`OUT_DIR` 分别覆盖模型、注册数据、普通缓存、输出路径。默认输出为 `storage/runs/rebuild-th3k-c312`。重新运行同一输出目录时，脚本跳过已经完成的阶段，CLI 从最近一个完整 checkpoint 续训。

三段分别固定 12 步 warmup 和相同 world size。`trainer_log.jsonl` 记录每步，`checkpoint-312` 用于评测。旧 TB2.1 verifier、Terminus 1.0.0 每题两次时，c117、c195、c312 分别有 53、57、62 个采样通过，分母均为 178。[评测协议对照](experiments/sft_results.md)

## 新数据如何进入下一轮 SFT

1. 按错误类型选择采集任务，记录每条源轨迹的任务、样本、段号与判分。同一条轨迹的 compaction 段共享标识，保留原始 JSONL 行号供映射。
2. `prep/prepare_terminal_horizon.py` 将 ShareGPT 数据转换为本项目 schema。`--no-chunk --max-tokens 262144 --bad-pair mask --overlong drop` 是 TH-3K 参照的数据策略。`prep/build_tokenized.py` 用 LF 的真实模板建立缓存。数据与权重使用配套 tokenizer；`checks/inspect_sft_labels.py` 可核对监督区间，`checks/check_prefix_consistency.py` 可核对训练与服务的历史轮模板。
3. 分段数据使用 `prep/build_traj_sched.py` 按 `src_row` 对齐源 manifest，再以完整轨迹为单位组织 batch。每步包含 64 条轨迹，同轨迹各段放在同一步，空位用全 -100 标签填充。默认 `global_sort` 按整轨迹长度降序、每 epoch 丢弃不足 64 条的尾组；`random` 每 epoch 随机排列。读取 schedule 的 `manifest.json` 中 `G`、`steps`、`steps_per_epoch` 作为 GA、max_steps 和存盘周期，同时启用 `disable_shuffling=true`。HF 长度分组和预排 schedule 对应不同实验。
4. 将数据策略、有效轨迹数、schedule、模型、warmup 步数及 checkpoint 写入实验记录。部署同一检查点，在固定 TB2.1 harness 下逐题与 c312 比较，然后为有代表性的退步任务回看轨迹。[评测流程](evaluation.md)

本机如需重建 TH-3K 普通缓存，在仓库根目录执行以下 CPU 分词命令，输出到新的目录。

```bash
ST=$(readlink -f storage)
CFG=coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml
DATA="$ST/data/tb_agent/th3k"
PYTHONPATH="$PWD/src" "$ST/conda/envs/lf/bin/python" coding_sft/prep/build_tokenized.py "$CFG" --set \
  model_name_or_path="$ST/models/Qwen__Qwen3.5-4B-text" \
  dataset_dir="$DATA" dataset=th3k_mask tokenized_path="$ST/runs/th3k_mask_flat_rebuild"
```

用上面新建的普通缓存可以重建 8 卡随机 schedule。调度器根据预处理文件中每行的 `src_row` 找到 manifest 中对应的源轨迹。输出目录的 `manifest.json` 给出实际步数和梯度累积数。

```bash
PYTHONPATH="$PWD/src" "$ST/conda/envs/lf/bin/python" coding_sft/prep/build_traj_sched.py \
  --cache "$ST/runs/th3k_mask_flat_rebuild" --manifest-src "$DATA/th3k_mask.manifest.jsonl" \
  --rows-jsonl "$DATA/th3k_mask.jsonl" --world 8 --epochs 8 --order random \
  --out "$ST/runs/th3k_mask_sched8_rand_rebuild"
```

训练端用 text-only 模型和 `qwen3_5_text` 模板。历史配方保留 assistant 历史轮的推理，部署需使用对应修改过的 Jinja 模板。text-only 转换与模板更新的入口为 `deploy/model/make_text_only.sh`。本地基座已存在。训练的 GatedDeltaNet 门与内核递归累加为 FP32，BF16 输入下 chunk 中间态为 BF16，可用 `checks/precision_audit.py` 审计本机运行路径。

## 已实施的数据与训练分支

| 分支 | 改动、观测与用途 |
| --- | --- |
| 坏动作与超长样本 | 测过整轨删除、只屏蔽坏轮、保留监督，以及截前缀或保留任务提示与尾段。保留上下文并屏蔽坏动作有利于保持可用任务覆盖。对新数据分别统计保留的任务、合法动作与可验证收尾。 |
| 采样、学习率、epoch | 测过随机、长度分组、packing 和学习率消融。3 至 8 epoch 的终点趋势向上，中间 c273 回落；继续扩 epoch 应同时观察同题重复样本与交付成功率。[分数记录](experiments/sft_results.md) |
| TH-3K solved-only | 录制命令重放后运行环境测试。2,766 条可重放轨迹中 660 条成功、1,990 条得零、116 条未判分；零分含命令解析和环境时序造成的假阴性。成功轨迹过滤后 579 条入训，`abl_th3k_solved_const_e8_w03ep_slurm2n` 共 72 步（每 epoch 9 步），未见 TB2.1 成绩。原集群证据位于 `code_sft_data/th3k_replay/{replay_scores.json,solved_rows.json}`。 |
| 教师数据与 tree | 已比较整轨迹、compaction 分段、教师判筛、solved 子集，并训练过 9B。tree 分支验证了条件对齐的实现，当前同 checkpoint 下游分数未超过 TH-3K mask。下一轮可按行为覆盖、交付前检查和错误恢复筛选数据，保持对照组同预算。[Tree 设计](methods/tree_sft.md) · [教师实验](experiments/sft_results.md) |
