# Tree-SFT 的实现与实验边界

部分采集 harness 在下一轮隐藏之前的 assistant 推理。整轨迹训练则会把那些推理放入后续轮次的前缀。Tree-SFT 用一条共享主干表达可见历史，每个当前轮的推理与动作各自形成受监督分支。用于比较的 per-turn 基准为每个 assistant 轮独立构造相同前缀、只监督该轮。

## 实现

- [建树](../../coding_sft/prep/prepare_tree_generic.py) 生成 trunk、fork、branch、position 与 labels，并与逐轮编码核对。超长时在分支之间切行，保留训练目标。历史性能试验采用 294,912 tokens 的 tree 行上限。
- [Attention 实现](../../coding_sft/patches/tree_attn.py) 为全注意力层构造可见性掩码，使分支仅访问分叉前的主干和自身。GatedDeltaNet 层保存分叉点状态，分别推进各分支，并补齐卷积的左上下文。训练通过 [train_tree.py](../../coding_sft/patches/train_tree.py) 安装补丁；权重按普通 Qwen 模型部署。
- [批次调度](../../coding_sft/prep/build_tree_sched.py) 把同一轨迹的所有子行放在同一优化步，按 rank 物理重排并用无监督行补齐。使用 schedule manifest 中的 GA 和总步数。旧实验会按固定 group schedule 取样，需以该实验 manifest 为准。

## 验证与成绩

合成模型用[跨分支检查](../../coding_sft/checks/check_tree_synth.py)检查隔离和数值。实际 4B 上的[逐轮对齐脚本](../../coding_sft/checks/check_tree_vs_perdh.py) 读取原集群保留的 tree 和 per-turn 对齐缓存；两条实测轨迹的 token 加权 NLL 分别是 1.458807 对 1.458593、1.436855 对 1.436867。32 条相同轨迹训练 8 步，逐步 loss 最大差 0.0010。单 L20X 的历史稳态吞吐为 2.8–3.0k tok/s，适用于该基准的长度和实现。

TB2.1 原生 @1.0.0 下，tree_th3k c117/c156/c195 分别为 0.202/0.292/0.247；整轨迹 mask 在对应步数为 0.298/0.298/0.320。数值等价验证回答的是训练时能否正确重现逐轮监督。选择上线配方还要逐题核对交付前检查及测试是否通过。[完整评测记录](../experiments/sft_results.md)

原集群 `tree_th3k_slurm2n_v4`（Slurm 989679）、`tree_flash_ais2n_v4`（AIS 352119780）、`tree_flash_solved_ais2n_v4`（AIS 352128120）分别对应 TH-3K、Flash 全量和 Flash solved。检查点和日志在原集群 `/storage/openpsi/users/chen/{checkpoints,runs}` 下。

## 在 157 上准备 tree 缓存

在仓库根目录执行。下面以每个输入行对应一条完整轨迹的 TH-3K 数据为例。同一轨迹拆成多个输入段的教师数据还需向调度器传 `--manifest-src`，将这些段按源轨迹分组。

```bash
ST=$(readlink -f storage)
PY="$ST/conda/envs/lf/bin/python"
TREE="$ST/runs/th3k_tree_rebuild"
export PYTHONPATH="$PWD/src"
"$PY" coding_sft/prep/prepare_tree_generic.py \
  --input "$ST/data/tb_agent/th3k/th3k_mask.jsonl" --out "$TREE" --name th3k_mask \
  --model "$ST/models/Qwen__Qwen3.5-4B-text" --bad-pair mask --overlong-conv drop --split-tokens 294912
"$PY" coding_sft/prep/build_tree_sched.py \
  --cache "$TREE" --out "${TREE}_sched8" --world 8 --epochs 8
```

启动 tree 训练时用 `torchrun` 调用 `coding_sft/patches/train_tree.py`，并用[tree 配方](../../coding_sft/configs/qwen3_5_tree_sft.yaml)指定模型、数据和输出路径。`tokenized_path` 指向刚建立的 schedule，梯度累积数和总步数取自该目录的 `manifest.json`。157 的单节点 H100 检查使用整轨迹训练，其结果见 [H100 实验记录](../experiments/h100_offload.md)。
