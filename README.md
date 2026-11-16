# Terminal Agent SFT

用 TerminalHorizon 的终端交互轨迹微调 Qwen3.5，训练能根据命令输出继续操作、检查交付结果的小模型。本仓库包含 SFT 数据准备、训练、数值检查与 H100 显存实验，也包含训练依赖的定制 LlamaFactory 源码。

| 入口 | 用途 |
| --- | --- |
| [单节点 H100](coding_sft/h100/launch.py) | 在 8 张 H100 上使用随机调度缓存，按 304 步学习率计划验证前 10 步及 checkpoint |
| [双节点 c312](coding_sft/launch/train_th3k_c312.sh) | 在两台各 8 卡的节点上，用普通 flat cache 分三段训练到 117、195、312 步 |
| [数据转换](coding_sft/prep/prepare_terminal_horizon.py) · [分词缓存](coding_sft/prep/build_tokenized.py) · [轨迹调度](coding_sft/prep/build_traj_sched.py) | 把原始对话转为 ShareGPT、应用真实训练模板、为分段数据构建按轨迹组织的 batch |
| [训练配置](coding_sft/configs/) · [检查工具](coding_sft/checks/) · [Tree-SFT](coding_sft/patches/) | 复查监督标签、模板、packing 和 tree attention |

## 环境与输入

在 Linux 仓库根目录运行。原环境使用 Python 3.11、PyTorch 2.6.0、CUDA 12.4、Transformers 5.12.1。已有配套环境时，将当前 checkout 的 `src/` 放在 `PYTHONPATH` 首位。从头安装时，在已激活的 Python 3.11 环境中运行：

```bash
TORCH_CUDA_ARCH_LIST=9.0 bash deploy/install_env.sh   # H100；A100 使用 8.0
python deploy/verify_env.py
```

设置 `SQZ_ROOT` 为已有的共享存储根；157 默认可使用项目的 `storage` 链接。Git 只保存程序和配方。模型、轨迹、缓存及 GPU 运行产物单独保存在共享存储。

```bash
export SQZ_ROOT=/path/to/shared/storage
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

默认输入为 `$SQZ_ROOT/models/Qwen__Qwen3.5-4B-text`、`$SQZ_ROOT/data/tb_agent/th3k` 下注册名为 `th3k_mask` 的数据集，以及 `$SQZ_ROOT/runs/abl_mask_rand_const_e8_ais2n-repro/` 下的缓存。TH3K 主线使用 2,441 条整轨迹，屏蔽异常回复的监督标签，过滤超过 262,144 tokens 的轨迹。普通分词缓存与 H100 预排缓存分别使用 `flat` 和 `sqz8/sched8_rand`。分段调度还需要与数据行对应的 manifest。

## 单节点 H100

使用新 Ceph 输出目录。先通过 `--prepare-only` 生成有效 YAML 与启动文件；实际训练时选择另一个新目录，去掉该选项。

```bash
PY="$SQZ_ROOT/conda/envs/lf/bin/python"
"$PY" coding_sft/h100/launch.py --prepare-only --output "$SQZ_ROOT/runs/h100-prepare-$(date +%Y%m%d-%H%M%S)"
"$PY" coding_sft/h100/launch.py --output "$SQZ_ROOT/runs/h100-train-$(date +%Y%m%d-%H%M%S)"
```

默认卸载 28 层激活，8 卡、GA8、每步 64 条轨迹。历史运行完成 10 步，累计 81,642,520 tokens，最终 loss 1.28295。成功运行的 `driver.log` 末行是 `check10 passed`。[原始验证实现](coding_sft/h100/reference/)

比较卸载层数可运行 `bash coding_sft/h100/bench.sh "$SQZ_ROOT/runs/bench-$(date +%Y%m%d-%H%M%S)" 20-vs-24`。已有运行可用 `python coding_sft/h100/summarize.py RUN_DIR` 离线汇总。20 层在 65k 配对短测比 24 层快约 6.2%，但在最长批次反向传播时 OOM；24 层完成了该批次的单步检查。

## 双节点 c312

两台节点应能访问同一项目、flat cache 与输出目录。将 `MASTER_ADDR` 设成首节点的可达地址，在两个节点分别运行对应的一条命令。

```bash
# 首节点
NODE_RANK=0 MASTER_ADDR="$MASTER_ADDR" bash coding_sft/launch/train_th3k_c312.sh
# 第二个节点
NODE_RANK=1 MASTER_ADDR="$MASTER_ADDR" bash coding_sft/launch/train_th3k_c312.sh
```

两节点共 16 卡，微批 1、GA4，按长度分组，恒定学习率 1e-5、warmup 12 步。重复使用同一 `OUT_DIR` 会从已保存的进度续训。可用 `MODEL_DIR`、`DATA_DIR`、`FLAT_CACHE`、`OUT_DIR` 分别覆盖输入和输出位置。[基础配方](coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml)

历史 TB2.1 旧 verifier、Terminus 1.0.0 下，4B base 为 50/267，c312 为 62/178。同一 c312 在 mirror verifier 1.0.1 下为 73/178。H100 十步验证衡量训练链路，未产生新的 TB2.1 成绩。

## 原集群实验与来源

`coding_sft/launch/slurm/` 和 `coding_sft/launch/ais/` 保留原集群的入口与资源设置，使用时需要对应的分区、镜像和共享目录。AIS 提交从环境读取 `AISTUDIO_TOKEN`。`archive/` 收录历史消融脚本与配方。

训练框架基于 LlamaFactory，包含本项目使用的 loss、模板、packing 和 checkpoint 修改。安装时使用本仓库源码。保留上游及随附第三方源码的版权声明。[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) · [Apache License 2.0](LICENSE)
