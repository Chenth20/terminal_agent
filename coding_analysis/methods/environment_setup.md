# 157 训练环境与模型准备

## 使用共享环境

157 的训练代码位于 `/home/chentinghong/llama-factory`。仓库的 `storage` 链接指向 Ceph，共享模型、数据、Conda 环境和运行产物都在这里。H100 单节点检查使用 8 张卡和选择性 CPU offload；c312 原参照使用两节点共 16 张卡。[训练入口](../training.md) · [H100 实验记录](../experiments/h100_offload.md)

```bash
cd /home/chentinghong/llama-factory
ST=$(readlink -f storage)
findmnt -T "$ST" -o TARGET,FSTYPE
source "$ST/conda/miniconda3/etc/profile.d/conda.sh"
conda activate "$ST/conda/envs/lf"
export PYTHONPATH="$PWD/src"
export CUDA_HOME="$ST/conda/envs/cuda124"
export PATH="$CUDA_HOME/bin:$PATH"
python deploy/verify_env.py
```

验证脚本检查包版本、源码和 GPU 内核。只核对 CPU 环境时运行 `python deploy/verify_env.py --no-gpu`。评测使用单独的 vLLM 与 Harbor 运行环境。[评测指南](../evaluation.md)

## 重建训练环境

在新前缀中运行以下命令，沿用上面的 `ST` 和 CUDA 12.4 工具链。安装脚本按[依赖清单](../../deploy/requirements-image.lock)安装 Python 轮包和本仓库源码，为 H100 编译 `causal-conv1d` 时使用 sm_90。

```bash
conda create --prefix "$ST/conda/envs/lf-rebuild" python=3.11.11 -y
conda activate "$ST/conda/envs/lf-rebuild"
TORCH_CUDA_ARCH_LIST=9.0 bash deploy/install_env.sh
python deploy/verify_env.py
```

### 训练依赖基线

| 组件 | 版本 | 备注 |
|---|---|---|
| python | 3.11 | 环境 ABI |
| torch | 2.6.0+cu124 | 决定 flash-attn / causal-conv1d 的 ABI |
| transformers | **5.12.1** | Qwen3.5 支持 + `train_sampling_strategy` |
| datasets | 4.0.0 | `loss_tag` 的 arrow schema 要求每个 message 都有 key |
| accelerate / peft / trl | 1.11.0 / 0.18.1 / 0.24.0 | pyproject 上限 |
| deepspeed | 0.18.4 | ZeRO-3 |
| liger-kernel | 0.8.4 | fused linear CE |
| flash-linear-attention + fla-core | 0.5.2 | GatedDeltaNet 融合核 |
| causal-conv1d | 1.5.3.post1 | 快路径必需 |
| flash-attn | 2.7.4.post1 | FA2，官方 cu12/torch2.6/cp311 预编译 wheel |
| wandb | 0.28.2 | 自建服务端兼容 |
| triton | 3.2.0 | 随 torch；liger / fla 内核依赖 |
| nvidia-nccl-cu12 | 2.21.5 | 随 torch |

原集群复现参照使用 8 × L20X 的节点，训练镜像的 SHA-256 为 `23cd938f8217fae75ff2373f62cf1c84e5a7ab3749ae9d5891af6f136107ed3b`。镜像中的包版本是锁定训练依赖的依据。

## Qwen3.5 text-only 基座与历史轮模板

157 已有 `storage/models/Qwen__Qwen3.5-4B-text`。需要从多模态基座重新导出时，在激活的训练环境中使用新输出目录。

```bash
bash deploy/model/make_text_only.sh \
  "$ST/models/Qwen__Qwen3.5-4B" "$ST/models/Qwen__Qwen3.5-4B-text-rebuild" --verify
```

导出脚本使用 `Qwen3_5ForCausalLM` 保存文本模型，保留语言权重、tokenizer 与生成配置。文本类支持按监督 token 数归一化 loss，终端记录中的字面 `<image>` 也可直接按文本处理。`--verify` 比较两种模型的 logits 与贪心解码结果。

项目模板在每个历史 assistant 轮渲染推理。服务端需接收 `reasoning_content` 或含 `<think>` 的历史内容，让训练和部署看到相同前缀。可用实际样本核对两条编码路径。

```bash
python coding_sft/checks/check_prefix_consistency.py \
  --model "$ST/models/Qwen__Qwen3.5-4B-text" --data "$ST/data/tb_agent/th3k/th3k_mask.jsonl" --rows 0 1 2
```
