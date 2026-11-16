#!/usr/bin/env bash
# Run on each of two 8-GPU nodes, sharing the same TH3K flat cache and output.
set -euo pipefail

cd "$(dirname "$0")/../.."
storage_root=${SQZ_ROOT:-$PWD/storage}
model_dir=${MODEL_DIR:-$storage_root/models/Qwen__Qwen3.5-4B-text}
data_dir=${DATA_DIR:-$storage_root/data/tb_agent/th3k}
flat_cache=${FLAT_CACHE:-$storage_root/runs/abl_mask_rand_const_e8_ais2n-repro/flat}
output_dir=${OUT_DIR:-$storage_root/runs/rebuild-th3k-c312}
: "${MASTER_ADDR:?set the rank-0 host address}" "${NODE_RANK:?set the node rank to 0 or 1}"
if [[ ! -f "$flat_cache/dataset_dict.json" ]]; then
    echo "TH3K flat cache not found: $flat_cache/dataset_dict.json" >&2
    exit 1
fi

export FORCE_TORCHRUN=1 NNODES=2 NPROC_PER_NODE=8 NODE_RANK MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export CUDA_HOME=${CUDA_HOME:-$storage_root/conda/envs/cuda124}
export PATH="$storage_root/conda/envs/lf/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>'
export VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>'
export AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'

# Match LlamaFactory's resume marker. Its CLI restores the latest complete step.
last_step=0
for state in "$output_dir"/checkpoint-*/trainer_state.json; do
    [[ -f "$state" ]] || continue
    checkpoint=${state%/*}
    step=${checkpoint##*/checkpoint-}
    if [[ "$step" =~ ^[0-9]+$ ]] && ((step > last_step)); then
        last_step=$step
    fi
done

for epoch in 3 5 8; do
    if ((last_step >= 39 * epoch)); then
        continue
    fi
    llamafactory-cli train coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml \
        model_name_or_path="$model_dir" dataset_dir="$data_dir" dataset=th3k_mask \
        tokenized_path="$flat_cache" output_dir="$output_dir" overwrite_output_dir=false report_to=none \
        gradient_accumulation_steps=4 train_sampling_strategy=group_by_length \
        lr_scheduler_type=constant_with_warmup warmup_ratio=null warmup_steps=12 \
        save_steps=1 keep_weights_every=39 num_train_epochs="$epoch"
done
