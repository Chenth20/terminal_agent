#!/bin/bash
# Slurm (interactive job) ablation: VARIANT=<name> JOBID=983941 bash coding_sft/launch/slurm/run_ablation_slurm.sh
set -euo pipefail
VARIANT=${VARIANT:?}; JOBID=${JOBID:?}
REPO=/storage/openpsi/users/chen/llama-factory; RUNS=/storage/openpsi/users/chen/runs/ablation; mkdir -p $RUNS
read -r DD DS OV < <(VARIANT=$VARIANT bash $REPO/coding_sft/launch/ablation.sh)
KEY=$(awk '/8.150.1.98/{f=1} f&&/password/{print $2; exit}' /root/.netrc)
WANDB_API_KEY=$KEY JOBID=$JOBID CPUS=64 GPUS=8 nohup bash $REPO/coding_sft/launch/slurm/in_container.sh "export WANDB_BASE_URL=http://8.150.1.98:8080 WANDB_PROJECT=chenth_terminal_agent_sft WANDB_DIR=/storage/openpsi/users/chen/runs/wandb RECORD_VRAM=1; llamafactory-cli train coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml dataset_dir=$DD dataset=$DS tokenized_path=$DD/tokenized_qwen3_5_text_262144 output_dir=/storage/openpsi/users/chen/checkpoints/abl_${VARIANT}_slurm overwrite_output_dir=false overwrite_cache=true gradient_accumulation_steps=8 report_to=wandb run_name=abl_${VARIANT}_slurm ${OV:-}" > $RUNS/abl_${VARIANT}_slurm.log 2>&1 &
echo "launched $VARIANT -> $RUNS/abl_${VARIANT}_slurm.log"
