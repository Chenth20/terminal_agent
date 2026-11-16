#!/bin/bash
#SBATCH --job-name=chenth_sched
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# Build the TRAJECTORY schedule cache for a flat (compaction-segmented) tokenized cache: step s = 64 complete
# trajectories, their rows bound to the same step, dummy rows filling the rest (see prep/build_traj_sched.py).
#   CACHE=<flat tokenized cache> SRC=<source manifest jsonl> OUT=<dir>_sched16 sbatch <this>
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
CACHE=${CACHE:?}; SRC=${SRC:?}; ROWS=${ROWS:?}; OUT=${OUT:?}; WORLD=${WORLD:-16}; EPOCHS=${EPOCHS:-8}; ORDER=${ORDER:-global_sort}; TAIL=${TAIL:-drop}
# ORDER=replicate (Loss-Length Replication) needs the finished gsort probe run: PROBE_SCHED=<gsort manifest.json> PROBE_LOG=<trainer_log.jsonl>
REPL=""
if [ "$ORDER" = "replicate" ]; then
    REPL="--probe-sched ${PROBE_SCHED:?} --probe-log ${PROBE_LOG:?} --probe-epoch ${PROBE_EPOCH:-2} --alpha ${ALPHA:-4} --beta ${BETA:-0.5} --rmin ${RMIN:-2} --rmax ${RMAX:-16}"
fi
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
cat > "$INNER" <<EOS
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface ALLOW_EXTRA_ARGS=1
cd ${REPO}; export PYTHONPATH=${REPO}/src:${REPO}
python coding_sft/prep/build_traj_sched.py --cache ${CACHE} --manifest-src ${SRC} --rows-jsonl ${ROWS} --world ${WORLD} \\
    --epochs ${EPOCHS} --order ${ORDER} --tail ${TAIL} ${REPL} --out ${OUT}
EOS
echo "Job ${SLURM_JOB_ID} on $(hostname) start $(date) cache=${CACHE} world=${WORLD} out=${OUT}"
srun --mpi=none --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} \
    bash -c "exec singularity exec --no-home --writable-tmpfs --bind /storage:/storage --cleanenv '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"