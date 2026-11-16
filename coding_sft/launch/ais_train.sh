#!/bin/bash
# Multi-node LlamaFactory launcher for AIStudio (runtime="pytorch", one container per node, same
# command on every node). Maps the platform's env into LF's torchrun env (NNODES / NODE_RANK /
# MASTER_ADDR / MASTER_PORT) and runs THIS checkout (PYTHONPATH=repo/src), not the llamafactory
# snapshot baked into the image.
#
#   N_NODES=4 CONFIG=coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml RUN_NAME=... \
#   EXTRA="output_dir=..." bash coding_sft/launch/ais_train.sh
set -uo pipefail   # no -e: `cmd | head` triggers SIGPIPE under pipefail and would kill the script silently
REPO=${REPO:-/storage/openpsi/users/chen/llama-factory}
CONFIG=${CONFIG:-coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml}
N_NODES=${N_NODES:?}
EXTRA=${EXTRA:-}
RUN_NAME=${RUN_NAME:-}
LOG_DIR=${LOG_DIR:-/storage/openpsi/users/chen/runs/ais}
mkdir -p "$LOG_DIR"
# everything this script prints goes to a per-node file on /storage from the first line (the AIS
# console log is not reachable from the cluster), file name resolved after the rank is known
EARLY=$LOG_DIR/${RUN_NAME:-lf}_$(hostname)_$$.log
exec > >(tee -a "$EARLY") 2>&1

log() { echo "[$(date '+%F %T')] [$(hostname)] $*"; }
log "platform env:"; env | sort | grep -E '^(MASTER|RANK|WORLD|PET|NODE|PAI|AISTUDIO|POD|KUBERNETES|NCCL|CUDA|NVIDIA|LOCAL|GPU)_?[A-Z_]*=' || true
log "all env keys: $(env | cut -d= -f1 | sort | tr '\n' ' ')"

# ---- node rank / master ---------------------------------------------------------------------
if   [ -n "${NODE_RANK:-}" ]; then :
elif [ -n "${RANK:-}" ]; then NODE_RANK=$RANK
elif [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then NODE_RANK=$OMPI_COMM_WORLD_RANK
elif [[ "${PAI_CURRENT_TASK_ROLE_NAME:-}${AISTUDIO_TASK_ROLE_NAME:-}${POD_NAME:-}" =~ [Mm]aster ]]; then NODE_RANK=0
else NODE_RANK=${PAI_CURRENT_TASK_ROLE_INDEX:-${AISTUDIO_TASK_ROLE_INDEX:-1}}; fi
if [ -z "${MASTER_ADDR:-}" ]; then
    MASTER_ADDR=${PET_MASTER_ADDR:-${AREAL_MASTER_ADDR:-}}
    if [ -z "$MASTER_ADDR" ]; then log "ERROR: no MASTER_ADDR from the platform; env dumped above"; sleep 600; exit 1; fi
fi
MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-29500}}
export NNODES=$N_NODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE=8 FORCE_TORCHRUN=1
log "NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT"

# ---- same env as the Slurm launchers ----------------------------------------------------------
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_NVLS_ENABLE=0
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface
# Compiled-kernel caches (triton + inductor/flex autotune) on /storage so they survive pre-emption restarts,
# but ONE DIRECTORY PER RANK: train_tree.py maps TREE_KCACHE_BASE -> <base>/rank<RANK>/{triton,inductor}.
# A single shared dir had 16 ranks on 2 nodes racing on the same files (mkdir + rename of tmp.pid_* + dlopen of
# __triton_launcher.so) -> FileNotFoundError / ESTALE on CPFS in every attempt (2026-10-28 16:15-17:50).
# Non-tree entries keep the node-local /tmp default.
export TREE_KCACHE_BASE=/storage/openpsi/users/chen/.cache/kernels/${RUN_NAME:-lf}
# arm-B attention split (verified equivalent 2026-10-29, see TREE_SFT.md 5.2): default ON for tree entries
export TREE_ATTN_SPLIT=${TREE_ATTN_SPLIT:-1} TREE_SPLIT_MULT=${TREE_SPLIT_MULT:-32768}
# GLOBAL autotune seed, shared by every tree run of this model (artifacts are content-addressed: triton hash
# dirs / inductor fx hashes -> a key miss recompiles and rank0 publishes the new entry; split=0/1 coexist)
export TREE_KCACHE_SEED=/storage/openpsi/users/chen/.cache/kernels/_seed/qwen3_5_4b_tmult${TREE_T_MULT:-8192}
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ALLOW_EXTRA_ARGS=1
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_BASE_URL=${WANDB_BASE_URL:-http://8.150.1.98:8080} WANDB_PROJECT=${WANDB_PROJECT:-chenth_terminal_agent_sft} WANDB_DIR=/storage/openpsi/users/chen/runs/wandb
    # One wandb run per experiment across pre-emption restarts: HF's WandbCallback calls wandb.init()
    # with a fresh random id on every process start, so each AIS restart used to open a new run.
    # A fixed id + WANDB_RESUME=allow makes every restart append to the same run (history is keyed
    # by train/global_step, so the resumed steps line up). The id is resolved in this order and
    # persisted in <output_dir>/wandb_run_id so later restarts (and re-submissions) find it:
    #   1. WANDB_RUN_ID from the environment   2. <output_dir>/wandb_run_id   3. derived from RUN_NAME
    # To continue an experiment in a run that already exists on the server, write its id into
    # <output_dir>/wandb_run_id before (re-)submitting.
    if [ -n "$RUN_NAME" ]; then
        OUT_DIR=$(echo " $EXTRA" | grep -oE ' output_dir=[^ ]+' | cut -d= -f2)
        if [ -z "${WANDB_RUN_ID:-}" ] && [ -n "$OUT_DIR" ] && [ -s "$OUT_DIR/wandb_run_id" ]; then
            WANDB_RUN_ID=$(tr -d '[:space:]' < "$OUT_DIR/wandb_run_id")
        fi
        export WANDB_RUN_ID=${WANDB_RUN_ID:-$(echo "$RUN_NAME" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n' '-')} WANDB_RESUME=${WANDB_RESUME:-allow}
        if [ -n "$OUT_DIR" ] && [ "$NODE_RANK" = "0" ]; then mkdir -p "$OUT_DIR"; echo "$WANDB_RUN_ID" > "$OUT_DIR/wandb_run_id"; fi
        log "wandb: run id $WANDB_RUN_ID (resume=$WANDB_RESUME)"
    fi
    EXTRA="$EXTRA report_to=wandb"
else
    export WANDB_DISABLED=true
fi
[ -n "$RUN_NAME" ] && EXTRA="$EXTRA run_name=$RUN_NAME"

cd "$REPO"
export PYTHONPATH=$REPO/src:$REPO                          # run THIS checkout, not the image snapshot
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 | head -3 || true
cg() { cat /sys/fs/cgroup/$1 2>/dev/null || cat /sys/fs/cgroup/memory/$2 2>/dev/null || echo "?"; }
log "cgroup limits: memory.max=$(cg memory.max memory.limit_in_bytes) pids.max=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo ?) cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo ?) nproc=$(nproc)"
( while true; do echo "[hb $(date +%T)] alive, load=$(cut -d' ' -f1 /proc/loadavg) mem=$(free -g | awk '/Mem/{print $3"/"$2"G"}') cg_mem=$(( $(cg memory.current memory.usage_in_bytes | tr -dc 0-9 || echo 0) / 1073741824 ))G procs=$(ls /proc | grep -c '^[0-9]')"; sleep 15; done ) &
HB=$!
log "which python: $(which python) $(python -V 2>&1); llamafactory-cli: $(which llamafactory-cli); id: $(id -u); PYTHONPATH=$PYTHONPATH"
for m in torch transformers deepspeed liger_kernel fla causal_conv1d flash_attn llamafactory; do
    log "import $m ..."; timeout 300 python -u -c "import $m, sys; print('   ok', $m.__name__, getattr($m, '__version__', ''), getattr($m, '__file__', ''))" || log "   FAILED import $m (exit $?)"
done
timeout 300 python -u -c 'import torch; print("cuda", torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)' || log "FAILED cuda check"

# ---- tokenized cache: build once on node 0, everyone else waits -------------------------------
# Without a pre-built cache every rank tokenizes independently and the ranks dead-lock on the
# rank-0 save barrier. Node 0 builds into a temp dir and renames it atomically; other nodes wait.
TP=$(echo " $EXTRA" | grep -oE ' tokenized_path=[^ ]+' | tail -n1 | cut -d= -f2)   # last wins, like the OmegaConf CLI (variants such as *_pack override it)
if [ -n "$TP" ] && [ ! -f "$TP/dataset_dict.json" ]; then
    if [ "$NODE_RANK" = "0" ]; then
        log "tokenized cache $TP missing -> building it on this node (CPU)"
        rm -rf "$TP" "$TP.building"
        # the platform exports RANK/WORLD_SIZE/MASTER_*; with them set, Accelerate inside
        # get_train_args() would init a process group and wait for node 1 forever -> strip them
        TOKENIZERS_PARALLELISM=false env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u MASTER_ADDR -u MASTER_PORT -u NNODES -u NODE_RANK -u NPROC_PER_NODE -u FORCE_TORCHRUN \
            python -u coding_sft/prep/build_tokenized.py "$CONFIG" --set $(echo "$EXTRA" | tr ' ' '\n' | grep -E '^(dataset_dir|dataset|cutoff_len|template|packing|neat_packing)=' ) \
            tokenized_path="$TP.building" overwrite_cache=true preprocessing_num_workers=${PREP_WORKERS:-8} \
            && mv "$TP.building" "$TP" || { log "ERROR: building the tokenized cache failed"; sleep 600; kill $HB; exit 1; }
        log "tokenized cache ready: $TP"
    else
        log "waiting for node 0 to build the tokenized cache $TP ..."
        for i in $(seq 1 360); do [ -f "$TP/dataset_dict.json" ] && break; sleep 10; done
        [ -f "$TP/dataset_dict.json" ] || { log "ERROR: cache not ready after 60 min"; kill $HB; exit 1; }
        log "cache ready"
    fi
fi

# ---- single-workflow lock: two AIS records of the same run (a stopped record's zombie pods + its
# resubmission) must not train into the same output_dir / wandb run at once. The master node takes a
# heartbeat lock; a fresh lock held by ANOTHER workflow record means we wait for it to die (stopped pods
# are reaped within minutes; a stale lock > 5 min is taken over).
REC_ID=${EXECUTIONRECORD_ID:-${AISTUDIO_JCS_JOB_ID:-pid$$}}
LOCK_DIR=$(echo " $EXTRA" | grep -oE ' output_dir=[^ ]+' | tail -n1 | cut -d= -f2)
if [ "$NODE_RANK" = "0" ] && [ -n "$LOCK_DIR" ]; then
    mkdir -p "$LOCK_DIR"; LOCK=$LOCK_DIR/.active_workflow
    for _i in $(seq 1 20); do
        OTHER=$(cat "$LOCK" 2>/dev/null || true)
        AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
        { [ -z "$OTHER" ] || [ "$OTHER" = "$REC_ID" ] || [ $AGE -gt 300 ]; } && break
        log "output_dir locked by workflow $OTHER (age ${AGE}s) - waiting 30s ($_i/20)"
        sleep 30
    done
    echo "$REC_ID" > "$LOCK"
    ( while true; do echo "$REC_ID" > "$LOCK" 2>/dev/null; sleep 60; done ) &
    LOCK_HB=$!
    log "workflow lock taken: $REC_ID -> $LOCK"
fi

# ---- GPU keepalive during autotune --------------------------------------------------------------
# The low-priority pool reaps pods whose GPUs look idle, and the first tree step spends 20-70 min in
# triton/inductor autotune with long idle gaps -> every v4 attempt so far was killed before step 1.
# Until a FRESH trainer_log entry appears (mtime < 10 min; a stale file from a previous attempt does
# not count), every 5 min burn a matmul for ~60 s on every GPU that is currently IDLE (<10% util).
# Each burn is a short-lived subprocess: no persistent CUDA context, no VRAM held between burns, and
# busy GPUs (real autotune/training compute) are never touched. Stops permanently once training steps
# are flowing; also killed when torchrun exits.
if [ "${GPU_KEEPALIVE:-1}" = "1" ]; then
    TRAINER_LOG=${LOCK_DIR:+$LOCK_DIR/trainer_log.jsonl}
    ( sleep 180   # let torchrun/model-load start first
      while true; do
        if [ -n "$TRAINER_LOG" ] && [ -f "$TRAINER_LOG" ]; then
            AGE=$(( $(date +%s) - $(stat -c %Y "$TRAINER_LOG" 2>/dev/null || echo 0) ))
            [ "$AGE" -lt 600 ] && { echo "[keepalive $(date +%T)] fresh trainer_log (age ${AGE}s) -> training started, stopping for good"; break; }
        fi
        IDLE=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '$2 < 10 {printf "%s ", $1}')
        if [ -n "$IDLE" ]; then
            echo "[keepalive $(date +%T)] burning idle GPUs for 60s: $IDLE"
            KA_GPUS="$IDLE" timeout 180 python -u - <<'PYKA'
import os, time, torch
ts = {}
for g in os.environ.get("KA_GPUS", "").split():
    try:
        ts[int(g)] = torch.randn(4096, 4096, device=f"cuda:{int(g)}")
    except Exception as e:
        print(f"[keepalive] cuda:{g} skipped: {e}", flush=True)
t0 = time.time()
while ts and time.time() - t0 < 60:
    for g, a in ts.items():
        _ = a @ a          # async launch; all idle GPUs burn concurrently
for g in ts:
    torch.cuda.synchronize(g)
PYKA
        else
            echo "[keepalive $(date +%T)] all GPUs busy, skipping burn"
        fi
        sleep 300
      done ) &
    KEEPALIVE=$!
    log "GPU keepalive started (pid $KEEPALIVE): 60s burn on idle GPUs every 5 min until first fresh trainer_log entry"
fi

if [ -n "${TREE_ENTRY:-}" ]; then
    log "launching: torchrun ${TREE_ENTRY} $CONFIG $EXTRA  (node rank $NODE_RANK, log: $EARLY)"
    torchrun --nnodes "$NNODES" --nproc_per_node 8 --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
        "$TREE_ENTRY" "$CONFIG" $EXTRA || { log "ERROR: training exited $?; keeping container 10 min for inspection"; sleep 600; kill $HB ${KEEPALIVE:-} 2>/dev/null; exit 1; }
else
    log "launching: llamafactory-cli train $CONFIG $EXTRA  (node rank $NODE_RANK, log: $EARLY)"
    llamafactory-cli train "$CONFIG" $EXTRA || { log "ERROR: training exited $?; keeping container 10 min for inspection"; sleep 600; kill $HB ${KEEPALIVE:-} 2>/dev/null; exit 1; }
fi
kill $HB ${LOCK_HB:-} ${KEEPALIVE:-} 2>/dev/null
log "training finished"
