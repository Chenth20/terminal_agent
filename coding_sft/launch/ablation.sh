#!/bin/bash
# Ablation launcher: one named variant -> dataset / overrides. Used by both the Slurm (in_container.sh)
# and the AIS (ais_train.sh) paths. Effective batch is always 64 trajectories: ga = 64 / n_gpus.
#
#   VARIANT=<name> bash coding_sft/launch/ablation.sh   -> prints "DATASET_DIR DATASET EXTRA" (eval'able)
#
# Variants (all: Qwen3.5-4B-text, 256K, 3 epochs, warmup 0.1, lr 1e-5, bs 64, group_by_length unless noted):
#   base         bad actions isolated (drop),  over-long isolated (drop)     [= the first full run's data]
#   mask         bad actions MASKED (loss=0),  over-long isolated (drop)     <- default for later ablations
#   keep         bad actions KEPT & supervised, over-long isolated (drop)
#   mask_cut     bad actions masked,           over-long PREFIX-truncated
#   mask_tail    bad actions masked,           over-long TAIL window (task prompt + suffix)
#   <best>_pack  + packing: true, neat_packing: true (hybrid-model fix in collator). Packing happens at
#                tokenization time, so this variant uses its OWN cache <dir>/tokenized_qwen3_5_text_262144_pack
#                (build it first: VARIANT=mask_pack sbatch coding_sft/launch/slurm/build_tokenized.slurm).
#   <best>_rand  + train_sampling_strategy: random (no length grouping)
#   <best>_const + lr_scheduler_type: constant_with_warmup (same lr 1e-5, same 10 % warmup, no cosine decay)
#   <best>_lr2e5 + learning_rate: 2e-5
#   <best>_lr5e6 + learning_rate: 5e-6
#   <best>_eN    + num_train_epochs: N, N = 1-9 (continue a finished run: OUT_DIR=<new dir> holding a hard-linked copy of its last checkpoint)
# Suffix overrides are appended AFTER the launcher's own `tokenized_path=` etc.; OmegaConf CLI parsing is
# last-wins, so a variant may override those too.
set -euo pipefail
D=/storage/openpsi/users/chen/data
case "${VARIANT:?}" in
  base)      echo "$D/terminal_horizon_3k_256k th3k_256k" ;;
  mask)      echo "$D/terminal_horizon_th3k_mask th3k_mask" ;;
  mask_9b)   echo "$D/terminal_horizon_th3k_mask th3k_mask" ;;   # same data + same 4B tokenized cache (4B/9B tokenizer.json, tokenizer_config, chat_template md5-identical); "_9b" makes submit_ablation_ais.sh pick the 9B yaml
  keep)      echo "$D/terminal_horizon_th3k_keep th3k_keep" ;;
  mask_cut)  echo "$D/terminal_horizon_th3k_mask_cut th3k_mask_cut" ;;
  mask_tail) echo "$D/terminal_horizon_th3k_mask_tail th3k_mask_tail" ;;
  qwen38_flash) echo "$D/terminal_horizon_qwen38_flash qwen38_flash" ;;
  qwen38_flash_solved) echo "$D/terminal_horizon_qwen38_flash_solved qwen38_flash_solved" ;;
  dsv41_history) echo "$D/terminal_horizon_dsv41_history dsv41_history" ;;
  dsv41_history_solved) echo "$D/terminal_horizon_dsv41_history_solved dsv41_history_solved" ;;   # same, score==1 segments only (-solved.jsonl, 4,228 segments / 1,253 trajs)
  th3k_solved) echo "$D/terminal_horizon_th3k_solved th3k_solved" ;;
  dsv41_judged)    echo "$D/terminal_horizon_dsv41_judged dsv41_judged" ;;        # dsv41 tool-mode, system-error-clean + Opus-4.8 judge KEEP (2,541 trajs / 8,876 compaction segments; code_sft_data/dsv41_judged), 4B tokenizer
  dsv41_judged_9b) echo "$D/terminal_horizon_dsv41_judged_9b dsv41_judged_9b" ;;  # same source, prepped/tokenized with the Qwen3.5-9B tokenizer; submit_ablation_ais.sh picks the 9B yaml for *_9b*
  dsv41_json262k) echo "$D/terminal_horizon_dsv41_json262k dsv41_json262k" ;;            # dsv41-flash JSON-mode WHOLE trajectories (thall/th500/threst @1.0.0 262k + csweep 128k/256k/512k), overlong drop
  dsv41_json262k_tc20k) echo "$D/terminal_horizon_dsv41_json262k_tc20k dsv41_json262k_tc20k" ;;   # same + think-cap 20k chars (over-long reasoning turns loss=0)
  dsv41_json262k_solved_tc20k) echo "$D/terminal_horizon_dsv41_json262k_solved_tc20k dsv41_json262k_solved_tc20k" ;;   # solved subset + think-cap 20k
  dsv41_hist_lin_solved) echo "$D/terminal_horizon_dsv41_hist_lin_solved dsv41_hist_lin_solved" ;;   # dsv41 tool-mode, SOLVED, LINEAR (one conversation per trajectory, compaction handoff inline, 92% end with task_complete), overlong DROP
  dsv41_hist_lin_solved_tail) echo "$D/terminal_horizon_dsv41_hist_lin_solved_tail dsv41_hist_lin_solved_tail" ;;
  dsv41_hist_lin_full) echo "$D/terminal_horizon_dsv41_hist_lin_full dsv41_hist_lin_full" ;;
  facet_final) echo "$D/terminal_horizon_facet_final facet_final" ;;   # FACET arena JSON-mode whole trajectories (code_sft_data/facet_final, 5,829 trajs incl score=0), prepped --bad-pair mask --overlong drop, 4B tokenizer   # dsv41 tool-mode, LINEAR (whole trajectory, 92% end with task_complete), ALL trajs incl score=0, overlong TAIL window   # same but overlong -> TAIL window (task prompt + suffix) so the task_complete tail survives   # TH-3K trajectories whose replay (cth-th3k-replay job) scored 1; same prep rules as th3k_mask
  *_w03ep)   base=${VARIANT%_w03ep}; read -r dd ds < <(VARIANT=$base bash "$0"); echo "$dd $ds warmup_ratio=0.0375" ;;   # warmup = 0.3 epoch of an 8-epoch run (0.3/8); the launcher may override with warmup_steps   # DeepSeek-V4.1-Flash tool-call mode (history reasoning VISIBLE), code_sft_data/dsv41_history, compaction-split segments   # same, score==1 segments only (-solved.jsonl)   # qwen3-8 flash-history, compaction-split segments, whole-segment loss (prep: prep_dataset.slurm NAME=qwen38_flash)
  *_pack)    base=${VARIANT%_pack}; read -r dd ds < <(VARIANT=$base bash "$0")
             echo "$dd $ds packing=true neat_packing=true tokenized_path=$dd/tokenized_qwen3_5_text_262144_pack" ;;
  *_rand)    base=${VARIANT%_rand}; read -r dd ds < <(VARIANT=$base bash "$0"); echo "$dd $ds train_sampling_strategy=random" ;;
  *_const)   base=${VARIANT%_const}; read -r dd ds < <(VARIANT=$base bash "$0"); echo "$dd $ds lr_scheduler_type=constant_with_warmup" ;;
  *_lr2e5)   base=${VARIANT%_lr2e5}; read -r dd ds < <(VARIANT=$base bash "$0"); echo "$dd $ds learning_rate=2.0e-5" ;;
  *_lr5e6)   base=${VARIANT%_lr5e6}; read -r dd ds < <(VARIANT=$base bash "$0"); echo "$dd $ds learning_rate=5.0e-6" ;;
  *_e[0-9])  base=${VARIANT%_e*}; read -r dd ds < <(VARIANT=$base bash "$0"); echo "$dd $ds num_train_epochs=${VARIANT##*_e}.0" ;;
  *) echo "unknown VARIANT $VARIANT" >&2; exit 1 ;;
esac
