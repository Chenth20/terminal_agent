# TB2.1 ablation results (tb21-terminus2-full@2.1.3, harness cth-terminus-2-stream@1.0.0)

> 本表保存 10 月旧 verifier 的逐题比较。11 月 mirror 与本地评测的得分、样本数和复现条件见[实验索引](README.md)。

> **归档说明（11-06）**：这份是 10-25/26 的快照，已被 [sft_results](sft_results.md) 的汇总表取代；保留它是因为里面有逐 job 的 serving 细节和 pass@done 列。

Snapshot 2026-10-25 10:19 CST; const 5-epoch rows added 2026-10-26 03:40; const 8-epoch rows (ckpt-234/273/312, jobs `cth-tb21-stream-abl-mask-const-c{234,273,312}-261026091{6,6,20}`, AIS 350860763/350861086/350861164) updated 21:35 — ep6 DONE, ep7/ep8 stopped 21:39 with their last 2+1 retried tasks cancelled (4th attempt on password-recovery / db-wal-recovery / gpt2-codegolf; all three had failed model-side three times already) — counted 0. slurm `cth-abl-g` cancelled. Epoch sweep write-up: [sft_results](sft_results.md). 89 envs; abl-* jobs = 2 samples/env (178 tasks), Qwen base jobs = 3 samples/env (267). Serving identical for every row: sglang 85b53914, tp1 engines, fp32 SSM state, EAGLE MTP (steps 3 / draft 4, base mtp head), KV patch, sampling T=1 / top_p 0.95 / top_k 20 / reasoning_effort xhigh, ctx 262144. SFT checkpoints served as VLM-format dirs (SFT language_model weights + base visual/mtp tensors + SFT chat template).

pass@done = passed / DONE tasks; **pass@all = passed / all tasks (HARNESS_FAILED / TIMEOUT counted 0 — the number to report)**. 1 task = 0.0056 for abl-* rows; differences below ~0.02 (3-4 tasks) are noise at 2 samples.

## Summary

| variant | job | status | DONE | passed | pass@done | pass@all | non-DONE | mean agent min | train loss |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B base (x3) | `4b` | DONE | 262/267 | 50 | 0.191 | **0.187** | HARNESS_FAILED=4, TIMEOUT=1 | 47 |  |
| Qwen3.5-9B base (x3) | `9b` | DONE | 257/267 | 66 | 0.257 | **0.247** | HARNESS_FAILED=10 | 38 |  |
| A1 base (drop bad traj), final | `base-final` | DONE | 175/178 | 51 | 0.291 | **0.287** | HARNESS_FAILED=3 | 114 | 0.562 |
| A1 base, ckpt-70 | `base-c70` | DONE | 171/178 | 43 | 0.251 | **0.242** | HARNESS_FAILED=7 | 90 |  |
| A2 mask (bad turns loss=0), final | `mask-final` | DONE | 168/178 | 51 | 0.304 | **0.287** | HARNESS_FAILED=10 | 114 | 0.820 |
| A2 mask, ckpt-70 | `mask-c70` | DONE | 172/178 | 42 | 0.244 | **0.236** | HARNESS_FAILED=5, TIMEOUT=1 | 87 |  |
| A3 keep (bad turns supervised), final | `keep-final` | DONE | 175/178 | 46 | 0.263 | **0.258** | HARNESS_FAILED=3 | 126 | 0.000 |
| A3 keep, ckpt-70 | `keep-c70` | DONE | 160/178 | 42 | 0.263 | **0.236** | EVAL_FAILED=1, HARNESS_FAILED=17 | 111 |  |
| B2 mask_cut (over-long prefix-truncated) | `mask-cut` | DONE | 174/178 | 43 | 0.247 | **0.242** | HARNESS_FAILED=4 | 132 | 1.076 |
| B3 mask_tail (prompt + suffix window) | `mask-tail` | DONE | 165/178 | 42 | 0.255 | **0.236** | EVAL_FAILED=1, HARNESS_FAILED=10, TIMEOUT=2 | 118 | 0.320 |
| C2 mask_rand (random sampler) | `mask-rand` | DONE | 170/178 | 43 | 0.253 | **0.242** | HARNESS_FAILED=8 | 119 | 1.137 |
| C3 mask_pack (neat packing) | `mask-pack` | DONE | 171/178 | 42 | 0.246 | **0.236** | HARNESS_FAILED=7 | 123 | 1.144 |
| D1 mask_const (constant lr 1e-5) | `mask-const` | DONE | 174/178 | 53 | 0.305 | **0.298** | HARNESS_FAILED=4 | 109 | 1.128 |
| D2 mask_lr2e5 (lr 2e-5) | `mask-lr2e5` | DONE | 167/178 | 51 | 0.305 | **0.287** | HARNESS_FAILED=11 | 119 | 1.115 |
| D3 mask_lr5e6 (lr 5e-6) | `mask-lr5e6` | DONE | 172/178 | 48 | 0.279 | **0.270** | HARNESS_FAILED=5, TIMEOUT=1 | 138 | 1.154 |
| D1 mask_const, 5 ep: ckpt-156 (epoch 4) | `mask-const-c156` | DONE | 171/178 | 53 | 0.310 | **0.298** | HARNESS_FAILED=7 | 122 | last-10 1.048 (step 150-160) |
| D1 mask_const, 5 ep: ckpt-195 (epoch 5, final) | `mask-const-c195` | DONE | 167/178 | 57 | 0.341 | **0.320** | HARNESS_FAILED=10, TIMEOUT=1 | 100 | last-10 1.021 |
| D1 mask_const, 8 ep: ckpt-234 (epoch 6) | `mask-const-c234` | DONE | 176/178 | 59 | 0.335 | **0.331** | HARNESS_FAILED=2 | 99 | last-10 0.997 |
| D1 mask_const, 8 ep: ckpt-273 (epoch 7) | `mask-const-c273` | STOPPED (final) | 172/178 | 53 | 0.308 | **0.298** | HARNESS_FAILED=4, CANCELLED=2 | 95 | last-10 0.981 |
| D1 mask_const, 8 ep: ckpt-312 (epoch 8, final) | `mask-const-c312` | STOPPED (final) | 175/178 | 62 | 0.354 | **0.348** | HARNESS_FAILED=2, CANCELLED=1 | 98 | last-10 0.947 |
| D1 mask_const 8 ep ckpt-312, **harness 1.0.0-json** (force response_format=json_object), ×3 | `mask-const-c312-json` | DONE | 257/267 | 88 | 0.342 | **0.330** | HARNESS_FAILED=9, TIMEOUT=1 | 125 | (same weights) |

## Notes

- **2026-10-27 force-JSON**: same epoch-8 weights, harness `1.0.0-json` (= 1.0.0 + `TERMINUS_FORCE_JSON_OBJECT=true`), 3 samples: 0.330 vs 0.348 plain — no gain, less consistent (3/3 solves 14 vs 2/2 21), +27 min/episode; 977 wasted "junk JSON" turns (`{"status":"final"}` etc.). 4B base gives the same picture (0.169 vs 0.187 plain); harness `1.0.0-json` deprecated + deleted 2026-10-28. Write-up [sft_results](sft_results.md). Not comparable row-for-row with the ×2 rows (different harness); listed for reference only.

- **2026-10-26: `mask_const` 5-epoch run** (`abl_mask_const_e5_slurm2n`, 195 steps, same constant lr 1e-5): ckpt-156 (epoch 4) 0.298 = the 3-epoch final; ckpt-195 (epoch 5) **0.320** (57/178), +4 tasks over 3 epochs — the best row so far. Failures: all model-side (tmux/`C-d` envs + context filled at ~258k mid-stream: video-processing, sqlite-db-truncate, circuit-fibsqrt, make-doom-for-mips, protein-assembly, extract-elf; one TIMEOUT mailman:0). AIS 350622444 / 350622741 stopped.

- All 15 jobs are DONE (last task finished 2026-10-25 00:05 CST). Every non-DONE task was inspected via `arena job log <job> <task> harness`:
  - **model-side, counted 0 (not retried)**: tmux `rc=-15` = agent killed at the 1770 s episode limit; `no server running` = model sent `C-d`; TIMEOUT; `upstream stream request failed` after `max_tokens limit` = the model filled its 256K context mid-stream (engine logs `Abort running request`, no outage). These cluster on the same envs for every model: git-multibranch, install-windows-3.11, configure-git-webserver, rstan-to-pystan, make-mips-interpreter.
  - **infra**: 1 task only (mask_pack extract-elf:1, gateway 503 at 00:02 when the slurm serving jobs were cancelled - the arena job was already at 177/178). Not re-run; worst case +0.006.
  - the 10 HARNESS_FAILED on `mask-final` and 17 on `keep-c70` are from the earlier AIS pre-emption period and were retried once; what remains is model-side.
- Serving for the new rows: lr2e5 / const on slurmd-64 (4 GPUs each), lr5e6 / pack on slurmd-76 (4 GPUs each), rand on one AIS node (8 GPUs); the earlier rows ran on 1 AIS node or 2 slurm GPUs. Concurrency does not change per-task behaviour (each episode is a single sequential agent), so the rows are comparable.
- Training loss is `train_loss` from all_results.json (mean over the whole run); the Stage A/B runs that resumed from checkpoints report a partial-run mean (keep = 0.0, tail = 0.32, base = 0.56) and are not comparable - see coding_analysis/experiments/sft_results.md for the last-10-step losses (all 1.088-1.103).
- Reference: earlier Qwen3.5-4B x3 runs on 10-21: bf16 SSM 0.150, fp32 SSM 0.224 at 210/267.

## Envs solved at least once (any sample)

| variant | envs solved | of 89 | status |
|---|---|---|---|
| `4b` | 30 | 0.337 | DONE |
| `9b` | 37 | 0.416 | DONE |
| `base-final` | 33 | 0.371 | DONE |
| `base-c70` | 32 | 0.360 | DONE |
| `mask-final` | 35 | 0.393 | DONE |
| `mask-c70` | 30 | 0.337 | DONE |
| `keep-final` | 33 | 0.371 | DONE |
| `keep-c70` | 29 | 0.326 | DONE |
| `mask-cut` | 30 | 0.337 | DONE |
| `mask-tail` | 29 | 0.326 | DONE |
| `mask-rand` | 28 | 0.315 | DONE |
| `mask-pack` | 30 | 0.337 | DONE |
| `mask-const` | 37 | 0.416 | DONE |
| `mask-lr2e5` | 35 | 0.393 | DONE |
| `mask-lr5e6` | 33 | 0.371 | DONE |

## Per-env pass (passed/DONE; `-` = nothing DONE yet)

| env | 4b | 9b | base-final | base-c70 | mask-final | mask-c70 | keep-final | keep-c70 | mask-cut | mask-tail | mask-rand | mask-pack | mask-const | mask-lr2e5 | mask-lr5e6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive-rejection-sampler | 1/3 | 0/3 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| bn-fit-modify | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 |
| break-filter-js-from-html | 0/3 | 0/3 | 1/2 | 0/2 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 1/2 | 0/2 |
| build-cython-ext | 0/3 | 0/3 | 2/2 | 1/2 | 0/2 | 1/2 | 1/2 | 2/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 |
| build-pmars | 1/3 | 0/3 | 2/2 | 2/2 | 2/2 | 0/2 | 1/2 | 2/2 | 2/2 | 1/2 | 1/2 | 1/2 | 2/2 | 1/2 | 1/2 |
| build-pov-ray | 0/3 | 1/3 | 1/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| caffe-cifar-10 | 0/3 | 0/3 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 |
| cancel-async-tasks | 1/3 | 1/3 | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 | 1/2 | 2/2 | 0/2 | 1/2 | 0/2 | 1/2 |
| chess-best-move | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| circuit-fibsqrt | 0/3 | 0/1 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/1 | 0/2 | 0/1 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 |
| cobol-modernization | 2/3 | 1/3 | 2/2 | 0/2 | 1/2 | 0/2 | 2/2 | 1/2 | 1/2 | 2/2 | 2/2 | 1/2 | 1/2 | 1/2 | 2/2 |
| code-from-image | 2/3 | 1/3 | 1/2 | 1/2 | 1/2 | 0/2 | 1/2 | 1/2 | 0/2 | 1/2 | 1/2 | 1/2 | 2/2 | 0/2 | 2/2 |
| compile-compcert | 1/3 | 2/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| configure-git-webserver | 2/2 | 2/3 | 0/2 | - | 1/1 | 1/1 | 0/1 | - | 2/2 | 1/1 | 1/1 | 1/2 | 2/2 | 1/1 | 1/1 |
| constraints-scheduling | 1/3 | 1/3 | 1/2 | 0/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 1/2 | 1/2 | 2/2 | 2/2 | 1/2 | 2/2 |
| count-dataset-tokens | 0/3 | 0/3 | 1/2 | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 |
| crack-7z-hash | 0/3 | 1/3 | 1/2 | 1/2 | 1/2 | 0/2 | 1/2 | 1/1 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 |
| custom-memory-heap-crash | 1/3 | 0/3 | 0/2 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| db-wal-recovery | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| distribution-search | 2/3 | 2/3 | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 | 1/2 | 1/2 | 2/2 | 1/2 | 1/2 | 1/2 | 2/2 | 1/2 |
| dna-assembly | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| dna-insert | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| extract-elf | 0/3 | 1/3 | 0/2 | 1/2 | 0/1 | 0/1 | 0/2 | 0/1 | 0/2 | 0/1 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 |
| extract-moves-from-video | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| feal-differential-cryptanalysis | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| feal-linear-cryptanalysis | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 |
| filter-js-from-html | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 |
| financial-document-processor | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| fix-code-vulnerability | 2/3 | 3/3 | 2/2 | 1/2 | 2/2 | 1/2 | 2/2 | 0/2 | 1/2 | 1/2 | 1/2 | 0/2 | 1/2 | 2/2 | 1/2 |
| fix-git | 2/3 | 3/3 | 2/2 | 2/2 | 1/2 | 2/2 | 1/2 | 0/2 | 0/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| fix-ocaml-gc | 1/3 | 0/3 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 1/1 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 2/2 | 0/2 |
| gcode-to-text | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| git-leak-recovery | 2/3 | 1/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| git-multibranch | 1/3 | 2/3 | 0/2 | - | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | - | 0/1 | 0/1 | - | - |
| gpt2-codegolf | 0/2 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| headless-terminal | 1/3 | 1/3 | 1/2 | 0/2 | 2/2 | 0/2 | 1/2 | 0/2 | 2/2 | 1/2 | 0/2 | 0/1 | 2/2 | 1/2 | 0/2 |
| hf-model-inference | 2/3 | 1/3 | 1/2 | 1/2 | 1/2 | 2/2 | 1/2 | 2/2 | 1/2 | 2/2 | 0/2 | 1/2 | 2/2 | 2/2 | 2/2 |
| install-windows-3.11 | 0/3 | 0/3 | - | 0/1 | 0/2 | - | 0/2 | 0/1 | 0/2 | - | - | - | - | - | 0/2 |
| kv-store-grpc | 3/3 | 3/3 | 2/2 | 2/2 | 2/2 | 0/2 | 1/2 | 1/2 | 2/2 | 2/2 | 1/2 | 2/2 | 1/2 | 1/2 | 1/2 |
| large-scale-text-editing | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 |
| largest-eigenval | 0/3 | 3/3 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 0/2 | 1/2 | 2/2 | 2/2 |
| llm-inference-batching-scheduler | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| log-summary-date-ranges | 1/3 | 3/3 | 1/2 | 1/2 | 2/2 | 2/2 | 1/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 1/2 | 0/2 | 1/2 |
| mailman | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| make-doom-for-mips | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| make-mips-interpreter | 0/3 | 0/3 | 0/2 | 0/1 | 0/1 | 0/2 | 0/1 | 0/1 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | - |
| mcmc-sampling-stan | 0/3 | 1/3 | 0/2 | 2/2 | 2/2 | 1/2 | 1/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 2/2 | 1/2 | 0/2 |
| merge-diff-arc-agi-task | 1/3 | 2/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 | 1/2 | 2/2 | 2/2 | 1/2 | 2/2 | 1/2 | 2/2 |
| model-extraction-relu-logits | 0/3 | 1/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| modernize-scientific-stack | 3/3 | 2/3 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 1/2 | 1/2 | 1/2 | 2/2 | 2/2 |
| mteb-leaderboard | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 |
| mteb-retrieve | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 |
| multi-source-data-merger | 3/3 | 3/3 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 |
| nginx-request-logging | 0/3 | 2/3 | 0/2 | 0/2 | 1/2 | 1/2 | 1/2 | 0/2 | 2/2 | 0/2 | 0/2 | 2/2 | 0/2 | 2/2 | 2/2 |
| openssl-selfsigned-cert | 3/3 | 1/3 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 |
| overfull-hbox | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 |
| password-recovery | 1/3 | 3/3 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 1/2 | 0/2 | 2/2 | 0/1 | 0/2 |
| path-tracing | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| path-tracing-reverse | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| polyglot-c-py | 0/2 | - | 2/2 | 1/2 | 1/2 | 1/2 | 0/2 | 2/2 | 1/2 | 1/2 | 2/2 | 1/2 | 1/2 | 1/2 | 2/2 |
| polyglot-rust-c | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| portfolio-optimization | 2/3 | 2/3 | 2/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 | 1/2 | 2/2 |
| protein-assembly | 0/3 | 0/3 | 0/1 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 |
| prove-plus-comm | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| pypi-server | 0/3 | 2/3 | 2/2 | 1/2 | 1/2 | 2/2 | 2/2 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 |
| pytorch-model-cli | 2/3 | 2/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| pytorch-model-recovery | 0/3 | 2/3 | 0/2 | 0/2 | 1/2 | 1/2 | 1/2 | 1/2 | 2/2 | 0/2 | 1/2 | 1/2 | 1/2 | 2/2 | 0/2 |
| qemu-alpine-ssh | 0/3 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 |
| qemu-startup | 1/3 | 3/3 | 0/2 | 0/2 | 0/1 | 0/2 | 1/2 | 1/2 | 1/2 | 1/2 | 0/2 | 1/2 | 1/2 | 1/2 | 0/1 |
| query-optimize | 1/3 | 1/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 |
| raman-fitting | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| regex-chess | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| regex-log | 0/3 | 0/3 | 1/2 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| reshard-c4-data | 0/3 | 0/3 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 | 1/2 | 1/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 1/2 |
| rstan-to-pystan | 0/3 | 1/3 | 2/2 | 1/1 | 1/2 | 1/2 | 1/2 | 0/1 | 0/1 | 1/2 | 0/1 | 1/2 | 0/1 | 1/1 | 1/2 |
| sam-cell-seg | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| sanitize-git-repo | 0/3 | 0/3 | 1/2 | 1/2 | 1/2 | 0/2 | 1/2 | 1/2 | 0/2 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 1/2 |
| schemelike-metacircular-eval | 0/1 | 0/3 | 0/2 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 | 0/2 | 1/2 | 0/2 |
| sparql-university | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| sqlite-db-truncate | 0/3 | 0/3 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| sqlite-with-gcov | 2/3 | 2/3 | 2/2 | 1/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 |
| torch-pipeline-parallelism | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| torch-tensor-parallelism | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| train-fasttext | 0/3 | 0/3 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| tune-mjcf | 0/3 | 1/3 | 1/2 | 1/2 | 1/2 | 2/2 | 0/2 | 0/2 | 1/2 | 0/2 | 2/2 | 0/2 | 1/2 | 1/2 | 1/2 |
| video-processing | 0/3 | 0/3 | 0/2 | 0/2 | 0/1 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| vulnerable-secret | 2/3 | 2/3 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 |
| winning-avg-corewars | 0/3 | 0/2 | 1/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| write-compressor | 0/3 | 0/3 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |

## Per-env pass count (of 2 samples), envs with ≥1 pass in any epoch

| env | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 |
|---|---|---|---|---|---|---|
| adaptive-rejection-sampler | 0 | 0 | 1 | 1 | 1 | 0 |
| bn-fit-modify | 1 | 0 | 0 | 0 | 1 | 1 |
| break-filter-js-from-html | 0 | 1 | 2 | 1 | 1 | 1 |
| build-cython-ext | 2 | 1 | 1 | 0 | 2 | 1 |
| build-pmars | 2 | 1 | 0 | 2 | 2 | 1 |
| caffe-cifar-10 | 0 | 1 | 0 | 0 | 1 | 2 |
| cancel-async-tasks | 1 | 2 | 0 | 0 | 0 | 0 |
| cobol-modernization | 1 | 0 | 2 | 2 | 1 | 2 |
| code-from-image | 2 | 1 | 1 | 2 | 1 | 2 |
| compile-compcert | 1 | 0 | 2 | 0 | 0 | 0 |
| configure-git-webserver | 2 | 1 | 1 | 1 | 2 | 2 |
| constraints-scheduling | 2 | 1 | 2 | 1 | 1 | 1 |
| count-dataset-tokens | 2 | 2 | 1 | 2 | 1 | 2 |
| crack-7z-hash | 0 | 1 | 2 | 1 | 1 | 2 |
| custom-memory-heap-crash | 0 | 0 | 1 | 0 | 1 | 0 |
| distribution-search | 1 | 2 | 2 | 2 | 1 | 2 |
| extract-elf | 0 | 0 | 1 | 1 | 0 | 1 |
| financial-document-processor | 0 | 0 | 1 | 0 | 0 | 0 |
| fix-code-vulnerability | 1 | 1 | 1 | 2 | 2 | 1 |
| fix-git | 1 | 1 | 2 | 1 | 1 | 1 |
| fix-ocaml-gc | 0 | 0 | 1 | 1 | 0 | 1 |
| git-leak-recovery | 2 | 2 | 2 | 1 | 2 | 2 |
| git-multibranch | 0 | 0 | 0 | 0 | 0 | 1 |
| headless-terminal | 2 | 1 | 1 | 1 | 1 | 0 |
| hf-model-inference | 2 | 1 | 2 | 2 | 1 | 2 |
| install-windows-3.11 | 0 | 0 | 0 | 0 | 1 | 0 |
| kv-store-grpc | 1 | 2 | 2 | 2 | 2 | 2 |
| largest-eigenval | 1 | 2 | 2 | 2 | 2 | 2 |
| log-summary-date-ranges | 1 | 1 | 2 | 1 | 2 | 2 |
| mcmc-sampling-stan | 2 | 0 | 2 | 1 | 1 | 1 |
| merge-diff-arc-agi-task | 2 | 2 | 1 | 2 | 2 | 2 |
| modernize-scientific-stack | 1 | 2 | 1 | 2 | 1 | 2 |
| mteb-leaderboard | 0 | 0 | 0 | 0 | 0 | 1 |
| multi-source-data-merger | 2 | 1 | 2 | 2 | 2 | 2 |
| nginx-request-logging | 0 | 2 | 2 | 1 | 0 | 2 |
| openssl-selfsigned-cert | 2 | 2 | 1 | 2 | 2 | 2 |
| overfull-hbox | 0 | 1 | 0 | 1 | 0 | 0 |
| password-recovery | 2 | 1 | 1 | 1 | 0 | 0 |
| polyglot-c-py | 1 | 0 | 2 | 2 | 1 | 1 |
| portfolio-optimization | 2 | 2 | 2 | 2 | 2 | 2 |
| pypi-server | 1 | 1 | 2 | 2 | 1 | 2 |
| pytorch-model-cli | 0 | 1 | 0 | 1 | 0 | 0 |
| pytorch-model-recovery | 1 | 2 | 0 | 1 | 1 | 1 |
| qemu-startup | 1 | 0 | 1 | 1 | 0 | 1 |
| query-optimize | 1 | 0 | 0 | 0 | 0 | 0 |
| regex-log | 1 | 1 | 0 | 1 | 1 | 1 |
| reshard-c4-data | 1 | 1 | 1 | 1 | 1 | 1 |
| rstan-to-pystan | 0 | 0 | 1 | 1 | 0 | 2 |
| sanitize-git-repo | 0 | 2 | 0 | 0 | 0 | 0 |
| schemelike-metacircular-eval | 0 | 0 | 0 | 1 | 0 | 0 |
| sqlite-db-truncate | 0 | 0 | 0 | 0 | 1 | 0 |
| sqlite-with-gcov | 2 | 2 | 1 | 2 | 1 | 1 |
| train-fasttext | 0 | 1 | 0 | 0 | 0 | 1 |
| tune-mjcf | 1 | 1 | 0 | 1 | 1 | 2 |
| vulnerable-secret | 1 | 2 | 2 | 2 | 2 | 1 |
| winning-avg-corewars | 1 | 1 | 0 | 0 | 2 | 0 |
