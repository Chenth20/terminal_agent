# 从训练 checkpoint 到 TB2.1 得分

命令从项目根目录执行。训练机有 Qwen3.5-4B 原始基座；11 月 7 日导出的 [276 个 Arena 作业快照](experiments/evidence/all_jobs_offline_2026-11-07.tsv) 可直接查分。训练机未安装 Arena CLI，也未挂载旧集群上的逐样本轨迹、c312 权重和 Slurm 本地评测目录。提交与拉轨迹的命令在已登录 Arena、可访问已注册模型的工作节点执行。

## 1. 准备模型

已有模型版本可直接进入第 2 步。对于新的 4B 全参文本 checkpoint，设 `CKPT` 为包含 `model.safetensors`、tokenizer 和模板的目录，`SERVE_DIR` 为模型部署目录。

```bash
storage/conda/envs/lf/bin/python coding_eval/tools/build_qwen35_serve_dir.py \
    "$CKPT" "$SERVE_DIR" --base storage/models/Qwen__Qwen3.5-4B
```

脚本对齐 426 个语言模型张量，使用训练时的 tokenizer 和历史推理模板，并补入基座的 312 个视觉和 MTP 张量。部署端加载该目录后，在 Arena 登记版本化的模型引用。历史生产引擎是 SGLang 0.5.18（85b53914），TP=1、SSM fp32、262144 上下文、EAGLE MTP。迁移包只含评测资产；AIS 的启动、路由注册和 Slurm 的 `serve_multi.sh` 属于原平台。记录新模型时同时保存 checkpoint、模型引用和引擎参数。[平台资产](arena_assets/README.md)

历史压测中，开启 fp32 SSM 的 4B 单引擎在修正 MTP draft KV 层数（1 层）后可容纳约 174 万 KV token。相同设备上，短提示词吞吐与实际长轨迹并发差异很大：上下文中位数约 90k 时，8 引擎节点实测并发约 50–60。单卡上下文约 40k/75k/125k 时，高效并发约 24/14/8。短请求运行 `python3 coding_eval/stress/load.py N MAX_TOKENS`，长请求运行 `python3 coding_eval/stress/load_long.py N MAX_TOKENS`，持续负载与独立 KV 前缀扫描分别用 `sustain.py` 和 `stress_kv.py`。旧 AIS 日志保存在原平台 `/storage/openpsi/users/chen/code_data/arena/logs/`。

## 2. 提交、收集、计分

复测 4B c312 的 73/178 分数使用 mirror verifier、1.0.0 harness、89 题 × 2 次。保留平台返回的实际 job ID，历史平台可能在申请的 ID 后追加后缀。

```bash
model=abl-mask-const-c312-chen-ais@1.0.0
arena job create --model-ref "$model" --job-id "c312-mirror-$(date +%y%m%d%H%M)" \
  --benchmark cth-tb21-full89-mirror@1.0.1 --harness cth-terminus-2-stream@1.0.0 \
  --samples 2 --max-parallel 32 --max-retries 1 --retry-backoff 60 --tags chen,tb21 --json

# 将 JOB 设为上一步返回的 data.job_id，OUT 设为可写的快照目录
bash coding_analysis/arena_assets/legacy/scripts/pull_trajectories.sh "$JOB" "$OUT"
export OUT
PYTHONPATH=coding_eval/tools PYTHONDONTWRITEBYTECODE=1 python3 -c \
  'import os, offline_results as r; r.OUT_ROOT=os.environ["OUT"]; r.main()' --out "$OUT/_summary"
```

157 的 `offline_results.py` 将原集群输入目录固定为 `/storage/openpsi/users/chen/output`，CLI 的 `--out` 指定汇总输出。上例仅在当前 Python 进程将 `OUT_ROOT` 指向拉取的目录。原集群直接运行 `python3 coding_eval/tools/offline_results.py` 即可。

`$OUT/$JOB/job.json` 保存任务列表，`tasks/<env>/s<idx>/` 保存轨迹及 harness、runner、reward、pytest 日志。汇总输出在 `$OUT/_summary/all_jobs.tsv` 和 `all_jobs.json`。对比全量任务中的通过样本数（DONE 且 score ≥ 1），同时记 DONE 数、总样本数、每题至少过一次的题目数和失败状态。非 DONE 计零。快照为 RUNNING 或 STOPPED 时在结果中写明状态。

旧基线复测时显式切换 benchmark 并恢复 ×3。镜像修复后的新模型与旧基线应各自在相同 benchmark/harness 下运行。

```bash
arena job create --model-ref qwen35-4b-fp32-chen-ais@1.0.0 --job-id "base4b-$(date +%y%m%d%H%M)" \
  --benchmark tb21-terminus2-full@2.1.3-gepa-pytorch-cache-v2 --harness cth-terminus-2-stream@1.0.0 \
  --samples 3 --max-parallel 32 --max-retries 1 --retry-backoff 60 --tags chen,tb21 --json
```

| 对照目的 | benchmark / harness | 已有结果 |
|---|---|---|
| 与旧 4B base 比 SFT | 旧 TB2.1 / 1.0.0 | base 50/267，mask_const c312 62/178，相差 16.1 个百分点 |
| 修复 verifier 后比较数据配方 | mirror 1.0.1 / 1.0.0 | c312 73/178（170 DONE），dsv41j A c224 58/178（171 DONE） |
| 新环境整轨迹复测 | mirror 1.0.1 / 1.6.262 | `--harness cth-terminus-2-stream@1.6.262`；保留该环境下的 c312 对照 |
| 96k 分段或教师对照 | mirror 1.0.1 / 1.6.96 | 96k 触发、512 步上限、usage 计数，单独报告 |

原始版本设置见 [harness 配置](arena_assets/harness/configs)。压缩档改变了计数方式、步数上限和镜像环境。local TB2.1 的 instruction、tests、oracle 投递时机、核数和内存也要对齐。11 月 14 日作废过旧题面和旧沙箱样本；后续有完整 178 个有效样本时才发布该臂终值。[本地运行记录](experiments/local_tb21_runs.md) · [本地评测流程](methods/tb21_local_pipeline.md)

## 3. 从失败回到训练

```bash
python3 coding_eval/tools/verifier_audit.py "$OUT/$JOB"
python3 coding_eval/tools/classify_tb21_errors.py "$OUT/$JOB"
python3 coding_eval/tools/traj_stats.py "$OUT/$JOB"
```

先看 pytest 是否真正运行，再看失败样本的交卷时刻、测试输出和 terminal 观察。c312 的 41 条 TTL 是 4 小时预算耗尽；有长推理循环，也有真实依赖和任务难度因素。分类脚本以 agent 耗时 ≥235 min 近似 TTL，以 ≥500 步近似 512 步上限。缺失日志或导入异常需结合 reward 与工作目录核实。非 DONE 可在有 Arena 的节点运行 `classify_tb21_failures.py JOB`；模糊的 stream 中断需查服务日志及末轮 token，曾有单轮失控输出填满 262k 导致 400。[逐题失败复盘](experiments/c312_failure_cases.md)

下一轮用同一 benchmark、harness、采样数和温度复测，在总分旁列「提前交卷后测试失败」「达到时间/步数上限」和逐题得失。对失败轨迹的虚假交卷轮做 loss mask、补充修正后自测通过的轨迹，可先验证交卷后失败是否减少。保留已成功样本的收尾自审，避免把有效的验证步骤一并裁掉。[训练轨迹与错误模式](experiments/trajectory_failures.md) · [实验总表](experiments/sft_results.md)
