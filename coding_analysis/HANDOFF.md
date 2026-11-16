# Terminal Agent 项目交接（157）

**截至 2026-11-16，项目处于第一阶段：用错误分析指导下一轮 SFT。** 已完成 Qwen3.5-4B 的训练、TB2.1 评测、逐轨迹归因和多组复测，也做过 9B 对照。当前最值得验证的是模型交卷前能否发现漏掉的要求。项目希望最终得到能完成爬取、数据整理等真实自动化任务的小模型；阶段目标分数尚未指定。RL 与面向实际场景的专项训练是后续阶段。已有能力分析和论文设计依据见[能力图谱](experiments/c312_capabilities.md)和[论文目录与阅读笔记](paper/README.md)。

## 目前可以引用的分数

|运行|TB2.1 协议|通过 / 全部采样|用途|
|---|---|---:|---|
|Qwen3.5-4B base|旧 verifier，Terminus 1.0.0|50/267，18.73%|4B 训练前基线|
|4B TH3K mask_const，checkpoint-312|同一旧协议|62/178，34.83%|相对 4B base 提高 **16.10 个百分点**|
|同一 c312|mirror 1.0.1，Terminus 1.0.0|**73/178，41.01%**|当前已验证的 4B SFT 参照，48/89 题至少成功一次|
|4B DeepSeek chat 整轨迹 Arm A，c224|同一 mirror 协议|58/178，32.58%|低 8.43 个百分点；TTL 74 条，c312 为 41 条|

分母包含未完成样本。mirror 的 4B base 尚缺，不能用 41.01% 减旧基线估算 SFT 增益。原集群本地 9B c312 曾记 81/178，后因题面和沙箱修正作废了部分样本；教师 dsv41-v2 的 150/178 是本地评测管线的对照。TB4 也只完成部分样本。[实验账本](experiments/README.md) · [本地评测记录](experiments/local_tb21_runs.md)

## 从数据到成绩

从服务器项目根 `/home/chentinghong/llama-factory` 工作。现有模型、数据、缓存和本机训练日志在 `storage/`；旧报告中的 `/storage/openpsi/users/chen/` 属于原集群。本机有 4B 基座和 2441 条 TH3K flat cache。本机 8 × H100、28 层激活卸载的随机调度短跑已完成 10 步，并通过全部 rank 的校验。历史 c312 属于双节点长度分组实验；重测需要原权重或按原配方重训，逐任务 Arena 轨迹需从原平台取回。[本机产物与日志](experiments/server_training_checks.md) · [H100 卸载对照](experiments/h100_offload.md)

1. **整理训练输入。** `coding_sft/prep/prepare_terminal_horizon.py` 将教师交互转成 ShareGPT，保留每轮终端反馈。TH3K 的 2852 条中有 411 条超过 262144 token，余下 2441 条保留完整轨迹，仅屏蔽坏回复的 loss。`prep/build_tokenized.py` 按训练模板分词。过程 judge、成功样本筛选都有实验；评分为零的轨迹仍可能含有有效纠错示范。[数据与教师筛选](methods/teacher_selection.md)
2. **重建 c312。** `coding_sft/configs/` 放训练配方，`coding_sft/launch/` 放提交入口。使用 Qwen3.5-4B-text、全参 BF16、ZeRO-3、FA2、长度分组、有效 batch 64、LR 1e-5、warmup 12 步。两台各 8 卡的节点从同一个 flat cache 训练，依次完成 117、195、312 步。[训练命令与环境](training.md)
3. **部署并复测。** `coding_eval/tools/build_qwen35_serve_dir.py` 将 SFT 语言权重与基座视觉/MTP 权重组合，保留训练模板。在可用的 SGLang 服务加载后注册模型。已有注册模型可以直接提交。复测使用同一 benchmark、harness 和两次采样。提交后保存平台返回的 job ID。[评测命令与口径](evaluation.md)
4. **从失败回到数据。** 先核对 verifier 是否运行，再对照同题的轨迹、终端输出与失败断言。将可改变的行为写成训练样本与复测指标，保持对照组配置一致。一轮结束后把数据版本、checkpoint、job、分子分母与逐题得失写入[实验账本](experiments/README.md)。

**本机常用命令**（从项目根目录执行）。第一条可从已有快照复算历史分数。第二条在单节点 8 卡环境中重跑 H100 十步验证，输出到新的 Ceph 目录。历史双节点 c312 的分段训练命令见[训练指南](training.md)。

```bash
cd /home/chentinghong/llama-factory
awk -F '\t' '$2=="cth-tb21m-mask-const-c312-2611051616" {print $11 "/" $8, $12}' \
  coding_analysis/experiments/evidence/all_jobs_offline_2026-11-07.tsv

ST=$(readlink -f storage)
"$ST/conda/envs/lf/bin/python" coding_sft/h100/launch.py \
  --output "$ST/runs/h100-th3k-$(date +%Y%m%d-%H%M%S)"
```

**重新评测 c312**（在已登录 Arena、可访问注册模型的节点执行）。从响应的 `data.job_id` 取得实际作业号，再拉取轨迹和运行评分汇总。[评测指南](evaluation.md)

```bash
arena job create --benchmark cth-tb21-full89-mirror@1.0.1 \
  --harness cth-terminus-2-stream@1.0.0 --model-ref abl-mask-const-c312-chen-ais@1.0.0 \
  --samples 2 --max-parallel 32 --max-retries 1 --retry-backoff 60 \
  --job-id "c312-mirror-$(date +%y%m%d%H%M)" --tags chen,tb21 --json
```

## 下一步先做什么

- **补足对照。** 用同一 mirror / 1.0.0 设置补测 4B base 和已训好的 Arm A c248。复核已有 c312 结果，记录负载、TTL 与各题两次采样的得失。独立轨迹集和隐藏测试用于验证泛化。
- **先验证交卷核对，再改数据。** c312 的 105 条失败中，42 条交卷后部分通过，其中 24 条只差一个测试；人工复盘另按主因归出 26 条漏需求或未运行即交卷。先只改确认提示，要求为每项显式需求给出执行证据，与原 c312 配对复测。若有效，在独立任务中采集“发现漏项→修复→复验”轨迹做 SFT；同时看 pass@all、部分通过数和交卷后通过率。[逐条失败依据](experiments/c312_failure_cases.md)
- **再验证终端恢复。** 13 条失败来自前台程序占用、续行或交互 shell 提前退出。加入能判断当前状态、恢复提示符并继续交付的短轨迹，复测终端失效数、成功恢复数和 TTL。构建打包和服务配置在 c312 的各 7 题中均通过 11/14 个样本，可作为以后选择实用方向的线索。爬虫方向需要另设有验收条件的站点采集任务，用字段正确性、分页完整性和重复率验证。[能力与场景证据](experiments/c312_capabilities.md)
