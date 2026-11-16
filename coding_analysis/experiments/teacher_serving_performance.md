# Qwen3.8-27B 推理服务性能：SGLang vs vLLM

报告整理日期：2026-08-26
报告中的 SGLang 服务镜像均为 /storage/openpsi/images/glm-5-2-2fbb8e.sif，不包含、也不代表新镜像 deepep-base-85b53914.sif 的性能
1. 结论摘要
1. 完整性与数据质量：矩阵应有 240 个 case，240/240 均有最终原始结果；204 个成功，24 个经 KV 容量 preflight 判定为不支持，12 个实际运行失败。逐一核验后，原始结果与标准化报告之间没有状态、维度、数据集 SHA 或性能指标不一致；所有成功 case 都完成了预期请求数、输出均严格为 256 tokens、case 前 cache reset 成功。
2. SGLang 的 prefill 整体比 vLLM 慢，旧镜像下差距稳定但并非数量级差距。在双方均成功的 case 上，SGLang 相对 vLLM 的 TTFT 中位变化为：MTP-off T1D8 +7.8%、MTP-off T2D4 +7.7%、MTP-on T2D4 +8.0%；对应输入吞吐分别为 -6.9%、-6.0%、-2.2%。256K/C1 的 T2D4 TTFT 为：MTP-on 114.3s vs 104.9s，MTP-off 107.9s vs 100.1s（SGLang vs vLLM）。
3. 不能概括为“MTP 一定更快、吞吐一定更高”。在完整 30 对 T2D4 配对上，开启 MTP 后：T1D8 上 MTP 的 decode 收益较明显，但 SGLang 只剩 6 个 32K 配对，且 64K 以上因 KV 容量不足全部不支持；vLLM MTP-on 也有 CUDA illegal-memory-access 相关失败。因此 MTP 的收益高度依赖并行策略、上下文、并发和实现，不能用 T1D8 的少数 case 外推完整矩阵
  ○ SGLang：TTFT +6.6%、E2E +1.5%、输入吞吐 -2.0%，只有 TPOT 改善 -5.6%；
  ○ vLLM：TTFT +6.4%、E2E +4.0%、输入吞吐 -5.0%、TPOT +1.9%，中位数反而全面变差。
4. T2D4 是更适合长上下文 prefill 的策略，但 TPOT 存在取舍。T2D4 相对 T1D8 的配对中位数：TTFT 改善 16.7%–30.8%，E2E 改善 5.9%–30.2%，输入吞吐提高 29.2%–50.3%。但是 TPOT 并非总是改善：SGLang MTP-on +53.6%、vLLM MTP-on +15.4%、vLLM MTP-off +8.5%（正值表示更慢）；只有 SGLang MTP-off 的 TPOT 改善 10.7%
5. SGLang 不应笼统地说“所有阶段都比 vLLM 慢”。旧镜像下 SGLang 的主要弱项是 prefill/TTFT 和整体输入吞吐；decode TPOT 在 T2D4 上反而优于 vLLM：MTP-on 中位 -13.7%，MTP-off -2.7%。这说明“prefill 慢”比“后端整体慢”更准确
6. 共享前缀命中率衡量的是整套 DP 系统的有效命中，不是单 worker 理论上限。C1/C4/C8 没有组内重复，命中应为零；C16/C32/C64 分别约 2/4/8 个请求共享同一前缀。测试没有发送 routing key（没有亲和路由），同组请求可能被分发到不同 DP worker。C64 的上下文中位有效命中率，T2D4 为 SGLang 35.9%/37.3%（MTP off/on），vLLM 32.9%/32.9%。但 vLLM 的 query-counter 与 source-accounting 分母不同，因此跨后端绝对命中率只能作为系统行为信号，不能单独用于判定 cache 引擎优劣
7. 稳定性差异明显：四组 MTP-off 全部成功（120/120）；SGLang T2D4 MTP-on 也 30/30 成功。SGLang T1D8 MTP-on 因单 worker KV 容量仅 39,688 tokens，只有 32K 的 6 个 case 可运行；vLLM MTP-on 则有 10 个 bench_exit_1 和 2 个输出长度门禁失败，日志多次出现 CUDA illegal memory access
2. 实验组与矩阵
2.1 四个变量
变量	取值
后端	SGLang、vLLM
并行策略	TP1/DP8（T1D8）、TP2/DP4（T2D4），均使用 8 GPU
MTP	on、off
名义上下文	32K、64K、128K、200K、256K
并发	1、4、8、16、32、64
每个“后端 × 并行策略 × MTP”实验组包含 5 × 6 = 30 个 case，共 2 × 2 × 2 × 30 = 240 个 case。每个请求固定生成 256 tokens。
2.2 纳入报告的 8 个有效运行
后端	MTP	策略	run ID	case 数
SGLang	on	T1D8	20260825-aligned-mtp-on-t1d8-v4	30
SGLang	on	T2D4	20260825-aligned-mtp-on-t2d4-v4	30
SGLang	off	T1D8	20260826-aligned-mtp-off-t1d8-v1	30
SGLang	off	T2D4	20260826-aligned-mtp-off-t2d4-v1	30
vLLM	on	T1D8	20260825-aligned-mtp-on-t1d8-v4	30
vLLM	on	T2D4	20260825-aligned-mtp-on-t2d4-v4	30
vLLM	off	T1D8	20260826-aligned-mtp-off-t1d8-v1	30
vLLM	off	T2D4	20260826-aligned-mtp-off-t2d4-v1	30
早期 v1/v2/v3 MTP-on 运行不满足对齐条件，未纳入任何结论。尤其是旧 vLLM TP1/DP8 200K/C1 曾出现约 49.8% prefix-cache hit、TTFT 仅 1.31s 的 warm-cache 污染数据；该数据已排除。
2.3 请求与共享前缀设计
● workload：generated_shared_prefix，单轮请求，不是多轮对话。
● 共享内容：每组请求的 system prompt 约占输入的 80%，每个 case 最多 8 个前缀组。
● C1/C4/C8：组数等于请求数，没有组内重复，理论上不应命中；用于 C1 冷 prefill 门禁以及低并发基线。
● C16/C32/C64：固定最多 8 组，每组约 2/4/8 个并发请求共享前缀，可在同一请求波次中产生复用；不需要构造成多轮对话。
● 没有发送 routing key，因此命中率包含 DP scheduler 是否把同组请求放到同一 worker 的影响。
● 数据集由共享、版本化的确定性缓存生成；同一 context/concurrency 在所有后端、MTP、并行策略之间 SHA 一致。共核验 30 个 context/concurrency SHA 键，无不一致。
● 每个服务先进行 1 个 32K/C1 warmup 请求；warmup 后再次执行后端原生 cache reset。每个正式 case 前还会再次 reset，case 内不做额外 warmup。
上下文 token 含义：nominal 是实验标签对应的名义窗口；target total context 是输入加输出的目标；input 是目标减去固定 256 输出；generator input 是生成器的预估输入，经过模板/tokenizer 后得到实际输入。256K 为服务窗口留了 1,024 token 头部余量：nominal 262,144、target 261,120、requested input 260,864、generator input 243,647。最终性能计算使用服务/客户端记录的实际输入 token，不使用 generator 预估值。
3. 模型、镜像与实际服务参数
3.1 模型身份
项目	值
模型	Qwen3.8-27B
权重路径	/storage/openpsi/models/Qwen__Qwen3.8-27B
served model / benchmark model ID	qwen3-8-27b-smoke
最大服务上下文	262,144
输出长度	256
请求入口	OpenAI-compatible chat serving benchmark
3.2 镜像
用途	镜像
SGLang 服务（本报告旧基线）	/storage/openpsi/images/glm-5-2-2fbb8e.sif
vLLM 服务	/storage/openpsi/images/areal-vllm0202-torch211.sif
两端 benchmark client	/storage/openpsi/images/glm-5-2-2fbb8e.sif
vLLM 服务日志确认引擎版本为 v0.17.0。旧 SGLang 服务日志未打印可靠的 package version，因此以不可变镜像路径作为复现标识。本报告明确不包含 /storage/openpsi/images/deepep-base-85b53914.sif。
3.3 对齐的共同参数
配置	SGLang	vLLM
TP/DP	1/8 或 2/4	1/8 或 2/4
pipeline parallel	1	1
context length	262,144	262,144
GPU memory fraction	0.85	0.85
KV cache dtype	fp8_e4m3	fp8_e4m3
attention backend	FlashInfer	FlashInfer
chunked prefill / batch token ceiling	32,768	32,768
最大运行请求/序列	64	64
CUDA graph batch ceiling	64	64
reasoning parser	qwen3	qwen3
tool-call parser	qwen3_coder	qwen3_coder
prefix cache	开启	开启
trust remote code	开启	开启
3.4 SGLang 实际后端特有参数
● --load-balance-method total_tokens、--base-gpu-id 0。
● --enable-metrics --enable-cache-report --log-requests，请求日志为 JSON、level 1。
● cache reset endpoint：/flush_cache。
● Mamba scheduler：extra_buffer；--max-mamba-cache-size 256。
● 启动命令中先出现 --mamba-ssm-dtype bfloat16，随后又出现 --mamba-ssm-dtype float32；按 CLI 后值覆盖前值，实际有效值按 parity metadata 和最终参数记为 float32。
● MTP-on：--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4。
● MTP-off：完全不传 speculative flags。
3.5 vLLM 实际后端特有参数
● T1D8 使用 --distributed-executor-backend uni，T2D4 使用 mp。
● --dtype auto、--enable-prefix-caching --enable-chunked-prefill --stream-interval 1。
● cache reset endpoint：/reset_prefix_cache。
● Mamba：--mamba-cache-dtype auto --mamba-ssm-cache-dtype float32 --mamba-cache-mode align。
● MTP-on：--speculative-config {"method":"mtp","model":"/storage/openpsi/models/Qwen__Qwen3.8-27B","num_speculative_tokens":3}。
● MTP-off：不传 speculative config。
3.6 仍无法完全对齐的差异
维度	SGLang	vLLM	影响
推测解码	EAGLE，3 steps、4 draft tokens、top-k 1	原生 MTP，3 speculative tokens	不是同一实现的严格 A/B
DP 调度	total_tokens	vLLM DP controller	prefix affinity、排队与吞吐不同
混合 Mamba cache	extra_buffer，max cache 256	align	长上下文容量与缓存行为不同
cache 指标	cache/compute token 计数	query/hit 与 source-accounting 两套计数	命中率绝对值口径不同
streaming	约逐 token 事件	MTP 下可能一个事件带多个 token	原始 ITL 不可直接横比
4. 指标定义与采集口径
指标	口径
TTFT	从客户端发出请求到收到首个输出 token/流事件；报告 mean/p95，单位 ms 或 s
E2E	请求从发出到完整响应结束；报告 mean/p95
Decode TPOT	(E2E - TTFT) / (输出 tokens - 1) 的逐请求统计；固定输出 256，使跨 case 可比较
ITL	客户端相邻流事件间隔；因后端流事件携带 token 数不同，只保留原始记录，不跨后端比较
输入吞吐	全部实际输入 tokens / 整个 case 持续时间，tok/s；包含共享 prefix 被 cache 命中后的系统效果
输出吞吐	全部输出 tokens / 整个 case 持续时间，tok/s
冷 prefill throughput/req	C1 实际输入 tokens / TTFT；用于单请求冷 prefill
prefix hit	每个 case 开始后的 counter delta；不是 Prometheus 从进程启动以来的累计值
GPU 显存	采样窗口内所有 GPU 的 memory_used_mib 峰值；报告中的“峰值单卡显存”是组内成功 case 的最大采样值
GPU util	每个 case GPU utilization 样本的均值，再按实验组汇总
KV cache	后端 metrics 暴露的 worker 使用率、容量/可用 token；用于容量与压力解释
MTP/EAGLE	SGLang 使用 per-worker acceptance/accept length 和 verify calls；vLLM 使用 accepted/draft token delta
4.1 Prefix cache 的具体计算
● SGLang：cached_delta / (cached_delta + prefill_compute_delta)；同时保留 sglang:cache_hit_rate gauge 的 worker 汇总作交叉检查。
● vLLM 主口径：prefix_cache_hits_delta / prefix_cache_queries_delta；另存 local_hit / (local_hit + local_compute) 的 source-accounting 命中率。
● 例如同一 32K/C64、T2D4、MTP-on case：
  ○ SGLang：cached 212,288、computed 1,865,548、queried 2,077,836，客户端实际输入 2,077,826，主命中率 10.22%；
  ○ vLLM：cached 244,800、computed 1,749,547、query counter 3,362,261，主命中率 7.28%，但 source-accounting 命中率 12.20%。
这说明 vLLM query counter 的分母不能简单等同于客户端输入 token。报告的 prefix 表沿用各后端已定义的主口径，主要用于观察同后端随并发、DP 策略的变化。
4.2 成功门禁
成功 case 必须同时满足：cache reset 成功；客户端退出码为 0；完成请求数等于并发；输出条数等于并发；每条输出恰好 256 tokens；数据集 SHA 符合 manifest；服务参数符合实验 parity。C1 还要求 prefix hit 不超过 1%。本轮实际所有成功 C1 命中率均严格为 0。
5. 完整性、容量与失败
后端	MTP	策略	最终结果	成功	容量不支持	实际运行失败
SGLang	on	T1D8	30/30	6	24	0
SGLang	on	T2D4	30/30	30	0	0
SGLang	off	T1D8	30/30	30	0	0
SGLang	off	T2D4	30/30	30	0	0
vLLM	on	T1D8	30/30	22	0	8
vLLM	on	T2D4	30/30	26	0	4
vLLM	off	T1D8	30/30	30	0	0
vLLM	off	T2D4	30/30	30	0	0
SGLang T1D8 MTP-on 的单 worker KV capacity 只有 39,688 tokens，因此 64K/128K/200K/256K 的全部 24 个 case 在发请求前被明确判为 context_exceeds_kv_capacity_per_worker，不是请求超时。
vLLM MTP-on 的实际失败为：
● T1D8：7 个 bench_exit_1，另有 256K/C8 的输出长度门禁失败；
● T2D4：3 个 bench_exit_1，另有 200K/C32 的输出长度门禁失败；
● 两组服务日志均多次记录 torch.AcceleratorError: CUDA error: an illegal memory access was encountered，所以 bench_exit_1 有明确引擎侧崩溃证据。
6. 实验结果
6.1 冷缓存 C1：TTFT、E2E 与 prefill
后端	MTP	策略	上下文	TTFT mean (s)	E2E mean (s)	prefill tok/s/req
SGLang	on	T1D8	32K	6.0	7.6	5,402
vLLM	on	T1D8	32K	5.6	7.6	5,860
SGLang	on	T2D4	32K	3.7	4.9	8,817
vLLM	on	T2D4	32K	3.1	5.0	10,485
SGLang	off	T1D8	32K	5.7	9.6	5,732
vLLM	off	T1D8	32K	5.2	9.2	6,262
SGLang	off	T2D4	32K	3.1	5.5	10,583
vLLM	off	T2D4	32K	2.9	5.5	11,257
vLLM	on	T1D8	64K	17.0	18.9	3,828
SGLang	on	T2D4	64K	9.7	11.1	6,693
vLLM	on	T2D4	64K	9.4	11.0	6,957
SGLang	off	T1D8	64K	17.4	21.5	3,746
vLLM	off	T1D8	64K	16.1	20.1	4,062
SGLang	off	T2D4	64K	9.4	11.8	6,976
vLLM	off	T2D4	64K	8.7	11.4	7,495
vLLM	on	T1D8	128K	56.6	60.4	2,309
SGLang	on	T2D4	128K	32.0	33.5	4,087
vLLM	on	T2D4	128K	29.7	32.2	4,394
SGLang	off	T1D8	128K	58.0	62.3	2,251
vLLM	off	T1D8	128K	53.9	58.0	2,422
SGLang	off	T2D4	128K	30.4	33.0	4,298
vLLM	off	T2D4	128K	28.4	31.1	4,602
vLLM	on	T1D8	200K	129.4	133.2	1,578
SGLang	on	T2D4	200K	72.9	75.6	2,802
vLLM	on	T2D4	200K	67.2	70.1	3,037
SGLang	off	T1D8	200K	132.9	137.4	1,536
vLLM	off	T1D8	200K	123.0	127.3	1,660
SGLang	off	T2D4	200K	69.0	71.7	2,961
vLLM	off	T2D4	200K	64.5	67.1	3,165
vLLM	on	T1D8	256K	204.0	208.1	1,277
SGLang	on	T2D4	256K	114.3	116.3	2,279
vLLM	on	T2D4	256K	104.9	107.7	2,484
SGLang	off	T1D8	256K	209.7	214.3	1,243
vLLM	off	T1D8	256K	194.0	198.3	1,343
SGLang	off	T2D4	256K	107.9	110.7	2,414
vLLM	off	T2D4	256K	100.1	102.8	2,602
SGLang T1D8 MTP-on 的 64K+ 行缺失是容量不支持，不应视作零值或超时。C1 的 TTFT p95 等于 mean，因为每个 case 只有一个请求。
6.2 32K/C64 与 256K/C64 的系统吞吐
上下文	后端	MTP	策略	E2E mean (s)	输入吞吐 (ktok/s)	输出吞吐 (tok/s)	TPOT mean (ms)
32K	SGLang	on	T1D8	36.5	28.9	227.5	7.4
32K	vLLM	on	T1D8	50.1	40.1	316.2	49.3
32K	SGLang	on	T2D4	50.3	39.2	309.2	61.3
32K	vLLM	on	T2D4	49.0	39.2	308.9	71.5
32K	SGLang	off	T1D8	51.2	40.1	315.8	69.4
32K	vLLM	off	T1D8	49.3	41.7	328.5	58.1
32K	SGLang	off	T2D4	50.5	40.2	317.2	71.8
32K	vLLM	off	T2D4	48.9	41.8	329.8	74.5
256K	SGLang	on	T2D4	1,297.3	12.3	12.1	1,880.5
256K	vLLM	on	T2D4	1,200.8	13.6	13.3	1,998.7
256K	SGLang	off	T1D8	1,066.2	11.4	11.2	801.2
256K	vLLM	off	T1D8	1,356.5	10.7	10.5	2,166.0
256K	SGLang	off	T2D4	1,202.9	13.0	12.7	1,590.6
256K	vLLM	off	T2D4	1,135.8	14.6	14.4	1,820.8
256K/C64 是很重的单次波次，E2E 达到 18–23 分钟。它体现的是共享前缀、DP 调度、长 prefill、decode 和排队共同作用后的整套系统吞吐，不应与 C1 冷 prefill 混为一个指标。
6.3 MTP on 相对 off：配对成功 case 的中位变化
负的 TTFT/E2E/TPOT 表示更快；正的输入吞吐表示更高。
后端	策略	配对数	TTFT	E2E	输入吞吐	TPOT
SGLang	T1D8	6	-1.0%	-25.0%	-2.6%	-67.6%
SGLang	T2D4	30	+6.6%	+1.5%	-2.0%	-5.6%
vLLM	T1D8	22	+5.8%	+1.7%	-3.1%	-35.3%
vLLM	T2D4	26	+6.4%	+4.0%	-5.0%	+1.9%
T1D8 配对存在明显选择偏差：SGLang 只有 32K 六个 case，vLLM 也只包含 MTP-on 成功的 22 个 case。最可信、覆盖最完整的是 SGLang T2D4 的 30 对；它显示 MTP 主要改善 decode TPOT，并未提高输入吞吐或端到端中位性能。
6.4 T2D4 相对 T1D8：配对成功 case 的中位变化
后端	MTP	配对数	TTFT	E2E	输入吞吐	TPOT
SGLang	on	6	-16.7%	-5.9%	+46.3%	+53.6%
SGLang	off	30	-30.8%	-30.2%	+50.3%	-10.7%
vLLM	on	20	-29.9%	-19.6%	+38.3%	+15.4%
vLLM	off	30	-21.3%	-14.7%	+29.2%	+8.5%
T2D4 的 prefill/TTFT 与总输入吞吐优势一致；TPOT 则依赖后端与 MTP，仅 SGLang MTP-off 在完整矩阵上同时改善 TPOT。
6.5 SGLang 相对 vLLM：配对成功 case 的中位变化
负的 TTFT/E2E/TPOT 表示 SGLang 更快；正的输入吞吐表示 SGLang 更高。
MTP	策略	配对数	TTFT	E2E	输入吞吐	TPOT
on	T1D8	6	+0.4%	-7.4%	-21.2%	-19.9%
on	T2D4	26	+8.0%	+1.7%	-2.2%	-13.7%
off	T1D8	30	+7.8%	+4.2%	-6.9%	+5.6%
off	T2D4	30	+7.7%	+4.1%	-6.0%	-2.7%
最稳健的 MTP-off 全矩阵表明：SGLang 的 TTFT/E2E/输入吞吐均落后 vLLM；decode TPOT 则在 T1D8 略慢、T2D4 略快。MTP-on T1D8 只有 6 个共同成功的 32K case，不代表长上下文。
6.6 共享前缀有效命中率
下表是各上下文成功 case 的主口径命中率中位数。
后端	MTP	策略	C1	C4	C8	C16	C32	C64
SGLang	on	T1D8	0.0%	0.0%	0.0%	0.0%	0.0%	0.0%
SGLang	on	T2D4	0.0%	0.0%	0.0%	14.1%	15.1%	37.3%
SGLang	off	T1D8	0.0%	0.0%	0.0%	9.4%	11.8%	17.9%
SGLang	off	T2D4	0.0%	0.0%	0.0%	3.1%	16.6%	35.9%
vLLM	on	T1D8	0.0%	0.0%	0.0%	0.0%	1.5%	2.2%
vLLM	on	T2D4	0.0%	0.0%	0.0%	0.0%	6.5%	32.9%
vLLM	off	T1D8	0.0%	0.0%	0.0%	0.0%	3.0%	7.3%
vLLM	off	T2D4	0.0%	0.0%	0.0%	0.0%	11.5%	32.9%
理论上，在忽略模板尾部和 token 化差异时，8 组、80% 共享前缀的最大复用比例约为：C16 40%、C32 60%、C64 70%。实际通常显著低于该上限，说明无 routing key 时 DP placement/prefix affinity 是主要限制之一。C16 的某些后端中位数仍为零，也说明“一轮内存在相同前缀”不等于一定落到可复用的同一 worker。
6.7 推测解码接受率
后端	策略	成功 case	平均 acceptance	平均 accept length
SGLang	T1D8	6	0.466	3.602
SGLang	T2D4	30	0.760	3.335
vLLM	T1D8	22	0.680	—
vLLM	T2D4	26	0.687	—
SGLang 的 acceptance/accept length 是 EAGLE per-worker 指标；vLLM 是 accepted/draft token 计数比。定义和 speculative budget 都不同，不能据此直接宣布某后端的 MTP 算法更好。
6.8 GPU、KV cache 与容量
后端	MTP	策略	成功 case	峰值单卡显存 (MiB)	峰值 worker KV 使用率	平均 GPU util	worker KV 容量 tokens
SGLang	on	T1D8	6	141,994	0.830	25.7%	39,688
SGLang	on	T2D4	30	138,084	0.660	51.7%	3,859,426
SGLang	off	T1D8	30	135,496	0.980	43.0%	998,080
SGLang	off	T2D4	30	131,166	0.530	50.9%	4,817,818
vLLM	on	T1D8	22	137,098	0.513	33.6%	464,000
vLLM	on	T2D4	26	133,110	0.533	45.8%	1,358,400
vLLM	off	T1D8	30	138,810	1.000	43.9%	498,624
vLLM	off	T2D4	30	131,510	0.444	51.3%	1,448,832
这里的“worker KV 容量 tokens”是各后端 metrics 所报的内部容量，不等价于“必定支持该长度 × 并发”的简单乘法上限。混合 Mamba state、radix/prefix cache、调度、预留空间和 DP 分布都会影响实际可用性。最显著异常是 SGLang T1D8 MTP-on 容量从 MTP-off 的 998,080 降到 39,688，直接导致 64K+ 不支持。
7. 数据异常、排除项与局限
1. C1 warm-cache 污染已排除：早期无效 vLLM 运行使用错误 cache reset 路径，200K/C1 曾出现 49.8% hit、1.31s TTFT。报告只使用修复 reset、共享确定性 dataset cache 和 C1 gate 后的 v4/v1 运行；选中数据的成功 C1 最大 hit 为 0。
2. 旧数据集跨节点随机缓存问题已排除：早期 client 节点各自使用 pickle cache，可能在相同名义长度下得到不同文本/token 数。有效运行改为共享、版本化数据集和 SHA manifest；本报告 30 个 context/concurrency 数据集键全部跨组一致。
3. ITL 粒度不同：SGLang 大致一个 token 一个流事件；vLLM 在 MTP 下可能一个事件携带多个 tokens。原始 ITL 数百毫秒或逐事件分布不应跨后端比较；报告使用基于总输出长度的 decode TPOT。
4. 配置无法完全等价：SGLang EAGLE 3 steps/4 draft tokens 与 vLLM 3 MTP speculative tokens 不同；Mamba cache、DP scheduler、prefix 指标实现也不同。因此后端表是“整套配置对比”，不是同一 speculative 算法的消融。
5. Prefix cache 计数分母不同：SGLang 查询 token 与客户端输入基本闭合；vLLM query counter 可能高于客户端输入，并另有 source-accounting 口径。跨后端只能谨慎比较趋势；同后端不同并发/策略的相对变化更可靠。
6. 没有 routing key：测得的是实际 DP 调度条件下的系统有效 cache hit。它适合回答“当前服务整体能命中多少”，不适合单独测单 worker radix/prefix cache 的理论效率。
7. 单次、不同节点实验：每个实验组是不同 Slurm 节点上的一次完整运行，没有跨节点重复和置信区间。节点硬件虽同类，但温度、频率、邻接拓扑及系统噪声可能带来波动；小于几个百分点的差异不宜过度解读。
8. 配对成功选择偏差：所有百分比只在两边都成功的同 context/concurrency case 上计算。SGLang MTP-on T1D8 只有 6 对，vLLM MTP-on 也丢失高压力 case，所以其配对中位数不能代表完整 30-case 矩阵。
9. 失败与不支持必须分开：SGLang 24 个 case 是根据实际 worker KV capacity 的 preflight，不是性能样本；vLLM 12 个 case 是发出请求后的实际失败。两者都不应填零参与吞吐/延迟聚合。
10. GPU/KV 为采样指标：GPU peak、util 和 KV fraction 依赖采样周期，适合定位容量/负载趋势，不等同于逐 kernel profiler 或精确显存组成分析。