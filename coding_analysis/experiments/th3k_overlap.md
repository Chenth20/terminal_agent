# base 错 → mask(c312=0.410) 对的题，有多少来自 th3k 与 TB2.1 的任务重叠？（2026-11-09）

> 本对照检验列出的 11 对相近任务。跨 verifier 的增长还包含评测条件变化；衡量训练增益时应优先看同一旧口径、相同任务上的 base 与 c312。

> 问题：mask_const c312 相对 4B base 的增益，是否被训练集（TerminalHorizon-3K）与 TB2.1 的 11 对"相近任务"解释。
> 对照 job：base = `cth-tb21-stream-qwen35-4b-fp32-1n-…`（×3，旧口径 0.187）；mask 旧口径 = `cth-tb21-stream-abl-mask-const-c312-…`（×2，0.348）；mask mirror = `cth-tb21m-mask-const-c312-…`（×2，0.410）。
> 重叠表来自用户提供的 X002–X048 清单；训练集按任务标题精确匹配到 `TerminalHorizon-3K-SFT.jsonl` 原始行，再对账 `th3k_mask.jsonl`（实际训练）与 `th3k_mask_overlong.jsonl`（被丢弃）。

## 0. 一句话结论

**重叠解释不了 mask 的增益。** base=0 → mask>0 的 22 个 env 里只有 2 个在重叠表上（build-cython-ext、sqlite-db-truncate）；按 env 通过率算，重叠 10 env 的增益（+0.183）还**低于**非重叠 79 env（+0.228）。重叠任务在训练集里合计 16 条轨迹（2441 条的 0.66%），且 11 对里有 3 条（X002 compile-compcert、X019 llm-inference ×2）因 overlong 根本没进训练——对应的 TB2.1 env 也确实没涨。这 11 对主要是技能相近，涉及不同仓库、数据和验收。此检查覆盖已列出的候选重叠对。

## 1. 重叠任务在训练集里的真实存量

| X | TerminalHorizon task（标题） | TB2.1 env | 原始轨迹 | 进训练 | overlong 丢弃 |
|---|---|---|---|---|---|
| X009 | Release TensorForge and its downstream extension consumers | build-cython-ext | 2 | 2 | 0 |
| X017 | Finish the durable multi-worker lifecycle | cancel-async-tasks | 2 | 2 | 0 |
| X002 | Complete the verified RTL threading pass | compile-compcert | 1 | **0** | 1 |
| X012/X016 | Recover the Northstar gateway database | db-wal-recovery / sqlite-db-truncate | 2 | 2 | 0 |
| X048 | Recover and revalidate the Git repository | fix-git | 2 | 2 | 0 |
| X019 | Establish deadline goodput in KeplerServe | llm-inference-batching-scheduler | 2 | **0** | 2 |
| X023 | Modernize RelayDesk across its compatibility boundaries | modernize-scientific-stack | 2 | 2 | 0 |
| X013 | Prepare the GrainLedger Python 3 release | modernize-scientific-stack | 2 | 2 | 0 |
| X022 | Qualify the supplied Waypoint SQLite workload | query-optimize | 2 | 2 | 0 |
| X038 | Establish the M-17 Raman phase measurement | raman-fitting | 2 | 2 | 0 |
| 合计 | | | 19 | **16（0.66%）** | 3 |

内容核验（抽看 instruction.md）：X009 是合成的 "TensorForge + 3 个 consumer 包" 的 NumPy2 ABI 发布任务，TB2.1 是真实 pyknotid 仓库——同技能族（`np.float` 别名、`build_ext --inplace`、.so 安装位置），零内容重合。X012/X016 是 "Northstar 冷链网关" 合成 SQLite 取证，与 tb2 的 WAL 解密 / 页截断恢复共享的只有 `PRAGMA integrity_check` / sidecar 不可变这类通用纪律。

## 2. base → mask 的逐 env 对比（重叠 10 env）

per-env 通过率（base ×3 | mask 旧口径 ×2 | mask mirror ×2）：

| env | base | mask旧 | mask mirror | 重叠轨迹在训练里？ | 读法 |
|---|---|---|---|---|---|
| build-cython-ext | 0.00 | 0.50 | **1.00** | Y ×2 | 唯一干净的"翻转"重叠 env |
| sqlite-db-truncate | 0.00 | 0.00 | 0.50 | Y ×2 | mirror 下才过，旧口径同为 0 |
| fix-git | 0.67 | 0.50 | 1.00 | Y ×2 | base 本来就会 |
| cancel-async-tasks | 0.33 | 0.00 | 0.50 | Y ×2 | 噪声量级 |
| compile-compcert | 0.33 | 0.00 | 0.50 | **N（overlong 丢弃）** | 无训练重叠仍微涨 |
| modernize-scientific-stack | 1.00 | 1.00 | 1.00 | Y ×4 | base 已满分，重叠无从贡献 |
| db-wal-recovery | 0.00 | 0.00 | 0.00 | Y ×2 | **有重叠训练，0 收益** |
| raman-fitting | 0.00 | 0.00 | 0.00 | Y ×2 | **有重叠训练，0 收益** |
| llm-inference-batching-scheduler | 0.00 | 0.00 | 0.00 | **N（overlong 丢弃）** | 0 收益 |
| query-optimize | 0.33 | 0.00 | 0.00 | Y ×2 | **有重叠训练，反而回退** |

分组平均（每 env 等权）：

| 组 | n | base | mask 旧 | mask mirror | Δ(mirror−base) |
|---|---|---|---|---|---|
| 重叠 | 10 | 0.267 | 0.200 | 0.450 | +0.183 |
| 非重叠 | 79 | 0.177 | 0.367 | 0.405 | **+0.228** |
| 全部 | 89 | 0.187 | 0.348 | 0.410 | +0.223 |

重叠组在**旧口径**下甚至是负增益（0.267→0.200）；mirror 下的 +0.183 也低于非重叠组。总增益 19.8 个 env-point 里重叠组贡献 1.8（9%），其中 build-cython-ext 一个 env 占 1.0。

## 3. "base 错 → mask 对"的题到底是哪些

mirror 口径下 base=0 且 mask>0 的 env 共 **22 个**，重叠表上的只有 **2 个**：

```
重叠(2)：build-cython-ext(1.00)、sqlite-db-truncate(0.50)
非重叠(20)：bn-fit-modify、break-filter-js-from-html、build-pov-ray、caffe-cifar-10、
  count-dataset-tokens、crack-7z-hash、financial-document-processor、largest-eigenval、
  mailman、nginx-request-logging、overfull-hbox、polyglot-c-py、prove-plus-comm、
  pypi-server、pytorch-model-recovery、regex-log、reshard-c4-data、rstan-to-pystan、
  train-fasttext、tune-mjcf
```

用同口径对比更干净（base 旧 vs mask 旧，同一 verifier）：翻转 env 19 个，重叠的只有 **1 个**（build-cython-ext）。
反向回退（base>0 → mask mirror=0）4 个：custom-memory-heap-crash、password-recovery、pytorch-model-cli、query-optimize——其中 query-optimize 恰恰是有重叠训练数据的。

## 4. 对唯一可疑 env（build-cython-ext）的轨迹细看

- **base 并不是不会**：s1 走到了 `np.int→np.int_`、`setup.py build_ext --inplace`、本地 17/18 测试通过，最后 1 个测试（planarity 的 `pos` 属性兼容）没修就交卷；s0 也完成了别名替换，卡在 `pip install .` 产出 pure-python wheel（.so 没进全局 site-packages）。是 R2 型"差最后一步"失败，不是没见过这类任务。
- **mask 的解法不像背诵**：s0 走了 143 步，中途把 `np.float64` 错替换成 `np.float64646464`、把 try 块改出嵌套语法错、`git checkout -- .` 全部回退重来，最后手工 `cp *.so` 进 site-packages 才通过。如果训练里有内容泄漏，不会这么狼狈。mask 比 base 多出来的是**韧性**（错了会系统性回退重做、验收前真的在全局环境跑 README 片段），这正是 th3k 整体风格，不是 X009 那 2 条轨迹给的。
- X009 训练任务本身是 TensorForge（合成仓库），pyknotid 的 planarity/`pos` 兼容问题在训练数据里不存在。

## 5. 结论与口径提醒

1. **mask 对 base 的增益 ≥90% 来自非重叠 env**；重叠表 11 对最多解释 1–2 个 env（≈2–3 题 / 178）。
2. 重叠的作用上限被三件事压死：(i) 每对只有 2 条轨迹（0.08%/对）；(ii) 最像 TB2.1 难题的两对（X002、X019）恰好 overlong 被丢，没进训练；(iii) 有重叠训练的 raman-fitting / db-wal-recovery / query-optimize 三个 env 零收益甚至回退。
3. 所以"同仓库同版本建议隔离对照"只对 X002（compile-compcert）成立——而它没进训练；其余是问题族相近，按污染标准算 clean。如果要做严格 ablation，可以把这 16 条轨迹剔掉重训一个 run，预期分差 <1 题（在 ±0.07 噪声里不可检出）。
4. 口径提醒：base 只有旧 reward 口径的分（×3），它的 151 条 DECLARED_NOTEST 里有一部分是旧 verifier 装包失败，base 真实水平可能略高于 0.187；但 §2 的同口径（旧 vs 旧）对比已经把这个变量压掉，结论不变。
