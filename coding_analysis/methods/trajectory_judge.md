 # 过程质量 LLM-Judge 轨迹过滤 — 完整评价流程(trajectory_judge.md)— 2026-11-06

> 训练数据清洗过程及数据量见[采集与清洗](trajectory_collection.md)，后续训练效果见[SFT 实验](../experiments/sft_results.md)。

> 来源:TerminalHorizon 论文(arXiv 2610.13166v2,即 th3k 数据的论文),Appendix F(三准则定义)+ Appendix J.1(原始 prompt)。
> 原文抽取自 `coding_analysis/paper/2610.13166v2.pdf`,prompt 为逐句核对的原文(§4)。
> 用途:对我们的 dsv41 整轨迹数据(`dsv41_json262k` 2037 条 + 96k 组 ~1900 条)复制 th3k 的过滤步骤(计划 P1,见 [README](../experiments/README.md))。

---

## 1. 流程总览

```
轨迹 jsonl(完整多轮对话)
   │  1) 构造输入记录:record_id + 完整对话原文
   │     — 剥掉 score/reward/success 字段(judge 不得见)
   │     — record_id 不得编码成败信息
   ▼
LLM judge(论文用 gpt-5.6-luna, reasoning effort = high)
   │  2) 读完整对话(不许摘录/关键词计数/程序化摘要代替)
   │  3) 只评三个维度:①轨迹有效性 ②环境接地的进展 ③多轮训练价值
   │  4) 输出 KEEP/DROP + reason + 2–4 条具体证据(JSON)
   ▼
(可选)争议复审:把先前 review 结论作为"待验证假设"喂回,
   judge 回到原文核实并裁决(论文 "Disputed cases" 条款)
   ▼
KEEP 集 → 进入 SFT;DROP 集 + 全部证据记录存档
```

论文结果:3,070 条教师轨迹 → 2,852 条保留(**92.9% keep rate**)。
关键设定:**judge 不接收最终 reward 或成功/失败标签**,判断必须接地在轨迹记录的动作与观察上。

## 2. 三个过滤准则(Appendix F,完整定义)

### ① 轨迹有效性(Trajectory validity)

有效轨迹应保持连贯上下文和可追溯的"动作–观察"对应,使决策能连接到其执行结果。

**排除**:
- 核心解题过程无法可靠解读:严重截断、持续的协议/格式损坏、错误的执行环境;
- 依赖 private answer 泄漏、未授权操纵内部状态、绕过规定接口。

**不排除**:局部交互错误本身不使轨迹无效——若模型识别到 parsing/命令错误、恢复有效交互并继续推进,该序列仍是有用的**纠错示范**。

该准则区分"可恢复的错误"与"损害轨迹完整性或违反任务约束的实质缺陷"。

### ② 环境接地的进展(Environment-grounded progress)

高质量轨迹应表明动作由环境证据驱动:是否用文件内容、日志、命令输出、报错、测试结果来指导决策,并在必要时修正假设或计划。

**有意义的进展不要求立即产生正确结果**:排除一条不可行路径、定位根因、发现新约束,同样是有价值的学习信号。

**排除**:被以下行为主导的轨迹——持续无视反馈、无依据的试错、重复无效动作、对错误结论的无支撑坚持。

评估对象是**整个动作序列的可靠性**,不要求每次尝试都正确。

### ③ 多轮训练价值(Multi-turn training value)

评估轨迹是否展示了实质关联的多轮终端工作:取证、实现、验证、假设修正、错误恢复。

**价值在于交互如何影响后续决策,而不在轨迹长度、命令数或输出量。**
- 等待、重复轮询、机械循环不因增加轮数而获得任何加分;
- 仅用于连接/管理执行环境的基础设施操作(SSH/Docker/Harbor)不计入解题深度;
- **不设固定轮数门槛**,只评估是否提供了可靠、非冗余、值得学习的交互过程。

### 总体判定(Overall decision)

- 满足准则的轨迹**即使没完成任务、或停在未解决的障碍上,也保留**。失败之后的"反馈驱动的诊断、修正、恢复"不应仅因含有错误而被丢弃;局部错误在不损害主动作链时同样不是拒绝理由。
- **拒绝**当:核心解法依赖被禁止的捷径;错误假设贯穿主要解题过程;或交互始终无产出、无法提供可靠学习信号。
- 这套过程导向的准则保留真实解题中的探索与纠错,同时把系统性错误和无效交互排除出监督示范。

## 3. Judge 的读法约定(prompt 内嵌,帮助理解 Terminus-2 轨迹)

1. 首条 user 消息通常含任务描述与 agent 交互协议;
2. 后续 user 消息通常是前序命令产生的**终端观察**,不是新的用户请求;
3. assistant 消息含推理、计划与终端动作;一轮可含多条命令;
4. 推理内容(`<think>`、analysis、plan)**本身不是执行证据**,须与实际动作及后续观察一起评估;
5. Terminus-2 轨迹以最后一批命令的终端观察结尾是**正常结束**,不需要额外的 assistant 确认;
6. 最终成败不足以决定训练价值;未完成的轨迹或"诚实停在未解决障碍"仍可能是有用示范。

## 4. 原始 Prompt(Appendix J.1,逐句原文)

> 使用说明(论文原文):"The following prompt implements the three filtering criteria in Appendix F by reviewing complete trajectories and returning an evidence-backed KEEP/DROP decision for each input record."

```text
You are an expert reviewer of terminal-agent trajectories for supervised fine-tuning. Your task is to
determine whether each complete multi-turn trajectory provides a reliable and useful training example.

Read the entire conversation. Do not substitute excerpts, keyword counts, command counts, or
programmatic summaries for the original trajectory. Base your judgment exclusively on the trajectory
itself. Do not access task packages, private verifiers, external benchmarks, or the internet.

Understanding the trajectory
- The initial user message typically contains the task description and the agent interaction protocol.
- Subsequent user messages usually contain terminal observations produced by preceding commands,
rather than new user requests.
- Assistant messages contain reasoning, plans, and terminal actions. One assistant turn may contain
multiple commands.
- Reasoning, including <think>, analysis, and plan content, is not itself evidence of execution.
Evaluate it together with the actual actions and subsequent observations.
- A Terminus-2 trajectory may end with the terminal observation from its final command batch. This is
a normal ending and does not require an additional assistant confirmation.
- Final success or failure is not sufficient to determine training value. An incomplete trajectory or an
honest stopping point at an unresolved obstacle may still provide a useful demonstration.

Evaluate only the following three dimensions.

1. Trajectory validity
Is the trajectory coherent and correctly ordered? Can actions be reliably associated with their terminal
observations? Check for genuine context truncation, severe format corruption, execution in the wrong
container, infrastructure failures, explicit refusal, or private-answer leakage that compromises the
demonstration.
Do not reject an entire trajectory because of one or two isolated invalid JSON responses, malformed
commands, or parser errors. If the agent recognizes the problem, restores valid interaction, and continues
with a clear action–observation chain, this constitutes a learnable recovery process. Treat protocol errors
as an overall validity defect when they recur without reliable recovery or make action–observation
correspondence uninterpretable.
Protocol recovery does not excuse independent integrity violations. Reject trajectories whose core
solution relies on private answers or hidden-test leakage, internal-state tampering, deleting runtime or
database state to reset the task, or bypassing required task interfaces.
Distinguish failure to finish from lack of training value. An unfinished trajectory can be retained
when its main content demonstrates reliable environment understanding, root-cause diagnosis, justified
implementation, feedback-based elimination of alternatives, correction of assumptions, or identification
of unresolved obstacles.
An isolated incorrect final judgment or unsupported completion claim does not automatically invalidate
all preceding work. Determine whether the error compromises the core action chain. Retention is
possible when the main process remains reliable and useful. Reject the trajectory when an incorrect
strategy dominates the solution, central conclusions contradict observations, or the overall example
primarily demonstrates an invalid way to complete the task.

2. Environment-grounded progress
Are important actions supported by actual terminal feedback? Does the agent use files, logs, command
outputs, errors, and test results to advance the task and adjust its approach when necessary?
Identify persistent disregard of feedback, repeated ineffective commands, random attempts, unmotivated
minor edits, and unsupported completion claims. Evaluate these behaviors in context: unsuccessful
attempts can be valuable when their observations inform subsequent diagnosis, revision, or recovery.

3. Multi-turn training value
Does the trajectory demonstrate a multi-turn terminal problem-solving process worth learning? Do
interactions contribute meaningful evidence, implementation, state changes, decisions, verification, or
recovery?
Consider whether the apparent interaction could be trivially replaced by one obvious edit, a fixed script,
or a mechanical loop. Do not count SSH, Docker, or Harbor infrastructure operations, waiting, repeated
polling, or irrelevant directory browsing as evidence of task difficulty.
Do not judge training value solely from trajectory length, command count, the presence of mistakes, or
final task success.

Decision
KEEP: The trajectory is valid, grounded in the environment, and provides sufficient multi-turn training
value. Final failure, incomplete execution, or an honest stopping point does not preclude KEEP when the
main process is reliable and instructive.
DROP: The trajectory is globally invalid; its core solution relies on leakage, state tampering, or interface
bypass; its behavior substantially degenerates or lacks environmental grounding; or errors dominate the
main action chain with little reliable multi-turn learning value.

Do not assign DROP solely because the task was not completed.

Disputed cases
If prior review claims are provided, treat them as hypotheses to verify, not as facts or votes. Return to the
original conversation to establish whether the relevant commands were executed, whether observations
support the claims, whether the issue was subsequently repaired, and whether it compromises the core
solution. Explain how the trajectory evidence resolves the disagreement.

Output
Return exactly one review for each input record:
{
"record_id": "<input record ID>",
"verdict": "KEEP or DROP",
"reason": "<concise justification>",
"evidence": [
"<specific evidence from the trajectory>",
"<specific evidence from the trajectory>"
]
}
Copy record id exactly. Provide two to four concrete pieces of evidence. Do not omit, duplicate, or
invent records.
```

## 5. 判定规则速查

|                | 条件                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **KEEP**       | 有效 + 接地 + 多轮价值足够。最终失败、未完成、诚实停止**不**排除 KEEP                                                           |
| **DROP**(任一) | (a) 全局无效;(b) 核心解法依赖泄漏/状态篡改/接口绕过;(c) 行为实质退化或缺乏环境接地;(d) 错误主导主动作链、几乎无可靠多轮学习价值 |
| **硬约束**     | 不得仅因任务未完成而 DROP                                                                                                       |

容易误判的边界(prompt 内明确):

| 情形                                                       | 判定                                                                |
| ---------------------------------------------------------- | ------------------------------------------------------------------- |
| 1–2 次孤立坏 JSON / 坏命令 / parser 错,随后恢复            | 不是 DROP 理由("learnable recovery process")                        |
| 协议错误反复出现且无可靠恢复,或动作–观察无法对应           | 有效性缺陷 → DROP                                                   |
| 格式恢复了,但核心解法有泄漏/篡改/绕过                      | 照样 DROP("protocol recovery does not excuse integrity violations") |
| 末尾一次错误宣告/无依据 completion claim                   | 看是否损害核心动作链;主过程可靠可 KEEP                              |
| 错误策略主导全程、核心结论与观察矛盾                       | DROP                                                                |
| 失败的尝试但其观察推动了后续诊断/修正                      | 有价值,倾向 KEEP                                                    |
| 等待、轮询、机械循环、SSH/Docker/Harbor 运维、无关目录浏览 | 不计入任务难度/训练价值                                             |
| 交互可被一次编辑/一个固定脚本/一个机械循环替代             | 无多轮价值 → DROP                                                   |

## 6. 争议复审机制(Disputed cases)

prompt 自带二轮复审入口:把先前的 review 结论作为输入重新喂给 judge 时,judge 须将其**当作待验证的假设而非事实或投票**,回到原始对话核实:
1. 相关命令是否真的执行;
2. 观察是否支持该主张;
3. 问题是否在后续被修复;
4. 是否损害核心解法;
并解释轨迹证据如何解决分歧。

(论文未报告 reviewer 数量、分歧触发条件、复审轮数——实现时自定,见 §8。)

## 7. 输出与存档格式

每条输入记录恰好一条 review(JSON):

```json
{
  "record_id": "<原样复制输入 ID>",
  "verdict": "KEEP | DROP",
  "reason": "<简要理由>",
  "evidence": ["<轨迹中的具体证据>", "<2–4 条>"]
}
```

要求:record_id 原样复制;证据 2–4 条且必须具体(引用轨迹内容);不得遗漏、重复或编造记录。
存档:KEEP 集进 SFT;**DROP 集与全部 review(含证据)永久留存**,用于审计与复审。

## 8. 移植到 dsv41 数据的适配(论文未覆盖,我们自定)

论文设置与我们的差异,以及对应决定:

| #   | 差异                                                                                                                                                      | 适配决定                                                                                                                                                                                                                                                                                      |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | **Judge 模型**:论文用 gpt-5.6-luna (high)。                                                                                                               | 用手头最强可用 API 模型,reasoning effort 最高档。关键不变量是:读全文、不给 reward、三准则;模型差异是二阶。                                                                                                                                                                                    |
| A2  | **输入防泄漏**:我们的 manifest 带 score/solved 字段。                                                                                                     | 构造输入时剥掉一切成败信号;record_id 用与成败无关的哈希(如 `src_row` 哈希),不得编码 score。                                                                                                                                                                                                   |
| A3  | **compaction handoff**:th3k 轨迹无压缩;我们线性化整轨迹内联了 harness 的 compaction 交接摘要。原 prompt 会把它当 "genuine context truncation" 误判 DROP。 | 在 "Understanding the trajectory" 一节**追加一条**(这是对原 prompt 的唯一改动):`- This harness may compact earlier context into a handoff summary when the context window fills. A handoff summary followed by continued work is a normal harness event, not context truncation or a defect.` |
| A4  | **超长轨迹**:prompt 要求读全文;我们的整轨迹可达 262k token(被丢的 1,556 条更长),可能超 judge 窗口。                                                       | 窗口装得下的(≤ judge ctx)正常评;装不下的单独标记 `OVERLONG_UNJUDGED`,不混入 KEEP(它们属于 P2 长任务回收臂,另行处理)。不做摘要代读——那违反 prompt 第一原则。                                                                                                                                   |
| A5  | **批量与格式**:论文一次多条("each input record")。                                                                                                        | 每请求 1 条轨迹(轨迹本身已占满上下文,批量无意义);温度 0 或最低;JSON 强校验,解析失败重试 ≤2 次,仍失败标 `JUDGE_ERROR` 不默认 KEEP。                                                                                                                                                            |
| A6  | **复审策略**:论文未报告。                                                                                                                                 | 第一轮单 judge。抽样人工复核 DROP 集 ≥30 条 + KEEP 集 ≥30 条;若 DROP 中误杀 >10%,用 §6 机制带着第一轮 reason/evidence 做第二轮复审。不做多数投票。                                                                                                                                            |
| A7  | **保留率预期**:th3k 92.9% 是在 challenge-verified 环境上得到的;我们的采集环境没有该保证。                                                                 | 预期 KEEP 85–92%;若 <80%,先抽查 DROP 证据分布(泄漏/轮询/无接地各占多少)再决定是否调整,**不得为了保留率放松准则**。                                                                                                                                                                            |
| A8  | **与 per-env 去重的顺序**:计划 P1 要求每 env 最多 2 条。                                                                                                  | 先 judge 后去重:同 env 的多条按 KEEP 优先;都 KEEP 时的排序规则在去重脚本里定(与 judge 无关,judge 不输出分数,只有 KEEP/DROP)。                                                                                                                                                                 |

## 9. 验收清单(过滤作业完成的标准)

- [ ] 每条输入轨迹有且仅有一条 review,record_id 一一对应(无遗漏/重复/编造);
- [ ] 所有 review 的 evidence ≥2 条且引用了轨迹中的具体内容(抽查 30 条);
- [ ] judge 输入确认不含 score/reward/成败字段(脚本断言);
- [ ] DROP 原因分布统计表(按 §5 四类归档);
- [ ] 人工抽查:DROP 误杀率 ≤10%,KEEP 漏杀(明显泄漏/轮询被留)≤5%;
- [ ] KEEP 集 fingerprint 复核(`sft_data_fingerprint.py`):think p50 / cmd/步 / 交卷率 / >20k 步占比,与过滤前对比记录在案;
- [ ] `OVERLONG_UNJUDGED` 与 `JUDGE_ERROR` 清单单独存档。
