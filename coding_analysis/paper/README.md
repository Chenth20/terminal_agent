# 终端智能体训练与评测论文

本项目现有证据最充分的是 Qwen3.5-4B 在 TerminalHorizon 教师轨迹上的 SFT，再用 Terminal-Bench 2.1 评测。以下论文的数字均为各作者在其模型和评测设置下的结果。项目自己的结果、命令和评测口径见[交接文档](../HANDOFF.md)。

### 从能执行的长轨迹学习

**TerminalHorizon** 用独立求解发现环境里被预先完成的工作，再逐步加入依赖上一步产物的新要求。论文从 1,535 个环境获得训练数据，使用 2,852 条轨迹的 27B 模型在 TB2.1 从 46.82% 升至 60.67%（表 2、图 6）。本项目使用其中经处理的 2,441 条整轨迹训练 4B，是最直接的论文落地。论文表 3 在 100 个环境上比较教师的交互轮数、反馈链长度和命令重复率。本地扫描显示 qwen38 flash 动作链更长，但许多轮是带不同参数的轮询。比较新教师时，可先抽同一批任务，检查失败后有没有根据输出改变动作，再决定是否扩采。[论文原文](2610.13166v2.pdf)

**Data Recipes for Agentic Models** 将任务来源、混合、增强、教师、轨迹过滤分开做消融。表 6 显示教师本身的 benchmark 排名和训练出的学生排名可以不同。表 7 与附录 A.3 显示，作者的数据中过滤少于 5 轮的轨迹在对齐 token 预算后仍有收益。本项目在同一基准上已有不同教师产生不同学生表现的观察，但教师、任务分布和训练配置同时变化。下一轮可在同一批训练任务上更换教师，固定有效监督 token 预算、底座与评测设置，比较完成率和超时率。[论文原文](2606.24855v1.pdf)

### 让错误能被观测、能被修正

**FACET** 先构建可运行的初始环境，再依据相同环境生成题目、参考解和 verifier。作者在自己的语料中观察到，失败轨迹的 54% 只漏掉一至两个检查项；4B 在论文 TB2.1 的三次采样平均从 17.60% 升至 24.72%（表 2、图 2a）。本项目 c312 也有完成主交付却遗漏边界条件的案例。可在独立训练题中给教师明确的需求，采集“验证边界条件→发现遗漏→修正→复验”的轨迹。用留出题的整题得分和失败断言类型分别衡量是否学会覆盖完整需求。[论文原文](2608.18580v1.pdf)

**Terminal-Universe** 从轨迹重建工作区，再重解已有需求或生成新需求。表 9 比较持续多轮任务在有无轮级验证反馈时的效果；表 10 在同为 35.1k 条记录的实验中，扩充环境、同环境扩题、同题多解分别得到 56.0%、53.8%、53.9%（27B、Terminus2-XML）。本项目现有 TB2.1 的一次任务有多次 shell 交互，只有一次最终 reward；其间每个终端动作没有论文中每次用户需求的独立 verifier 标签。可优先在新训练环境采集反馈明确的重解轨迹，和同任务增加采样作等预算对照。直接按尾部思考长度裁剪会丢失真实的交付前复验。[论文原文](2609.04148v1.pdf)

### SFT 后的 RL 实验

**TMax** 用有可执行终局奖励的环境训练 terminal agent，按难度、领域和验证方式生成任务。作者用 DPPO 在 Qwen3.5-9B 上直接做 RL，自建 harness 上 TB2.1 从 20.5% 升至 29.6%（表 2）；其常被引用的 33% 来自 TB2.0。Data Recipes 表 11 的另一组结果表明，Qwen3-8B 可从先 SFT 再 RL 中获益。两组实验的底座和环境不同。本项目第二阶段可以从已复现的 SFT checkpoint 起步，在独立训练环境先测每题多次采样的奖励分布与可训练题数，训练后用相同预算和同一套 TB2.1 设置与 SFT 起点配对复测。RL 和面向爬虫等实际场景的专项训练均是后续计划。[论文原文](2606.23321v2.pdf)

## 157 节点补充论文

| 主题 | 论文 |
|---|---|
| 交互轨迹的训练效果 | [What Makes Interaction Trajectories Effective for Training Terminal Agents?](2606.03461v1.pdf) |
| 长程任务的细粒度评分 | [Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading](2607.08964v3.pdf) |
| 面向 4B 编程模型的在线任务合成 | [FrogNano: Training a 4B Coding Agent via Online Task Synthesis](2609.07925v5.pdf) |
| 长程终端任务的强化学习 | [T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks](2609.11042v1.pdf) |
| 终端智能体的递归自我改进 | [Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine](2610.22192v1.pdf) |
