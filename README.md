EvoCore 系统结构与模块功能总览

项目简介：EvoCore = 可成长的 AI 胚胎体

EvoCore 并非传统意义的 AI 模型，而是一个具备 **生命周期、结构演化、能量机制、自主学习与死亡重生** 能力的智能体。

> 它像一颗 AI 胚胎，随着环境刺激、能量流动与学习反馈，不断**成长、分化、合并、死亡与重生**，是一种“活”的模型架构。

核心组成如下：

---

项目结构概览

* `coggraph.py`：CogGraph 主控模块，负责细胞网络构建、连接、生命周期调度、能量调控、结构维护。
* `CogUnit.py`：单个细胞 CogUnit 的行为逻辑，包括状态更新、能量代谢、分裂死亡、基因记忆等。
* `env.py`：环境模块，定义了输入状态与奖励机制。
* `transformer_policy.py`：轻量 Transformer 策略网络。
* `rl_agent.py`：策略代理，负责使用 Transformer 进行强化学习。
* `train_self_driven.py`：主训练入口。
* `eval_policy.py`：策略评估脚本。
* `utils.py`：辅助工具函数。

---

CogUnit（细胞单元）结构与机制详解 \[`CogUnit.py`]

核心特性

* 拥有独立 `state`, `energy`, `age`, `gene`, `memory_pool`
* 拥有自己的 `function` 网络：一个浅层 MLP，自主处理输入 → 输出
* 每个 CogUnit 角色为：`sensor`, `processor`, `emitter`

能量机制

* 每步根据输入复杂度、连接强度、调用频率动态计算能量消耗
* 若能量高于阈值 → 可分裂
* 若能量为 0 或老化/退化 → 死亡
* 能量税机制由 `coggraph.py` 控制（限制系统总能）

分裂机制

* 满足角色门槛（min\_energy + min\_calls）后，可执行 `clone()`
* clone 时会融合局部记忆、基因变异（突变只允许 hidden\_size 上升）

基因记忆系统

* 每个 CogUnit 维护 local memory pool，记忆高质量输出片段
* 复制时融合自身与记忆的基因 + 行为偏好（如 sensor\_bias）
* 记忆可评分并逐步淘汰弱记忆

---

CogGraph（细胞图谱）主控逻辑 \[`coggraph.py`]

生命周期调度

* 每步调用 `step()`，更新所有单元状态，触发生长、连接、剪枝、死亡等
* 模拟生命周期阶段（从 0 开始，每步推进）

能量调控

* 系统总能量上限控制（如 250）
* 超过时触发累进能量税，或转移至 energy\_pool 储存池
* energy\_pool 可反哺低能细胞（救活机制）

成长机制

* 定期触发 `rebalance_cell_types()`：根据比例 1:2:1 自动调节 sensor/processor/emitter 数量
* 自动连接 + 随机突变连接 + 死连接剪除
* 超能细胞强制分裂，新增结构

结构演化

* 相似单元 merge：`merge_redundant_units()`
* 相似子图重构：`restructure_common_subgraphs()` → processor + emitter 成为复合单元
* 子系统发现：自动识别高密度子图 → 标记 subsystem\_id

死亡机制

* 低能 / 长寿 / 输出退化 → `should_die()` 被移除
* 死亡可触发能量分配 + 遗传信息留存

重启机制（文档规划中）

* 若系统陷入崩溃、模块频死、能量失控 → 模拟“死亡 + 重启”阶段
* 通过保存“摘要记忆”实现重生后的快速恢复

---

学习机制：策略学习 & 自我优化

`transformer_policy.py`

* 使用 TransformerEncoder + 可学习位置编码
* 输入为多个 CogUnit 的状态序列（如 processor 输出）
* 输出为 logits（分类动作为上下左右等）

`rl_agent.py`

* 使用 `REINFORCE + baseline` 策略
* 包含：策略网络 + 值函数（value head）
* 每步缓存状态、动作 log\_prob、reward → episode 结束后优化策略

---

训练与评估流程

训练入口 \[`train_self_driven.py`]

* 初始化 `CogGraph`、环境、agent
* 每步运行：env → graph.step() → agent.select\_action()
* 每集结束后执行策略梯度更新 `agent.finish_episode()`

评估入口 \[`eval_policy.py`]

* 加载已训练模型 checkpoint
* 运行指定 episode 数量，评估平均 reward

---

系统核心理念回顾

EvoCore 架构目标是：

1. 不再使用“静态网络”，而是从原始细胞自动发育结构
2. 使用能量系统调控行为活跃性与系统规模
3. 具备复制、变异、死亡、遗传等“生命式演化机制”
4. 学习方式可进化（结合强化学习、遗传融合、自我反馈）
5. 长期目标是形成结构可塑性 + 功能多样性的智能体

最终将朝着：环境驱动、自主成长、结构演化、强化学习融合的“AI 胚胎体方向发展。




**Project Introduction: EvoCore = A Growing AI Embryo**
EvoCore is not a traditional AI model. It is an intelligent agent with life cycle, structural evolution, energy regulation, self-learning, and death-rebirth ability.

It works like an AI embryo. With environmental inputs, energy flow, and learning feedback, it keeps growing, differentiating, merging, dying, and rebirthing. It is a “living” model architecture.

**Project Structure Overview**
coggraph.py: Main controller of CogGraph, managing unit network structure, connections, lifecycle scheduling, energy control, and structural evolution.

CogUnit.py: Behavior logic of single cell unit, including state update, energy metabolism, splitting, death, genetic memory, etc.

env.py: Environment module, defines state input and reward mechanism.

transformer_policy.py: Lightweight Transformer-based policy network.

rl_agent.py: Reinforcement learning agent that uses Transformer for decision-making.

train_self_driven.py: Main training script.

eval_policy.py: Evaluation script for policy.

utils.py: Helper functions.

**CogUnit (Cell Unit) Design & Mechanisms** [CogUnit.py]
Core Features

Each unit has its own state, energy, age, gene, and memory_pool.

Contains its own shallow MLP as function network to process input → output.

Roles: sensor, processor, emitter.

Energy System

Each step, energy is consumed based on input complexity, connection strength, and call frequency.

If energy > threshold → can split().

If energy = 0 or aging → die().

Energy tax is managed by coggraph.py (to control total system energy).

Splitting Mechanism

If unit meets energy and usage thresholds → it can clone().

During cloning, it mixes local memory and mutates its gene (only allows increase in hidden size).

Genetic Memory System

Each unit keeps a local memory pool, saving high-quality output fragments.

When cloning, it combines self-gene with memory preference (e.g., sensor_bias).

Low-score memories will be removed gradually.

**CogGraph (Main Graph Logic)** [coggraph.py]
Lifecycle Scheduling

Each step calls step(), updating all units, triggering growth, connection, pruning, or death.

Simulates a living system step by step.

Energy Regulation

Total energy of the system has an upper limit (e.g., 250).

If over limit → apply progressive energy tax or move excess to energy_pool.

energy_pool can feed low-energy units (like rescue).

Growth System

Runs rebalance_cell_types() regularly to adjust sensor:processor:emitter ratio to 1:2:1.

Auto-connect + random mutate + prune dead connections.

Over-energy units must split and add new structure.

Structural Evolution

Similar units will be merged by merge_redundant_units().

Similar subgraphs reconstructed by restructure_common_subgraphs() → processor + emitter become composite units.

High-density subgraph will be marked as subsystem_id.

Death Mechanism

Low energy / old age / weak output → should_die() → remove the unit.

Death can trigger energy redistribution and genetic memory saving.

Reboot Mechanism (Planned)

If system crashes, frequent deaths, or energy chaos → enter "death + restart" phase.

Will save abstract memory for fast recovery after reboot.

**Learning Mechanism: Strategy Learning & Self-Optimization**
[transformer_policy.py]

Uses TransformerEncoder with learnable positional encoding.

Input is a sequence of CogUnit states (e.g., processor outputs).

Output is action logits (e.g., move up/down/left/right).

[rl_agent.py]

Uses REINFORCE with baseline method.

Includes policy network + value function (value head).

Caches state, action log_prob, reward at each step → update policy after each episode.

**Training & Evaluation Flow**
Training Entry [train_self_driven.py]

Initializes CogGraph, environment, and agent.

Each step: environment → graph.step() → agent.select_action().

After each episode → policy is updated with agent.finish_episode().

**Evaluation Entry** [eval_policy.py]

Loads saved model checkpoint.

Runs specified number of episodes and calculates average reward.

**The goals of EvoCore architecture:**

Replace static networks with self-growing structures from original cells.

Use energy to control activity and system size.

Include life-like mechanisms: clone, mutate, die, inherit.

Support evolutionary learning: reinforcement + memory + feedback.

Long-term goal is to create intelligent agents with structural plasticity and functional diversity.

The final direction is to become a truly self-developing AI embryo — driven by environment, growing its structure, evolving, and learning continuously.