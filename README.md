**郭宗磊独立制作
Independently created by Zonglei Guo**

# EvoCore 系统结构与模块功能总览

---

## 项目简介：EvoCore = 可成长的 AI 胚胎体

EvoCore 并非传统意义的 AI 模型，而是一个具备 **生命周期、结构演化、能量机制、自主学习与死亡重生** 能力的智能体。

> 它像一颗 AI 胚胎，随着环境刺激、能量流动与学习反馈，不断 **成长、分化、合并、死亡与重生**，是一种 “活” 的模型架构。

---

## 项目结构概览

| 文件                      | 作用简述                                       |
| ----------------------- | ------------------------------------------ |
| `coggraph.py`           | CogGraph 主控模块，管理细胞网络构建、连接、生命周期、能量调控与结构演化   |
| `CogUnit.py`            | 单细胞 **CogUnit** 行为逻辑：状态更新、能量代谢、分裂/死亡、基因记忆等 |
| `env.py`                | 环境模块：定义状态输入、资源/陷阱分布与奖励机制                   |
| `transformer_policy.py` | 轻量级 Transformer 策略网络                       |
| `rl_agent.py`           | 强化学习代理，调用 Transformer 作决策                  |
| `train_self_driven.py`  | 主训练入口                                      |
| `eval_policy.py`        | 策略评估脚本                                     |
| `utils.py`              | 辅助工具函数                                     |
| `config_runtime.py`     | 运行期可调参数（如共享 Transformer、FP16、编译模式等）        |

---

##  新版本重要变更一览

1. **目标向量二通道化**

   * `TaskInjector.encode_goal()` 现返回 **`(2, env²)`** 的 one‑hot：<br>  `vec[0]` = 资源层，`vec[1]` = 陷阱层。
   * `INPUT_CHANNELS = 4 (state) + 2 (goal) = 6`，全链路已对齐。
   * `coggraph.step()` 顶部旧的 `goal_tensor = ...` 已删除，避免混淆。
2. **共享 Transformer 头数自适应**

   * 通过 `math.gcd(embed_dim, RF.shared_tx_heads)` 计算可整除的 **最大公因数** 作为实际多头数。
   * 若请求的头数无法整除，则自动降到满足整除的最接近值，并给出 warning。
   * 默认在 6 通道 \* env² 的 embed\_dim 下，往往得到 **2 头** —— 不是硬编码，而是 GCD 恰好为 2。
3. **环境扩容内存占用说明**

   * 每 1000 步触发一次 curriculum：`env_size += 5`，同时 **input\_size = env² × 6** 线性暴涨。
   * 为保持“细胞不降维”，`upscale_old_units()` 会 **复制旧权重并 zero‑pad 新维度**，导致显存/内存瞬时增加。
   * 可通过调小 `max_total_energy` 或延长扩容间隔来减缓内存高峰。

---

## CogUnit（细胞单元）结构与机制详解  \[`CogUnit.py`]

### 核心特性

* 独立持有 `state`, `energy`, `age`, `gene`, `memory_pool`
* 内置浅层 **MLP** (`Linear → ReLU → Linear`) 处理输入→输出，输入维度随环境自动升维
* 角色固定三类：`sensor` / `processor` / `emitter`

### 能量机制

* 动态消耗：输入方差 + 连接强度 + 调用密度 叠加加权
* 能量 > 阈值 ⇒ 允许 **split()** 复制
  能量 ≤0 或老化/退化 ⇒ **die()** 移除
* 系统层面的 **累进能量税** 限制总规模，超额能量转入 `energy_pool`

### 分裂 & 基因记忆

* 满足 `min_energy`+`min_calls` 条件方可克隆
* 克隆时混合 local memory，`hidden_size` 仅允许 **增加**（演化方向单向向上）
* 记忆池存储高分片段，低分逐渐淘汰；克隆时可注入 `*_bias` 基因

---

## CogGraph（细胞图谱）主控逻辑  \[`coggraph.py`]

### 生命周期调度

* 每步 `step()`：更新全部单元 → 分裂/死亡/连接/剪枝 等
* 环境扩容（curriculum）与共享 Transformer 调度均在此统一

### 能量调控

* `max_total_energy` 设置系统能量天花板
* 超标 ⇒ 累进税 或 转移至 `energy_pool`，后者可喂养弱细胞

### 成长 & 结构演化

* `rebalance_cell_types()` 保持 **1 : 2 : 1** (sensor\:processor\:emitter)
* `auto_connect()` + 随机突变 + 死连接剪除 = 连边自组织
* `merge_redundant_units()` 合并同质单元；`restructure_common_subgraphs()` 重构高相似子图
* 发现 **子系统**：局部高密度区域自动打上 `subsystem_id`

### 死亡 & 遗产

* `should_die()` 评估低能 / 老化 / 输出退化
* “寿终”细胞能量按角色分给年轻同类；优秀输出写入他人记忆池

---

## 学习机制：策略学习 & 自我优化

### `transformer_policy.py`

* **TransformerEncoder + 可学习 PE**
* 输入：选定若干 CogUnit 的状态序列（默认所有 processor 输出）
* 输出：动作 logits（上下左右）

### `rl_agent.py`

* **REINFORCE + baseline**
* 缓存 (state, log\_prob, reward)，episode 结束后统一更新

---

## 训练与评估

### 训练入口  \[`train_self_driven.py`]

1. 初始化 `CogGraph` / `GridEnvironment` / `RLAgent`
2. `env.step()` → `graph.step()` → `agent.select_action()`
3. Episode 结束 → `agent.finish_episode()` 做策略梯度

### 评估入口  \[`eval_policy.py`]

* 加载 checkpoint，跑若干 episode，输出平均 reward

---

## 系统核心理念回顾

1. **自生长**：不依赖预定义架构，细胞自组织出拓扑
2. **能量驱动**：用能量守恒调节活跃度与规模
3. **生命式演化**：复制 × 变异 × 死亡 × 遗传
4. **多重学习**：强化学习 + 记忆融合 + 元反馈
5. **可塑结构**：长期目标是功能多样且可持续重构的智能体

最终愿景：在真实或模拟环境中，EvoCore 将像胚胎一样，

> **被环境驱动 → 自主成长 → 结构演化 → 不断学习**，
> 成为真正的 **Self‑Developing AI Embryo**。

---

# EvoCore — System Overview & Module Guide

*(English section keeps the original “lively” tone while mirroring the Chinese content)*

## What is EvoCore? — A Growing **AI Embryo**

EvoCore is **not** a frozen neural net.
It is an *organism‑like* agent equipped with **life‑cycle, structural evolution, energy economy, self‑learning and death‑rebirth**.

> Think of it as an AI embryo: fed by environmental signals and energy flow, it keeps **growing, splitting, merging, dying, reviving**. The architecture lives and breathes.

---

## Repo Layout

| File                    | Brief Description                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| `coggraph.py`           | Top‑level **CogGraph** controller: builds the cell graph, manages connections, life‑cycle, energy & evolution |
| `CogUnit.py`            | Single‑cell logic: state update, metabolism, split/death, genetic memory                                      |
| `env.py`                | Grid environment with *resources* & *hazards*                                                                 |
| `transformer_policy.py` | Lightweight Transformer policy network                                                                        |
| `rl_agent.py`           | RL agent wrapping the Transformer                                                                             |
| `train_self_driven.py`  | Main training entry                                                                                           |
| `eval_policy.py`        | Policy evaluation script                                                                                      |
| `utils.py`              | Helper utilities                                                                                              |
| `config_runtime.py`     | Runtime switches (shared‑Tx, fp16, compile, etc.)                                                             |

---

##  What’s New

1. **Two‑Channel Goal Map**

   * `TaskInjector.encode_goal()` now returns **`(2, env²)` one‑hot**: channel‑0 resource, channel‑1 hazard.
   * `INPUT_CHANNELS = 4 (state) + 2 (goal) = 6` everywhere.
   * Legacy `goal_tensor` var in `coggraph.step()` is gone.
2. **Head‑count Auto‑Tuning** for shared Transformer

   * We pick the **greatest common divisor** between embed\_dim and the requested `RF.shared_tx_heads`.
   * If not divisible, we gracefully downgrade (with a warning).
   * With 6·env² dims the gcd often equals **2**, hence the observed "2 heads" — it is *data‑driven*, not hard‑coded.
3. **Memory Spikes on Curriculum Expansions**

   * Every 1000 steps the grid grows (+5), thus **input\_size = env² × 6** inflates quadratically.
   * `upscale_old_units()` keeps all historic params via zero‑padding → sudden RAM/GPU peaks.
   * Mitigate by lowering `max_total_energy` or stretching the expansion interval.

---

## CogUnit — Anatomy & Mechanics

* Private `state`, `energy`, `age`, `gene`, `memory_pool`
* Personal MLP (*Linear‑ReLU‑Linear*), auto‑resized when env grows
* Roles: **sensor / processor / emitter**

### Energy

* Consumption = f(input variance, connection strength, call density)
* `energy > thresh` ⇒ **split()**; `energy ≤ 0` or senescence ⇒ **die()**
* Progressive tax & `energy_pool` governed by CogGraph

### Split & Genes

* Need both energy & call quotas → `clone()`
* Clone mixes local memory, mutates gene, allows *only* larger hidden\_size
* Memory pool keeps high‑score chunks, bad ones decay

---

## CogGraph — The Big Orchestrator

* Central `step()` drives life‑cycle, curriculum, shared‑Tx
* Energy cap via `max_total_energy` + tax + pool
* `rebalance_cell_types()` → keeps **1:2:1** ratio
* Auto‑connect, mutate, prune dead links
* Merge redundant units; rebuild common sub‑graphs; detect *subsystems*
* Heritage: dying elders donate energy & memory

---

## Learning Pipeline

* **TransformerEncoder + learnable PE** (`transformer_policy.py`)
* RL agent: **REINFORCE w/ baseline** (`rl_agent.py`)
* Stores (state, log\_prob, reward) per step → updates after each episode

---

## Train & Eval

* **Train**: see `train_self_driven.py` (env → graph → action loop, then optimize)
* **Eval**: run `eval_policy.py` with saved checkpoints

---

## Core Vision

1. From *static* nets to **self‑grown** structures
2. Energy as the *budget* throttling size & activity
3. Life‑like evolution: clone × mutate × die × inherit
4. Fusion of RL, memory, meta‑feedback
5. Towards structural plasticity & functional diversity

> The grand goal: an **environment‑driven, self‑developing AI embryo** that never stops evolving.
