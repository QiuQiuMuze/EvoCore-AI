#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_self_driven.py
====================
启动自驱动强化学习训练：
    GridEnvironment  ←→  CogGraph  ←→  RLAgent (TransformerPolicyNetwork)

❖ 主要假设
- env.step(action) **原实现不返回** 新状态 / 奖励 / done，所以：
    • 执行 env.step(action) 后，直接用 env.get_state() 取下一状态；
    • 奖励 = env.agent_energy_gain - env.agent_energy_penalty（env.step 内已更新）；
    • 每个 episode 固定 MAX_STEPS 步视为终止。
- CogGraph 需要你补充 sensor_forward / processor_forward / emitter_forward。
  若暂未实现，则脚本会 fallback：直接把环境状态张量当作 sensor / processor 输出。

❖ 运行示例
$ python train_self_driven.py --episodes 5000 --max-steps 256
"""
from __future__ import annotations
from env import logger
import argparse
import os
import time
import torch

from env import GridEnvironment
from coggraph import CogGraph, N_GOAL_CHANNELS          # 需确保已实现 forward 三接口
from agents.rl_agent import RLAgent
from utils import IntrinsicCuriosityModule
from collections import deque
import torch.nn as nn
import torch.nn.functional as F

def resize_input_proj(net: nn.Module, new_dim: int, device):
    """
    把 net.policy_net.input_proj 从（old_dim→d_model）换成 (new_dim→d_model)，
    并把旧权重的前 min(old_dim,new_dim) 列搬过去，bias 完全复制。
    """
    old = net.policy_net.input_proj
    d_model = old.out_features
    new = nn.Linear(new_dim, d_model).to(device)
    with torch.no_grad():
        # 复制旧权重到新权重的前半部分
        cols = min(old.in_features, new_dim)
        new.weight[:, :cols].copy_(old.weight[:, :cols])
        new.bias.copy_(old.bias)
    net.policy_net.input_proj = new

# -------------------------------------------------------------------------- #
#                              参数解析                                      #
# -------------------------------------------------------------------------- #
def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-curiosity", action="store_true",
                        help="enable intrinsic curiosity reward")
    parser.add_argument("--no-curiosity", action="store_false", dest="use_curiosity")
    parser.set_defaults(use_curiosity=True)

    parser.add_argument("--episodes", type=int, default=10_000,
                        help="total training episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="steps per episode")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate for policy network")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="save checkpoint every N episodes (0 = never)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu | cuda")
    parser.add_argument(
        "--fastflag", action="store_true",
        help="关闭 torch.compile / amp / CUDA-Graph 等所有加速开关（debug 或 CPU 模式用）"
    )
    return parser.parse_args()


# -------------------------------------------------------------------------- #
#                          动态维度探测工具                                   #
# -------------------------------------------------------------------------- #
@torch.no_grad()
def _infer_input_dim(graph: CogGraph, env_state: torch.Tensor) -> int:
    """
    尝试通过 graph.sensor_forward / processor_forward 推断 transformer 输入维度。
    如果 graph 还未实现接口，则直接返回 env_state.size(0)。
    """
    if hasattr(graph, "sensor_forward"):
        s_out = graph.sensor_forward(env_state)
        if isinstance(s_out, torch.Tensor):
            return s_out.numel()
    return env_state.numel()


# -------------------------------------------------------------------------- #
#                              训练主函数                                    #
# -------------------------------------------------------------------------- #
def main(cfg):
    import logging
    logging.debug("✅ Logger 测试 Debug")
    logging.info("✅ Logger 测试 Info")
    # ────────── ① 根据 fastflag 决定是否全关加速 ──────────
    from config_runtime import RF          # ★ 必须在 disable 前 import
    if not cfg.fastflag:                   # 没加 --fastflag → 启用全部加速
        pass                               # 什么都不做，保持 RF 里的默认优化
    else:                                  # 加了 --fastflag → 彻底关闭
        RF.disable_all()                   # 包括 compile / amp / cudagraph / batch …
    # ──────────────────────────────────────────────────────
    device = torch.device(cfg.device)

    # 1) 初始化 CogGraph 内部环境 & 图
    # 先用一个临时 env 推断初始输入维度（Graph 还没挂 agent，所以不用它）
    temp_env = GridEnvironment(size=5)
    init_state = temp_env.get_state().float()
    init_dim = _infer_input_dim(None, init_state)  # None 会让 _infer_input_dim 返回 env.size

    # 2) 根据推断维度创建 RLAgent
    agent = RLAgent(
        input_dim=init_dim,
        num_actions=temp_env.action_space_n,
        lr=cfg.lr,
        gamma=cfg.gamma,
        d_model=64,
        device=device
    )

    # 3) 现在把 agent 注入 CogGraph，再取它的 env
    graph = CogGraph(agent, device=cfg.device)
    graph.debug = True
    env = graph.env

    # ---- 新增 ----
    import CogUnit
    CogUnit.MAX_OUTPUT_DIM = graph.processor_hidden_size
    # --------------------

    # 2) 动态推断 processor 输出维度 + 显式拼接目标向量后 rebuild Agent
    init_state = env.get_state().float()
    proc_dim = _infer_input_dim(graph, init_state)                   # e.g. 125
    goal_dim = env.size * env.size * N_GOAL_CHANNELS
    full_dim = proc_dim + goal_dim                                   # e.g. 125+50=175
    agent = RLAgent(
        input_dim=full_dim,                                          # ← 用 full_dim
        num_actions=4,
        lr=cfg.lr,
        gamma=cfg.gamma,
        d_model=64,
        device=device)

    graph.rl_agent = agent

    # 在创建 agent 之后 —— 初始化 Intrinsic Curiosity Module
    icm = IntrinsicCuriosityModule(
        state_dim=full_dim,
        action_dim=agent.policy_net.fc_out.out_features,
        hidden_dim= 128,    # 隐藏层大小，你可以改成 d_model 或者 128
        lr=1e-4           # ICM 学习率
    ).to(device)
    curiosity_beta = 0.3 if cfg.use_curiosity else 0.0  # 内在奖励权重

    last_dim = full_dim

    print(f"[Init] transformer input_dim = {full_dim}, device = {device}")

    # 3) 训练循环 —— 伪 Episode 截断（环境只 reset 一次，每隔 max_steps 步做一次更新）
    reward_history = []
    # 只做一次环境初始化
    if hasattr(graph, "reset_state"):
        graph.reset_state()
    env.reset()
    state = env.get_state().float()
    # —— 新增：用滑动窗口保存最近 4 步的带目标向量的特征 —— #
    history = deque(maxlen=4)
    # 构造一个固定长度 = proc_dim + goal_dim 的 init_feat
    init_sensor    = graph.sensor_forward(state)                      # (1, state_dim)
    init_processor = graph.processor_forward(init_sensor).squeeze(0)  # (proc_dim,)
    tv = graph.target_vector.to(device)
    goal_vec = tv.view(-1)
    init_feat = torch.cat([init_processor, goal_vec], dim=-1)
    # --- 保守起见：若将来 proc_dim 变小，也对 init_feat 右补零 ---
    if init_feat.numel() < last_dim:  # last_dim 之前已经设为 full_dim
        init_feat = F.pad(init_feat, (0, last_dim - init_feat.numel()))
    for _ in range(history.maxlen):
        history.append(init_feat)




    ep_reward = 0.0
    step_in_horizon = 0

    # 总步数 = episodes * max_steps
    TOTAL_STEPS = cfg.episodes * cfg.max_steps
    horizon_id = 0
    for global_step in range(1, TOTAL_STEPS + 1):
        step_in_horizon += 1

        # —— ① 打印信息 ——
        logger.warning(f"\n==== Step {global_step} (within horizon step {step_in_horizon}) ====")

        # —— ② 构造 CogGraph.step() 的输入 ——
        goal_vec = graph.task.encode_goal(graph.env.size).float().to(device)
        flat_state = state.view(-1)                                       # (env_size*env_size*C,)
        inp = state.view(1, -1).to(device)     # (1, D)
        graph.step(inp)

        # 如果 processor_hidden_size 变化，则同时更新到 full_dim
        proc_dim = graph.processor_hidden_size
        goal_dim = env.size * env.size * N_GOAL_CHANNELS
        new_full = proc_dim + goal_dim
        if new_full != last_dim:
            print(f"[Resize Input] dim: {last_dim} → {new_full}")
            agent.resize_state_dim(new_full)
            icm.expand_state_dim(new_full)
            last_dim = new_full


            # —— 隐藏维度变更时，重置 history —— #
            history = deque(maxlen=history.maxlen)
            init_sensor    = graph.sensor_forward(state)
            init_processor = graph.processor_forward(init_sensor).squeeze(0)
            tv = graph.target_vector.to(device)
            goal_vec = tv.view(-1)
            init_feat = torch.cat([init_processor, goal_vec], dim=-1)
            if init_feat.numel() < last_dim:  # new_full 已经赋给 last_dim
                init_feat = F.pad(init_feat, (0, last_dim - init_feat.numel()))
            for _ in range(history.maxlen):
                history.append(init_feat)




        # —— ④ 拿 Transformer 输入 ——
        sensor_out    = graph.sensor_forward(state)      # (1, state_dim)
        processor_out = graph.processor_forward(sensor_out)
        env.render()

        # —— 新增：加上本轮目标向量（无类别提示）的位置编码 —— #
        tv = graph.target_vector.to(device)
        goal_vec  = tv.view(-1)
        step_feat = torch.cat([processor_out.squeeze(0), goal_vec], dim=-1)
        if step_feat.numel() < last_dim:  # 防止 < last_dim
            step_feat = F.pad(step_feat, (0, last_dim - step_feat.numel()))

        history.append(step_feat)

        # 用最近 history.maxlen 步的 processor_out 序列作为 Transformer 输入

        # --- 对齐 history 中的张量 ---
        max_len = max(t.numel() for t in history)  # 找到最长的
        aligned = []
        for t in history:
            if t.numel() < max_len:  # 右侧补零
                t = F.pad(t, (0, max_len - t.numel()))
            aligned.append(t)

        seq = torch.stack(aligned, dim=0).unsqueeze(0).to(device)  # (1, L, max_len)
        state_seq = seq


        # —— ⑤ 选动作 & 执行环境步 ——
        action = agent.select_action(state_seq)
        # ——— 使用 capture 形式执行一步环境，获取 raw_reward ———

        # —— ⑥ 与环境交互，获取原始奖励 ——
        next_state_np, raw_reward, done, _ = env.step(action, cog_step=graph.current_step)

        # —— ⑦ 组合最终外在奖励 —— #
        ext_reward = raw_reward
        # —— ⑧ 下一状态转张量 ——
        next_raw = next_state_np.float().to(device)

        # —— ⑩ 计算 Intrinsic Curiosity Reward —— #
        # 1) 下一个 Processor 特征
        next_sensor = graph.sensor_forward(next_raw)  # (1, proc_dim)
        next_processor = graph.processor_forward(next_sensor)  # (1, proc_dim)
        # 2) 下一个 Goal 向量
        tv_next = graph.target_vector.to(device)
        goal_vec_next = tv_next.view(-1)
        # 3) 拼接成 full_dim 特征
        next_feat = torch.cat([next_processor.squeeze(0), goal_vec_next], dim=-1)
        if next_feat.numel() < last_dim:
            next_feat = F.pad(next_feat, (0, last_dim - next_feat.numel()))
        # 4) 用 step_feat (上面已构造) 和 next_feat 计算 IC 奖励
        ic_reward = icm.compute_intrinsic_reward(
            step_feat,
            next_feat,
            torch.tensor([action], device=device)
        ) if cfg.use_curiosity else 0.0

        # —— ⑧ 计算衰减因子 & 合并奖励 ——
        progress = min(global_step, 5000) / 5000
        decay    = 1.0 - 0.35 * progress
        total_reward = (ext_reward + curiosity_beta * ic_reward) * decay

        agent.store_reward(total_reward)
        ep_reward += total_reward
        # —— 新增：每200步也更新 ICM —— #
        if cfg.use_curiosity and global_step % 200 == 0:
            icm.update_parameters()


        # —— ⑨ 更新 state ——
        state = next_raw

        # —— ⑩ 达到截断长度，做一次“Episode”更新 ——
        if step_in_horizon >= cfg.max_steps:
            horizon_id += 1
            agent.finish_episode()
            if cfg.use_curiosity:
                icm.update_parameters()
            reward_history.append(ep_reward)
            step_in_horizon = 0
            ep_reward = 0.0
            # 注意：不调用 env.reset()
        if cfg.save_every and horizon_id % cfg.save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = f"checkpoints/agent_h{horizon_id}.pth"
            agent.save(ckpt)
            print(f"[Save] saved {ckpt}")

        # —— （可选）每隔 10 个 horizon 打印一次平均奖励 ——
        if global_step % (cfg.max_steps * 10) == 0 and reward_history:
            last10 = reward_history[-10:]
            print(f"[Step {global_step}] avg_reward(last10 horizons) = {sum(last10)/len(last10):.4f}")

    # 收尾：如果最后一段未满 max_steps，也要做一次更新
    if step_in_horizon > 0:
        horizon_id += 1
        agent.finish_episode()
        if cfg.use_curiosity:
            icm.update_parameters()
        reward_history.append(ep_reward)

    # —— 训练总结 & 最终保存 ——
    print(f"Training finished ✓  total horizons = {horizon_id}")
    os.makedirs("checkpoints", exist_ok=True)
    agent.save("checkpoints/agent_final.pth")
    print("Saved final model to checkpoints/agent_final.pth")


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = get_cfg()
    main(cfg)
    t0 = time.time()
    print(f"Total runtime: {time.time() - t0:.1f} s")

"""
关键点说明
位置	说明
input_dim 自动推断	通过 graph.sensor_forward() 探测输出维度；若接口未实现，则退化为环境 state 大小。
Episode 终止	因 GridEnvironment 当前无 done 标志，采用固定 MAX_STEPS（可通过 --max-steps 调整）。
奖励计算	直接使用环境在 step() 内更新的 agent_energy_gain / agent_energy_penalty 字段。
断点续训	--save-every 控制周期性保存，文件包含网络参数 + 优化器状态。

下一步

确认 CogGraph 已实现 sensor_forward / processor_forward / emitter_forward。

若想引入 early-stop（如达到目标点）、或更复杂的奖励，可在 env 内部扩展 done 与 info 返回值，再更新脚本对应部分。
"""

"""
python train_self_driven.py --episodes 3 --max-steps 1000 --save-every 1 --device cpu
"""
"""
# 纯 CPU，什么都不用装
python train_self_driven.py --episodes 10 --max-steps 500 --save-every 2 --device cpu  # RF.use_shared_tx=True 也能跑，走官方 nn.Transformer

# GPU + Flash-Attn / TE
pip install flash-attn --no-build-isolation
pip install transformer-engine
python train_self_driven.py --episodes 3 --max-steps 1000 --save-every 1 --device cuda          # 默认每步跑一次共享 Tx
# 或每 4 步跑一次，省 Python
sed -i 's/shared_tx_interval = 1/shared_tx_interval = 4/' config_runtime.py
python train_self_driven.py --episodes 3 --max-steps 1000 --save-every 1 --device cuda
要完全关掉这条加速路径（比如纯 Debug）：
from config_runtime import RF
RF.use_shared_tx = False
"""

