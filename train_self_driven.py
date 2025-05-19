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
from coggraph import CogGraph          # 需确保已实现 forward 三接口
from agents.rl_agent import RLAgent
from utils import IntrinsicCuriosityModule

import torch.nn as nn

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

    device = torch.device(cfg.device)

    # 1) 初始化 CogGraph 内部环境 & 图
    # 先用一个临时 env 推断初始输入维度（Graph 还没挂 agent，所以不用它）
    from env import GridEnvironment
    temp_env = GridEnvironment(size=5)
    init_state = torch.from_numpy(temp_env.get_state()).float()
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

    # 2) 动态推断输入维度后再建 Agent
    init_state = torch.from_numpy(env.get_state()).float()
    input_dim = _infer_input_dim(graph, init_state)
    agent = RLAgent(
        input_dim=input_dim,
        num_actions=4,
        lr=cfg.lr,
        gamma=cfg.gamma,
        d_model=64,
        device=device)

    graph.rl_agent = agent

    # 在创建 agent 之后 —— 初始化 Intrinsic Curiosity Module
    icm = IntrinsicCuriosityModule(
        state_dim=input_dim,
        action_dim=agent.policy_net.fc_out.out_features,
        hidden_dim= 128,    # 隐藏层大小，你可以改成 d_model 或者 128
        lr=1e-4           # ICM 学习率
    ).to(device)
    curiosity_beta = 0.3 if cfg.use_curiosity else 0.0  # 内在奖励权重

    last_dim = graph.processor_hidden_size  # 初始 D_old

    print(f"[Init] transformer input_dim = {input_dim}, device = {device}")

    # 3) 训练循环 —— 伪 Episode 截断（环境只 reset 一次，每隔 max_steps 步做一次更新）
    reward_history = []
    # 只做一次环境初始化
    if hasattr(graph, "reset_state"):
        graph.reset_state()
    env.reset()
    state = torch.from_numpy(env.get_state()).float().to(device)
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
        inp = torch.cat([flat_state, goal_vec], dim=0).unsqueeze(0)      # (1, D)
        graph.step(inp)

        # —— ③ 如果隐藏维度变了，一次性调整 policy_net 和 value_head ——
        new_dim = graph.processor_hidden_size
        if new_dim != last_dim:
            print(f"[Resize] dim: {last_dim} → {new_dim}")
            agent.resize_state_dim(new_dim)  # ← 这里自动重建并拷贝策略网络＋价值网络
            icm.expand_state_dim(new_dim)
            last_dim = new_dim

        # —— ④ 拿 Transformer 输入 ——
        sensor_out    = graph.sensor_forward(state)      # (1, state_dim)
        processor_out = graph.processor_forward(sensor_out)
        env.render()
        state_seq = torch.stack([sensor_out, processor_out], dim=0).unsqueeze(0).to(device)  # (1,2,dim)

        # —— ⑤ 选动作 & 执行环境步 ——
        action = agent.select_action(state_seq)
        env.step(action, cog_step=graph.current_step)

        # —— ⑥ Reward shaping ——
        agent_pos   = tuple(env.agent_pos)
        goal_pos    = graph.task.target_position
        dist_res    = abs(agent_pos[0]-goal_pos[0]) + abs(agent_pos[1]-goal_pos[1])
        proximity_bonus = 0.01 if dist_res <= 2 else 0.0
        danger_dist = env.distance_to_nearest_danger(agent_pos)
        if danger_dist <= 2:
            danger_shaping = -0.05
        elif danger_dist >= 3:
            danger_shaping = 0.0

        # —— ⑦ 计算外在奖励 & 内在奖励 ——
        ext_reward = (
            getattr(env, "agent_energy_gain", 0.0)
          - getattr(env, "agent_energy_penalty", 0.0)
          + proximity_bonus + danger_shaping
        )
        next_raw    = torch.from_numpy(env.get_state()).float().to(device)
        next_sensor = graph.sensor_forward(next_raw)   # (1, state_dim)
        ic_reward = (
            icm.compute_intrinsic_reward(
                sensor_out.squeeze(0),
                next_sensor.squeeze(0),
                torch.tensor([action], device=device)
            ) if cfg.use_curiosity else 0.0
        )

        # —— ⑧ 计算衰减因子 & 合并奖励 ——
        progress = min(global_step, 2500) / 2500
        decay    = 1.0 - 0.7 * progress
        total_reward = (ext_reward + curiosity_beta * ic_reward) * decay

        agent.store_reward(total_reward)
        ep_reward += total_reward

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
python train_self_driven.py --episodes 10 --max-steps 1000 --save-every 2 --device cpu
"""