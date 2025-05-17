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
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="total training episodes")
    parser.add_argument("--max-steps", type=int, default=256,
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
    graph = CogGraph(device=cfg.device)  # 内部已包含 env = GridEnvironment(size=5)
    graph.debug = True
    # —— 让 train 循环也用同一个 env 实例 ——
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

    # 在创建 agent 之后 —— 初始化 Intrinsic Curiosity Module
    icm = IntrinsicCuriosityModule(
        state_dim=input_dim,
        action_dim=agent.policy_net.fc_out.out_features,
        hidden_dim=64,    # 隐藏层大小，你可以改成 d_model 或者 128
        lr=1e-4           # ICM 学习率
    ).to(device)
    curiosity_beta = 0.05  # 内在奖励权重

    last_dim = graph.processor_hidden_size  # 初始 D_old

    print(f"[Init] transformer input_dim = {input_dim}, device = {device}")

    # 3) 训练循环
    reward_history = []
    for ep in range(1, cfg.episodes + 1):


        # 如 graph 有 reset_state() 请调用；否则新建实例或跳过
        if hasattr(graph, "reset_state"):
            graph.reset_state()
        env.reset()

        state = torch.from_numpy(env.get_state()).float().to(cfg.device)
        ep_reward = 0.0

        from env import logger
        for t in range(cfg.max_steps):
            # ——— 打印当前 Episode/Step 信息 ———
            logger.info(f"\n==== Episode {ep}  Step {t+1} ====")

            # ——— 构造 CogGraph.step() 的输入（含目标向量） ———
          # 这里我们复用 graph.task.encode_goal，
            # 生成 (env_size*env_size*INPUT_CHANNELS,) 的 one-hot goal
            goal_vec = graph.task.encode_goal(graph.env.size).float().to(device)

            # 把环境状态展平，再拼上目标向量
            flat_state = state.view(-1)  # (env_size*env_size*INPUT_CHANNELS,)
            inp = torch.cat([flat_state, goal_vec], dim=0).unsqueeze(0)  # (1, D)

           # ——— 真正调用 step() ———
            # 这一步会：
            #   1) 按序 sensor→processor→emitter
            #   2) 更新 self.current_step
            #   3) 触发所有 logger.info/debug
            graph.step(inp)
            # ——— 检测隐藏维度变化 & 重置 input_proj ———
            new_dim = graph.processor_hidden_size
            if new_dim != last_dim:
                print(f"[Resize] input_proj: {last_dim} → {new_dim}")
                resize_input_proj(agent, new_dim, device)
                icm.expand_state_dim(new_dim)
                last_dim = new_dim

            # ——— 拿出前向输出，构造 transformer 输入序列 ———
            sensor_out    = graph.sensor_forward(state)
            processor_out = graph.processor_forward(sensor_out)

            # ——— 可视化环境 ——（可选，和你最初版本对齐）———
            env.render()


            # --- 构造 transformer 输入 ---
            state_seq = torch.stack([sensor_out, processor_out], dim=0)  # (seq_len=2, dim)
            state_seq = state_seq.unsqueeze(0).to(device)                # (1, 2, dim)

            # --- 选动作 & 环境交互 ---
            action = agent.select_action(state_seq)
            env.step(action, cog_step=graph.current_step)

            # ---------- reward shaping ----------
            agent_pos = tuple(env.agent_pos)                    # (x, y)
            goal_pos  = graph.task.target_position              # 资源目标
            dist_res  = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
            proximity_bonus = 0.2 if dist_res <= 2 else 0.0     # 靠近资源

            danger_dist = env.distance_to_nearest_danger(agent_pos)
            if danger_dist <= 1:
                danger_shaping = -0.2                           # 靠近危险
            elif danger_dist >= 3:
                danger_shaping = 0.1                            # 远离危险
            else:
                danger_shaping = 0.0
            # --------------------------------------

            # # 奖励：环境内部字段决定
            # reward = (getattr(env, "agent_energy_gain", 0.0)
            #           - getattr(env, "agent_energy_penalty", 0.0)
            #           + proximity_bonus + danger_shaping)
            #
            # agent.store_reward(reward)
            # ep_reward += reward
            #
            # # 更新下一状态
            # state = torch.from_numpy(env.get_state()).float().to(cfg.device)
            # —— 1) 计算外在奖励 ——
            ext_reward = (
                    getattr(env, "agent_energy_gain", 0.0)
                    - getattr(env, "agent_energy_penalty", 0.0)
                    + proximity_bonus + danger_shaping
            )
            # —— 2) 拿到 raw 下一状态 & 对应的 sensor 特征 ——
            next_raw = torch.from_numpy(env.get_state()).float().to(device)
            next_sensor = graph.sensor_forward(next_raw)  # shape = (1, state_dim)
            # —— 3) 计算内在奖励 ——
            #    用当前 step 的 sensor_out (shape=(1,state_dim)) 而非 raw state
            ic_reward = icm.compute_intrinsic_reward(
                sensor_out.squeeze(0),  # (state_dim,)
                next_sensor.squeeze(0),  # (state_dim,)
                torch.tensor([action],
                             dtype=torch.long,
                             device=device)
            )
            # —— 4) 合并奖励并存储 ——
            total_reward = ext_reward + curiosity_beta * ic_reward
            agent.store_reward(total_reward)
            ep_reward += total_reward

            # —— 5) 更新下一步：raw state & (下一步的) sensor_out ——
            state = next_raw
            # 下次循环开始会重新计算 sensor_out = graph.sensor_forward(state)

        # --- Episode 结束：策略更新 ---
        agent.finish_episode()
        reward_history.append(ep_reward)
        # —— 每集结束后，优化 ICM 模型 ——
        icm.update_parameters()

        # --- 日志 & Checkpoint ---
        if ep % 100 == 0:
            avg_r = sum(reward_history[-100:]) / 100
            print(f"[Ep {ep:>5}]  avg_reward(100) = {avg_r:.4f}")

        if cfg.save_every and ep % cfg.save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/agent_ep{ep}.pth"
            agent.save(ckpt_path)
            print(f"[Save] {ckpt_path} saved")

    # 全程训练完成后再存一份最终模型
    os.makedirs("checkpoints", exist_ok=True)
    agent.save("checkpoints/agent_final.pth")
    print("Training finished ✓")


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