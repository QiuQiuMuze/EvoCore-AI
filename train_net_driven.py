#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_net_driven.py
===================
使用 ImmuneCogGraph 在 GridSecurityEnv 上进行强化学习训练：
  - env: 病毒 + 黑客双重威胁
  - graph: 免疫识别 + 抗体记忆 + 特攻单元
  - reward: 基于新感染数量和新黑客事件数量的惩罚
"""
import argparse
import os
import time
import torch
import random
from collections import deque

from env_net import GridSecurityEnv
from ImmuneCogGraph import ImmuneCogGraph
from agents.rl_agent import RLAgent

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int,   default=200,   help="训练的 Episode 数")
    parser.add_argument("--max-steps",   type=int,   default=500,   help="每个 Episode 最大步数")
    parser.add_argument("--lr",          type=float, default=1e-4,  help="Transformer 学习率")
    parser.add_argument("--gamma",       type=float, default=0.99,  help="折扣因子（如果用 RLAgent）")
    parser.add_argument("--d-model",     type=int,   default=128,   help="Transformer d_model")
    parser.add_argument("--size",        type=int,   default=10,    help="Grid 大小")
    parser.add_argument("--ramp",        type=int,   default=5000,  help="difficulty_ramp")
    parser.add_argument("--spawn-int",   type=int,   default=200,   help="病毒 spawn_interval")
    parser.add_argument("--hack-penalty",type=float, default=1.0,   help="每个新黑客事件的惩罚系数")
    parser.add_argument("--device",      type=str,   default="cpu", help="cpu | cuda")
    parser.add_argument("--save-every",  type=int,   default=50,    help="每 N episodes 保存一次模型 (0=不保存)")
    parser.add_argument("--save-dir",    type=str,   default="checkpoints", help="模型保存目录")
    return parser.parse_args()

def main(cfg):
    global_step = 0
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 1) 构造环境
    env = GridSecurityEnv(
        size=cfg.size,
        device=device,
        difficulty_ramp=cfg.ramp,
        spawn_interval=cfg.spawn_int
    )
    env.reset()

    # 2) 构造一个 dummy RLAgent 传给 ImmuneCogGraph （它内部实际用的是 TransformerPolicyNetwork）
    dummy_agent = RLAgent(
        input_dim=1,           # 不真正使用，随便填
        num_actions=1,         # 同上
        lr=cfg.lr,
        gamma=cfg.gamma,
        d_model=cfg.d_model,
        device=device
    )

    # 3) 构造 ImmuneCogGraph
    graph = ImmuneCogGraph(
        rl_agent=dummy_agent,
        device=device,
        env=env
    )

    episode_rewards = []

    for ep in range(1, cfg.episodes + 1):
        # Episode 开始前，只 reset env，memory 保持累积
        total_reward = 0.0

        # 记录前一步的感染 & 黑客统计
        prev_inf  = env.infected_map.sum().item()
        prev_hack = env.hack_history.sum().item()
        # 记录前一状态
        prev_inf_total = env.infected_map.sum().item()
        prev_hack_total = env.hack_history.sum().item()

        for step in range(1, cfg.max_steps + 1):
            global_step += 1
            # 环境当前状态张量
            state_tensor = env.get_state_tensor().view(1, -1)  # (1, C×H×W)

            # 把它送入 immune-graph，让它内部推进一次（包含环境 .step() 和单元决策）
            graph.step(state_tensor)
            # 当前状态
            curr_inf_total = env.infected_map.sum().item()
            curr_hack_total = env.hack_history.sum().item()

            # 清除量 = 之前有、现在没有
            delta_inf_cleared = max(0, prev_inf_total - curr_inf_total)
            delta_hack_cleared = max(0, prev_hack_total - curr_hack_total)

            # 更新历史记录
            prev_inf_total = curr_inf_total
            prev_hack_total = curr_hack_total

            # 计算新感染数 & 新黑客事件数
            curr_inf  = env.infected_map.sum().item()
            curr_hack = env.hack_history.sum().item()

            delta_inf  = curr_inf  - prev_inf
            delta_hack = curr_hack - prev_hack
            if delta_inf < 0:
                delta_inf *= 2

            # 简单的 reward 设计：每个新感染 -1，每个新黑客事件 -hack_penalty
            delta_inf_cleared = max(0, prev_inf_total - curr_inf_total)
            reward = delta_inf_cleared * 2.0 - curr_inf_total * 0.5 - delta_hack * cfg.hack_penalty

            total_reward += reward
            print(f"[Step {global_step:3d} | Ep {ep:3d} Step {step:3d}] "
                  f"病毒数={curr_inf_total:.0f}, 黑客数={curr_hack_total:.0f}, "
                  f"清除病毒={delta_inf_cleared:.0f}, 清除黑客={delta_hack_cleared:.0f}, "
                  f"total_reward={reward:.2f}")

            # 更新历史
            prev_inf  = curr_inf
            prev_hack = curr_hack

        episode_rewards.append(total_reward)
        avg_last10 = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
        print(f"[Episode {ep:3d}] reward = {total_reward: .3f}    avg(last10)= {avg_last10: .3f}")

        # 周期性保存 Transformer 模型
        if cfg.save_every > 0 and ep % cfg.save_every == 0:
            path = os.path.join(cfg.save_dir, f"transformer_ep{ep}.pth")
            torch.save(graph.transformer.state_dict(), path)
            print(f"  → Saved transformer to {path}")

    # 训练结束
    overall_avg = sum(episode_rewards) / len(episode_rewards)
    print(f"\nTraining finished: {len(episode_rewards)} episodes, overall avg reward = {overall_avg:.3f}")

    # 最后再保存一次
    final_path = os.path.join(cfg.save_dir, "transformer_final.pth")
    torch.save(graph.transformer.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    cfg = get_args()
    start = time.time()
    main(cfg)
    print(f"Total runtime: {time.time() - start:.1f}s")
