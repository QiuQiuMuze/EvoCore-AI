#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_self_driven.py
====================
启动自驱动强化学习训练：
    GridEnvironment  ←→  CogGraph  ←→  RLAgent (TransformerPolicyNetwork)

- env.step(action) 返回 (next_state, reward, done, info)，在训练循环中直接使用返回值并在 done=True 时提前结束回合。
- CogGraph 需要你补充 sensor_forward / processor_forward / emitter_forward。
  若暂未实现，则脚本会 fallback：直接把环境状态张量当作 sensor / processor 输出。

❖ 运行示例
$ python train_self_driven.py --episodes 5000 --max-steps 256
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque

import torch

from env import GridEnvironment
from coggraph import CogGraph          # 需确保已实现 forward 三接口
from agents.rl_agent import RLAgent


logger = logging.getLogger(__name__)


LOG_INTERVAL = 20


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


def _energy_snapshot(graph: CogGraph) -> tuple[float, float, float, int]:
    """返回 (总能量+能量池, 细胞总能, 能量池, 单元数量)。"""
    total_cells = graph.total_energy() if hasattr(graph, "total_energy") else 0.0
    pool = getattr(graph, "energy_pool", 0.0)
    units = len(getattr(graph, "units", []))
    return total_cells + pool, total_cells, pool, units


# -------------------------------------------------------------------------- #
#                              训练主函数                                    #
# -------------------------------------------------------------------------- #
def main(cfg):
    device = torch.device(cfg.device)

    # 1) 初始化环境 & 图
    env = GridEnvironment(size=5)
    graph = CogGraph(device=cfg.device)  # 若需传入参数，请自行修改

    # ---- 新增 ----
    import CogUnit
    CogUnit.MAX_OUTPUT_DIM = graph.processor_hidden_size
    # --------------------

    # 2) 动态推断输入维度后再建 Agent
    init_state = torch.as_tensor(env.get_state(), dtype=torch.float32)
    input_dim = _infer_input_dim(graph, init_state)
    agent = RLAgent(
        input_dim=input_dim,
        num_actions=4,                   # GridEnvironment 定义了 4 个动作
        lr=cfg.lr,
        gamma=cfg.gamma,
        device=device
    )

    logger.info(f"[Init] transformer input_dim = {input_dim}, device = {device}")

    # 3) 训练循环
    reward_window: deque[float] = deque(maxlen=100)
    energy_window: deque[float] = deque(maxlen=100)
    composite_window: deque[float] = deque(maxlen=100)
    use_graph_pipeline = all(
        hasattr(graph, attr) and callable(getattr(graph, attr))
        for attr in ("sensor_forward", "processor_forward", "emitter_forward")
    )
    sensor_forward = getattr(graph, "sensor_forward", None)
    processor_forward = getattr(graph, "processor_forward", None)
    emitter_forward = getattr(graph, "emitter_forward", None)

    for ep in range(1, cfg.episodes + 1):

        state_np = env.reset()
        # 如 graph 有 reset_state() 请调用；否则新建实例或跳过
        if hasattr(graph, "reset_state"):
            graph.reset_state()

        state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
        ep_reward = 0.0
        episode_steps = 0
        energy_start, *_ = _energy_snapshot(graph)

        for t in range(cfg.max_steps):

            # --- CogGraph 前向 ---
            if use_graph_pipeline:
                with torch.no_grad():
                    sensor_out = sensor_forward(state)
                    processor_out = processor_forward(sensor_out)
                    emitter_forward(processor_out)
            else:
                # fallback：直接用 env 状态当作序列输入
                sensor_out = processor_out = state

            # --- 构造 transformer 输入 ---
            if isinstance(sensor_out, torch.Tensor):
                sensor_tensor = sensor_out.to(device)
            else:
                sensor_tensor = torch.as_tensor(sensor_out, dtype=torch.float32, device=device)

            if isinstance(processor_out, torch.Tensor):
                processor_tensor = processor_out.to(device)
            else:
                processor_tensor = torch.as_tensor(processor_out, dtype=torch.float32, device=device)

            state_seq = torch.stack((sensor_tensor, processor_tensor), dim=0)  # (seq_len=2, dim)
            state_seq = state_seq.unsqueeze(0)                                 # (1, 2, dim)

            # --- 选动作 & 环境交互 ---
            action = agent.select_action(state_seq)
            next_state_np, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            ep_reward += reward
            episode_steps = t + 1

            # 更新下一状态
            state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device)

            if done:
                break

        # --- Episode 结束：策略更新 ---
        agent.finish_episode()
        energy_end, total_cell_energy, _, unit_count = _energy_snapshot(graph)
        energy_delta = energy_end - energy_start
        composite_score = ep_reward + energy_delta

        reward_window.append(ep_reward)
        energy_window.append(energy_delta)
        composite_window.append(composite_score)

        # --- 日志 & Checkpoint ---
        if ep % LOG_INTERVAL == 0 or ep == 1:
            avg_reward = sum(reward_window) / len(reward_window)
            avg_energy = sum(energy_window) / len(energy_window)
            avg_composite = sum(composite_window) / len(composite_window)
            per_unit_energy = total_cell_energy / max(unit_count, 1)
            logger.info(
                f"[Ep {ep:>5}] steps={episode_steps:>3} reward={ep_reward:+.4f} "
                f"energyΔ={energy_delta:+.4f} combined={composite_score:+.4f} | "
                f"avg_reward(100)={avg_reward:+.4f} avg_energyΔ(100)={avg_energy:+.4f} "
                f"avg_combined(100)={avg_composite:+.4f} mean_unit_energy={per_unit_energy:.3f}"
            )

        if cfg.save_every and ep % cfg.save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/agent_ep{ep}.pth"
            agent.save(ckpt_path)
            logger.info(f"[Save] {ckpt_path} saved")

    # 全程训练完成后再存一份最终模型
    os.makedirs("checkpoints", exist_ok=True)
    agent.save("checkpoints/agent_final.pth")
    logger.info("Training finished ✓")


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = get_cfg()
    t0 = time.time()
    main(cfg)
    logger.info(f"Total runtime: {time.time() - t0:.1f} s")

"""
关键点说明
位置	说明
input_dim 自动推断	通过 graph.sensor_forward() 探测输出维度；若接口未实现，则退化为环境 state 大小。
Episode 终止	使用 env.step 返回的 done 信号提前截断，否则最多执行 --max-steps 步。
奖励计算	直接采用 env.step 返回的 reward。
断点续训	--save-every 控制周期性保存，文件包含网络参数 + 优化器状态。

下一步

确认 CogGraph 已实现 sensor_forward / processor_forward / emitter_forward。

若想引入 early-stop（如达到目标点）、或更复杂的奖励，可在 env 内部扩展 done 与 info 返回值，再更新脚本对应部分。
"""