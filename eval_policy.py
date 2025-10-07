#!/usr/bin/env python3
"""
eval_policy.py

Continuous evaluation script for trained RL agents on the GridEnvironment.

Usage:
    python eval_policy.py \
        --ckpt <checkpoint_path> \
        --episodes <num_segments> \
        --max-steps <steps_per_segment> \
        --device <cpu|cuda> [--seed <random_seed>]

Example:
    python eval_policy.py \
        --ckpt checkpoints/agent_final.pth \
        --episodes 20 \
        --max-steps 500 \
        --device cpu \
        --seed 42

Metrics reported:
总步数（Total Steps）：环境中实际执行的总步数，等于 episodes × max_steps

平均每步奖励（Avg Reward/Step）：每一步的平均净能量收益（即 energy_gain - energy_penalty），反映整体奖励效率

资源采集率（Resources/Step）：在所有步骤中，采集到资源的步数占比，表示单位步的资源获取频率

状态覆盖率（State Coverage）：智能体曾经访问过的不同格子数，占整个地图格子总数的比例，衡量探索范围
"""
import argparse
import time
import random
import statistics
from collections import deque

import torch
from env import GridEnvironment
from coggraph import CogGraph, INPUT_CHANNELS
from agents.rl_agent import RLAgent


def pad_or_trunc(state_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Flatten a state tensor and pad or truncate to target length"""
    flat = state_tensor.view(-1)
    cur = flat.numel()
    if cur < target_len:
        pad = flat.new_zeros(target_len - cur)
        return torch.cat([flat, pad], dim=0)
    return flat[:target_len]


def build_feature(raw_state: torch.Tensor, graph: CogGraph, raw_dim: int, device: str, saved_input_dim: int):
    """
    Construct the input feature by running sensor->processor. 目标向量不再透明提供，
    因此直接依赖传感器/处理器输出即可。Truncate 或 pad 到 saved_input_dim。
    """
    flat = pad_or_trunc(raw_state, raw_dim)
    inp = flat.float().view(1, -1).to(device)
    s_out = graph.sensor_forward(inp)
    p_out = graph.processor_forward(s_out)
    if p_out.dim() > 1:
        p_out = p_out.squeeze(0)

    feat = p_out
    if feat.numel() < saved_input_dim:
        feat = torch.nn.functional.pad(feat, (0, saved_input_dim - feat.numel()))
    else:
        feat = feat[:saved_input_dim]
    return feat


def evaluate(ckpt_path: str, episodes: int, max_steps: int, device: str, seed: int):
    # 0) load model checkpoint
    chkpt = torch.load(ckpt_path, map_location=device)
    saved_input_dim = chkpt["policy_state_dict"]["input_proj.weight"].shape[1]
    saved_d_model   = chkpt["policy_state_dict"]["input_proj.weight"].shape[0]

    # infer environment size and channel count
    def infer_env(inp_dim, max_s=40):
        cands = [(s, inp_dim // (s*s)) for s in range(4, max_s+1) if inp_dim % (s*s)==0]
        if not cands:
            raise ValueError(f"Cannot infer env from input_dim={inp_dim}")
        ideal = (inp_dim / INPUT_CHANNELS)**0.5
        return min(cands, key=lambda x: abs(x[0]-ideal))

    env_size, ckpt_ch = infer_env(saved_input_dim)
    raw_dim = env_size * env_size * ckpt_ch
    if ckpt_ch != INPUT_CHANNELS:
        print(f"⚠️ ckpt uses {ckpt_ch} channels, code defines {INPUT_CHANNELS}; pad/trunc accordingly.")

    # set random seeds
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # initialize environment and graph
    env = GridEnvironment(size=env_size, max_steps=max_steps)
    env.reset()
    graph = CogGraph(None, device=device)
    graph.env = env
    graph.env_size = env_size
    graph.processor_hidden_size = saved_input_dim
    graph.upscale_old_units(raw_dim)
    graph._target_buf = graph._target_buf.new_zeros((2, env_size*env_size))
    graph.target_vector = graph._target_buf.clone()
    graph.reset_state()

    # initialize agent and disable exploration
    agent = RLAgent(input_dim=saved_input_dim,
                    num_actions=env.action_space_n,
                    d_model=saved_d_model,
                    device=device)
    agent.load(ckpt_path, map_location=device)
    agent.policy_net.eval()
    agent.use_epsilon = False

    # prepare history buffer
    N_HIST = 4
    history = deque(maxlen=N_HIST)
    initial_raw = env.get_state()
    feat0 = build_feature(initial_raw, graph, raw_dim, device, saved_input_dim)
    for _ in range(N_HIST):
        history.append(feat0)

    total_steps = episodes * max_steps
    total_reward = 0.0
    collected = 0
    visited = set()
    step_count = 0

    # run continuous segments without reset
    for ep in range(episodes):
        for _ in range(max_steps):
            # build state sequence for agent
            seq = []
            maxlen = max(t.numel() for t in history)
            for t in history:
                if t.numel() < maxlen:
                    t = torch.nn.functional.pad(t, (0, maxlen - t.numel()))
                seq.append(t)
            state_seq = torch.stack(seq, dim=0).unsqueeze(0).to(device)

            action = agent.select_action(state_seq)

            # step graph to update target_vector
            raw_state = env.get_state()
            graph.step(pad_or_trunc(raw_state, raw_dim).float().view(1, -1).to(device))

            # execute action in environment
            next_raw, _, _, _ = env.step(action, cog_step=graph.current_step)
            r = env.agent_energy_gain - env.agent_energy_penalty
            total_reward += r
            if env.agent_energy_gain > 0:
                collected += 1
            visited.add(tuple(env.agent_pos))

            # update history
            feat = build_feature(next_raw, graph, raw_dim, device, saved_input_dim)
            history.append(feat)

            step_count += 1

        # clear agent buffers but do not reset env or graph state
        agent.log_probs.clear()
        agent.saved_states.clear()
        agent.saved_logits.clear()

    # calculate and print metrics
    avg_reward_step = total_reward / step_count
    resources_per_step = collected / step_count
    coverage = len(visited) / (env_size * env_size)

    print("✔ Continuous Evaluation Summary:")
    print(f"  Total Steps        : {step_count}")
    print(f"  Avg Reward/Step    : {avg_reward_step:.4f}")
    print(f"  Resources/Step     : {resources_per_step:.4f}")
    print(f"  State Coverage     : {coverage*100:.1f}%")
    return avg_reward_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/agent_final.pth')
    parser.add_argument('--episodes', type=int, default=10,
                        help='number of truncation segments')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='steps per segment')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    start_time = time.time()
    evaluate(args.ckpt, args.episodes, args.max_steps, args.device, args.seed)
    print(f"Elapsed Time: {time.time() - start_time:.1f}s")
