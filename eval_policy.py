# eval_policy.py
import torch
from env import GridEnvironment
from coggraph import CogGraph
from agents.rl_agent import RLAgent
import argparse, statistics, time
from env import logger
import random
import numpy as np

def evaluate(ckpt_path: str, episodes: int = 100, max_steps: int = 256, device: str = "cpu", seed: int = None):
    # ——— 0) 加载 checkpoint，推断训练时的输入维 (input_dim) 和 Transformer 隐藏维 (d_model) ———
    checkpoint = torch.load(ckpt_path, map_location=device)
    saved_input_dim = checkpoint["policy_state_dict"]["input_proj.weight"].shape[1]
    saved_d_model = checkpoint["policy_state_dict"]["input_proj.weight"].shape[0]

    # ——— 1) 推断训练时的环境 size（env_size），重建环境 & 重写 graph 的尺寸参数 ———
    from coggraph import INPUT_CHANNELS, TaskInjector
    saved_env_size = int((saved_input_dim / INPUT_CHANNELS) ** 0.5)
    assert saved_env_size * saved_env_size * INPUT_CHANNELS == saved_input_dim, (
        f"Cannot infer env_size from input_dim={saved_input_dim}"
    )
    env = GridEnvironment(size=saved_env_size, max_steps=max_steps)

    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)
    # 1️⃣ 先创建并加载 Agent
    agent = RLAgent(
        input_dim=saved_input_dim,
        num_actions=env.action_space_n,
        d_model=saved_d_model,
        device=device
    )
    agent.load(ckpt_path, map_location=device)
    # —— 切到 eval 模式，关闭 ε-greedy ——
    agent.policy_net.eval()
    agent.use_epsilon = False
    # —— 确保缓存一开始为空 ——
    agent.log_probs.clear()
    agent.saved_states.clear()
    agent.saved_logits.clear()

    # —— 构造 CogGraph & 挂环境 ——
    graph = CogGraph(agent, device=device)

    # 强制还原训练时的输入维度
    graph.env_size = saved_env_size
    graph.processor_hidden_size = saved_input_dim
    graph.env = GridEnvironment(size=saved_env_size, max_steps=max_steps)
    # graph.task = TaskInjector(target_position=(saved_env_size - 1, saved_env_size - 1))
    # graph.target_vector = graph.task.encode_goal(saved_env_size)
    graph.upscale_old_units(saved_input_dim)

    graph.debug = True

    rewards, lengths = [], []
    per_step_rewards = []  # 每步回报列表
    coverages = []  # 每回合状态覆盖率
    resources_collected = []  # 每回合资源采集总数
    for ep in range(episodes):
        env.reset()
        if hasattr(graph, "reset_state"):
            graph.reset_state()
        state = torch.from_numpy(env.get_state()).float().to(device)
        ep_reward = 0.0
        ep_length = 0
        collected = 0
        visited = set()
        for _ in range(max_steps):
            s_out = graph.sensor_forward(state)  # (input_dim,)
            p_out = graph.processor_forward(s_out)  # (input_dim,)
            graph.emitter_forward(p_out)

            raw_seq = torch.stack([s_out, p_out], dim=0)  # (2, input_dim)
            state_seq = raw_seq.unsqueeze(0).to(device)  # (1, 2, input_dim)


            action = agent.select_action(state_seq)

            env.step(action, cog_step=graph.current_step)

            r = env.agent_energy_gain - env.agent_energy_penalty
            ep_reward += r
            ep_length += 1
            if env.agent_energy_gain > 0:
                collected += 1
            # 状态覆盖：记录 agent_pos
            visited.add(tuple(env.agent_pos))

            state = torch.from_numpy(env.get_state()).float().to(device)
        rewards.append(ep_reward)
        lengths.append(ep_length)
        per_step_rewards.append(ep_reward / ep_length)
        coverages.append(len(visited) / (env.size * env.size))
        resources_collected.append(collected)
        agent.log_probs.clear()
        agent.saved_states.clear()
        agent.saved_logits.clear()

    mean_r = statistics.mean(rewards)
    std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    avg_len = statistics.mean(lengths)
    if per_step_rewards:
        mean_per_step = statistics.mean(per_step_rewards)
    else:
        mean_per_step = 0.0

    mean_coverage = statistics.mean(coverages) if coverages else 0.0
    mean_collected = statistics.mean(resources_collected) if resources_collected else 0.0

    print("✔ Evaluation Summary:")
    print(f"  Episodes        : {episodes}")
    print(f"  Avg Return      : {mean_r:.4f}")
    print(f"  Return Std      : {std_r:.4f}")
    print(f"  Avg Return/Step      : {mean_per_step:.4f}")
    print(f"  Avg Ep Length        : {avg_len:.1f} steps")
    print(f"  State Coverage       : {mean_coverage * 100:.1f}%")
    print(f"  Avg Resources Collected: {mean_collected:.1f}")
    return mean_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/agent_ep2000.pth")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=4000) # 检测的最长步数
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fastflag", action="store_true",
                        help="关闭则禁用所有 RuntimeFlags 加速")
    args = parser.parse_args()
    import random, numpy as np, torch, statistics, time

    seeds = [100, 200, 300]
    all_means = []
    for seed in seeds:
        # —— 固定这一轮的随机种子 ——
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # （如果 env 支持 seed()，也要在这里调用一次）

        t0 = time.time()
        mean_r = evaluate(
            args.ckpt,
            episodes = args.episodes,
            max_steps = args.max_steps,
            device = args.device,
            seed = seed
        )
        elapsed = time.time() - t0
        print(f"Seed={seed}: Avg Return={mean_r:.4f}, Time={elapsed:.1f}s\n")
        all_means.append(mean_r)

    # 汇总多 seed 下的表现
    mean_of_means = statistics.mean(all_means)
    std_of_means  = statistics.stdev(all_means)
    print("===== Across seeds =====")
    print(f"Seeds: {seeds}")
    print(f"Mean of Avg Returns = {mean_of_means:.4f}")
    print(f"Std  of Avg Returns = {std_of_means:.4f}")

"""
python eval_policy.py \
  --ckpt checkpoints/agent_h4.pth \
  --episodes 1\
  --max-steps 1000\
  --device cpu
"""