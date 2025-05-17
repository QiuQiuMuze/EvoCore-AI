# eval_policy.py
import torch
from env import GridEnvironment
from coggraph import CogGraph
from agents.rl_agent import RLAgent
import argparse, statistics, time
from env import logger

def evaluate(ckpt_path: str, episodes: int = 100, max_steps: int = 256, device: str = "cpu"):
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
    graph = CogGraph(device=device)
    graph.env_size = saved_env_size
    graph.env = env
    graph.processor_hidden_size = saved_input_dim
    # 如果后续逻辑需要 goal 向量，也可以同步重置：
    graph.task = TaskInjector(target_position=(saved_env_size - 1, saved_env_size - 1))
    graph.target_vector = graph.task.encode_goal(saved_env_size)
    graph.debug = True

    # ——— 2) 直接用 checkpoint 上的 input_dim & d_model 构造 Agent，并加载权重 ———
    agent = RLAgent(
        input_dim=saved_input_dim,
        num_actions=env.action_space_n,
        d_model=saved_d_model,
        device=device
    )
    agent.load(ckpt_path, map_location=device)

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
            s_out = graph.sensor_forward(state)
            p_out = graph.processor_forward(s_out)
            graph.emitter_forward(p_out)
            state_seq = torch.stack([s_out, p_out], dim=0).unsqueeze(0).to(device)

            action = agent.select_action(state_seq)  # no grad needed, but keep default
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/agent_ep2000.pth")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=4000) # 检测的最长步数
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    t0 = time.time()
    evaluate(args.ckpt,
            episodes = args.episodes,
            max_steps = args.max_steps,  # ← 传递
            device = args.device)
    print(f"⏱  finished in {time.time()-t0:.1f}s")
"""
python eval_policy.py \
  --ckpt checkpoints/agent_final.pth \
  --episodes 20 \
  --max-steps 1000 \
  --device cpu
"""