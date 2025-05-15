# eval_policy.py
import torch
from env import GridEnvironment
from coggraph import CogGraph
from agents.rl_agent import RLAgent
import argparse, statistics, time

def evaluate(ckpt_path: str, episodes: int = 100, max_steps: int = 256, device: str = "cpu"):
    # 1) 重建组件
    env   = GridEnvironment(size=5)
    graph = CogGraph(device=device)
    graph.debug = True
    input_dim = graph.processor_hidden_size
    agent = RLAgent(input_dim=input_dim, num_actions=env.action_space_n, device=device)
    agent.load(ckpt_path, map_location=device)

    rewards = []
    for ep in range(episodes):
        env.reset()
        if hasattr(graph, "reset_state"):
            graph.reset_state()
        state = torch.from_numpy(env.get_state()).float().to(device)
        ep_reward = 0.0
        for _ in range(max_steps):
            s_out = graph.sensor_forward(state)
            p_out = graph.processor_forward(s_out)
            graph.emitter_forward(p_out)
            state_seq = torch.stack([s_out, p_out], dim=0).unsqueeze(0).to(device)

            action = agent.select_action(state_seq)  # no grad needed, but keep default
            env.step(action)

            r = env.agent_energy_gain - env.agent_energy_penalty
            ep_reward += r
            state = torch.from_numpy(env.get_state()).float().to(device)
        rewards.append(ep_reward)

    print(f"✔ Evaluated {episodes} episodes  |  "
          f"mean={statistics.mean(rewards):.4f}  "
          f"std={statistics.stdev(rewards):.4f}")

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
