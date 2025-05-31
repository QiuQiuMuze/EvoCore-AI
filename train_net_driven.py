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
from collections import Counter
import torch
import random
from collections import deque

from env_net import GridSecurityEnv
from ImmuneCogGraph import ImmuneCogGraph
from agents.rl_agent import RLAgent
from contextlib import nullcontext

# tools.py 或直接放在 train_net_driven.py 顶部
def can_compile(device: torch.device) -> bool:
    return hasattr(torch, "compile") and device.type == "cuda"

torch.set_float32_matmul_precision("high")  # 让 matmul 用高精度实现

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

def get_virus_type_stats(attacks: dict) -> str:
    """
    统计每类病毒的当前活跃数量
    """
    virus_counter = Counter()
    for (_, _), info in attacks.items():
        vtype = info.get("type", "unknown")
        virus_counter[vtype] += 1
    return ", ".join(f"{k}:{v}" for k, v in virus_counter.items())

def summarize_kills_by_type(kill_stats_by_type: dict) -> str:
    """
    将 {"worm": {"self_direct": 3, "guided": 1}, ...} → "worm:4, trojan:1, ..."
    """
    return ", ".join(
        f"{k}:{v['self_direct'] + v['guided']}"
        for k, v in kill_stats_by_type.items()
        if v["self_direct"] + v["guided"] > 0
    ) or "None"

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

    if can_compile(device):  # 仅在 CUDA + PyTorch>=2 时启用
        graph = torch.compile(graph)

    episode_rewards = []

    # ----------- 选 autocast 上下文 ----------- #
    from contextlib import nullcontext
    if device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

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
            if step == 1:  # 本 Episode 第一次进入循环
                virus_cleared_roll = 0  # 1000 步累计清除病毒
                hack_cleared_roll = 0  # 1000 步累计清除黑客
                last_reset_step = global_step
            global_step += 1
            # 环境当前状态张量
            with (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                  if device.type == "cuda" else nullcontext()):
                state_tensor = env.get_state_tensor().view(1, -1)  # (1, C×H×W)
                graph.step(state_tensor)

            # 当前状态
            # --- 当前活跃威胁 ---
            curr_inf_total = env.infected_map.sum().item()  # 仍然用感染格计数
            curr_hack_total = (env.privilege_level > 0.05).sum().item()  # <— 只统计仍在提权的格子

            delta_inf_cleared = max(0, prev_inf_total - curr_inf_total)
            delta_hack_cleared = max(0, prev_hack_total - curr_hack_total)

            # 累加真正击杀病毒的数量（按类型统计总和）
            true_virus_kills = sum(
                v["self_direct"] + v["guided"]
                for v in graph.virus_kill_stats_by_type.values()
            )

            virus_cleared_roll = true_virus_kills
            hack_cleared_roll = sum(
                v["self_direct"] + v["guided"]
                for v in graph.hack_kill_stats_by_type.values()
            )

            # 1000 步清零一次
            if global_step - last_reset_step >= 1000:
                virus_cleared_roll = 0
                hack_cleared_roll = 0
                last_reset_step = global_step
            # <<< PATCH-END

            # 更新历史记录
            prev_inf_total = curr_inf_total
            prev_hack_total = curr_hack_total

            # 计算新感染数 & 新黑客事件数
            curr_inf  = env.infected_map.sum().item()
            curr_hack = env.hack_history.sum().item()

            delta_inf = curr_inf_total - prev_inf_total
            delta_hack = curr_hack_total - prev_hack_total

            if delta_inf < 0:
                delta_inf *= 2

            new_infections = max(0, delta_inf)  # Δinf >0 才算新增
            reward = (
                    delta_inf_cleared * 1.0  # +1 / 成功清除
                    - new_infections * 1.0  # -1 / 新增感染
                    - delta_hack * cfg.hack_penalty  # 黑客惩罚保持不变
            )
            # --- 统计黑客 ---
            hack_stats = env.get_hack_stats()
            hack_msg = ", ".join(f"{k}:{v}" for k, v in hack_stats['per_type'].items())

            # 从 graph 拿到分类型统计

            hack_kill_msg = summarize_kills_by_type(graph.hack_kill_stats_by_type)
            virus_kill_msg = summarize_kills_by_type(graph.virus_kill_stats_by_type)

            total_reward += reward
            # policy_update(state, flat_action_index, reward)
            virus_msg = get_virus_type_stats(env.attacks)

            if step % 50 == 0:
                print(
                    f"[Step {global_step} | Ep {ep} Step {step}]\n"
                    f"感染点数 = {curr_inf_total:.0f}，黑客点数 = {curr_hack_total:.0f}\n"
                    f"病毒类型统计 [{virus_msg}]\n"
                    f"黑客类型统计 [{hack_msg}]\n"
                    f"累计清除病毒 = {virus_cleared_roll:.0f}，累计清除黑客 = {hack_cleared_roll:.0f}\n"
                    f"消灭的病毒分类 = [{virus_kill_msg}]\n"
                    f"消灭的黑客分类 = [{hack_kill_msg}]\n"
                    f"step 奖励 = {reward:.2f}，总奖励 = {total_reward:.2f}\n"
                    f"权限总和 = {hack_stats['total_priv']:.1f}，威胁度 = {hack_stats['threat_score']:.2f}"
                )


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
