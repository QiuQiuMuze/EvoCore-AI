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

    # key = (x,y) 坐标，value = global_step（第一次检测到此 hack 时的全局步数）
    hack_spawn_times: dict[tuple[int, int], int] = {}

    # 2) 构造一个 dummy RLAgent 传给 ImmuneCogGraph
    dummy_agent = RLAgent(
        input_dim=1,
        num_actions=1,
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
    if can_compile(device):
        graph = torch.compile(graph)

    episode_rewards = []

    # ------------- 训练循环 -------------
    for ep in range(1, cfg.episodes + 1):
        # 如果你希望每个 Episode 都干净开始，就在这里重置环境：
        # env.reset()

        # （可选）如果每集都要从零开始计 hack_spawn_times，就在此处清空
        hack_spawn_times.clear()

        total_reward = 0.0

        # 记录上一步环境中“已有多少感染 / 有多少活跃 hack”
        # （这些变量只用来给打印看，真正清除数下面通过 graph.kill_stats_by_type 计算）
        last_inf  = env.infected_map.sum().item()
        last_hack = (env.privilege_level > 0.04).sum().item()

        # 用于“每 1000 步将 累计清除计数 滚动归零”的辅助量
        last_reset_step = global_step

        # 设定“超时阈值”：只有当感染／黑客点在网格里连续存留超过下面这几步，才开始扣分
        VIRUS_TIMEOUT  = 10       # 感染格在网格里连续存在超过 10 步，才算“超时”
        HACK_TIMEOUT   = 8        # 黑客点在网格里连续存在超过 8 步，才算“超时”
        VIRUS_PENALTY  = 2      # 每个超时感染格每步扣 0.5 分
        HACK_PENALTY   = cfg.hack_penalty  # 黑客超时每步扣 cfg 里指定的分

        for step in range(1, cfg.max_steps + 1):
            if step == 1:
                virus_cleared_roll = 0
                hack_cleared_roll  = 0
                last_reset_step    = global_step

            global_step += 1

            # —— 1) 先记录“前一步”累计击杀数  —— #
            old_virus_kills = sum(
                v["self_direct"] + v["guided"]
                for v in graph.virus_kill_stats_by_type.values()
            )
            old_hack_kills = sum(
                v["self_direct"] + v["guided"]
                for v in graph.hack_kill_stats_by_type.values()
            )

            # —— 2) 让 ImmuneCogGraph 跑一步（内部会先 env.step() 再 emitter 动作） —— #
            with (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                  if device.type == "cuda" else nullcontext()):
                state_tensor = env.get_state_tensor().view(1, -1)
                graph.step(state_tensor)

            # —— 3) 紧接着记录“本步后”累计击杀数  —— #
            new_virus_kills = sum(
                v["self_direct"] + v["guided"]
                for v in graph.virus_kill_stats_by_type.values()
            )
            new_hack_kills = sum(
                v["self_direct"] + v["guided"]
                for v in graph.hack_kill_stats_by_type.values()
            )

            # 本步“真正清除”的病毒+黑客数量 = 差值
            cleared_virus = new_virus_kills - old_virus_kills
            cleared_hack  = new_hack_kills - old_hack_kills

            # —— 4) 更新 hack_spawn_times：记录“当前环境里哪些 hack 第一次出现”
            current_hacks = set(env.hacks.keys())
            for pos in current_hacks:
                if pos not in hack_spawn_times:
                    hack_spawn_times[pos] = global_step
            # 把已经消失（被清除）的 hack 从字典里删掉
            for pos in list(hack_spawn_times.keys()):
                if pos not in current_hacks:
                    del hack_spawn_times[pos]

            # —— 5) 读当前环境里的“活跃感染数”和“活跃 hack 数” —— #
            curr_inf  = env.infected_map.sum().item()
            curr_hack = len(current_hacks)  # 等价于 (env.privilege_level>0.05).sum().item()

            # —— 6) 计算“及时清除奖励”（只要本步清除了就 +1）
            reward = 0.0
            if cleared_virus > 0:
                reward += 1.0 * cleared_virus
            if cleared_hack > 0:
                reward += 1.0 * cleared_hack

            # —— 7) 计算“超时惩罚” —— #
            #   7.1) 病毒：若某个感染格子在网格里连续存留超过 VIRUS_TIMEOUT 步，就扣分
            overdue_virus_mask = (env.infected_duration_map > VIRUS_TIMEOUT)
            num_overdue_virus = int(overdue_virus_mask.sum().item())
            reward -= VIRUS_PENALTY * num_overdue_virus

            #   7.2) 黑客：若某条 hack 在 hack_spawn_times 里记录的时间超过 HACK_TIMEOUT，就扣分
            num_overdue_hack = 0
            for pos, spawn_step in hack_spawn_times.items():
                if global_step - spawn_step > HACK_TIMEOUT:
                    num_overdue_hack += 1
            reward -= HACK_PENALTY * num_overdue_hack

            total_reward += reward

            # —— 8) 更新“累计清除数”（纯打印用） —— #
            virus_cleared_roll = new_virus_kills
            hack_cleared_roll  = new_hack_kills

            # —— 9) 如果要“每 1000 步清一次”滚动统计，就写在这里 —— #
            if global_step - last_reset_step >= 1000:
                virus_cleared_roll = 0
                hack_cleared_roll  = 0
                last_reset_step    = global_step

            # —— 10) 打印状态 —— #
            hack_stats     = env.get_hack_stats()
            hack_msg       = ", ".join(f"{k}:{v}" for k, v in hack_stats['per_type'].items())
            virus_msg      = get_virus_type_stats(env.attacks)
            hack_kill_msg  = summarize_kills_by_type(graph.hack_kill_stats_by_type)
            virus_kill_msg = summarize_kills_by_type(graph.virus_kill_stats_by_type)

            if step % 50 == 0:
                print(
                    f"[Step {global_step} | Ep {ep} Step {step}]\n"
                    f"当前感染格 = {curr_inf:.0f}，当前活跃黑客 = {curr_hack:.0f}\n"
                    f"病毒类型统计 [{virus_msg}]\n"
                    f"黑客类型统计 [{hack_msg}]\n"
                    f"累计清除病毒 = {virus_cleared_roll:.0f}，累计清除黑客 = {hack_cleared_roll:.0f}\n"
                    f"消灭的病毒分类 = [{virus_kill_msg}]\n"
                    f"消灭的黑客分类 = [{hack_kill_msg}]\n"
                    f"step 奖励 = {reward:.2f}，总奖励 = {total_reward:.2f}\n"
                    f"权限总和 = {hack_stats['total_priv']:.1f}，威胁度 = {hack_stats['threat_score']:.2f}"
                )

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

"""
python train_net_driven.py --episodes 500 --max-steps 500 --device cpu
"""