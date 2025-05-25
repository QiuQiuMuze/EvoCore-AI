# goal_generator.py
import random
import torch

def sample_unvisited(env_size, visit_counts, exclude=None):
    """
    从所有格子中挑一个访问次数最少（==0）或者随机的格子
    visit_counts: dict[(x,y)] -> count
    exclude: set[(x,y)] -> 被排除的目标点（可选）
    """
    exclude = exclude or set()

    # 首先收集还没访问过的、并且不在 exclude 内的
    unvisited = [
        (x, y)
        for x in range(env_size)
        for y in range(env_size)
        if visit_counts.get((x, y), 0) == 0 and (x, y) not in exclude
    ]
    if unvisited:
        return random.choice(unvisited)

    # 全部都访问过，则按最少访问次数挑一个（也排除 exclude）
    min_cnt = min(visit_counts.values())
    rare = [pos for pos, c in visit_counts.items() if c == min_cnt and pos not in exclude]
    if rare:
        return random.choice(rare)

    # 如果 exclude 太多导致无法选，兜底随便选一个
    return (random.randint(0, env_size - 1), random.randint(0, env_size - 1))


def make_onehot(env_size, pos, device):
    """
    pos: (x,y) 元组
    返回 shape=(env²,) 的 one-hot tensor
    """
    idx = pos[1]*env_size + pos[0]
    vec = torch.zeros(env_size*env_size, device=device)
    vec[idx] = 1.0
    return vec
