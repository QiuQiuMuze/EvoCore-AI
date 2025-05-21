# goal_generator.py
import random
import torch

def sample_unvisited(env_size, visit_counts):
    """
    从所有格子中挑一个访问次数最少（==0）或者随机的格子
    visit_counts: dict[(x,y)] -> count
    """
    # 首先收集还没访问过的
    unvisited = [(x,y) for x in range(env_size) for y in range(env_size)
                 if visit_counts.get((x,y),0)==0]
    if unvisited:
        return random.choice(unvisited)
    # 全部都访问过，则按最少访问次数挑一个
    min_cnt = min(visit_counts.values())
    rare = [pos for pos,c in visit_counts.items() if c==min_cnt]
    return random.choice(rare)

def make_onehot(env_size, pos, device):
    """
    pos: (x,y) 元组
    返回 shape=(env²,) 的 one-hot tensor
    """
    idx = pos[1]*env_size + pos[0]
    vec = torch.zeros(env_size*env_size, device=device)
    vec[idx] = 1.0
    return vec
