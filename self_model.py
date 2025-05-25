# self_model.py

import torch
import statistics
from typing import List

def build_self_model(unit) -> torch.Tensor:
    """
    返回一个 9 维向量：
      [is_sensor, is_processor, is_emitter,
       avg_reward, std_reward, reward_trend,
       avg_energy, num_connections, action_count]
    """

    # 1. 角色 one-hot
    role = unit.get_role()
    is_sensor    = 1.0 if role == "sensor"    else 0.0
    is_processor = 1.0 if role == "processor" else 0.0
    is_emitter   = 1.0 if role == "emitter"   else 0.0

    # 2. 最近 reward 轨迹特征（假设 unit.reward_trace 是一个 list）
    reward_trace: List[float] = getattr(unit, "reward_trace", [])
    if len(reward_trace) >= 1:
        window = reward_trace[-10:]  # 取最近 10 次
        avg_reward = float(statistics.mean(window))
        std_reward = float(statistics.pstdev(window))  # 总体标准差
    else:
        avg_reward = 0.0
        std_reward = 0.0

    # 3. reward 趋势：最后一次变化
    if len(reward_trace) >= 2:
        reward_trend = float(reward_trace[-1] - reward_trace[-2])
    else:
        reward_trend = 0.0

    # 4. 平均能量
    avg_energy = float(getattr(unit, "energy", 0.0))

    # 5. 下游连接数
    num_connections = float(getattr(unit, "connection_count", 0.0))

    # 6. 行为频次：单位时间内 action 调用次数
    action_count = float(getattr(unit, "recent_actions", 0.0))

    vec = torch.tensor(
        [
            is_sensor, is_processor, is_emitter,
            avg_reward, std_reward, reward_trend,
            avg_energy, num_connections, action_count
        ],
        dtype=torch.float32,
        device=unit.state.device
    )
    return vec
