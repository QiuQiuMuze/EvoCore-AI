# self_model.py
import torch

def build_self_model(unit) -> torch.Tensor:
    """
    返回一个 6 维向量：
      [is_sensor, is_processor, is_emitter, recent_success_rate, avg_energy, num_connections]
    """
    # 一．角色 one-hot
    role = unit.get_role()
    is_sensor    = 1.0 if role == "sensor"    else 0.0
    is_processor = 1.0 if role == "processor" else 0.0
    is_emitter   = 1.0 if role == "emitter"   else 0.0

    # 二．近期成功率（你可以根据自己的“成就记录”改成合适的计算）
    #    这里假设你在 CogUnit 或 Graph 里维护了 unit.attempts, unit.successes
    attempts = getattr(unit, "attempts", 1.0)
    successes = getattr(unit, "successes", 0.0)
    recent_success_rate = successes / attempts

    # 三．平均能量
    avg_energy = float(getattr(unit, "energy", 0.0))

    # 四．连接数（下游出边数）
    num_connections = float(getattr(unit, "connection_count", 0.0))

    vec = torch.tensor(
        [is_sensor, is_processor, is_emitter,
         recent_success_rate, avg_energy, num_connections],
        dtype=torch.float32,
        device=unit.state.device  # 保持和 state 同设备
    )
    return vec
