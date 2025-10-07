"""Constants and thresholds used by :mod:`CogUnit` logic."""
from __future__ import annotations

ENABLE_MINI_LEARN = False  # ← 关闭自编码训练
FOLLOW_INPUT_DEVICE = True  # ← 自动把内部张量跟随输入 device（GPU/CPU）
MAX_OUTPUT_DIM = None       # ← 若设为 int，则 get_output() 强截断

# === Split-Gate 动态阈值表===========================
# 比例 k_es：Emitter <-> Sensor   ／  k_p： 相对 Processor/2
SPLIT_HI_ES_TABLE = {50: 1.30, 200: 1.20, 500: 1.12, float("inf"): 1.05}
SPLIT_HI_P_TABLE = {50: 1.20, 200: 1.15, 500: 1.08, float("inf"): 1.03}

TOL_FRAC_SPLIT = 0.05  # 至少差值 Δ≥ceil(total×5 %) （且 ≥1）

# ── 角色分裂最低能量阈值 以及 最低调用频率 ────────────
ROLE_SPLIT_RULE = {
    "sensor": {"min_e": 0.9, "min_calls": 0},
    "processor": {"min_e": 0.9, "min_calls": 1},
    "emitter": {"min_e": 0.6, "min_calls": 0},
}


def get_hi(table: dict[float, float], total: int) -> float:
    """按照总细胞数返回当前阶段的 hi 阈值."""
    for lim, val in table.items():
        if total < lim:
            return val
    return table[float("inf")]
