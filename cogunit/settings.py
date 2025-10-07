"""Configuration flags and thresholds for CogUnit behaviors."""

ENABLE_MINI_LEARN = False  # ← 关闭自编码训练
FOLLOW_INPUT_DEVICE = True  # ← 自动把内部张量跟随输入 device（GPU/CPU）
MAX_OUTPUT_DIM = None       # ← 若设为 int，则 get_output() 强截断

SPLIT_HI_ES_TABLE = {50: 1.30, 200: 1.20, 500: 1.12, float("inf"): 1.05}
SPLIT_HI_P_TABLE = {50: 1.20, 200: 1.15, 500: 1.08, float("inf"): 1.03}
TOL_FRAC_SPLIT = 0.05

ROLE_SPLIT_RULE = {
    "sensor": {"min_e": 0.9, "min_calls": 0},
    "processor": {"min_e": 0.9, "min_calls": 1},
    "emitter": {"min_e": 0.6, "min_calls": 0},
}


def get_hi_threshold(table: dict[int | float, float], total: int) -> float:
    """按照总细胞数返回当前阶段的 hi 阈值"""
    for limit, value in table.items():
        if total < limit:
            return value
    return table[float("inf")]

__all__ = [
    "ENABLE_MINI_LEARN",
    "FOLLOW_INPUT_DEVICE",
    "MAX_OUTPUT_DIM",
    "SPLIT_HI_ES_TABLE",
    "SPLIT_HI_P_TABLE",
    "TOL_FRAC_SPLIT",
    "ROLE_SPLIT_RULE",
    "get_hi_threshold",
]
