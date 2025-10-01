"""Constants used by the immune cognition graph implementation."""

HIT_THRESH = 0.15          # 越大越宽松
MAX_CONNECTIONS = 4  # 每个单元最多连接 4 个下游
N_GOAL_CHANNELS = 3
FLASH_ATTN_AVAILABLE = False
MIN_PATROL_DIST = 3      # 巡逻目标与当前位置的最小曼哈顿距离
HIT_BONUS       = 1.0     # 命中立刻奖励
MISS_PENALTY    = 0.00    # 打空扣分
LEARNED_FACTOR  = 0.3     # learned assignment reward bias
