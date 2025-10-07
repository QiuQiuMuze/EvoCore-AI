"""Constants shared across coggraph modules."""
from __future__ import annotations

HIT_THRESH = 0.15
MAX_CONNECTIONS = 4
N_STATE_CHANNELS = 4
N_GOAL_CHANNELS = 3
INPUT_CHANNELS = N_STATE_CHANNELS + N_GOAL_CHANNELS

IDEAL_RATIO = {"emitter": 1, "processor": 2, "sensor": 1}
DENOM = sum(IDEAL_RATIO.values())
MAX_CONV_FRAC = 0.6
TOL_FRAC = 0.05
