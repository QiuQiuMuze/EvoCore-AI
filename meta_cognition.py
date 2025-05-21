# meta_cognition.py

from collections import deque

class MetaCognition:
    def __init__(self, history_len=50):
        self.last_action = None
        self.reward_trace = deque(maxlen=history_len)
        self.success_count = 0
        self.failure_count = 0

    def record(self, action, reward, success_threshold=0.5):
        self.last_action = action
        self.reward_trace.append(reward)
        if reward >= success_threshold:
            self.success_count += 1
        else:
            self.failure_count += 1

    def recent_success_rate(self):
        total = len(self.reward_trace)
        if total == 0:
            return None
        return sum(1 for r in self.reward_trace if r >= 0.5) / total

    def needs_mutation(self, min_rate=0.3):
        rate = self.recent_success_rate()
        return rate is not None and rate < min_rate
