# per_memory.py
import numpy as np, torch
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_step=1, gamma=0.99):
        self.capacity   = capacity
        self.alpha      = alpha
        self.gamma      = gamma
        self.n_step     = n_step
        self.pos        = 0
        self.buffer     = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        # 用于临时存 n 步 transition
        from collections import deque
        self._n_buffer = deque(maxlen=n_step)

    def append(self, transition, priority):
        # 1) 先把新 transition 推入 n 步队列
        self._n_buffer.append(transition)
        # 2) 只有当队列满了 n_step，才合成一个 n-step transition
        if len(self._n_buffer) < self.n_step:
            return

        # 3) 计算 n-step 累积回报 R = sum_{i=0}^{n-1} γ^i r_i
        R = 0.0
        for idx, tr in enumerate(self._n_buffer):
            R += (self.gamma ** idx) * tr['reward']

        # 4) 用队首的 state/action，用队尾的 next_state/done
        first = self._n_buffer[0]
        last  = self._n_buffer[-1]
        n_step_tr = {
            'state':      first['state'],
            'action':     first['action'],
            'reward':     R,
            'next_state': last['next_state'],
            'done':       last['done'],
            'label': last.get('label', 0),
        }
        # 5) 真正存入主 buffer
        self._store(n_step_tr, priority)

    # —— 新增：支持 len(buffer) 和 buffer[idx] —— #
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def _store(self, transition, priority):
        transition["priority"] = priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity


    def sample(self, batch_size, beta=0.4):
        # 按 priority^α 采样
        N = len(self)
        if N == 0:
            # Buffer 为空时，不要去 np.random.choice(0,...)
            return [], [], torch.tensor([], dtype=torch.float32)
        probs = self.priorities[:N] ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # importance-sampling weights
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
