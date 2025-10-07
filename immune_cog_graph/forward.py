"""Forward mixin for ImmuneCogGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from collections import deque

from env import logger

if TYPE_CHECKING:
    from .core import ImmuneCogGraph

class ForwardMixin:
    def sensor_forward(self, flat_state: torch.Tensor):
        # 保证 [B, D_full]
        if flat_state.dim() == 1:
            flat_state = flat_state.unsqueeze(0)
        D_full = self._D_env + self._D_hack + self._D_goal
        L = flat_state.shape[1]
        if L < D_full:
            flat_state = F.pad(flat_state, (0, D_full - L))
        elif L > D_full:
            flat_state = flat_state[:, :D_full]

        # 直接调用原始 sensor_net，不再使用 torch.compile
        return self.sensor_net(flat_state)

    def processor_forward(self, sensor_out: torch.Tensor):
        if sensor_out.dim() == 1:
            sensor_out = sensor_out.unsqueeze(0)

        # —— 共享网络模式 —— #
        if self.use_shared_unit_nets:
            if not hasattr(self, "_compiled_processor_net"):
                try:
                    self._compiled_processor_net = torch.compile(
                        self.processor_net, fullgraph=False, dynamic=True
                    )
                except:
                    self._compiled_processor_net = self.processor_net
            out = self._compiled_processor_net(sensor_out)

            # hotspot 合并
            if not hasattr(self, "_hotspot_queue"):
                self._hotspot_queue = deque(maxlen=20)
            for p in (u for u in self.units if u.role == "processor"):
                for pos in getattr(p, "hotspots", set()):
                    if pos not in self._hotspot_queue:
                        self._hotspot_queue.append(pos)
                p.hotspots = set()
            return out

        # —— 独立网络模式 —— #
        proc_units = [u for u in self.units if u.role == "processor"]
        outs = []
        for idx, u in enumerate(proc_units):
            inp = sensor_out[idx:idx+1]
            # 如果 u 没有自己的网络，就降级用全局的 self.processor_net
            net = u.processor_net if hasattr(u, "processor_net") else self.processor_net
            po  = net(inp)
            outs.append(po)
        proc_out = torch.cat(outs, dim=0) if outs else None

        # hotspot 合并
        if not hasattr(self, "_hotspot_queue"):
            self._hotspot_queue = deque(maxlen=20)
        for p in proc_units:
            for pos in getattr(p, "hotspots", set()):
                if pos not in self._hotspot_queue:
                    self._hotspot_queue.append(pos)
            p.hotspots = set()
        return proc_out

    def emitter_forward(self, proc_out: torch.Tensor):
        """
        支持「全局共享」或「各自网络」两种模式。
        现在 proc_out → emitter_net → [batch, 3*seq_len]。
        把 e.last_output 设置为长度 3*seq_len 的一维张量，供解码使用。
        """
        if proc_out.dim() == 1:
            proc_out = proc_out.unsqueeze(0)

        # 先对 vec 做对齐（pad/trunc）
        vec = proc_out[0]
        H_e = self.emitter_hidden_size
        if vec.shape[0] < H_e:
            vec = F.pad(vec, (0, H_e - vec.shape[0]))
        elif vec.shape[0] > H_e:
            vec = vec[:H_e]

        emitters = [u for u in self.units if u.role == "emitter"]

        # —— 共享 emitter_net —— #
        if self.use_shared_unit_nets:
            batch_in = []
            for e in emitters:
                self.expand_unit_dim(e, H_e)
                batch_in.append(vec.unsqueeze(0))
            if not batch_in:
                return None
            batch = torch.cat(batch_in, dim=0)  # [num_emitters, H_e]
            logits = self.emitter_net(batch)  # [num_emitters, 3*seq_len]
            for e, lg in zip(emitters, logits):
                e.last_output = lg.detach()  # 一维 [3*seq_len]
            return logits

        # —— 独立 emitter_net —— #
        logits_list = []
        for e in emitters:
            self.expand_unit_dim(e, H_e)
            net = e.emitter_net if hasattr(e, "emitter_net") else self.emitter_net
            lg = net(vec.unsqueeze(0))  # [1, 3*seq_len]
            e.last_output = lg.detach().squeeze(0)  # [3*seq_len]
            logits_list.append(lg)
        if not logits_list:
            return None

        logits = torch.cat(logits_list, dim=0)  # [num_emitters, 3*seq_len]
        return logits
