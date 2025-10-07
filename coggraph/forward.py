"""Forward pass mixin for CogGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from config_runtime import RF
from env import logger

if TYPE_CHECKING:
    from .core import CogGraph

class ForwardMixin:
    def sensor_forward(self, env_state_np):
        """
        Args:
            env_state_np : np.ndarray 或 torch.Tensor (size=N)
        Returns:
            torch.Tensor (size = env_state_np.size) —— 作为 sensor 输出
        """
        dev = self.device  # ← 统一目标设备
        x = torch.as_tensor(env_state_np, dtype=torch.float32, device=dev).view(-1)

        # ① —— 保底：传进来多少，就以 graph.processor_hidden_size 为准 ——
        if any(u.input_size < self.processor_hidden_size for u in self.units if u.role == "sensor"):
            for s in (u for u in self.units if u.role == "sensor"):
                if s.input_size < self.processor_hidden_size:  # 只升不降
                    self.expand_unit_dim(s, self.processor_hidden_size)
        # —— 对齐到 processor_hidden_size ——
        D = x.numel()
        if D < self.processor_hidden_size:
            pad = (0, self.processor_hidden_size - D)
            x = torch.nn.functional.pad(x, pad)
        elif D > self.processor_hidden_size:
            x = x[: self.processor_hidden_size]

        sensors = [u for u in self.units if u.get_role() == "sensor"]
        if not sensors:
            return x            # 无 sensor 时直接返回

        if not RF.batch_sensor:  # ← 用 config 统一控制是否 batch 模式
            for s in sensors:
                s.update(x.unsqueeze(0))
            return torch.stack([s.last_output for s in sensors], dim=0).mean(dim=0)

        # ---- ⚡ 批量前向，仅做前向推理 ----
        # ① 把 N 份输入堆在一起
        batch_in = x.unsqueeze(0).repeat(len(sensors), 1)        # [N, D]

        # ② 用 **首个 sensor 的网络结构** 复制一个临时 net
        #    假设所有 sensor 都是相同的  Linear → ReLU → Linear
        net = sensors[0].function
        batch_out = net(batch_in)                                # [N, D]

        # ③ 分别写回每个 sensor 的 last_output（不调用 update()，
        #    省掉 split/代谢等逻辑；这些逻辑已经在 Graph.step() 外部显式调用）
        for s, o in zip(sensors, batch_out):
            s.last_output = o.detach()

        return batch_out.mean(dim=0)       # 仍返回合并输出

    def processor_forward(self, sensor_out):
        """
        Args:
            sensor_out : torch.Tensor 1-D
        Returns:
            torch.Tensor (size = self.processor_hidden_size)
        """
        """批量或逐个执行 processor.update()."""
        dev = self.device
        sensor_out = sensor_out.to(dev)                   # (1,D)

        # ① —— 保底：传进来多少，就以 graph.processor_hidden_size 为准 ——
        if any(u.input_size < self.processor_hidden_size for u in self.units if u.role == "processor"):
            for s in (u for u in self.units if u.role == "processor"):
                if s.input_size < self.processor_hidden_size:  # 只升不降
                    self.expand_unit_dim(s, self.processor_hidden_size)

        procs = [u for u in self.units if u.role == "processor"]
        if not procs:
            return sensor_out

        D = sensor_out.size(-1)
        if RF.batch_processor:
            # -------- 构造批输入 --------
            batch_in = sensor_out.expand(len(procs), -1)  # [N,D]

            # -------- 共享一张网络 --------
            net = procs[0].function

            ctx = (torch.autocast("cuda", dtype=torch.float16)
                   if (RF.use_fp16 and dev.type == "cuda") else nullcontext())
            with ctx, torch.inference_mode():
                batch_out = net(batch_in)                 # [N,D]

            # -------- 回写 last_output --------
            for u, o in zip(procs, batch_out):
                u.last_output = o.detach()
            merged = batch_out.mean(dim=0)
        else:
            # 回退：逐个 update
            outs = []
            for p in procs:
                p.update(sensor_out)
                outs.append(p.get_output().view(-1))
            merged = torch.stack(outs).mean(dim=0)

        # —— 与旧实现相同的 pad / truncate —— #
        if merged.numel() < self.processor_hidden_size:
            merged = torch.nn.functional.pad(
                merged, (0, self.processor_hidden_size - merged.numel()))
        else:
            merged = merged[: self.processor_hidden_size]
        return merged.to(dev)

    def emitter_forward(self, proc_out: torch.Tensor):
        """
        批量版：把 for-loop --> 单次 repeat + Linear，逻辑不变
        """
        # ---- 0) 统一输入形状 ----
        if proc_out.dim() == 1:
            proc_out = proc_out.unsqueeze(0)  # [1, H_p]

        # ---- 1) 对齐到 emitter_hidden_size ----
        vec = proc_out[0]
        H_e = self.emitter_hidden_size
        if vec.shape[0] < H_e:
            vec = F.pad(vec, (0, H_e - vec.shape[0]))
        elif vec.shape[0] > H_e:
            vec = vec[:H_e]

        # ---- 2) 收集 emitter 索引（一次性）----
        em_idx = [i for i, u in enumerate(self.units) if u.role == "emitter"]
        if not em_idx:  # 没有 emitter
            return None

        # 一次性把所有 emitter 线性层升维
        for i in em_idx:
            self.expand_unit_dim(self.units[i], H_e)

        # ---- 3) 单次 forward ----
        batch_in = vec.repeat(len(em_idx), 1)  # [N_emit, H_e]
        logits = self.emitter_net(batch_in)  # [N_emit, seq_len]

        # ---- 4) 写回各 emitter.last_output ----
        for idx, lg in zip(em_idx, logits):
            self.units[idx].last_output = lg.detach()

        return logits

    def collect_emitter_outputs(self):
        """收集所有 emitter 输出并自动对齐到目标维度"""
        aligned = []
        for unit in self.units:
            if unit.get_role() != "emitter":
                continue

            raw = unit.get_output().squeeze(0) if unit.get_output().dim() == 2 else unit.get_output()
            vec = self._align_to_goal_dim(raw)

            if vec.shape[-1] != self._goal_dim():
                # 理论不会发生，安全检查
                logger.warning(f"[警告] 对齐失败 {unit.id} 长度 {vec.shape[-1]}")
                continue

            aligned.append(vec.unsqueeze(0))

        if aligned:
            stacked = torch.cat(aligned, dim=0).to(self.device, non_blocking=True)      # [N, goal_dim]
            logger.debug(
                "[输出检查] Emitter 对齐后均值(前5) : %s",
                stacked.mean(dim=0)[:5]
            )

            return stacked
        else:
            logger.debug("[输出检查] 当前没有活跃的 emitter 单元")
            return None
