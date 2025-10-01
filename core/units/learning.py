import torch

from env import logger
from config_runtime import RF

from .constants import ENABLE_MINI_LEARN, FOLLOW_INPUT_DEVICE


class LearningMixin:
    def mini_learn(self, input_tensor, target_tensor, lr=0.001):
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)
        output = self.function(input_tensor)
        loss = torch.nn.functional.mse_loss(output, target_tensor)
        self.function.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.function.parameters():
                if param.grad is not None:
                    param.copy_(param - lr * param.grad)
        logger.debug(f"[Mini-Learn] {self.id} loss={loss.item():.4f} (lr={lr})")

    def compute_self_reward(self, input_tensor, output_tensor):
        if input_tensor.shape != output_tensor.shape:
            output_tensor = output_tensor[:, : input_tensor.shape[1]]
        error = torch.mean((input_tensor - output_tensor) ** 2)
        reward = 0.01 * (self.input_size / 50) * (1.0 - error.item())
        return max(reward, 0.0)

    def update(self, input_tensor: torch.Tensor):
        from contextlib import nullcontext

        input_tensor = input_tensor.squeeze()
        if not FOLLOW_INPUT_DEVICE:
            if self.function[0].weight.device != input_tensor.device:
                self.to(input_tensor.device)
        if input_tensor.dim() == 3 and input_tensor.size(1) == 1:
            input_tensor = input_tensor.squeeze(1)
        if self.call_history:
            self.avg_recent_calls = sum(self.call_history) / len(self.call_history)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() == 3 and input_tensor.size(1) == 1:
            input_tensor = input_tensor.squeeze(1)
        if hasattr(self, "goal_vec") and self.goal_vec is not None:
            gv = self.goal_vec
            if gv.dim() == 2:
                gv = gv.reshape(1, -1)
            elif gv.dim() == 1:
                gv = gv.unsqueeze(0)
            elif gv.dim() == 3 and gv.shape[1] == 1:
                gv = gv.squeeze(1)
            else:
                raise RuntimeError(f"[goal_vec 异常] 当前 shape={gv.shape}")
            if input_tensor.dim() == 3 and input_tensor.size(1) == 1:
                input_tensor = input_tensor.squeeze(1)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = torch.cat([input_tensor, gv], dim=-1)
        D = input_tensor.shape[-1]
        if D < self.input_size:
            pad = (0, self.input_size - D)
            input_tensor = torch.nn.functional.pad(input_tensor, pad)
        elif D > self.input_size:
            input_tensor = input_tensor[..., : self.input_size]
        step = getattr(self, "current_step", 0)
        self.memory_limit = 5 + (step // 500) * 5
        global_step = getattr(self, "current_step", 0)
        self.memory_pool_limit = min(50 + (global_step // 500) * 30, 1000)
        if FOLLOW_INPUT_DEVICE:
            if self.function[0].weight.device != input_tensor.device:
                self.to(input_tensor.device)
        current_input_size = input_tensor.shape[-1]
        if current_input_size > self.input_size:
            old_l1, old_l2 = self.function[0], self.function[2]
            new_layer1 = torch.nn.Linear(current_input_size, self.hidden_size).to(old_l1.weight.device)
            new_layer2 = torch.nn.Linear(self.hidden_size, current_input_size).to(old_l2.weight.device)
            with torch.no_grad():
                new_layer1.weight[:, : old_l1.weight.shape[1]].copy_(old_l1.weight)
                new_layer1.bias.copy_(old_l1.bias)
                new_layer2.weight[:, : old_l2.weight.shape[1]].copy_(old_l2.weight)
                new_layer2.bias.copy_(old_l2.bias)
            self.function[0] = new_layer1
            self.function[2] = new_layer2
            self.input_size = current_input_size
            self.last_output = torch.zeros((1, self.input_size), device=self.device)
        elif current_input_size < self.input_size:
            pad = (0, self.input_size - current_input_size)
            input_tensor = torch.nn.functional.pad(input_tensor, pad)
        use_grad = ENABLE_MINI_LEARN
        ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if (RF.use_fp16 and self.device.type == "cuda")
            else nullcontext()
        )
        with ctx, torch.inference_mode():
            raw_output = self.function(input_tensor)
        self.last_output = raw_output.detach().clone()
        self.state = self.last_output.clone()
        if self.output_history_tensor.shape[1] != self.last_output.shape[0]:
            self.output_history_tensor = torch.zeros((5, self.last_output.shape[0]), device="cpu")
            self.output_history_ptr = 0
        out = self.last_output.detach().cpu().view(-1)
        if out.shape[0] != self.output_history_tensor.shape[1]:
            self.output_history_tensor = torch.zeros((5, out.shape[0]), device="cpu")
            self.output_history_ptr = 0
        self.output_history_tensor[self.output_history_ptr] = out
        self.output_history_ptr = (self.output_history_ptr + 1) % self.output_history_tensor.shape[0]
        self.age += 1
        self.state_memory.append(self.state.detach().cpu())
        if len(self.state_memory) > self.memory_limit:
            self.state_memory.pop(0)
        input_var = float(input_tensor.var())
        recent_call_freq = getattr(self, "recent_calls", 1)
        connection_count = getattr(self, "connection_count", 1)
        avg_recent_calls = getattr(self, "avg_recent_calls", 0.0)
        if avg_recent_calls >= 4.0 and self.energy > 0.0:
            self.energy += 0.05
            logger.debug(f"[奖励] {self.id} 平均调用频率 {avg_recent_calls:.2f} → 能量 +0.04")
        if hasattr(self, "current_step"):
            if self.get_role() == "emitter" and self.current_step < 20:
                noise = torch.randn_like(self.last_output) * 0.2
                self.last_output += noise
                logger.debug(f"[扰动] emitter {self.id} 输出加入扰动")
            elif self.get_role() == "processor" and self.current_step < 5:
                noise = torch.randn_like(self.last_output) * 0.1
                self.last_output += noise
                logger.debug(f"[扰动] processor {self.id} 输出加入扰动")
        self_reward = self.compute_self_reward(input_tensor, self.last_output) * 0.03
        self.energy += self_reward
        if self_reward > 0:
            logger.debug(
                f"[内部奖励] {self.id} 自评奖励 +{self_reward:.4f} 能量 (现有能量 {self.energy:.2f})"
            )
        if self.role == "emitter" and hasattr(self, "goal_vec"):
            out_vec = self.last_output.view(-1)
            idx = torch.argmax(out_vec).item()
            x, y = idx % self.env_size, idx // self.env_size
            hazard = getattr(self, "current_hazard_xy", None)
            if hazard is None:
                hx, hy = None, None
            else:
                hx, hy = hazard
            if hx is not None and (x, y) == (hx, hy):
                self.is_hazard_confirmed = True
                self.goal_vec.zero_()
            else:
                self.is_hazard_confirmed = False
        if self.get_role() == "emitter":
            bias = self.gene.get("emitter_bias", 1.0)
            lr = 0.001 * (2.0 - min(1.5, bias))
            if ENABLE_MINI_LEARN:
                self.mini_learn(input_tensor, self.last_output.detach(), lr=lr)
        else:
            bias = (
                self.gene.get("processor_bias", 1.0)
                if self.role == "processor"
                else self.gene.get("sensor_bias", 1.0)
            )
            lr = 0.001 * (2.0 - min(1.5, bias))
            if ENABLE_MINI_LEARN:
                self.mini_learn(input_tensor, input_tensor, lr=lr)
