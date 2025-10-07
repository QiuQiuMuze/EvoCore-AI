"""Learning related helpers for :class:`CogUnit`."""
from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn as nn

from config_runtime import RF
from env import logger

from .constants import ENABLE_MINI_LEARN, FOLLOW_INPUT_DEVICE


def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def perform_mini_learn(unit, input_tensor, target_tensor, lr: float = 0.001) -> None:
    input_tensor = _ensure_batch(input_tensor)
    target_tensor = _ensure_batch(target_tensor)

    output = unit.function(input_tensor)
    loss = torch.nn.functional.mse_loss(output, target_tensor)

    unit.function.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in unit.function.parameters():
            if param.grad is not None:
                param.copy_(param - lr * param.grad)

    logger.debug(
        "[Mini-Learn] %s loss=%.4f (lr=%s)", unit.id, loss.item(), lr,
    )


def compute_self_reward(unit, input_tensor, output_tensor) -> float:
    if input_tensor.shape != output_tensor.shape:
        output_tensor = output_tensor[:, : input_tensor.shape[1]]
    error = torch.mean((input_tensor - output_tensor) ** 2)
    reward = 0.01 * (unit.input_size / 50) * (1.0 - error.item())
    return max(reward, 0.0)


def _align_goal_vector(goal_vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    gv = goal_vec
    if gv.dim() == 2:
        gv = gv.reshape(1, -1)
    elif gv.dim() == 1:
        gv = gv.unsqueeze(0)
    elif gv.dim() == 3 and gv.shape[1] == 1:
        gv = gv.squeeze(1)
    else:
        raise RuntimeError(f"[goal_vec 异常] 当前 shape={gv.shape}")

    if gv.dim() == 3 and gv.size(1) == 1:
        gv = gv.squeeze(1)
    if gv.dim() == 1:
        gv = gv.unsqueeze(0)

    if gv.shape[-1] != target_dim:
        if gv.shape[-1] < target_dim:
            pad = (0, target_dim - gv.shape[-1])
            gv = torch.nn.functional.pad(gv, pad)
        else:
            gv = gv[..., :target_dim]
    return gv


def _prepare_input_tensor(unit, input_tensor: torch.Tensor) -> torch.Tensor:
    tensor = input_tensor.squeeze()

    if not FOLLOW_INPUT_DEVICE:
        if unit.function[0].weight.device != tensor.device:
            unit.to(tensor.device)

    if tensor.dim() == 3 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)

    if unit.call_history:
        unit.avg_recent_calls = sum(unit.call_history) / len(unit.call_history)

    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)

    if hasattr(unit, "goal_vec") and unit.goal_vec is not None:
        gv = _align_goal_vector(unit.goal_vec, tensor.shape[-1])
        tensor = torch.cat([tensor, gv], dim=-1)

    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    D = tensor.shape[-1]
    if D < unit.input_size:
        pad = (0, unit.input_size - D)
        tensor = torch.nn.functional.pad(tensor, pad)
    elif D > unit.input_size:
        tensor = tensor[..., : unit.input_size]
    return tensor


def _maybe_follow_device(unit, input_tensor: torch.Tensor) -> None:
    if FOLLOW_INPUT_DEVICE:
        if unit.function[0].weight.device != input_tensor.device:
            unit.to(input_tensor.device)


def _expand_input_if_needed(unit, input_tensor: torch.Tensor) -> torch.Tensor:
    current_input_size = input_tensor.shape[-1]
    if current_input_size <= unit.input_size:
        if current_input_size < unit.input_size:
            pad = (0, unit.input_size - current_input_size)
            input_tensor = torch.nn.functional.pad(input_tensor, pad)
        return input_tensor

    old_l1, old_l2 = unit.function[0], unit.function[2]
    w1, b1 = old_l1.weight.data.clone(), old_l1.bias.data.clone()
    w2, b2 = old_l2.weight.data.clone(), old_l2.bias.data.clone()

    h = unit.hidden_size
    new_l1 = nn.Linear(current_input_size, h, device=old_l1.weight.device)
    new_l2 = nn.Linear(h, current_input_size, device=old_l2.weight.device)

    with torch.no_grad():
        new_l1.weight[:, : w1.shape[1]].copy_(w1)
        new_l1.bias.copy_(b1)
        new_l2.weight[: w2.shape[0], : w2.shape[1]].copy_(w2)
        new_l2.bias[: b2.shape[0]].copy_(b2)

    unit.function = nn.Sequential(new_l1, nn.ReLU(), new_l2)
    unit.input_size = current_input_size
    unit.output_history_tensor = torch.zeros((5, unit.input_size), device="cpu")
    unit.output_history_ptr = 0

    new_hist = []
    for out in unit.output_history:
        v = out.view(-1)
        pad = (0, current_input_size - v.shape[0])
        v2 = torch.nn.functional.pad(v, pad)
        new_hist.append(v2.unsqueeze(0))
    unit.output_history = new_hist

    new_mem = []
    for mem in unit.state_memory:
        pad = (0, current_input_size - mem.numel())
        new_mem.append(torch.nn.functional.pad(mem, pad))
    unit.state_memory = new_mem

    device = input_tensor.device
    unit.last_output = torch.zeros(unit.input_size, device=device)
    unit.state = torch.zeros(unit.input_size, device=device)

    return input_tensor


def _update_history(unit, output: torch.Tensor) -> None:
    if unit.output_history_tensor.shape[1] != output.shape[0]:
        unit.output_history_tensor = torch.zeros((5, output.shape[0]), device="cpu")
        unit.output_history_ptr = 0

    out = output.detach().cpu().view(-1)
    if out.shape[0] != unit.output_history_tensor.shape[1]:
        unit.output_history_tensor = torch.zeros((5, out.shape[0]), device="cpu")
        unit.output_history_ptr = 0
    unit.output_history_tensor[unit.output_history_ptr] = out
    unit.output_history_ptr = (unit.output_history_ptr + 1) % unit.output_history_tensor.shape[0]


def _apply_noise(unit, output: torch.Tensor) -> torch.Tensor:
    if not hasattr(unit, "current_step"):
        return output
    if unit.get_role() == "emitter" and unit.current_step < 20:
        noise = torch.randn_like(output) * 0.2
        output = output + noise
        logger.debug("[扰动] emitter %s 输出加入扰动", unit.id)
    elif unit.get_role() == "processor" and unit.current_step < 5:
        noise = torch.randn_like(output) * 0.1
        output = output + noise
        logger.debug("[扰动] processor %s 输出加入扰动", unit.id)
    return output


def _handle_emitter_goal(unit, output: torch.Tensor) -> None:
    if unit.get_role() != "emitter" or not hasattr(unit, "goal_vec"):
        return
    out_vec = output.view(-1)
    idx = torch.argmax(out_vec).item()
    x, y = idx % unit.env_size, idx // unit.env_size

    hazard = getattr(unit, "current_hazard_xy", None)
    if hazard is None:
        unit.is_hazard_confirmed = False
        return
    hx, hy = hazard
    if (x, y) == (hx, hy):
        unit.is_hazard_confirmed = True
        unit.goal_vec.zero_()
    else:
        unit.is_hazard_confirmed = False


def _maybe_learn(unit, input_tensor, output_tensor) -> None:
    if unit.get_role() == "emitter":
        bias = unit.gene.get("emitter_bias", 1.0)
        lr = 0.001 * (2.0 - min(1.5, bias))
        if ENABLE_MINI_LEARN:
            perform_mini_learn(unit, input_tensor, output_tensor.detach(), lr=lr)
    else:
        bias_key = "processor_bias" if unit.role == "processor" else "sensor_bias"
        bias = unit.gene.get(bias_key, 1.0)
        lr = 0.001 * (2.0 - min(1.5, bias))
        if ENABLE_MINI_LEARN:
            perform_mini_learn(unit, input_tensor, input_tensor, lr=lr)


def perform_update(unit, input_tensor: torch.Tensor) -> None:
    tensor = _prepare_input_tensor(unit, input_tensor)

    unit.memory_limit = 5 + (getattr(unit, "current_step", 0) // 500) * 5
    global_step = getattr(unit, "current_step", 0)
    unit.memory_pool_limit = min(50 + (global_step // 500) * 30, 1000)

    _maybe_follow_device(unit, tensor)
    tensor = _expand_input_if_needed(unit, tensor)

    use_autocast = RF.use_fp16 and unit.device.type == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_autocast else nullcontext()

    with ctx, torch.inference_mode():
        raw_output = unit.function(tensor)

    raw_output = _apply_noise(unit, raw_output)
    unit.last_output = raw_output.detach().clone()
    unit.state = unit.last_output.clone()

    _update_history(unit, unit.last_output)

    unit.age += 1
    unit.state_memory.append(unit.state.detach().cpu())
    if len(unit.state_memory) > unit.memory_limit:
        unit.state_memory.pop(0)

    input_var = float(tensor.var())
    recent_call_freq = getattr(unit, "recent_calls", 1)
    connection_count = getattr(unit, "connection_count", 1)
    _ = (input_var, recent_call_freq, connection_count)

    if getattr(unit, "avg_recent_calls", 0.0) >= 4.0 and unit.energy > 0.0:
        unit.energy += 0.05
        logger.debug("[奖励] %s 平均调用频率 %.2f → 能量 +0.04", unit.id, unit.avg_recent_calls)

    self_reward = compute_self_reward(unit, tensor, unit.last_output) * 0.03
    unit.energy += self_reward
    if self_reward > 0:
        logger.debug(
            "[内部奖励] %s 自评奖励 +%.4f 能量 (现有能量 %.2f)",
            unit.id,
            self_reward,
            unit.energy,
        )

    _handle_emitter_goal(unit, unit.last_output)
    _maybe_learn(unit, tensor, unit.last_output)
