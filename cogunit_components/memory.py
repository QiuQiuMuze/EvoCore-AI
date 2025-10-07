"""Memory management helpers for :class:`CogUnit`."""
from __future__ import annotations

import random
from typing import Iterable

import torch
import torch.nn.functional as F

from env import logger


def get_recent_outputs(unit, n: int = 5):
    L = min(n, unit.output_history_tensor.size(0))
    idxs = [(unit.output_history_ptr - i - 1) % L for i in reversed(range(L))]
    return [unit.output_history_tensor[i] for i in idxs]


def _aligned_history(output_history: Iterable[torch.Tensor]):
    if len(output_history) < 2:
        return None
    max_len = max(t.numel() for t in output_history)
    aligned = []
    for t in output_history:
        vec = t.view(-1)
        if vec.numel() < max_len:
            vec = F.pad(vec, (0, max_len - vec.numel()), value=0)
        else:
            vec = vec[:max_len]
        aligned.append(vec)
    return aligned


def is_worthy_of_memory(unit) -> bool:
    if unit.age < 100:
        return False

    history = unit.output_history
    if unit.role == "sensor":
        if len(history) < 2:
            return False
        changes = []
        for prev, curr in zip(history, history[1:]):
            p = prev.view(-1)
            c = curr.view(-1)
            L = min(p.numel(), c.numel())
            changes.append((c[:L] - p[:L]).abs().sum().item())
        if not changes:
            return False
        avg_change = sum(changes) / len(changes)
        SENSOR_CHANGE_THRESHOLD = 5.0
        return avg_change > SENSOR_CHANGE_THRESHOLD

    if unit.role == "processor":
        if getattr(unit, "avg_recent_calls", 0) < 0.75:
            return False
        if len(history) < 2:
            return False
        total_diff = 0.0
        count = 0
        for prev, curr in zip(history, history[1:]):
            if prev.shape == curr.shape:
                total_diff += torch.norm(curr - prev).item()
                count += 1
        if count == 0:
            return False
        variation = total_diff / count
        return variation > 0.05

    if unit.role == "emitter":
        if getattr(unit, "avg_recent_calls", 0) < 2.0:
            return False
        if len(history) < 2:
            return False
        diff = sum(
            torch.norm(history[i] - history[i + 1]).item()
            for i in range(len(history) - 1)
        ) / (len(history) - 1)
        return 0.01 < diff < 0.5

    return False


def add_to_local_memory(unit) -> None:
    unit.local_memory_pool = [m for m in unit.local_memory_pool if "score" in m]

    history = unit.output_history
    aligned_history = None
    if len(history) >= 3:
        aligned_history = _aligned_history(history)

    if unit.role == "sensor":
        if aligned_history and len(aligned_history) >= 2:
            diffs = []
            for prev, curr in zip(history, history[1:]):
                p = prev.view(-1)
                c = curr.view(-1)
                L = min(p.numel(), c.numel())
                diffs.append((c[:L] - p[:L]).abs().sum().item())
            variation = sum(diffs) / len(diffs)
            score = variation
        else:
            score = 0
    elif unit.role == "processor":
        if aligned_history:
            diffs = [
                torch.norm(aligned_history[i] - aligned_history[i + 1]).item()
                for i in range(len(aligned_history) - 1)
            ]
            diversity = sum(diffs) / len(diffs)
        else:
            diversity = 0
        score = diversity * 0.5 + getattr(unit, "avg_recent_calls", 0) * 0.3
    elif unit.role == "emitter":
        if aligned_history:
            diffs = [
                torch.norm(aligned_history[i] - aligned_history[i + 1]).item()
                for i in range(len(aligned_history) - 1)
            ]
            avg_diff = sum(diffs) / len(diffs)
            stability = 1.0 if 0.01 < avg_diff < 0.5 else 0.0
        else:
            stability = 0
        score = getattr(unit, "avg_recent_calls", 0) * 0.5 + stability * 0.3
    else:
        score = unit.energy + getattr(unit, "avg_recent_calls", 0)

    mem = {
        "gene": unit.gene.copy(),
        "output": unit.last_output.clone(),
        "role": unit.role,
        "age": unit.age,
        "hidden_size": unit.hidden_size,
        "score": score,
    }
    unit.local_memory_pool.append(mem)

    if len(unit.local_memory_pool) > unit.memory_pool_limit:
        unit.local_memory_pool.sort(key=lambda m: m["score"])
        unit.local_memory_pool.pop(0)
    logger.info(
        "[记忆加入] %s（%s，Age=%s）加入本地记忆池，评分=%.2f，当前共 %s 条",
        unit.id,
        unit.role,
        unit.age,
        mem["score"],
        len(unit.local_memory_pool),
    )


def fuse_memory(unit, clone_unit) -> None:
    if not hasattr(unit, "local_memory_pool") or len(unit.local_memory_pool) < 1:
        return
    memory = random.choice(unit.local_memory_pool[-5:])
    for key in ["sensor_bias", "processor_bias", "emitter_bias"]:
        g1 = unit.gene.get(key, 1.0)
        g2 = memory["gene"].get(key, 1.0)
        clone_unit.gene[key] = 0.7 * g1 + 0.3 * g2
    logger.debug("[记忆融合] %s 结合 local memory 基因 → 子基因：%s", unit.id, clone_unit.gene)

    if unit.last_output is None or memory.get("output") is None:
        if getattr(unit, "is_permanent_explorer", False):
            clone_unit.is_permanent_explorer = True
        return

    if "output" in memory:
        o1 = unit.last_output.squeeze(0) if unit.last_output.dim() == 2 else unit.last_output
        o2 = memory["output"].squeeze(0) if memory["output"].dim() == 2 else memory["output"]
        target_dim = max(o1.shape[0], o2.shape[0])
        if o1.shape[0] < target_dim:
            o1 = F.pad(o1, (0, target_dim - o1.shape[0]), value=0)
        if o2.shape[0] < target_dim:
            o2 = F.pad(o2, (0, target_dim - o2.shape[0]), value=0)
        clone_unit.last_output = 0.6 * o1 + 0.4 * o2
        logger.debug("[行为融合] 结合输出 → 前5维: %s", clone_unit.last_output[:5])

    if random.random() < unit.gene.get("mutation_rate", 0.01) * 2 and "hidden_size" in memory:
        h1 = unit.hidden_size
        h2 = memory.get("hidden_size", h1)
        new_hidden = max(h1, int(0.7 * h1 + 0.3 * h2))
        new_hidden = min(128, new_hidden)
        if new_hidden != unit.hidden_size:
            clone_unit.hidden_size = new_hidden
            clone_unit.function = torch.nn.Sequential(
                torch.nn.Linear(clone_unit.input_size, new_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(new_hidden, clone_unit.input_size),
            )
            clone_unit.gene["hidden_size_tag"] = new_hidden
            logger.debug("[网络融合] hidden_size 融合为 %s", new_hidden)

    if getattr(unit, "is_permanent_explorer", False):
        clone_unit.is_permanent_explorer = True

    if not hasattr(clone_unit, "visit_counts") or not clone_unit.visit_counts:
        clone_unit.visit_counts = unit.visit_counts.copy()
