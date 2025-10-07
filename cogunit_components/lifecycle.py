"""Lifecycle management utilities for :class:`CogUnit`."""
from __future__ import annotations

import random
from typing import Iterable

import torch

from env import logger

from .constants import ROLE_SPLIT_RULE, SPLIT_HI_ES_TABLE, SPLIT_HI_P_TABLE, TOL_FRAC_SPLIT, get_hi
from .memory import add_to_local_memory, fuse_memory, get_recent_outputs, is_worthy_of_memory


def should_split(unit) -> bool:
    emitter_count = getattr(unit, "global_emitter_count", 1)
    processor_count = getattr(unit, "global_processor_count", 1)
    sensor_count = getattr(unit, "global_sensor_count", 1)
    total = getattr(unit, "global_unit_count", sensor_count + processor_count + emitter_count)

    role = unit.get_role()

    if role == "emitter" and emitter_count <= 8:
        logger.warning("[紧急增殖] %s 是唯一 emitter，强制尝试分裂并补给", unit.id)
        unit.energy += 1
        return True
    if role == "processor" and processor_count <= 16:
        logger.warning("[紧急增殖] %s 是唯一 processor，强制尝试分裂并补给", unit.id)
        unit.energy += 1
        return True
    if role == "sensor" and sensor_count <= 8:
        logger.warning("[紧急增殖] %s 是唯一 sensor，强制尝试分裂并补给", unit.id)
        unit.energy += 1
        return True

    hi_es = get_hi(SPLIT_HI_ES_TABLE, total)
    hi_p = get_hi(SPLIT_HI_P_TABLE, total)
    half_p = processor_count / 2

    def _delta_enough(x, y):
        delta = x - y
        return delta >= max(1, int(total * TOL_FRAC_SPLIT))

    overpop = False
    if role == "emitter":
        if _delta_enough(emitter_count, sensor_count * hi_es) or _delta_enough(emitter_count, half_p * hi_p):
            overpop = True
    elif role == "sensor":
        if _delta_enough(sensor_count, emitter_count * hi_es) or _delta_enough(sensor_count, half_p * hi_p):
            overpop = True
    elif role == "processor":
        if _delta_enough(half_p, emitter_count * hi_p) or _delta_enough(half_p, sensor_count * hi_p):
            overpop = True

    if overpop:
        return False

    rule = ROLE_SPLIT_RULE[role]
    if unit.energy < rule["min_e"]:
        return False
    if role != "sensor" and unit.avg_recent_calls < rule["min_calls"]:
        return False

    history = get_recent_outputs(unit, 6)
    if len(history) >= 6 and all(torch.equal(history[0], h) for h in history[1:]):
        return False

    return True


def evaluate_self(unit, min_rate: float = 0.3) -> bool:
    rate = unit.meta.recent_success_rate()
    if rate is None:
        return False
    return rate < min_rate


def request_upgrade(unit, target_role=None, reason: str = "") -> None:
    unit.meta = unit.meta.__class__(history_len=unit.meta.reward_trace.maxlen)
    for k in ["sensor_bias", "processor_bias", "emitter_bias"]:
        unit.gene[k] += random.gauss(0, 0.05)
    for p in unit.function.parameters():
        p.data += torch.randn_like(p) * 0.01
    logger.info(
        "[Meta-升级] %s, %s 因“%s”触发自我进化，开始思考赛博人生，觉得自己又行了。新gene=%s",
        unit.id,
        unit.role,
        reason,
        unit.gene,
    )


def should_die(unit) -> bool:
    if unit.role == "processor":
        if unit.energy <= 0.0:
            return True
        if unit.age > 270:
            return True
        if 250 <= unit.age <= 270:
            death_chance = (unit.age - 250) / 20
            if random.random() < death_chance:
                logger.info("[衰老死亡] %s 年龄=%s，概率=%.2f → 死亡", unit.id, unit.age, death_chance)
                unit.death_by_aging = True
                if is_worthy_of_memory(unit):
                    add_to_local_memory(unit)
                return True

    if unit.energy <= 0.0:
        return True

    if unit.age > 270:
        return True

    if 250 <= unit.age <= 270:
        death_chance = (unit.age - 250) / 20
        if random.random() < death_chance:
            logger.info("[衰老死亡] %s 年龄=%s，概率=%.2f → 死亡", unit.id, unit.age, death_chance)
            unit.death_by_aging = True
            if is_worthy_of_memory(unit):
                add_to_local_memory(unit)
            return True

    graph = getattr(unit, "graph", None)
    if graph is not None:
        if 0 <= graph.current_step - graph.static_mode_exit_step <= 50:
            return False

    if unit.role in ["emitter"] and unit.inactive_steps > 20:
        return True

    if unit.role in ["processor", "emitter"] and getattr(unit, "current_step", 0) > 600 and unit.age > 150:
        if len(unit.output_history) >= 4:
            diffs = []
            for i in range(len(unit.output_history) - 1):
                a = unit.output_history[i]
                b = unit.output_history[i + 1]
                target_dim = max(a.shape[-1], b.shape[-1])
                if a.shape[-1] < target_dim:
                    padding = (0, target_dim - a.shape[-1])
                    a = torch.nn.functional.pad(a, padding, value=0)
                if b.shape[-1] < target_dim:
                    padding = (0, target_dim - b.shape[-1])
                    b = torch.nn.functional.pad(b, padding, value=0)
                diffs.append(torch.norm(a - b).item())
            if diffs and max(diffs) < 0.005:
                logger.info("[退化死亡] %s 输出变化极小 → 被淘汰", unit.id)
                return True
    return False


def clone_unit(
    unit,
    role_override=None,
    new_input_size=None,
    global_resources=None,
    global_hazards=None,
    free_positions=None,
):
    role = role_override or unit.role
    input_size = new_input_size if new_input_size is not None else unit.input_size

    clone = unit.__class__(
        input_size=input_size,
        hidden_size=unit.hidden_size,
        role=role,
        env_size=unit.env_size,
    )

    clone.gene = unit.gene.copy()
    if random.random() < unit.gene.get("mutation_rate", 0.01):
        key = random.choice(["sensor_bias", "processor_bias", "emitter_bias"])
        clone.gene[key] *= random.uniform(0.9, 1.1)

    clone.mutation_rate = unit.gene.get("mutation_rate", 0.01)
    clone.local_memory_pool = []
    clone.position = unit.position
    if free_positions:
        occupied = set(free_positions)
        graph = getattr(unit, "graph", None)
        if graph is not None:
            for u in graph.units:
                occupied.add(u.get_position())
        free = [
            (x, y)
            for x in range(unit.env_size)
            for y in range(unit.env_size)
            if (x, y) not in occupied
        ]
        if free:
            clone.position = random.choice(free)
        else:
            fx = random.randint(0, unit.env_size - 1)
            fy = random.randint(0, unit.env_size - 1)
            logger.warning("[出生回退] 未找到安全位置，随机在 (%s,%s)", fx, fy)
            clone.position = (fx, fy)

    clone.energy = unit.energy * 0.6
    clone.age = 0
    clone.state = unit.state.clone()

    if input_size != unit.input_size:
        clone.last_output = torch.zeros(input_size)
    else:
        clone.last_output = unit.last_output.clone()

    unit.energy *= 0.4

    scored_memories = [m for m in unit.local_memory_pool if "score" in m]
    if scored_memories:
        scored_memories.sort(key=lambda m: m["score"], reverse=True)
        top_half = scored_memories[: (len(scored_memories) + 1) // 2]
        clone.local_memory_pool = top_half
    else:
        clone.local_memory_pool = []

    clone.to(unit.device)

    fuse_memory(unit, clone)

    return clone
