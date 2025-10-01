import copy
import inspect
import random
from collections import defaultdict

import torch
import torch.nn as nn

from env import logger

from .constants import ROLE_SPLIT_RULE, SPLIT_HI_ES_TABLE, SPLIT_HI_P_TABLE, TOL_FRAC_SPLIT, _get_hi


_CLONE_PARAM_ALIASES = {
    "unit_id": "id",
}


class ReproductionMixin:
    def should_split(self):
        emitter_count = getattr(self, "global_emitter_count", 1)
        processor_count = getattr(self, "global_processor_count", 1)
        sensor_count = getattr(self, "global_sensor_count", 1)
        total = getattr(self, "global_unit_count", 1)
        role = self.get_role()
        if role == "emitter" and emitter_count <= 8:
            logger.warning(f"[紧急增殖] {self.id} 是唯一 emitter，强制尝试分裂并补给")
            self.energy += 1
            return True
        if role == "processor" and processor_count <= 16:
            logger.warning(f"[紧急增殖] {self.id} 是唯一 processor，强制尝试分裂并补给")
            self.energy += 1
            return True
        if role == "sensor" and sensor_count <= 8:
            logger.warning(f"[紧急增殖] {self.id} 是唯一 sensor，强制尝试分裂并补给")
            self.energy += 1
            return True
        total = getattr(self, "global_unit_count", sensor_count + processor_count + emitter_count)
        hi_es = _get_hi(SPLIT_HI_ES_TABLE, total)
        hi_p = _get_hi(SPLIT_HI_P_TABLE, total)
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
        if self.energy < rule["min_e"]:
            return False
        if role != "sensor" and self.avg_recent_calls < rule["min_calls"]:
            return False
        history = self.get_recent_outputs(6)
        if len(history) >= 6 and all(torch.equal(history[0], h) for h in history[1:]):
            return False
        return True

    def clone(
        self,
        role_override=None,
        new_input_size=None,
        global_resources=None,
        global_hazards=None,
        free_positions=None,
    ):
        role = role_override or self.role
        input_size = new_input_size if new_input_size is not None else self.input_size
        init_kwargs = {
            "input_size": input_size,
            "hidden_size": self.hidden_size,
            "role": role,
            "env_size": self.env_size,
        }
        if hasattr(self, "get_clone_init_kwargs"):
            extra_kwargs = self.get_clone_init_kwargs() or {}
            if not isinstance(extra_kwargs, dict):
                raise TypeError("get_clone_init_kwargs must return a dict of keyword arguments")
            init_kwargs.update(extra_kwargs)
        constructor = type(self)
        try:
            clone_unit = constructor(**init_kwargs)
        except TypeError as exc:
            missing_kwargs = {}
            signature = inspect.signature(constructor.__init__)
            for name, param in list(signature.parameters.items())[1:]:
                if param.kind not in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    continue
                if name in init_kwargs:
                    continue
                if param.default is not inspect.Parameter.empty:
                    continue
                if hasattr(self, name):
                    missing_kwargs[name] = getattr(self, name)
                    continue
                alias = _CLONE_PARAM_ALIASES.get(name)
                if alias and hasattr(self, alias):
                    missing_kwargs[name] = getattr(self, alias)
            if not missing_kwargs:
                raise
            init_kwargs.update(missing_kwargs)
            clone_unit = constructor(**init_kwargs)
        clone_unit.visit_counts = defaultdict(int)
        if input_size == self.input_size:
            clone_unit.function = copy.deepcopy(self.function)
        else:
            old_l1, old_relu, old_l2 = self.function
            w1, b1 = old_l1.weight.data.clone(), old_l1.bias.data.clone()
            w2, b2 = old_l2.weight.data.clone(), old_l2.bias.data.clone()
            h = self.hidden_size
            new_l1 = nn.Linear(input_size, h, device=w1.device)
            new_l2 = nn.Linear(h, input_size, device=w2.device)
            with torch.no_grad():
                cols = min(new_l1.in_features, w1.shape[1])
                new_l1.weight.data[:, :cols].copy_(w1[:, :cols])
                new_l1.bias.copy_(b1)
                rows = min(new_l2.out_features, w2.shape[0])
                cols2 = min(new_l2.in_features, w2.shape[1])
                new_l2.weight.data[:rows, :cols2].copy_(w2[:rows, :cols2])
                new_l2.bias.data[:rows].copy_(b2[:rows])
            clone_unit.function = nn.Sequential(new_l1, nn.ReLU(), new_l2)
        clone_unit.gene = {k: v for k, v in self.gene.items()}
        if random.random() < self.gene.get("mutation_rate", 0.01):
            delta = random.choice([2, 4])
            new_hidden = self.hidden_size + delta
            new_hidden = max(self.hidden_size, min(128, new_hidden))
            if new_hidden != self.hidden_size:
                clone_unit.hidden_size = new_hidden
                clone_unit.function = nn.Sequential(
                    nn.Linear(clone_unit.input_size, new_hidden),
                    nn.ReLU(),
                    nn.Linear(new_hidden, clone_unit.input_size),
                )
                logger.info(f"[突变升维] hidden_size ↑ 为 {new_hidden}")
        if random.random() < self.gene.get("mutation_rate", 0.005):
            for key in ["sensor_bias", "processor_bias", "emitter_bias"]:
                mutation = random.uniform(-0.1, 0.1)
                clone_unit.gene[key] = max(0.5, min(2.0, clone_unit.gene[key] + mutation))
            logger.info(f"[突变] gene 突变为 {clone_unit.gene}")
        if free_positions:
            clone_unit.position = random.choice(free_positions)
        else:
            occupied = set(global_resources or ()) | set(global_hazards or ())
            fps = [
                (x, y)
                for x in range(self.env_size)
                for y in range(self.env_size)
                if (x, y) not in occupied
            ]
            if fps:
                clone_unit.position = random.choice(fps)
            else:
                fx = random.randint(0, self.env_size - 1)
                fy = random.randint(0, self.env_size - 1)
                logger.warning(f"[出生回退] 未找到安全位置，随机在 ({fx},{fy})")
                clone_unit.position = (fx, fy)
        clone_unit.energy = self.energy * 0.6
        clone_unit.age = 0
        clone_unit.state = self.state.clone()
        if input_size != self.input_size:
            clone_unit.last_output = torch.zeros(input_size)
        else:
            clone_unit.last_output = self.last_output.clone()
        self.energy *= 0.4
        scored_memories = [m for m in self.local_memory_pool if "score" in m]
        if scored_memories:
            scored_memories.sort(key=lambda m: m["score"], reverse=True)
            top_half = scored_memories[: (len(scored_memories) + 1) // 2]
            clone_unit.local_memory_pool = top_half
        else:
            clone_unit.local_memory_pool = []
        clone_unit.to(self.device)
        if hasattr(self, "local_memory_pool") and len(self.local_memory_pool) >= 1:
            memory = random.choice(self.local_memory_pool[-5:])
            for key in ["sensor_bias", "processor_bias", "emitter_bias"]:
                g1 = self.gene.get(key, 1.0)
                g2 = memory["gene"].get(key, 1.0)
                clone_unit.gene[key] = 0.7 * g1 + 0.3 * g2
            logger.debug(f"[记忆融合] {self.id} 结合 local memory 基因 → 子基因：{clone_unit.gene}")
            if self.last_output is None or memory.get("output") is None:
                if getattr(self, "is_permanent_explorer", False):
                    clone_unit.is_permanent_explorer = True
                return clone_unit
            if "output" in memory:
                o1 = self.last_output.squeeze(0) if self.last_output.dim() == 2 else self.last_output
                o2 = memory["output"].squeeze(0) if memory["output"].dim() == 2 else memory["output"]
                target_dim = max(o1.shape[0], o2.shape[0])
                if o1.shape[0] < target_dim:
                    o1 = torch.nn.functional.pad(o1, (0, target_dim - o1.shape[0]), value=0)
                if o2.shape[0] < target_dim:
                    o2 = torch.nn.functional.pad(o2, (0, target_dim - o2.shape[0]), value=0)
                clone_unit.last_output = 0.6 * o1 + 0.4 * o2
                logger.debug(f"[行为融合] 结合输出 → 前5维: {clone_unit.last_output[:5]}")
            if random.random() < self.gene.get("mutation_rate", 0.01) * 2:
                if "hidden_size" in memory:
                    h1 = self.hidden_size
                    h2 = memory.get("hidden_size", h1)
                    new_hidden = max(h1, int(0.7 * h1 + 0.3 * h2))
                    new_hidden = min(128, new_hidden)
                    if new_hidden != self.hidden_size:
                        clone_unit.hidden_size = new_hidden
                        clone_unit.function = torch.nn.Sequential(
                            torch.nn.Linear(clone_unit.input_size, new_hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(new_hidden, clone_unit.input_size),
                        )
                        clone_unit.gene["hidden_size_tag"] = new_hidden
                        logger.debug(f"[网络融合] hidden_size 融合为 {new_hidden}")
        if getattr(self, "is_permanent_explorer", False):
            clone_unit.is_permanent_explorer = True
        if not hasattr(clone_unit, "visit_counts") or not clone_unit.visit_counts:
            clone_unit.visit_counts = self.visit_counts.copy()
        return clone_unit
