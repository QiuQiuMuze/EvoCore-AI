import random

import torch

from env import logger


class MortalityMixin:
    def should_die(self) -> bool:
        if self.role == "processor":
            if self.energy <= 0.0:
                return True
            if self.age > 270:
                return True
            if 250 <= self.age <= 270:
                death_chance = (self.age - 250) / 20
                if random.random() < death_chance:
                    logger.info(f"[衰老死亡] {self.id} 年龄={self.age}，概率={death_chance:.2f} → 死亡")
                    self.death_by_aging = True
                    if self.is_worthy_of_memory():
                        self.add_to_local_memory()
                    return True
        if self.energy <= 0.0:
            return True
        if self.age > 270:
            return True
        if 250 <= self.age <= 270:
            death_chance = (self.age - 250) / 20
            if random.random() < death_chance:
                logger.info(f"[衰老死亡] {self.id} 年龄={self.age}，概率={death_chance:.2f} → 死亡")
                self.death_by_aging = True
                if self.is_worthy_of_memory():
                    self.add_to_local_memory()
                return True
        graph = getattr(self, "graph", None)
        if graph is not None:
            if 0 <= graph.current_step - graph.static_mode_exit_step <= 50:
                return False
        if self.role in ["emitter"] and self.inactive_steps > 20:
            return True
        if self.role in ["processor", "emitter"] and getattr(self, "current_step", 0) > 600 and self.age > 150:
            if len(self.output_history) >= 4:
                diffs = []
                for i in range(len(self.output_history) - 1):
                    a = self.output_history[i]
                    b = self.output_history[i + 1]
                    target_dim = max(a.shape[-1], b.shape[-1])
                    if a.shape[-1] < target_dim:
                        padding = (0, target_dim - a.shape[-1])
                        a = torch.nn.functional.pad(a, padding, value=0)
                    if b.shape[-1] < target_dim:
                        padding = (0, target_dim - b.shape[-1])
                        b = torch.nn.functional.pad(b, padding, value=0)
                    diffs.append(torch.norm(a - b).item())
                if max(diffs) < 0.005:
                    logger.info(f"[退化死亡] {self.id} 输出变化极小 → 被淘汰")
                    return True
        return False
