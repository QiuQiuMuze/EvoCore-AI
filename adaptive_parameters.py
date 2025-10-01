import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ThreatStatistics:
    mean: float = 0.0
    std: float = 0.0
    density: float = 0.0
    pressure: float = 0.0


class CellularLearningController:
    """Adaptive parameter manager for ImmuneCogGraph.

    The controller keeps exponential moving averages over several threat
    statistics extracted from the environment and derives thresholds and
    weightings for the immune system.  The goal is to replace previously
    hard-coded constants with signals that respond automatically to how the
    environment evolves.
    """

    def __init__(
        self,
        env,
        device: str | torch.device = "cpu",
        alpha: float = 0.1,
        warmup_steps: int = 3000,
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self._step: int = 0

        self._infection = ThreatStatistics()
        self._hack = ThreatStatistics()
        self._infection_threshold: float = 0.08
        self._hack_threshold: float = 0.08
        self._quota_scale: float = 1.0
        self._privilege_scale: float = 1.5
        self._weight_scale: float = 1.0

        self.observe(env, step=0)

    # ------------------------------------------------------------------
    def observe(self, env=None, step: Optional[int] = None) -> None:
        env = env or self.env
        self.env = env
        if step is not None:
            self._step = step

        inf_map = env.infected_map.detach()
        hack_map = env.privilege_level.detach()
        stealth_map = getattr(env, "hack_strength", hack_map).detach()

        self._infection = self._update_stats(inf_map, self._infection)
        hack_stats = self._update_stats(hack_map, self._hack)
        stealth_stats = self._update_stats(stealth_map, self._hack)
        # Combine privilege + stealth pressure for hack response
        self._hack.mean = max(hack_stats.mean, stealth_stats.mean)
        self._hack.std = max(hack_stats.std, stealth_stats.std)
        self._hack.density = max(hack_stats.density, stealth_stats.density)
        self._hack.pressure = max(hack_stats.pressure, stealth_stats.pressure)

        env_scale = max(1.0, math.log1p(env.size))
        self._quota_scale = env_scale * (1.0 + self._total_pressure() * 0.5)
        self._privilege_scale = 1.0 + env_scale * (0.5 + self._hack.density)
        self._weight_scale = env_scale * (1.0 + self._total_pressure())

        self._infection_threshold = self._blend(
            self._infection_threshold,
            self._derive_threshold(self._infection),
        )
        self._hack_threshold = self._blend(
            self._hack_threshold,
            self._derive_threshold(self._hack),
        )

    # ------------------------------------------------------------------
    def on_env_resize(self, env) -> None:
        self.env = env
        self.observe(env, step=self._step)

    # ------------------------------------------------------------------
    def get_detection_threshold(self, step: Optional[int] = None) -> float:
        target = max(self._infection_threshold, self._hack_threshold)
        current_step = self._step if step is None else step
        if current_step < self.warmup_steps:
            ratio = current_step / max(1, self.warmup_steps)
            base = 0.5  # initial conservative detection threshold
            return float(self._blend(base, target, ratio))
        return float(target)

    def get_infection_threshold(self) -> float:
        return float(self._infection_threshold)

    def get_hack_threshold(self) -> float:
        return float(self._hack_threshold)

    def get_privilege_scale(self) -> float:
        return float(self._privilege_scale)

    def get_goal_biases(self) -> tuple[float, float]:
        env_scale = max(0.5, math.log1p(self.env.size))
        hack_bias = 1.0 + env_scale + self._scale_signal(self._hack.pressure, env_scale)
        infection_bias = 1.0 + 0.8 * env_scale + self._scale_signal(
            self._infection.pressure, env_scale
        )
        infection_bias = min(infection_bias, hack_bias * 0.95)
        return float(hack_bias), float(max(0.0, infection_bias))

    def get_candidate_quota(self, emitter_count: int, threat_count: int) -> int:
        if threat_count <= 0:
            return 0
        emitters = max(1, emitter_count)
        ratio = threat_count / emitters
        dynamic = ratio * self._quota_scale
        cap = max(1.0, self._quota_scale * 2.0)
        quota = int(math.ceil(min(dynamic, cap)))
        return max(1, min(threat_count, quota))

    def get_target_weights(self) -> torch.Tensor:
        total = self._total_pressure()
        device = self.device
        base_explore = 1.0 + 0.3 * math.log1p(self.env.size)

        if total <= 1e-8:
            return torch.tensor([base_explore, 0.0, 0.0], device=device)

        infection_ratio = self._infection.pressure / total
        hack_ratio = self._hack.pressure / total

        w1 = 0.0 if self._infection.pressure <= 1e-8 else base_explore + self._weight_scale * infection_ratio
        w2 = 0.0 if self._hack.pressure <= 1e-8 else base_explore + self._weight_scale * hack_ratio
        return torch.tensor([base_explore, w1, w2], device=device)

    # ------------------------------------------------------------------
    def _total_pressure(self) -> float:
        return self._infection.pressure + self._hack.pressure

    def _scale_signal(self, signal: float, env_scale: float) -> float:
        if signal <= 0.0:
            return 0.0
        return env_scale * math.tanh(signal * (1.0 + env_scale))

    def _derive_threshold(self, stats: ThreatStatistics) -> float:
        if stats.density <= 1e-8:
            return 0.02
        base = stats.mean + stats.std
        density_boost = 0.5 + stats.density
        return float(max(0.02, min(0.9, base * density_boost)))

    def _update_stats(self, tensor: torch.Tensor, stats: ThreatStatistics) -> ThreatStatistics:
        if tensor.numel() == 0:
            return stats
        flat = tensor.view(-1)
        mean = flat.mean().item()
        std = flat.std(unbiased=False).item()
        density = (flat > 1e-5).float().mean().item()
        pressure = density * (mean + std)

        updated = ThreatStatistics(
            mean=self._blend(stats.mean, mean),
            std=self._blend(stats.std, std),
            density=self._blend(stats.density, density),
            pressure=self._blend(stats.pressure, pressure),
        )
        return updated

    def _blend(self, old: float, new: float, ratio: Optional[float] = None) -> float:
        if ratio is None:
            ratio = self.alpha
        return old * (1.0 - ratio) + new * ratio

