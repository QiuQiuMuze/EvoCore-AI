"""Adaptive threat generation models for EvoCore-AI.

These lightweight evolution modules mutate base virus and hacker
profiles, track feedback from ImmuneCogGraph, and continuously adjust
spawn parameters so that the immune system must learn instead of relying
on hard-coded guidance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

import torch


@dataclass
class ThreatProfile:
    """Describes a single evolved threat instance."""

    name: str
    params: Dict[str, float]
    base_name: str
    novelty: float
    target: Optional[Tuple[int, int]] = None
    metadata: Dict[str, float] = field(default_factory=dict)


class _EMAStat:
    """Tracks exponential moving averages for threat outcomes."""

    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.avg_reward = 0.0
        self.avg_lifetime = 0.0
        self.num_samples = 0

    def update(self, reward: float, lifetime: float) -> None:
        self.num_samples += 1
        beta = 1.0 - self.momentum
        self.avg_reward = self.momentum * self.avg_reward + beta * reward
        self.avg_lifetime = self.momentum * self.avg_lifetime + beta * lifetime

    def score(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.avg_reward + 0.1 * self.avg_lifetime


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class VirusEvolutionModel:
    """Evolves new virus behaviour profiles from base templates."""

    def __init__(
        self,
        base_catalog: Dict[str, Dict[str, float]],
        device: torch.device | str = "cpu",
        mutation_rate: float = 0.12,
    ) -> None:
        self.device = torch.device(device)
        self.base_catalog = {k: dict(v) for k, v in base_catalog.items()}
        self.mutation_rate = mutation_rate
        self.stats: Dict[str, _EMAStat] = {name: _EMAStat() for name in base_catalog}
        self._rng = random.Random()
        self._mutation_id = 0

    def register_feedback(self, profile_name: str, reward: float, lifetime: float) -> None:
        if profile_name not in self.stats:
            self.stats[profile_name] = _EMAStat()
        self.stats[profile_name].update(reward, lifetime)

    def population_pressure(self, active_ratio: float, step_ratio: float) -> float:
        return _clamp(0.2 + 0.6 * active_ratio + 0.4 * step_ratio, 0.1, 1.5)

    def _select_base(self) -> str:
        names = list(self.stats)
        weights = []
        for name in names:
            stat = self.stats[name]
            base_weight = 1.0
            if stat.num_samples > 0:
                base_weight += stat.score()
            weights.append(max(base_weight, 0.05))
        total = sum(weights)
        if total <= 0:
            return self._rng.choice(names)
        probs = [w / total for w in weights]
        return self._rng.choices(names, probs)[0]

    def _mutate_value(self, value: float, pressure: float, scale: float = 1.0) -> float:
        noise = self._rng.gauss(0.0, self.mutation_rate * pressure * scale)
        return value * (1.0 + noise)

    def _mutate_additive(self, value: float, pressure: float, scale: float = 1.0) -> float:
        noise = self._rng.gauss(0.0, self.mutation_rate * pressure * scale)
        return value + noise

    def sample_profile(
        self,
        step: int,
        active_infections: int,
        grid_size: int,
    ) -> ThreatProfile:
        base_name = self._select_base()
        base = self.base_catalog.get(base_name, {})
        pressure = self.population_pressure(
            active_ratio=active_infections / max(grid_size * grid_size, 1),
            step_ratio=min(step / 5000.0, 1.0),
        )

        params = dict(base)
        params["spread_prob"] = _clamp(
            self._mutate_value(params.get("spread_prob", 0.1), pressure, scale=0.5),
            0.01,
            0.95,
        )
        params["stealth"] = _clamp(
            self._mutate_additive(params.get("stealth", 0.0), pressure, scale=0.4),
            0.0,
            0.99,
        )
        params["power"] = _clamp(
            self._mutate_value(params.get("power", 1.0), pressure, scale=0.3),
            0.2,
            5.0,
        )

        if params.get("burst", False) or self._rng.random() < 0.2 * pressure:
            params["burst"] = True
            base_chance = params.get("burst_chance", 0.25)
            params["burst_chance"] = _clamp(
                self._mutate_value(base_chance, pressure, scale=0.4),
                0.05,
                0.95,
            )
            base_area = params.get("burst_area", 2)
            params["burst_area"] = int(
                _clamp(round(self._mutate_additive(base_area, pressure, scale=1.0)), 1, 5)
            )
        else:
            params["burst"] = False
            params.pop("burst_chance", None)
            params.pop("burst_area", None)

        novelty = 0.0
        name = base_name
        if self._rng.random() < 0.35 * pressure:
            name = f"{base_name}_m{self._mutation_id}"
            self._mutation_id += 1
            novelty = 1.0
            self.base_catalog.setdefault(name, dict(params))
            self.stats.setdefault(name, _EMAStat())

        return ThreatProfile(name=name, params=params, base_name=base_name, novelty=novelty)


class HackerEvolutionModel:
    """Evolves hacker behaviours with contextual targeting."""

    def __init__(
        self,
        base_catalog: Dict[str, Dict[str, float]],
        device: torch.device | str = "cpu",
        mutation_rate: float = 0.18,
    ) -> None:
        self.device = torch.device(device)
        self.base_catalog = {k: dict(v) for k, v in base_catalog.items()}
        self.stats: Dict[str, _EMAStat] = {name: _EMAStat() for name in base_catalog}
        self.mutation_rate = mutation_rate
        self._rng = random.Random()
        self._mutation_id = 0

    def register_feedback(self, profile_name: str, reward: float, lifetime: float) -> None:
        if profile_name not in self.stats:
            self.stats[profile_name] = _EMAStat()
        self.stats[profile_name].update(reward, lifetime)

    def _select_base(self) -> str:
        names = list(self.stats)
        weights = []
        for name in names:
            stat = self.stats[name]
            weight = 1.0 + stat.score()
            weights.append(max(weight, 0.05))
        total = sum(weights)
        if total <= 0:
            return self._rng.choice(names)
        probs = [w / total for w in weights]
        return self._rng.choices(names, probs)[0]

    def _mutate(self, value: float, pressure: float, scale: float = 1.0) -> float:
        noise = self._rng.gauss(0.0, self.mutation_rate * pressure * scale)
        return value * (1.0 + noise)

    def _mutate_add(self, value: float, pressure: float, scale: float = 1.0) -> float:
        noise = self._rng.gauss(0.0, self.mutation_rate * pressure * scale)
        return value + noise

    def sample_batch(
        self,
        step: int,
        vulnerability: torch.Tensor,
        login_failures: torch.Tensor,
        privilege: torch.Tensor,
        batch_size: int = 3,
    ) -> List[ThreatProfile]:
        grid_size = vulnerability.shape[0]
        heat = (
            vulnerability.to(self.device)
            + 0.4 * login_failures.to(self.device)
            + 0.6 * privilege.to(self.device)
        )
        flat_heat = heat.view(-1)
        if flat_heat.numel() == 0:
            return []
        topk = min(batch_size, flat_heat.numel())
        values, indices = torch.topk(flat_heat, topk)
        pressure = _clamp(float(values.mean().item()), 0.1, 5.0)

        profiles: List[ThreatProfile] = []
        for idx in indices.tolist():
            x = idx % grid_size
            y = idx // grid_size
            base_name = self._select_base()
            base = self.base_catalog.get(base_name, {})

            params = dict(base)
            params["spawn_prob"] = _clamp(
                self._mutate(params.get("spawn_prob", 0.01), pressure, scale=0.5),
                0.001,
                0.35,
            )
            params["stealth"] = _clamp(
                self._mutate_add(params.get("stealth", 0.0), pressure, scale=0.3),
                0.0,
                0.99,
            )
            params["impact"] = _clamp(
                self._mutate(params.get("impact", 1.0), pressure, scale=0.4),
                0.2,
                6.0,
            )
            params["max_fail"] = max(
                1.0,
                self._mutate(params.get("max_fail", 3.0), pressure, scale=0.2),
            )

            novelty = 0.0
            name = base_name
            if self._rng.random() < 0.3 * pressure:
                name = f"{base_name}_m{self._mutation_id}"
                self._mutation_id += 1
                novelty = 1.0
                self.base_catalog.setdefault(name, dict(params))
                self.stats.setdefault(name, _EMAStat())

            profiles.append(
                ThreatProfile(
                    name=name,
                    params=params,
                    base_name=base_name,
                    novelty=novelty,
                    target=(x, y),
                )
            )
        return profiles


__all__ = [
    "ThreatProfile",
    "VirusEvolutionModel",
    "HackerEvolutionModel",
]
