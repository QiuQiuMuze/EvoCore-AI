"""Utility classes describing the global energy economy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnergyConfig:
    """Configurable coefficients that shape the energy economy."""

    warmup_steps: int = 200
    warmup_increment: float = 0.01
    high_freq_call_threshold: float = 3.5
    high_freq_bonus: float = 0.02
    self_reward_scale: float = 0.015
    intrinsic_goal_bonus: float = 0.3
    hazard_penalty: float = 0.45
    hazard_processor_penalty: float = 0.2
    hazard_escape_bonus: float = 0.03
    resource_base_scale: float = 0.12
    resource_hit_bonus: float = 0.6
    resource_upstream_share: float = 0.2
    linger_penalty: float = 0.008
    monotony_penalty: float = 0.03
    monotony_bonus: float = 0.03
    inactivity_threshold: int = 35
    inactivity_decay: float = 0.004
    diversity_penalty: float = 0.03
    diversity_bonus: float = 0.03
    movement_penalties: Tuple[Tuple[int, int, float], ...] = (
        (3, 5, 0.06),
        (6, 8, 0.08),
        (9, 11, 0.1),
    )
    pool_primary_cap: float = 0.2
    pool_secondary_cap: float = 0.08
    pool_stability_factor: float = 0.04


class EnergyPolicy:
    """Helper that converts behavioural statistics into energy deltas."""

    def __init__(self, config: EnergyConfig | None = None) -> None:
        self.config = config or EnergyConfig()

    # ----- warmup / upkeep -------------------------------------------------
    def warmup_bonus(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.warmup_increment
        return 0.0

    # ----- intrinsic & local adjustments -----------------------------------
    def high_frequency_bonus(self, avg_calls: float) -> float:
        if avg_calls >= self.config.high_freq_call_threshold:
            return self.config.high_freq_bonus
        return 0.0

    def scale_self_reward(self, reward_signal: float) -> float:
        if reward_signal <= 0.0:
            return 0.0
        return reward_signal * self.config.self_reward_scale

    def intrinsic_completion_bonus(self) -> float:
        return self.config.intrinsic_goal_bonus

    # ----- hazards ---------------------------------------------------------
    def hazard_penalties(self) -> Tuple[float, float]:
        return (self.config.hazard_penalty, self.config.hazard_processor_penalty)

    def hazard_escape_bonus(self) -> float:
        return self.config.hazard_escape_bonus

    # ----- resources -------------------------------------------------------
    def resource_base_reward(self, proximity: float) -> float:
        closeness = max(1.0 - proximity, 0.0)
        return max(closeness * self.config.resource_base_scale, 0.0)

    def resource_hit_bonus(self) -> float:
        return self.config.resource_hit_bonus

    def resource_upstream_share(self) -> float:
        return self.config.resource_upstream_share

    # ----- behavioural penalties ------------------------------------------
    def linger_penalty(self) -> float:
        return self.config.linger_penalty

    def monotony_penalty(self) -> float:
        return self.config.monotony_penalty

    def monotony_bonus(self) -> float:
        return self.config.monotony_bonus

    def diversity_penalty(self) -> float:
        return self.config.diversity_penalty

    def diversity_bonus(self) -> float:
        return self.config.diversity_bonus

    def inactivity_threshold(self) -> int:
        return self.config.inactivity_threshold

    def inactivity_decay(self) -> float:
        return self.config.inactivity_decay

    def movement_penalty(self, manhattan: int) -> float:
        for low, high, penalty in self.config.movement_penalties:
            if low <= manhattan < high:
                return penalty
        return 0.0

    # ----- pool distribution -----------------------------------------------
    def pool_caps(self) -> Tuple[float, float]:
        return (self.config.pool_primary_cap, self.config.pool_secondary_cap)

    def pool_stability_factor(self) -> float:
        return self.config.pool_stability_factor

    # ----- helpers ---------------------------------------------------------
    def describe(self) -> dict:
        """Return a serialisable snapshot for debugging/telemetry."""
        return {
            "warmup_steps": self.config.warmup_steps,
            "warmup_increment": self.config.warmup_increment,
            "high_freq_call_threshold": self.config.high_freq_call_threshold,
            "high_freq_bonus": self.config.high_freq_bonus,
            "self_reward_scale": self.config.self_reward_scale,
            "intrinsic_goal_bonus": self.config.intrinsic_goal_bonus,
            "hazard_penalty": self.config.hazard_penalty,
            "hazard_processor_penalty": self.config.hazard_processor_penalty,
            "hazard_escape_bonus": self.config.hazard_escape_bonus,
            "resource_base_scale": self.config.resource_base_scale,
            "resource_hit_bonus": self.config.resource_hit_bonus,
            "resource_upstream_share": self.config.resource_upstream_share,
            "linger_penalty": self.config.linger_penalty,
            "monotony_penalty": self.config.monotony_penalty,
            "monotony_bonus": self.config.monotony_bonus,
            "inactivity_threshold": self.config.inactivity_threshold,
            "inactivity_decay": self.config.inactivity_decay,
            "diversity_penalty": self.config.diversity_penalty,
            "diversity_bonus": self.config.diversity_bonus,
            "movement_penalties": list(self.config.movement_penalties),
            "pool_primary_cap": self.config.pool_primary_cap,
            "pool_secondary_cap": self.config.pool_secondary_cap,
            "pool_stability_factor": self.config.pool_stability_factor,
        }
