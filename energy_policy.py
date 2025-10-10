"""Utility classes describing the global energy economy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EnergyConfig:
    """Configurable coefficients that shape the energy economy."""

    warmup_steps: int = 200
    warmup_increment: float = 0.012
    high_freq_call_threshold: float = 3.2
    high_freq_bonus: float = 0.025
    self_reward_scale: float = 0.018
    intrinsic_goal_bonus: float = 0.34
    hazard_penalty: float = 0.33
    hazard_processor_penalty: float = 0.11
    hazard_escape_bonus: float = 0.06
    resource_base_scale: float = 0.18
    resource_hit_bonus: float = 0.72
    resource_upstream_share: float = 0.24
    linger_penalty: float = 0.006
    monotony_penalty: float = 0.03
    monotony_bonus: float = 0.03
    inactivity_threshold: int = 35
    inactivity_decay: float = 0.003
    diversity_penalty: float = 0.025
    diversity_bonus: float = 0.035
    movement_penalties: Tuple[Tuple[int, int, float], ...] = (
        (4, 6, 0.05),
        (6, 9, 0.07),
        (9, 12, 0.09),
    )
    pool_primary_cap: float = 0.28
    pool_secondary_cap: float = 0.12
    pool_stability_factor: float = 0.06


class EnergyPolicy:
    """Helper that converts behavioural statistics into energy deltas."""

    def __init__(self, config: EnergyConfig | None = None) -> None:
        self.config = config or EnergyConfig()
        self._resource_share = self.config.resource_upstream_share
        self._hazard_escape = self.config.hazard_escape_bonus
        self._success_ratio = 0.5
        # Processor:Emitter 理想目标 = 2:1，记录当前偏差用于能量调整
        self._processor_balance = 1.0

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
        success = float(max(0.0, min(1.0, self._success_ratio)))
        primary = self.config.hazard_penalty * (0.85 + 0.3 * success)
        secondary = self.config.hazard_processor_penalty * (0.75 + 0.4 * success)

        # 若 processor 短缺，则降低其额外惩罚，避免雪上加霜
        shortage = max(0.0, 1.0 - float(self._processor_balance))
        if shortage > 0.0:
            secondary *= 1.0 - min(0.45, 0.35 + 0.35 * shortage)
            secondary = max(0.05, secondary)
        return (primary, secondary)

    def hazard_escape_bonus(self) -> float:
        return self._hazard_escape

    # ----- resources -------------------------------------------------------
    def resource_base_reward(self, proximity: float) -> float:
        closeness = max(1.0 - proximity, 0.0)
        return max(closeness * self.config.resource_base_scale, 0.0)

    def resource_hit_bonus(self) -> float:
        return self.config.resource_hit_bonus

    def resource_upstream_share(self) -> float:
        return self._resource_share

    def update_environment_feedback(
        self,
        *,
        reward_hits: int,
        danger_hits: int,
        processor_count: int,
        emitter_count: int,
        exploration_ratio: Optional[float] = None,
        last_cycle_success: Optional[float] = None,
    ) -> None:
        total = max(1, reward_hits + danger_hits)
        success = reward_hits / total
        if last_cycle_success is not None:
            success = 0.7 * success + 0.3 * float(last_cycle_success)
        self._success_ratio = float(max(0.0, min(1.0, success)))

        danger_pressure = 1.0 - self._success_ratio
        base_share = self.config.resource_upstream_share * (1.0 + 0.25 * danger_pressure)

        target_balance = 2.0  # processor : emitter = 2 : 1
        balance = processor_count / max(1, emitter_count)
        normalized = balance / target_balance if target_balance > 0 else 1.0
        normalized = float(max(0.2, min(normalized, 3.0)))
        if normalized < 1.0:
            base_share *= 1.0 + (1.0 - normalized) * 0.55
        else:
            base_share *= 1.0 - min(normalized - 1.0, 1.5) * 0.25

        self._processor_balance = normalized

        if exploration_ratio is not None:
            ratio = float(max(0.0, min(1.0, exploration_ratio)))
            base_share *= 0.9 + 0.2 * ratio

        self._resource_share = float(min(max(base_share, 0.16), 0.36))

        escape = self.config.hazard_escape_bonus * (1.0 + 0.6 * danger_pressure)
        self._hazard_escape = float(min(max(escape, self.config.hazard_escape_bonus * 0.8), self.config.hazard_escape_bonus * 1.8))

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
            "dynamic_resource_share": self._resource_share,
            "processor_balance_factor": self._processor_balance,
            "dynamic_hazard_escape_bonus": self._hazard_escape,
            "dynamic_success_ratio": self._success_ratio,
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
