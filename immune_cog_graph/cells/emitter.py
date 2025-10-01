"""Emitter cell controller and helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
from torch.distributions import Categorical

from adaptive_guidance import ASSIGNMENT_LEARNED, ASSIGNMENT_SELF
from emitter_actions import ACTION_BLOCK, ACTION_HACK_DEFENSE, ACTION_QUARANTINE
from env import logger

from ..constants import HIT_BONUS, LEARNED_FACTOR, MIN_PATROL_DIST

if TYPE_CHECKING:  # pragma: no cover
    from ..graph import ImmuneCogGraph


def _assign_curiosity_goal_impl(graph: "ImmuneCogGraph", unit) -> None:
    visited = graph.env.visited_map
    age_map = graph.visit_age_map
    never_visited_mask = ~visited
    cooldown = getattr(unit, "intrinsic_cooldown", 0)
    long_time_mask = age_map >= cooldown
    candidate_mask = never_visited_mask | long_time_mask
    candidates = torch.nonzero(candidate_mask)
    if candidates.numel() == 0:
        return

    ex, ey = unit.position
    far_mask = torch.tensor(
        [abs(xx - ex) + abs(yy - ey) >= MIN_PATROL_DIST for (yy, xx) in candidates.tolist()],
        dtype=torch.bool,
        device=candidates.device,
    )
    if far_mask.any():
        candidates = candidates[far_mask]

    sel = torch.randint(0, candidates.size(0), (1,), generator=graph._rng).item()
    ty, tx = candidates[sel].tolist()
    unit.personal_goal = (tx, ty)
    unit.goal_type = "curiosity"
    unit.assignment_source = ASSIGNMENT_SELF
    unit.assignment_trace = {
        "pos": (tx, ty),
        "step": graph.current_step,
        "meta": {"strategy": "curiosity"},
        "source": ASSIGNMENT_SELF,
    }
    unit._last_intrinsic_step = graph.current_step
    logger.info(f"[好奇点] 给 emitter {unit.id} 分配新好奇点：({tx},{ty})")


def _assign_goal_impl(graph: "ImmuneCogGraph", unit) -> None:
    size = graph.env.size
    seq_len = size * size

    old_goal = getattr(unit, "personal_goal", None)
    old_type = getattr(unit, "goal_type", None)
    if old_type in ("infection", "hack") and old_goal is not None:
        if old_type == "infection" and old_goal in graph.known_infections:
            return
        if old_type == "hack" and old_goal in graph.known_hacks:
            return
        unit.personal_goal = None
        unit.goal_type = None

    threat_list: List[int] = []
    for (x_inf, y_inf) in graph.known_infections:
        threat_list.append(y_inf * size + x_inf)
    for (x_hk, y_hk) in graph.known_hacks:
        threat_list.append(y_hk * size + x_hk)

    if threat_list:
        device = graph.device
        threat_vec = torch.zeros((1, seq_len), device=device)
        indices = torch.tensor(threat_list, dtype=torch.long, device=device)
        threat_vec[0].scatter_(0, indices, 1.0)

        logits = graph.goal_net(threat_vec)
        probs = torch.softmax(logits, dim=-1)

        hack_bias = 2.0
        threat_probs = probs[0, indices].clone()
        for idx_i, flat_i in enumerate(indices.tolist()):
            cx = flat_i % size
            cy = flat_i // size
            if (cx, cy) in graph.known_hacks:
                threat_probs[idx_i] *= hack_bias

        infected_count = len(graph.known_infections)
        if infected_count == 0:
            infection_bias = 0.0
        else:
            infection_bias = 1.2 + (max(0, 10 - infected_count) * 0.06)
            infection_bias = min(infection_bias, hack_bias * 0.8)

        for i, flat_i in enumerate(indices.tolist()):
            cx, cy = flat_i % size, flat_i // size
            if (cx, cy) in graph.known_infections:
                threat_probs[i] *= infection_bias

        max_k = 5
        _, sorted_idxs_in_threat = torch.sort(threat_probs, descending=True)
        filtered_candidates: List[int] = []
        for idx_in_threat in sorted_idxs_in_threat.tolist():
            flat_idx = indices[idx_in_threat].item()
            cx, cy = flat_idx % size, flat_idx // size
            occ = sum(
                1
                for emitter in graph.units
                if emitter.role == "emitter" and getattr(emitter, "personal_goal", None) == (cx, cy)
            )
            if occ >= 3:
                continue
            filtered_candidates.append(flat_idx)
            if len(filtered_candidates) == max_k:
                break

        score_lookup = {indices[i].item(): float(threat_probs[i].item()) for i in range(indices.numel())}

        if not filtered_candidates:
            _assign_curiosity_goal_impl(graph, unit)
            return

        choice, meta = graph.adaptive_guidance.select_goal(
            emitter_id=unit.id,
            emitter_pos=unit.position,
            candidates=filtered_candidates,
            threat_scores=score_lookup,
            step=graph.current_step,
        )

        if choice is None:
            _assign_curiosity_goal_impl(graph, unit)
            return

        goal_x, goal_y = choice % size, choice // size
        if (goal_x, goal_y) in graph.known_hacks:
            unit.goal_type = "hack"
        else:
            unit.goal_type = "infection"
        unit.personal_goal = (goal_x, goal_y)
        unit.assignment_source = ASSIGNMENT_LEARNED
        unit.assignment_trace = {
            "pos": (goal_x, goal_y),
            "step": graph.current_step,
            "meta": meta,
            "source": ASSIGNMENT_LEARNED,
        }
        unit._last_intrinsic_step = graph.current_step
        logger.info(
            f"[学习调度] emitter {unit.id} 目标 → ({goal_x},{goal_y}), 策略={meta.get('strategy', 'exploit')}"
        )
        return

    if getattr(unit, "goal_type", None) is None:
        _assign_curiosity_goal_impl(graph, unit)


def _emitter_forward_impl(graph: "ImmuneCogGraph", proc_out: torch.Tensor):
    if proc_out.dim() == 1:
        proc_out = proc_out.unsqueeze(0)

    vec = proc_out[0]
    hidden = graph.emitter_hidden_size
    if vec.shape[0] < hidden:
        vec = F.pad(vec, (0, hidden - vec.shape[0]))
    elif vec.shape[0] > hidden:
        vec = vec[:hidden]

    emitters = [u for u in graph.units if u.role == "emitter"]

    if graph.use_shared_unit_nets:
        batch_in = []
        for emitter in emitters:
            graph.expand_unit_dim(emitter, hidden)
            batch_in.append(vec.unsqueeze(0))
        if not batch_in:
            return None
        batch = torch.cat(batch_in, dim=0)
        logits = graph.emitter_net(batch)
        for emitter, lg in zip(emitters, logits):
            emitter.last_output = lg.detach()
        return logits

    logits_list: List[torch.Tensor] = []
    for emitter in emitters:
        graph.expand_unit_dim(emitter, hidden)
        net = emitter.emitter_net if hasattr(emitter, "emitter_net") else graph.emitter_net
        lg = net(vec.unsqueeze(0))
        emitter.last_output = lg.detach().squeeze(0)
        logits_list.append(lg)
    if not logits_list:
        return None
    return torch.cat(logits_list, dim=0)


def _argmax_position_impl(graph: "ImmuneCogGraph", output_vec: torch.Tensor) -> Tuple[int, int]:
    flat_idx = torch.argmax(output_vec).item()
    size = graph.env.size
    y, x = divmod(flat_idx, size)
    return x, y


def _decode_action_impl(graph: "ImmuneCogGraph", unit, output_vec: torch.Tensor):
    size = graph.env.size
    seq_len = size * size
    ux, uy = unit.position

    flat = torch.argmax(output_vec).item()
    act_type = flat // seq_len
    flat_idx = flat % seq_len
    tx = flat_idx % size
    ty = flat_idx // size

    if act_type == 0:
        if abs(tx - ux) + abs(ty - uy) > 1:
            nx = ux + (1 if tx > ux else -1) if tx != ux else ux
            ny = uy + (1 if ty > uy else -1) if ty != uy else uy
            nx = max(0, min(nx, size - 1))
            ny = max(0, min(ny, size - 1))
            return {"type": "move", "target": (nx, ny)}
        return {"type": "move", "target": (tx, ty)}
    if act_type == 1:
        return {"type": ACTION_BLOCK, "target": (tx, ty)}
    return {"type": ACTION_HACK_DEFENSE, "target": (tx, ty)}


def _apply_and_reward_impl(graph: "ImmuneCogGraph", unit, action: dict) -> float:
    size = graph.env.size
    H, W = graph.env.infected_map.shape

    orig_inf = graph.env.infected_map.clone()
    orig_priv = graph.env.privilege_level.clone()
    orig_stealth = graph.env.hack_strength.clone()

    raw_state = graph.env.get_state_tensor().cpu()
    state_tensor = raw_state.view(1, -1).to(graph.device)
    goal_flat = graph.tv_cached.view(1, -1)

    hack_maps = []
    for attack_type in graph.env.attack_types:
        mask = torch.zeros_like(graph.env.privilege_level, dtype=torch.float32, device=graph.device)
        for (hx, hy), info in graph.env.hacks.items():
            if info.get("type") == attack_type:
                mask[hy, hx] = 1.0
        hack_maps.append(mask)
    hack_flat = torch.stack(hack_maps, dim=0).view(1, -1)

    full_state = torch.cat([state_tensor, hack_flat, goal_flat], dim=1)
    with torch.no_grad():
        graph.value_head(full_state)

    kernel3 = torch.ones((1, 1, 3, 3), device=graph.device, dtype=torch.float32)

    hits = 0
    total_reward = 0.0

    if action["type"] == ACTION_BLOCK:
        cx0, cy0 = action["target"]
        seed_mask = torch.zeros_like(orig_inf, dtype=torch.float32, device=graph.device)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xx = cx0 + dx
                yy = cy0 + dy
                if 0 <= xx < W and 0 <= yy < H and orig_inf[yy, xx] > 0.04:
                    seed_mask[yy, xx] = 1.0

        seed_bin = seed_mask.unsqueeze(0).unsqueeze(0)
        neighbor_from_seed = conv2d(seed_bin, kernel3, padding=1)[0, 0]

        mask_to_clear = (neighbor_from_seed > 0) & (orig_inf > 1e-5)
        ys, xs = torch.nonzero(mask_to_clear, as_tuple=True)
        all_to_clear = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            info = graph.env.attacks.get((x, y), None)
            virus_type = "扩散点" if info is None else info.get("type", "virus")
            all_to_clear.append((x, y, virus_type))

        hits = len(all_to_clear)
        if hits > 0:
            src = getattr(unit, "assignment_source", ASSIGNMENT_SELF)
            if src not in graph.kill_stats:
                graph.kill_stats[src] = 0
            unit.cleared_positions = set()
            for (x, y, virus_type) in all_to_clear:
                graph.kill_stats[src] += 1
                bucket_v = graph.virus_kill_stats_by_type.setdefault(
                    virus_type, {ASSIGNMENT_SELF: 0, ASSIGNMENT_LEARNED: 0}
                )
                if src not in bucket_v:
                    bucket_v[src] = 0
                bucket_v[src] += 1
                unit.cleared_positions.add((x, y))

            for (x, y, _) in all_to_clear:
                graph.emitter_actions.perform({"type": ACTION_BLOCK, "target": (x, y)})

            curr_inf_count = int((graph.env.infected_map > 0.04).sum().item())
            scale = 1.0 / (1 + curr_inf_count)
            factor = LEARNED_FACTOR if src == ASSIGNMENT_LEARNED else 0.1

            weighted_hits = float(len(all_to_clear))
            if not any(vt == "扩散点" for (_, _, vt) in all_to_clear):
                multiplier = 5.0
            elif all(vt == "扩散点" for (_, _, vt) in all_to_clear):
                multiplier = 3.0
            else:
                multiplier = 1.0

            reward = HIT_BONUS * weighted_hits * factor * scale * multiplier
            unit.energy += reward
            unit.meta.record(action="defense", reward=reward)
            total_reward += reward

            trace = getattr(unit, "assignment_trace", None)
            if trace and trace.get("source") == ASSIGNMENT_LEARNED:
                latency = graph.current_step - trace.get("step", graph.current_step)
                target_pos = trace.get("pos", getattr(unit, "personal_goal", (0, 0)))
                graph.adaptive_guidance.register_feedback(target_pos, reward, latency)

            infected_points = torch.nonzero(graph.env.infected_map > 0.04).tolist()
            if infected_points and hasattr(unit, "position"):
                ux, uy = unit.position
                dists = [math.hypot(ux - px, uy - py) for (py, px) in infected_points]
                bonus = max(0, (5 - min(dists)) / 5) * 0.05 * hits
                unit.energy += bonus
                unit.meta.record(action="distance_bonus", reward=bonus)

            for pid in graph.reverse_connections.get(unit.id, ()):
                processor = graph.unit_map.get(pid)
                if processor and processor.role == "processor":
                    fb = reward * 0.6
                    processor.energy += fb
                    processor.meta.record(action="upstream", reward=fb)

    elif action["type"] == ACTION_HACK_DEFENSE:
        x0, y0 = action["target"]
        hack_seed = torch.zeros_like(orig_priv, dtype=torch.float32, device=graph.device)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xx = x0 + dx
                yy = y0 + dy
                if 0 <= xx < W and 0 <= yy < H and (
                    orig_priv[yy, xx] > 0.04 or orig_stealth[yy, xx] > 0.04
                ):
                    hack_seed[yy, xx] = 1.0

        hack_seed_bin = hack_seed.unsqueeze(0).unsqueeze(0)
        neighbor_hack = conv2d(hack_seed_bin, kernel3, padding=1)[0, 0]

        valid_mask = (neighbor_hack > 0) & ((orig_priv > 0.04) | (orig_stealth > 0.04))
        ys, xs = torch.nonzero(valid_mask, as_tuple=True)
        all_to_clear = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            info = graph.env.hacks.get((x, y), None)
            hack_type = "传播点" if info is None else info.get("type", "unknown")
            all_to_clear.append((x, y, hack_type))

        hits = len(all_to_clear)
        if hits > 0:
            src = getattr(unit, "assignment_source", ASSIGNMENT_SELF)
            if src not in graph.hack_kill_stats:
                graph.hack_kill_stats[src] = 0
            unit.cleared_hack = set()
            for (x, y, hack_type) in all_to_clear:
                graph.hack_kill_stats[src] += 1
                bucket_h = graph.hack_kill_stats_by_type.setdefault(
                    hack_type, {ASSIGNMENT_SELF: 0, ASSIGNMENT_LEARNED: 0}
                )
                if src not in bucket_h:
                    bucket_h[src] = 0
                bucket_h[src] += 1
                unit.cleared_hack.add((x, y))

            for (x, y, _) in all_to_clear:
                graph.emitter_actions.perform({"type": ACTION_HACK_DEFENSE, "target": (x, y)})

            curr_hack_count = int((graph.env.privilege_level > 0.04).sum().item())
            scale = 1.0 / (1 + curr_hack_count)
            factor = LEARNED_FACTOR if src == ASSIGNMENT_LEARNED else 0.1

            reward = HIT_BONUS * hits * factor * scale
            unit.energy += reward
            unit.meta.record(action="hack_defense", reward=reward)
            total_reward += reward

    elif action["type"] == ACTION_QUARANTINE:
        target = action.get("target")
        if target:
            graph.emitter_actions.perform(action)
        total_reward = 0.0

    return total_reward


def _run_emitter_actions_impl(graph: "ImmuneCogGraph") -> None:
    size = graph.env.size
    seq_len = size * size

    raw = graph.env.get_state_tensor()
    state_tensor = raw.view(1, -1).to(graph.device)

    hack_maps = []
    for attack_type in graph.env.attack_types:
        mask = torch.zeros_like(graph.env.privilege_level, dtype=torch.float32, device=graph.device)
        for (hx, hy), info in graph.env.hacks.items():
            if info.get("type") == attack_type:
                mask[hy, hx] = 1.0
        hack_maps.append(mask)
    hack_flat = torch.stack(hack_maps, dim=0).view(1, -1)
    goal_flat = graph.tv_cached.view(1, -1)

    batch_units = []
    batch_flats = []
    batch_rewards = []

    for unit in graph.units:
        if unit.role != "emitter" or not hasattr(unit, "get_output"):
            continue

        if not hasattr(unit, "assignment_source"):
            unit.assignment_source = ASSIGNMENT_SELF
        if not hasattr(unit, "assignment_trace"):
            unit.assignment_trace = None

        action_vec = unit.get_output()
        action = graph._decode_action_from_output(unit, action_vec)

        if action["type"] == "move":
            bx, by = getattr(unit, "personal_goal", action["target"])
            for _ in range(20):
                ux, uy = unit.position
                if (ux, uy) == (bx, by):
                    if getattr(unit, "goal_type", None) == "infection":
                        block_action = {"type": ACTION_BLOCK, "target": (bx, by)}
                        total_reward = graph._apply_and_reward(unit, block_action)
                        flat_idx_cell = by * size + bx
                        flat_for_rl = 1 * seq_len + flat_idx_cell
                        batch_units.append(unit)
                        batch_flats.append(flat_for_rl)
                        batch_rewards.append(total_reward)
                    elif getattr(unit, "goal_type", None) == "hack":
                        hack_action = {"type": ACTION_HACK_DEFENSE, "target": (bx, by)}
                        total_reward = graph._apply_and_reward(unit, hack_action)
                        flat_idx_cell = by * size + bx
                        flat_for_rl = 2 * seq_len + flat_idx_cell
                        batch_units.append(unit)
                        batch_flats.append(flat_for_rl)
                        batch_rewards.append(total_reward)
                    unit.personal_goal = None
                    unit.goal_type = None
                    break

                if abs(bx - ux) >= abs(by - uy):
                    nx = ux + (1 if bx > ux else -1)
                    ny = uy
                else:
                    nx = ux
                    ny = uy + (1 if by > uy else -1)
                nx = max(0, min(nx, size - 1))
                ny = max(0, min(ny, size - 1))
                unit.position = (nx, ny)

                if (nx, ny) == (bx, by):
                    if getattr(unit, "goal_type", None) == "infection":
                        block_action = {"type": ACTION_BLOCK, "target": (bx, by)}
                        total_reward = graph._apply_and_reward(unit, block_action)
                        flat_idx_cell = by * size + bx
                        flat_for_rl = 1 * seq_len + flat_idx_cell
                        batch_units.append(unit)
                        batch_flats.append(flat_for_rl)
                        batch_rewards.append(total_reward)
                    elif getattr(unit, "goal_type", None) == "hack":
                        hack_action = {"type": ACTION_HACK_DEFENSE, "target": (bx, by)}
                        total_reward = graph._apply_and_reward(unit, hack_action)
                        flat_idx_cell = by * size + bx
                        flat_for_rl = 2 * seq_len + flat_idx_cell
                        batch_units.append(unit)
                        batch_flats.append(flat_for_rl)
                        batch_rewards.append(total_reward)
                    unit.personal_goal = None
                    unit.goal_type = None
                    break
            continue

        if action["type"] in (ACTION_BLOCK, ACTION_HACK_DEFENSE):
            total_reward = graph._apply_and_reward(unit, action)
            bx, by = action["target"]
            flat_idx_cell = by * size + bx
            act_type = 1 if action["type"] == ACTION_BLOCK else 2
            flat_for_rl = act_type * seq_len + flat_idx_cell
            batch_units.append(unit)
            batch_flats.append(flat_for_rl)
            batch_rewards.append(total_reward)

    if not batch_units:
        return

    B = len(batch_units)
    flat_tensor = torch.tensor(batch_flats, dtype=torch.long, device=graph.device)
    reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=graph.device)

    fs_env = state_tensor.expand(B, -1)
    fs_hack = hack_flat.expand(B, -1)
    fs_goal = goal_flat.expand(B, -1)
    full_state_batch = torch.cat([fs_env, fs_hack, fs_goal], dim=1)

    sensor_out_b = graph._sensor_net(full_state_batch)
    proc_out_b = graph.processor_net(sensor_out_b)
    logits_b = graph.emitter_net(proc_out_b)
    probs_b = torch.softmax(logits_b, dim=-1)

    idx_batch = torch.arange(B, device=graph.device)
    logp_b = torch.log(probs_b[idx_batch, flat_tensor] + 1e-8)

    value_b = graph.value_head(full_state_batch).squeeze(-1)
    advantage_b = reward_tensor - value_b.detach()

    actor_loss_b = -(logp_b * advantage_b).mean()
    critic_loss_b = F.mse_loss(value_b, reward_tensor)
    entropy_b = Categorical(logits=logits_b).entropy().mean()

    total_loss = actor_loss_b + 0.5 * critic_loss_b - graph.entropy_coef * entropy_b

    graph.policy_optimizer.zero_grad()
    graph.value_optimizer.zero_grad()
    total_loss.backward()
    graph.policy_optimizer.step()
    graph.value_optimizer.step()


class EmitterCellController:
    """Encapsulates emitter-specific behaviours."""

    def __init__(self, graph: "ImmuneCogGraph") -> None:
        self.graph = graph

    def assign_goal(self, unit) -> None:
        _assign_goal_impl(self.graph, unit)

    def assign_curiosity_goal(self, unit) -> None:
        _assign_curiosity_goal_impl(self.graph, unit)

    def forward(self, proc_out: torch.Tensor):
        return _emitter_forward_impl(self.graph, proc_out)

    def argmax_position(self, output_vec: torch.Tensor) -> Tuple[int, int]:
        return _argmax_position_impl(self.graph, output_vec)

    def decode_action(self, unit, output_vec: torch.Tensor):
        return _decode_action_impl(self.graph, unit, output_vec)

    def apply_and_reward(self, unit, action: dict) -> float:
        return _apply_and_reward_impl(self.graph, unit, action)

    def run_actions(self) -> None:
        _run_emitter_actions_impl(self.graph)
