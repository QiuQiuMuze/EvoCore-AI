from typing import Dict, Tuple

import torch

from adaptive_guidance import ASSIGNMENT_SELF
from emitter_actions import ACTION_BLOCK, ACTION_HACK_DEFENSE


def run_emitter_actions(graph):
    size = graph.env.size
    seq_len = size * size
    raw = graph.env.get_state_tensor()
    state_tensor = raw.view(1, -1).to(graph.device)
    hack_maps = []
    for t in graph.env.attack_types:
        mask = torch.zeros_like(graph.env.privilege_level, dtype=torch.float32, device=graph.device)
        for (hx, hy), info in graph.env.hacks.items():
            if info.get("type") == t:
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
    critic_loss_b = torch.nn.functional.mse_loss(value_b, reward_tensor)
    entropy_b = torch.distributions.Categorical(logits=logits_b).entropy().mean()
    total_loss = actor_loss_b + 0.5 * critic_loss_b - graph.entropy_coef * entropy_b
    graph.policy_optimizer.zero_grad()
    graph.value_optimizer.zero_grad()
    total_loss.backward()
    graph.policy_optimizer.step()
    graph.value_optimizer.step()


def decode_action_from_output(graph, unit, output_vec: torch.Tensor) -> Dict:
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
        else:
            return {"type": "move", "target": (tx, ty)}
    elif act_type == 1:
        return {"type": ACTION_BLOCK, "target": (tx, ty)}
    else:
        return {"type": ACTION_HACK_DEFENSE, "target": (tx, ty)}


def argmax_position(graph, output_vec: torch.Tensor) -> Tuple[int, int]:
    flat_idx = torch.argmax(output_vec).item()
    size = graph.env.size
    y, x = divmod(flat_idx, size)
    return x, y
