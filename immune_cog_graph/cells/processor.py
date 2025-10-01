"""Processor cell controller and helpers."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from ..graph import ImmuneCogGraph


def _prepare_before_update_impl(graph: "ImmuneCogGraph", unit, full_state, expected_input):
    if unit.input_size < expected_input:
        graph.expand_unit_dim(unit, expected_input)

    if unit.role == "emitter":
        vec = getattr(graph, "tv_cached", graph.target_vector)
        unit.goal_vec = vec
        unit.current_hazard_xy = getattr(graph, "current_hazard_xy", None)

    unit.global_emitter_count = sum(1 for u in graph.units if u.role == "emitter")
    incoming = graph.reverse_connections.get(unit.id, ())
    unit.recent_calls = len(incoming)

    if unit.role == "sensor":
        return full_state

    if incoming:
        weighted, total_w = [], 0.0
        for uid in incoming:
            strength = graph.connections.get(uid, {}).get(unit.id, 0.0)
            if strength == 0.0 or uid not in graph.unit_map:
                continue
            out = graph.unit_map[uid].get_output().squeeze(0)
            if out.shape[0] < graph.processor_hidden_size:
                out = F.pad(out, (0, graph.processor_hidden_size - out.shape[0]))
            elif out.shape[0] > graph.processor_hidden_size:
                out = out[: graph.processor_hidden_size]
            weighted.append(out * strength)
            total_w += strength

        if total_w > 0:
            agg = torch.stack(weighted, dim=0).sum(dim=0) / total_w
            return agg.unsqueeze(0)

    return torch.zeros(1, expected_input, device=graph.device)


def _finalize_unit_update_impl(graph: "ImmuneCogGraph", unit, full_state, extra_dict, pending_dict, allow_clone=True):
    vec = getattr(unit, "last_output", None)
    if vec is not None:
        need = vec.shape[-1]
        graph.expand_unit_dim(unit, need)

        def first_linear(root):
            if isinstance(root, nn.Module):
                for module in root.modules():
                    if isinstance(module, nn.Linear):
                        return module
            return None

        lin = first_linear(unit) or first_linear(getattr(unit, "function", None))
        layer_in = lin.in_features if lin else need

        if need < layer_in:
            vec = F.pad(vec, (0, layer_in - need))
        elif need > layer_in:
            vec = vec[..., :layer_in]
        unit.last_output = vec

    return super(type(graph), graph)._finalize_unit_update(
        unit, full_state, extra_dict, pending_dict, allow_clone=allow_clone
    )


def _processor_forward_impl(graph: "ImmuneCogGraph", sensor_out: torch.Tensor):
    if sensor_out.dim() == 1:
        sensor_out = sensor_out.unsqueeze(0)

    if graph.use_shared_unit_nets:
        if not hasattr(graph, "_compiled_processor_net"):
            try:
                graph._compiled_processor_net = torch.compile(
                    graph.processor_net, fullgraph=False, dynamic=True
                )
            except Exception:
                graph._compiled_processor_net = graph.processor_net
        out = graph._compiled_processor_net(sensor_out)

        if not hasattr(graph, "_hotspot_queue"):
            graph._hotspot_queue = deque(maxlen=20)
        for proc_unit in (u for u in graph.units if u.role == "processor"):
            for pos in getattr(proc_unit, "hotspots", set()):
                if pos not in graph._hotspot_queue:
                    graph._hotspot_queue.append(pos)
            proc_unit.hotspots = set()
        return out

    proc_units = [u for u in graph.units if u.role == "processor"]
    outs: list[torch.Tensor] = []
    for idx, proc_unit in enumerate(proc_units):
        inp = sensor_out[idx:idx + 1]
        net = proc_unit.processor_net if hasattr(proc_unit, "processor_net") else graph.processor_net
        outs.append(net(inp))
    proc_out = torch.cat(outs, dim=0) if outs else None

    if not hasattr(graph, "_hotspot_queue"):
        graph._hotspot_queue = deque(maxlen=20)
    for proc_unit in proc_units:
        for pos in getattr(proc_unit, "hotspots", set()):
            if pos not in graph._hotspot_queue:
                graph._hotspot_queue.append(pos)
        proc_unit.hotspots = set()
    return proc_out


def _expand_unit_dim_impl(graph: "ImmuneCogGraph", unit, new_in: int):
    def first_linear(root):
        if isinstance(root, nn.Module):
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    return module
        return None

    def last_linear(root):
        if isinstance(root, nn.Module):
            for module in reversed(list(root.modules())):
                if isinstance(module, nn.Linear):
                    return module
        return None

    net_root = unit if isinstance(unit, nn.Module) else getattr(unit, "function", None)
    lin1 = first_linear(net_root)
    if lin1 is None:
        unit.input_size = max(getattr(unit, "input_size", 0), new_in)
        return

    if lin1.in_features >= new_in:
        unit.input_size = lin1.in_features
        return

    old_w = lin1.weight.data
    old_b = lin1.bias.data if lin1.bias is not None else None
    out_f, old_in = old_w.shape
    new_l1 = nn.Linear(new_in, out_f, bias=lin1.bias is not None).to(old_w.device)
    new_l1.weight.data[:, :old_in].copy_(old_w)
    if old_b is not None:
        new_l1.bias.data.copy_(old_b)

    def _replace_first(root):
        for name, mod in root.named_children():
            if mod is lin1:
                setattr(root, name, new_l1)
                return
            _replace_first(mod)

    _replace_first(net_root)

    lin_last = last_linear(net_root)
    if lin_last is not None and lin_last.out_features < new_in:
        w_last = lin_last.weight.data
        b_last = lin_last.bias.data if lin_last.bias is not None else None
        old_out, old_in = w_last.shape
        new_last = nn.Linear(old_in, new_in, bias=lin_last.bias is not None).to(w_last.device)
        new_last.weight.data[:old_out, :].copy_(w_last)
        if b_last is not None:
            new_last.bias.data[:old_out].copy_(b_last)

        def _replace_last(root):
            for name, mod in root.named_children():
                if mod is lin_last:
                    setattr(root, name, new_last)
                    return
                _replace_last(mod)

        _replace_last(net_root)

    unit.input_size = new_in
    unit.l1 = new_l1


class ProcessorCellController:
    """Encapsulates processor-specific behaviours."""

    def __init__(self, graph: "ImmuneCogGraph") -> None:
        self.graph = graph

    def prepare_before_update(self, unit, full_state, expected_input):
        return _prepare_before_update_impl(self.graph, unit, full_state, expected_input)

    def finalize_unit_update(self, unit, full_state, extra_dict, pending_dict, allow_clone=True):
        return _finalize_unit_update_impl(
            self.graph, unit, full_state, extra_dict, pending_dict, allow_clone=allow_clone
        )

    def forward(self, sensor_out: torch.Tensor):
        return _processor_forward_impl(self.graph, sensor_out)

    def expand_unit_dim(self, unit, new_in: int):
        return _expand_unit_dim_impl(self.graph, unit, new_in)
