"""Sensor cell controller and helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from ..graph import ImmuneCogGraph


def _run_scans_impl(graph: "ImmuneCogGraph") -> None:
    """Run distributed threshold scans across all sensor units."""
    thr = 0.04
    graph.known_infections.clear()
    graph.known_hacks.clear()

    sensors = [u for u in graph.units if getattr(u, "role", None) == "sensor"]
    num_sensors = len(sensors)
    H, W = graph.env.infected_map.shape
    total_cells = H * W

    if num_sensors == 0:
        inf_mask = graph.env.infected_map > thr
        ys_inf, xs_inf = torch.nonzero(inf_mask, as_tuple=True)
        for y, x in zip(ys_inf.tolist(), xs_inf.tolist()):
            graph.known_infections.add((int(x), int(y)))

        hack_mask = graph.env.privilege_level > thr
        ys_hack, xs_hack = torch.nonzero(hack_mask, as_tuple=True)
        for y, x in zip(ys_hack.tolist(), xs_hack.tolist()):
            graph.known_hacks.add((int(x), int(y)))
        return

    inf_flat = graph.env.infected_map.view(-1)
    hack_flat = graph.env.privilege_level.view(-1)
    chunk_size = math.ceil(total_cells / num_sensors)

    for i in range(num_sensors):
        start = i * chunk_size
        end = min(start + chunk_size, total_cells)

        segment_inf = inf_flat[start:end]
        idxs_local_inf = torch.nonzero(segment_inf > thr, as_tuple=True)[0]
        for local_idx in idxs_local_inf.tolist():
            flat_idx = local_idx + start
            x = flat_idx % W
            y = flat_idx // W
            graph.known_infections.add((int(x), int(y)))

        segment_hack = hack_flat[start:end]
        idxs_local_hack = torch.nonzero(segment_hack > thr, as_tuple=True)[0]
        for local_idx in idxs_local_hack.tolist():
            flat_idx = local_idx + start
            x = flat_idx % W
            y = flat_idx // W
            graph.known_hacks.add((int(x), int(y)))


def _sensor_forward_impl(graph: "ImmuneCogGraph", flat_state: torch.Tensor) -> torch.Tensor:
    if flat_state.dim() == 1:
        flat_state = flat_state.unsqueeze(0)
    D_full = graph._D_env + graph._D_hack + graph._D_goal
    L = flat_state.shape[1]
    if L < D_full:
        flat_state = F.pad(flat_state, (0, D_full - L))
    elif L > D_full:
        flat_state = flat_state[:, :D_full]
    return graph.sensor_net(flat_state)


class SensorCellController:
    """Encapsulates sensor-specific behaviours for :class:`ImmuneCogGraph`."""

    def __init__(self, graph: "ImmuneCogGraph") -> None:
        self.graph = graph

    def run_scans(self) -> None:
        """Run the distributed sensor scan routine."""
        _run_scans_impl(self.graph)

    def forward(self, flat_state: torch.Tensor) -> torch.Tensor:
        """Forward propagate the shared sensor network."""
        return _sensor_forward_impl(self.graph, flat_state)
