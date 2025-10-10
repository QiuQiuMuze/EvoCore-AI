import random
import logging
import torch
from collections import deque, Counter

class LimitedDebugHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__(level=logging.DEBUG)
        self.buffer = deque(maxlen=capacity)

    def emit(self, record):
        if record.levelno == logging.DEBUG:
            try:
                msg = self.format(record)
                self.buffer.append(msg)
            except Exception:
                pass

    def dump_to_console(self):
        print("\n==== [最近 Debug 日志] ====")
        for msg in self.buffer:
            print(msg)

# === 设置 root logger ===
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.handlers.clear()

# Debug 缓存 Handler
debug_handler = LimitedDebugHandler(capacity=100)
debug_handler.setFormatter(logging.Formatter('%(asctime)s [DEBUG] %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(debug_handler)

# 普通输出 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(console_handler)

class GridEnvironment:
    action_space_n = 4  # 上/下/左/右

    def __init__(self, size=5, max_steps: int | None = None, sensor_range: int = 2):
        self.size = size
        self.max_steps = max_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sensor_range = max(0, sensor_range)

        # 环境刷新调度
        self.refresh_cycle_steps = 1000
        self.refresh_chunk_steps = 200
        self._cycle_anchor_step = 0
        self._resource_chunk_plan: list[int] = []
        self._hazard_chunk_plan: list[int] = []
        self._chunk_index = 0
        self.cycle_resource_total = 0
        self.cycle_hazard_total = 0
        self.cycle_reward_hits = 0
        self.cycle_danger_hits = 0
        self.chunk_resource_total = 0
        self.chunk_hazard_total = 0
        self.chunk_reward_hits = 0
        self.chunk_danger_hits = 0
        self._chunk_reward_hit_anchor = 0
        self._chunk_danger_hit_anchor = 0

        # 经验点和危险点
        self.resources = Counter()
        self.hazards = Counter()

        # 计数和位置
        self.step_count = 0
        self.explored_cells_count = 0
        self.agent_pos = [random.randrange(self.size), random.randrange(self.size)]

        # 记忆保留 & 反馈追踪
        self.reward_hit_count = 0
        self.danger_hit_count = 0
        self._last_cycle_reward_hits = 0
        self._last_cycle_danger_hits = 0
        self._last_cycle_success_ratio = 0.5
        self._last_cycle_exploration_ratio = 0.0
        self.map_retention = 0.65
        self.knowledge_retention = 0.75
        self._hazard_contact_timer = 0

        # 初始化探索标记和状态缓冲
        self._init_buffers()

        # 初始化环境
        exclude = {tuple(self.agent_pos)}
        self.refresh_environment(step=0, explored_cells_count=0, exclude_positions=exclude)
        self.reset()

    def _init_buffers(self):
        shape = (self.size, self.size)
        self.visited_map = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.known_map = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.detected_resources_map = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.detected_hazards_map = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._state_buf = torch.zeros((5, self.size, self.size), dtype=torch.float32, device=self.device)
        self.explored_cells_count = 0

    def _build_chunk_plan(self, total_count: int) -> list[int]:
        if self.refresh_chunk_steps <= 0:
            return [total_count]
        chunks = max(1, self.refresh_cycle_steps // self.refresh_chunk_steps)
        base, remainder = divmod(total_count, chunks)
        plan = [base] * chunks
        for idx in range(remainder):
            plan[idx % chunks] += 1
        return plan

    def _apply_retention_mask(self, tensor: torch.Tensor, retention: float) -> None:
        if retention >= 1.0:
            return
        if retention <= 0.0:
            tensor.fill_(False)
            return
        drop_mask = torch.rand(tensor.shape, device=tensor.device) > retention
        tensor[drop_mask] = False

    def _apply_memory_retention(self) -> None:
        self._apply_retention_mask(self.visited_map, self.map_retention)
        self._apply_retention_mask(self.known_map, self.knowledge_retention)
        self._apply_retention_mask(self.detected_resources_map, self.knowledge_retention)
        self._apply_retention_mask(self.detected_hazards_map, self.knowledge_retention)
        ax, ay = self.agent_pos
        if self._in_bounds(ax, ay):
            self.visited_map[ay, ax] = True
            self.known_map[ay, ax] = True
        self.explored_cells_count = int(self.visited_map.sum().item())

    def _sample_weighted_positions(
        self,
        count: int,
        *,
        blocked: set[tuple[int, int]],
        frontier_weight: float,
        revisit_weight: float,
    ) -> list[tuple[int, int]]:
        if count <= 0:
            return []

        blocked = set(blocked)
        visited_cpu = self.visited_map.detach().to("cpu")
        known_cpu = self.known_map.detach().to("cpu")
        frontier: list[tuple[int, int]] = []
        fresh: list[tuple[int, int]] = []
        familiar: list[tuple[int, int]] = []

        for y in range(self.size):
            for x in range(self.size):
                pos = (x, y)
                if pos in blocked:
                    continue
                visited = bool(visited_cpu[y, x].item())
                known = bool(known_cpu[y, x].item())
                if not visited and known:
                    frontier.append(pos)
                elif not visited:
                    fresh.append(pos)
                else:
                    familiar.append(pos)

        if not frontier and not fresh and not familiar:
            return []

        weighted: list[tuple[int, int]] = []
        frontier_multiplier = max(1, int(round(frontier_weight * 2)))
        fresh_multiplier = max(1, int(round(frontier_weight)))
        familiar_multiplier = max(1, int(round(revisit_weight)))

        for pos in frontier:
            weighted.extend([pos] * frontier_multiplier)
        for pos in fresh:
            weighted.extend([pos] * fresh_multiplier)
        for pos in familiar:
            weighted.extend([pos] * familiar_multiplier)

        random.shuffle(weighted)

        selected: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for pos in weighted:
            if pos in seen:
                continue
            selected.append(pos)
            seen.add(pos)
            if len(selected) >= count:
                break

        return selected

    def _trim_points(self, counter: Counter, remove_count: int) -> None:
        if remove_count <= 0:
            return
        keys = list(counter.keys())
        if not keys:
            return
        random.shuffle(keys)
        for pos in keys[:remove_count]:
            counter.pop(pos, None)
            self.update_known_cell(pos)

    def _distribute_chunk(self, exclude_positions: set | None = None):
        if self._chunk_index >= len(self._resource_chunk_plan):
            self.chunk_resource_total = 0
            self.chunk_hazard_total = 0
            self._chunk_reward_hit_anchor = self.reward_hit_count
            self._chunk_danger_hit_anchor = self.danger_hit_count
            self.chunk_reward_hits = 0
            self.chunk_danger_hits = 0
            return

        exclude_positions = set(exclude_positions or set())
        exclude_positions.add(tuple(self.agent_pos))

        resource_chunk_total = self._resource_chunk_plan[self._chunk_index]
        hazard_chunk_total = self._hazard_chunk_plan[self._chunk_index]
        self.chunk_resource_total = resource_chunk_total
        self.chunk_hazard_total = hazard_chunk_total
        self._chunk_reward_hit_anchor = self.reward_hit_count
        self._chunk_danger_hit_anchor = self.danger_hit_count
        self.chunk_reward_hits = 0
        self.chunk_danger_hits = 0

        resource_blocked = exclude_positions | set(self.resources.keys())
        new_resources = self._sample_weighted_positions(
            resource_chunk_total,
            blocked=resource_blocked,
            frontier_weight=1.6,
            revisit_weight=1.0,
        )
        for pos in new_resources:
            self.resources[pos] = 1

        hazard_blocked = exclude_positions | set(self.hazards.keys())
        new_hazards = self._sample_weighted_positions(
            hazard_chunk_total,
            blocked=hazard_blocked,
            frontier_weight=2.2,
            revisit_weight=0.8,
        )
        for pos in new_hazards:
            self.hazards[pos] = 1

        self._chunk_index += 1
        self._sense_environment()

    def refresh_environment(self, step: int, explored_cells_count: int, exclude_positions: set = None):
        exclude_positions = exclude_positions or set()

        area = max(1, self.size * self.size)
        explored_ratio = explored_cells_count / area if explored_cells_count is not None else self.explored_cells_count / area
        explored_ratio = float(max(0.0, min(1.0, explored_ratio)))

        reward_hits = getattr(self, "reward_hit_count", 0)
        danger_hits = getattr(self, "danger_hit_count", 0)
        total_hits = reward_hits + danger_hits
        if total_hits > 0:
            success_ratio = reward_hits / total_hits
        else:
            success_ratio = getattr(self, "_last_cycle_success_ratio", 0.5)
        success_ratio = float(max(0.0, min(1.0, success_ratio)))

        self._last_cycle_reward_hits = reward_hits
        self._last_cycle_danger_hits = danger_hits
        self._last_cycle_success_ratio = success_ratio
        self._last_cycle_exploration_ratio = explored_ratio
        self.reward_hit_count = 0
        self.danger_hit_count = 0

        self._apply_memory_retention()

        extra = min(80, max(((self.step_count - 1200) // 300), 0))
        base_resource_density = 0.18
        growth_factor = 0.75 + 0.45 * explored_ratio
        resource_target = int(area * base_resource_density * growth_factor)
        resource_target += int(extra * (0.5 + 0.5 * success_ratio))

        base_hazard_density = 0.22
        hazard_adjust = 1.0 - 0.6 * (0.5 - success_ratio)
        hazard_adjust = max(0.7, min(1.25, hazard_adjust))
        caution_factor = 0.9 + 0.2 * (1.0 - explored_ratio)
        hazard_density = base_hazard_density * hazard_adjust * caution_factor
        hazard_density = max(0.14, min(0.28, hazard_density))
        hazard_target = int(area * hazard_density)
        hazard_target += int(extra * (0.6 + 0.4 * (1.0 - success_ratio)))

        self._trim_points(self.resources, max(0, len(self.resources) - resource_target))
        self._trim_points(self.hazards, max(0, len(self.hazards) - hazard_target))

        to_add_res = max(0, resource_target - len(self.resources))
        to_add_haz = max(0, hazard_target - len(self.hazards))

        self._resource_chunk_plan = self._build_chunk_plan(to_add_res)
        self._hazard_chunk_plan = self._build_chunk_plan(to_add_haz)
        self._chunk_index = 0
        self._cycle_anchor_step = step
        existing_resource_total = len(self.resources)
        existing_hazard_total = len(self.hazards)
        self.cycle_resource_total = existing_resource_total + sum(self._resource_chunk_plan)
        self.cycle_hazard_total = existing_hazard_total + sum(self._hazard_chunk_plan)
        self.cycle_reward_hits = 0
        self.cycle_danger_hits = 0
        self.chunk_resource_total = 0
        self.chunk_hazard_total = 0
        self.chunk_reward_hits = 0
        self.chunk_danger_hits = 0
        self._chunk_reward_hit_anchor = self.reward_hit_count
        self._chunk_danger_hit_anchor = self.danger_hit_count

        self._sense_environment()

        self._distribute_chunk(exclude_positions=exclude_positions)
        self.prev_dist_resource = self.distance_to_nearest_known_resource(tuple(self.agent_pos))
        self.prev_danger_dist = self.distance_to_nearest_known_danger(tuple(self.agent_pos))

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def _sense_environment(self):
        """更新感知范围内的资源 / 危险信息到探测图。"""
        ax, ay = self.agent_pos
        rng = self.sensor_range
        for dx in range(-rng, rng + 1):
            for dy in range(-rng, rng + 1):
                nx, ny = ax + dx, ay + dy
                if not self._in_bounds(nx, ny):
                    continue
                self.known_map[ny, nx] = True
                self.detected_resources_map[ny, nx] = bool(self.resources[(nx, ny)] > 0)
                self.detected_hazards_map[ny, nx] = bool(self.hazards[(nx, ny)] > 0)

    def update_known_cell(self, pos: tuple[int, int]):
        """当资源 / 危险状态变化时，刷新已知网格的感知记录。"""
        x, y = pos
        if not self._in_bounds(x, y):
            return
        if not self.known_map[y, x]:
            return
        self.detected_resources_map[y, x] = bool(self.resources[(x, y)] > 0)
        self.detected_hazards_map[y, x] = bool(self.hazards[(x, y)] > 0)

    def get_known_resource_positions(self) -> list[tuple[int, int]]:
        mask = (self.detected_resources_map & self.known_map).nonzero(as_tuple=False)
        if mask.numel() == 0:
            return []
        coords = mask.cpu().tolist()
        return [(int(x), int(y)) for y, x in coords]

    def get_known_hazard_positions(self) -> list[tuple[int, int]]:
        mask = (self.detected_hazards_map & self.known_map).nonzero(as_tuple=False)
        if mask.numel() == 0:
            return []
        coords = mask.cpu().tolist()
        return [(int(x), int(y)) for y, x in coords]

    def get_nearest_known_resource(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        known = self.get_known_resource_positions()
        if not known:
            return None
        return min(known, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def get_nearest_known_danger(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        known = self.get_known_hazard_positions()
        if not known:
            return None
        return min(known, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def distance_to_nearest_known_resource(self, pos):
        nearest = self.get_nearest_known_resource(pos)
        if nearest is None:
            return float("inf")
        return abs(pos[0] - nearest[0]) + abs(pos[1] - nearest[1])

    def distance_to_nearest_known_danger(self, pos):
        nearest = self.get_nearest_known_danger(pos)
        if nearest is None:
            return float("inf")
        return abs(pos[0] - nearest[0]) + abs(pos[1] - nearest[1])

    def reset(self):
        # 随机初始化 agent
        self.agent_pos = [random.randrange(self.size), random.randrange(self.size)]
        self.step_count = 0
        self.agent_energy_gain = 0.0
        self.agent_energy_penalty = 0.0
        self._hazard_contact_timer = 0
        # 当 episode 被上层截断时，step_count 会被重置为 0，但
        # refresh 周期的锚点 (_cycle_anchor_step) 仍然停留在上一轮的值
        #（例如 1000）。如果不同步更新，后续 progress = step_count - anchor
        # 会变成负数，从而导致 refresh / chunk 分发永远不会触发，
        # 给人一种资源 / 危险点变少的错觉。
        # 因此在 reset 时也重置锚点，确保新的 episode 能正常进入 refresh 节奏。
        self._cycle_anchor_step = self.step_count

        # 当资源被完全采集后，下一轮需要重新刷新环境，避免 "horizon step"
        # 在每一步都被立即截断
        if not self.resources:
            self.refresh_environment(
                step=0,
                explored_cells_count=0,
                exclude_positions={tuple(self.agent_pos)},
            )

        # 清空标记
        self.visited_map.fill_(False)
        self.known_map.fill_(False)
        self.detected_resources_map.fill_(False)
        self.detected_hazards_map.fill_(False)
        self.explored_cells_count = 0

        # 执行一次感知，初始化局部知识
        self._sense_environment()

        # 最近距离初始化（基于已感知的知识）
        self.prev_dist_resource = self.distance_to_nearest_known_resource(tuple(self.agent_pos))
        self.prev_danger_dist = self.distance_to_nearest_known_danger(tuple(self.agent_pos))

        # 清空命中统计
        self.reward_hit_count = 0
        self.danger_hit_count = 0
        self.cycle_reward_hits = 0
        self.cycle_danger_hits = 0
        self.chunk_reward_hits = 0
        self.chunk_danger_hits = 0
        self._chunk_reward_hit_anchor = 0
        self._chunk_danger_hit_anchor = 0

        return self.get_state()

    def step(self, action, cog_step: int | None = None):
        x, y = self.agent_pos
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.size - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.size - 1:
            x += 1
        self.agent_pos = [x, y]

        pos = (x, y)
        # 资源命中
        if self.resources[pos] > 0:
            self.reward_hit_count += 1
            self.agent_energy_gain = 0.1
            self.resources[pos] -= 1
            if self.resources[pos] <= 0:
                self.resources.pop(pos, None)
            self.update_known_cell(pos)
        else:
            self.agent_energy_gain = 0.0

        # 危险命中
        if self.hazards[pos] > 0:
            self.danger_hit_count += 1
            self.agent_energy_penalty = 0.1
            self.update_known_cell(pos)
        else:
            self.agent_energy_penalty = 0.0

        self.cycle_reward_hits = self.reward_hit_count
        self.cycle_danger_hits = self.danger_hit_count
        self.chunk_reward_hits = max(0, self.reward_hit_count - self._chunk_reward_hit_anchor)
        self.chunk_danger_hits = max(0, self.danger_hit_count - self._chunk_danger_hit_anchor)

        if self.agent_energy_penalty > 0.0:
            self._hazard_contact_timer = 4
        elif self._hazard_contact_timer > 0:
            self._hazard_contact_timer -= 1

        # 感知更新（移动后立即刷新可见区域）
        self._sense_environment()

        # 更新步数
        self.step_count += 1
        step_for_refresh = cog_step if cog_step is not None else self.step_count
        progress = step_for_refresh - self._cycle_anchor_step
        refresh_triggered = False
        if progress >= self.refresh_cycle_steps and self.refresh_cycle_steps > 0:
            self.refresh_environment(
                step_for_refresh,
                self.explored_cells_count,
                exclude_positions={tuple(self.agent_pos)},
            )
            progress = 0
            refresh_triggered = True

        if progress > 0 and self.refresh_chunk_steps > 0 and progress % self.refresh_chunk_steps == 0:
            self._distribute_chunk(exclude_positions={tuple(self.agent_pos)})

        # 计算 reward shaping
        base = self.agent_energy_gain - self.agent_energy_penalty
        dist_res = self.distance_to_nearest_known_resource(pos)
        delta_res = self.prev_dist_resource - dist_res
        if delta_res > 0:
            resource_shaping = 0.0015
        elif delta_res < 0:
            resource_shaping = -0.0005
        else:
            resource_shaping = 0.0
        self.prev_dist_resource = dist_res

        danger_dist = self.distance_to_nearest_known_danger(pos)
        delta_danger = self.prev_danger_dist - danger_dist
        if delta_danger > 0:
            danger_shaping = 0.0015
            if self._hazard_contact_timer > 0:
                danger_shaping += 0.0015
        elif delta_danger < 0:
            danger_shaping = -0.001
        else:
            danger_shaping = 0.0
        self.prev_danger_dist = danger_dist

        explore_bonus = 0.0
        if not self.visited_map[y, x]:
            self.visited_map[y, x] = True
            self.explored_cells_count += 1
            explore_bonus = 0.0012

        reward = base + resource_shaping + danger_shaping + explore_bonus
        next_state = self.get_state()

        done = False
        if self.max_steps is not None and self.step_count >= self.max_steps:
            done = True
        elif refresh_triggered and self.refresh_cycle_steps > 0:
            done = True
        elif not self.resources:
            if self._chunk_index < len(self._resource_chunk_plan) or \
               self._chunk_index < len(self._hazard_chunk_plan):
                self._distribute_chunk(exclude_positions={tuple(self.agent_pos)})

        return next_state, reward, done, {}

    def get_nearest_resource_to(self, pos):
        if not self.resources:
            return None
        return min(self.resources, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def get_state(self):
        # 就地更新状态缓冲

        buf = self._state_buf
        buf.fill_(0.0)

        x, y = self.agent_pos
        buf[0, y, x] = 1.0
        buf[1].copy_(self.detected_resources_map.to(buf.dtype))
        buf[2].copy_(self.detected_hazards_map.to(buf.dtype))
        buf[3].copy_(self.known_map.to(buf.dtype))
        buf[4].copy_(self.visited_map.to(buf.dtype))

        return buf.view(-1)

    def render(self):
        if not logger.isEnabledFor(logging.DEBUG):
            return

        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        x, y = self.agent_pos
        grid[y][x] = 'A'
        logger.debug('\n' + '\n'.join(' '.join(row) for row in grid) + '\n')

    def distance_to_nearest_danger(self, pos):
        if not self.hazards:
            return float("inf")
        return min(abs(pos[0] - hx) + abs(pos[1] - hy) for hx, hy in self.hazards)

    def distance_to_nearest_resource(self, pos):
        if not self.resources:
            return float("inf")
        return min(abs(pos[0] - rx) + abs(pos[1] - ry) for rx, ry in self.resources)

    def get_nearest_danger_to(self, pos):
        if not self.hazards:
            return None
        return min(self.hazards, key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def resize(self, new_size: int):
        self.size = new_size
        self._init_buffers()

        x, y = self.agent_pos
        x = min(x, new_size - 1)
        y = min(y, new_size - 1)
        self.agent_pos = [x, y]
        self.refresh_environment(step=self.step_count,explored_cells_count = self.explored_cells_count)

    def reset_with_size(self, new_size: int):
        self.size = new_size
        self.agent_pos = [random.randrange(self.size), random.randrange(self.size)]
        self._init_buffers()
        self.resources = Counter()
        self.hazards = Counter()
        self.step_count = 0
        self.explored_cells_count = 0
        self.reward_hit_count = 0
        self.danger_hit_count = 0
        self.refresh_environment(step=0, explored_cells_count=0, exclude_positions={tuple(self.agent_pos)})

    def inject_reward_map(self, reward_points: list[tuple[int,int]], punishment_points: list[tuple[int,int]]):
        self.resources = Counter(reward_points)
        self.hazards = Counter(punishment_points)
        self.visited_map.fill_(False)
        self.known_map.fill_(False)
        self.detected_resources_map.fill_(False)
        self.detected_hazards_map.fill_(False)
        self.explored_cells_count = 0
        self.reward_hit_count = 0
        self.danger_hit_count = 0
        self.cycle_resource_total = len(self.resources)
        self.cycle_hazard_total = len(self.hazards)
        self.cycle_reward_hits = 0
        self.cycle_danger_hits = 0
        self.chunk_resource_total = 0
        self.chunk_hazard_total = 0
        self.chunk_reward_hits = 0
        self.chunk_danger_hits = 0
        self._chunk_reward_hit_anchor = 0
        self._chunk_danger_hit_anchor = 0
        self._sense_environment()
        self.prev_dist_resource = self.distance_to_nearest_known_resource(tuple(self.agent_pos))
        self.prev_danger_dist = self.distance_to_nearest_known_danger(tuple(self.agent_pos))

    def get_cycle_statistics(self) -> dict:
        return {
            "cycle_total": {
                "resources": self.cycle_resource_total,
                "hazards": self.cycle_hazard_total,
            },
            "cycle_hits": {
                "resources": self.cycle_reward_hits,
                "hazards": self.cycle_danger_hits,
            },
            "chunk_total": {
                "resources": self.chunk_resource_total,
                "hazards": self.chunk_hazard_total,
            },
            "chunk_hits": {
                "resources": self.chunk_reward_hits,
                "hazards": self.chunk_danger_hits,
            },
        }

if __name__ == "__main__":
    env = GridEnvironment(size=10)
    env.render()
    for _ in range(5):
        action = random.randrange(4)
        env.step(action)
        env.render()
        print("State vector:", env.get_state())
