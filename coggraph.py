# coggraph.py
import uuid
from CogUnit import CogUnit
import torch
import random
from env import GridEnvironment

MAX_CONNECTIONS = 4  # 每个单元最多连接 4 个下游


class TaskInjector:
    def __init__(self, target_position):
        self.target_position = target_position  # 目标坐标 (x, y)

    def encode_goal(self, env_size):
        """将目标位置编码成 one-hot 向量（与输入同维度）"""
        index = self.target_position[1] * env_size + self.target_position[0]
        vec = torch.zeros(env_size * env_size)
        vec[index] = 1.0
        return vec

    def evaluate(self, env, emitter_outputs):
        """评估 emitter 是否“指向”目标位置"""
        if emitter_outputs is None:
            return False

        pred_index = torch.argmax(emitter_outputs.mean(dim=0)).item()
        x, y = pred_index % env.size, pred_index // env.size
        return (x, y) == self.target_position


class CogGraph:
    """
    CogGraph 管理所有 CogUnit 的集合和连接关系：
    - 添加 / 删除单元
    - 管理连接（可拓展为图）
    - 调度每一轮所有 CogUnit 的更新、分裂、死亡，并传递输出
    """
    def __init__(self):
        self.target_vector = torch.ones(50)
        self.connection_usage = {}  # {(from_id, to_id): last_used_step}
        self.current_step = 0
        self.units = []
        self.connections = {}  # {from_id: {to_id: strength_float}}
        self.unit_map = {}     # {unit_id: CogUnit 实例} 快速索引单元

    def add_unit(self, unit: CogUnit):
        # 将单元加入图结构中
        self.units.append(unit)
        self.unit_map[unit.id] = unit
        self.connections[unit.id] = {}

    def remove_unit(self, unit: CogUnit):
        # 从图中移除单元及其连接
        self.units = [u for u in self.units if u.id != unit.id]
        if unit.id in self.connections:
            del self.connections[unit.id]
        if unit.id in self.unit_map:
            del self.unit_map[unit.id]
        for k in self.connections:
            if unit.id in self.connections[k]:
                del self.connections[k][unit.id]

    def connect(self, from_unit: CogUnit, to_unit: CogUnit):
        if from_unit.id not in self.connections:
            self.connections[from_unit.id] = {}  # to_id → strength

        if to_unit.id in self.connections[from_unit.id]:
            return

        # 超过上限时，移除 strength 最弱的连接
        if len(self.connections[from_unit.id]) >= MAX_CONNECTIONS:
            weakest_id = min(
                self.connections[from_unit.id],
                key=lambda uid: self.connections[from_unit.id][uid]
            )
            del self.connections[from_unit.id][weakest_id]
            print(f"[连接替换] {from_unit.id} 移除最弱连接 {weakest_id}")

        # 建立新连接，初始权重为 1.0
        self.connections[from_unit.id][to_unit.id] = 1.0
        print(f"[连接建立] {from_unit.id} → {to_unit.id} (strength=1.0)")

    def auto_connect(self):
        for unit in self.units:
            # 只处理 processor 节点
            if unit.get_role() != "processor":
                continue

            # 获取已有连接数（下游）
            current_connections = self.connections[unit.id]
            if len(current_connections) < 2:
                # 随机找一个 emitter 或 processor 来连接
                def euclidean(p1, p2):
                    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

                u_pos = unit.get_position()
                candidates = [
                    u for u in self.units
                    if u.id != unit.id and u.get_role() in ["processor", "emitter"]
                       and u.id not in self.connections[unit.id]
                       and euclidean(u_pos, u.get_position()) < 3
                ]

                # 按能量从高到低排序，优先选择最有价值连接目标
                if candidates:
                    candidates.sort(key=lambda u: u.energy, reverse=True)
                    target = candidates[0]  # 能量最高者
                    self.connect(unit, target)
                    if target.id not in current_connections:
                        self.connect(unit, target)
                        print(f"[新连接] {unit.id} → {target.id}")
        # === 随机突变连接：processor 有小概率连接新目标 ===
        if random.random() < 0.1:  # 10% 概率触发突变
            from_candidates = [u for u in self.units if u.get_role() == "processor"]
            to_candidates = [u for u in self.units if u.get_role() in ["processor", "emitter"]]

            if from_candidates and to_candidates:
                from_unit = random.choice(from_candidates)
                to_unit = random.choice(to_candidates)

                if to_unit.id not in self.connections.get(from_unit.id, []):
                    self.connect(from_unit, to_unit)
                    print(f"[突变连接] {from_unit.id} → {to_unit.id}")

    def trace_info_paths(self):
        print(f"[信息路径追踪] 步数 {self.current_step}")
        for emitter in self.units:
            if emitter.get_role() != "emitter":
                continue

            # 追溯上游 processor
            emit_from = [pid for pid in self.unit_map if emitter.id in self.connections.get(pid, {})]
            for pid in emit_from:
                proc_from = [sid for sid in self.unit_map if pid in self.connections.get(sid, {})]
                for sid in proc_from:
                    print(f"  sensor:{sid} → processor:{pid} → emitter:{emitter.id}")

    def step(self, input_tensor: torch.Tensor):
        if self.current_step % 50 == 0:
            self.target_vector = torch.rand(50)
            print(f"[目标变化] 第 {self.current_step} 步，目标向量已更新")

        self.current_step += 1
        new_units = []  # 新生成的单元（复制）
        output_buffer = {}  # 缓存每个单元的输出 {unit_id: output_tensor}

        # === 第一阶段：单元更新处理 ===
        for unit in self.units[:]:
            unit_input = input_tensor.unsqueeze(0)  # 外部输入 [1, 8]


            # 如果该单元有上游连接（被其他单元指向）
            incoming = [uid for uid in self.unit_map if unit.id in self.connections.get(uid, [])]
            for uid in incoming:
                self.connection_usage[(uid, unit.id)] = self.current_step
                # 增强强度
                for uid in incoming:
                    self.connections[uid][unit.id] *= 1.05  # 增强 5%
                    self.connections[uid][unit.id] = min(self.connections[uid][unit.id], 5.0)  # 上限

            if unit.get_role() == "sensor":
                unit_input = input_tensor.unsqueeze(0)
            elif incoming:
                weighted_outputs = []
                total_weight = 0.0
                for uid in incoming:
                    strength = self.connections[uid][unit.id]
                    output = self.unit_map[uid].get_output().squeeze(0)  # 统一为 [8]
                    weighted_outputs.append(output * strength)
                    total_weight += strength

                if total_weight > 0:
                    unit_input = torch.stack(weighted_outputs).sum(dim=0, keepdim=True) / total_weight  # [1, 8]
                else:
                    unit_input = torch.zeros_like(input_tensor).unsqueeze(0)

                if total_weight > 0:
                    unit_input = torch.stack(weighted_outputs).sum(dim=0, keepdim=True) / total_weight
                else:
                    unit_input = torch.zeros_like(input_tensor).unsqueeze(0)
            else:
                continue  # processor 或 emitter 且无输入，跳过

            # 执行单元的更新逻辑
            unit.update(unit_input)
            # ✅ 加强连接权重（使用次数越多越强）
            for uid in incoming:
                if unit.id in self.connections.get(uid, {}):
                    self.connections[uid][unit.id] *= 1.05  # 增强
                    self.connections[uid][unit.id] = min(self.connections[uid][unit.id], 5.0)
            output_buffer[unit.id] = unit.get_output()
            print(unit)

            # === 判断是否需要复制 ===
            if unit.should_split():
                child = unit.clone()
                new_units.append(child)
                self.connect(unit, child)  # 父子单元建立连接

            # === 判断是否死亡 ===
            if unit.should_die():
                print(f"[死亡] {unit.id} 被移除")
                self.remove_unit(unit)

        # 将所有新生成的单元加入图结构
        for unit in new_units:
            self.add_unit(unit)

        self.auto_connect()
        # === 死连接清理 ===
        if self.current_step % 10 == 0:
            threshold = 10
            for from_id in list(self.connections.keys()):
                for to_id in list(self.connections[from_id].keys()):
                    last_used = self.connection_usage.get((from_id, to_id), -1)
                    if self.current_step - last_used > threshold:
                        del self.connections[from_id][to_id]  # ✅ 正确删除方式
                        print(f"[死连接清除] {from_id} → {to_id}")
                        # 删除连接后，给 from_unit 轻微能量惩罚
                        if from_id in self.unit_map:
                            self.unit_map[from_id].energy -= 0.01  # 可调参数
                            print(f"[惩罚] {from_id} 因连接失效，能量 -0.01")
                    else:
                        # ✅ 削弱仍在用但表现差的连接
                        self.connections[from_id][to_id] *= 0.95
                        if self.connections[from_id][to_id] < 0.1:
                            del self.connections[from_id][to_id]
                            print(f"[连接衰减清除] {from_id} → {to_id}")

        # 简易任务奖励：如果 emitter 输出靠近某个目标向量，则发放奖励
        target_vector = self.target_vector
        outputs = self.collect_emitter_outputs()
        if outputs is not None:
            avg_output = outputs.mean(dim=0)
            distance = torch.norm(avg_output - target_vector)
            if distance < 1.0:  # 满足目标
                for unit in self.units:
                    if unit.get_role() == "processor":
                        unit.energy += 0.05  # 奖励值可调
                        print(f"[奖励] {unit.id} 因接近目标输出获得能量 +0.05")
            # ✅ 增加多样性惩罚
            action_indices = [torch.argmax(out).item() for out in outputs]
            common_action = max(set(action_indices), key=action_indices.count)
            if action_indices.count(common_action) > len(action_indices) * 0.7:
                for unit in self.units:
                    if unit.get_role() == "emitter":
                        unit.energy -= 0.02
                        print(f"[惩罚] emitter {unit.id} 因输出单一行为被扣能量")
            if task.evaluate(env, outputs):
                print(f"[任务完成] 达成目标位置 {task.target_position}，奖励 +0.1")
                for unit in self.units:
                    if unit.get_role() == "processor":
                        unit.energy += 0.1

        self.trace_info_paths()

    def summary(self):
        # 打印当前图结构概况
        print(f"[图结构] 当前单元数: {len(self.units)}")
        for unit in self.units:
            print(f" - {unit} → 连接数: {len(self.connections[unit.id])}")

    def collect_emitter_outputs(self):
        outputs = []
        for unit in self.units:
            if unit.get_role() == "emitter":
                outputs.append(unit.get_output())
        if outputs:
            return torch.stack(outputs)
        else:
            return None  # 没有任何 emitter





def interpret_emitter_output(output_tensor):
    """
    将 emitter 的输出向量解释为动作。
    """
    action_names = [f"动作{i}" for i in range(50)]
    if output_tensor.dim() == 3:
        output_tensor = output_tensor.squeeze(1)  # 变成 [N, 8]

    for i, out in enumerate(output_tensor):
        action_index = torch.argmax(out).item()
        action = action_names[action_index]
        print(f"[行为触发] 第 {i+1} 个 emitter 执行动作: {action}")

def environment_feedback(output_tensor, graph):
    """
    环境对 emitter 输出的简单反馈机制：
    - 如果输出中出现特定模式（例如 ↑ 和 → 同时较强），奖励对应 emitter
    - 奖励通过提升 energy 实现
    """
    if output_tensor.dim() == 3:
        output_tensor = output_tensor.squeeze(1)  # [N, 8]

    for i, out in enumerate(output_tensor):
        # 示例条件：若 ↑ (index 0) 和 → (index 3) 输出值都大于 0.2
        if out[5] > 0.3 and out[13] < -0.1:  # 自定义规则
            emitter = [u for u in graph.units if u.get_role() == "emitter"][i]
            emitter.energy += 0.05  # 简单奖励
            print(f"[奖励] emitter {emitter.id} 因 ↑+→ 被奖励 +0.05 能量")



if __name__ == "__main__":
    env = GridEnvironment(size=5)
    # 初始化任务目标（例如目标位置在右下角 (4, 4)）
    task = TaskInjector(target_position=(4, 4))
    goal_tensor = task.encode_goal(env.size)  # 生成 25维 one-hot 向量
    graph = CogGraph()

    # 初始化单元
    sensor = CogUnit(role="sensor")
    processor = CogUnit(role="processor")
    emitter = CogUnit(role="emitter")

    graph.add_unit(sensor)
    graph.add_unit(processor)
    graph.add_unit(emitter)

    # 初始连接
    graph.connect(sensor, processor)
    graph.connect(processor, emitter)

    # 运行模拟
    for step in range(101):
        print(f"\n==== 第 {step+1} 步 ====")

        # 1️⃣ 获取环境状态，输入 sensor
        state = env.get_state()
        input_tensor = torch.cat([
            torch.from_numpy(state).float(),
            goal_tensor
        ], dim=0)

        # 2️⃣ 启动认知系统（感知 + 决策）
        graph.step(input_tensor)

        # 3️⃣ 收集 emitter 输出，转换为动作
        emitter_output = graph.collect_emitter_outputs()
        if emitter_output is not None:
            action_index = torch.argmax(emitter_output.mean(dim=0)).item()
            print(f"[行动决策] 执行动作: {action_index}")

            # 4️⃣ 执行动作，改变环境
            env.step(action_index)

        # 5️⃣ 打印环境图示
        env.render()

        # 如果没有单元剩下，退出
        if not graph.units:
            print("[终止] 所有单元死亡。")
            break
