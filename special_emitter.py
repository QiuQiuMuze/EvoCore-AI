# special_emitter.py

from CogUnit import CogUnit
from emitter_actions import EmitterActions
from typing import Tuple, Dict, List

class SpecialEmitter(CogUnit):
    """
    特殊攻击类型“特攻”单元，继承自 CogUnit。
    仅在检测到指定攻击类型时激活，并使用预定义策略进行高效拦截。
    支持：
      - 动态检测对应攻击位置
      - 执行专用防御动作
      - 记录成功次数，达到阈值后可自动复制自己（分化出更多特攻单元）
    """

    def __init__(self,
                 unit_id: int,
                 attack_type: str,
                 strategy: Dict,
                 env,
                 clone_threshold: int = 5,
                 **kwargs):
        """
        :param unit_id: 单元唯一标识
        :param attack_type: 该单元针对的攻击类型（如 'worm','apt'）
        :param strategy: 预定义的防御策略字典，如 {"type":"quarantine"}
        :param env: GridSecurityEnv 实例
        :param clone_threshold: 成功拦截次数达到此阈值即分化新特攻单元
        """
        kwargs["id"] = unit_id  # 显式塞入 CogUnit 能识别的 key
        kwargs["role"] = "emitter"  # 防止忘记设置 role
        super().__init__(**kwargs)
        self.attack_type = attack_type
        self.strategy = strategy
        self.env = env
        self.actions = EmitterActions(env)
        self.success_count = 0
        self.clone_threshold = clone_threshold

    def step(self, state):
        """
        覆盖父类 step 流程：
        1. 检测对应攻击
        2. 执行特攻策略
        3. 更新成功计数，判断是否分化
        4. 否则回落到普通 emitter 行为
        """
        # 1. 检测环境中所有该类型攻击位置
        targets = self._detect_attack_positions()
        if targets:
            for pos in targets:
                # 2. 对每个目标执行策略
                action = self._build_action(pos)
                self.actions.perform(action)
                # 3. 如果成功清除感染，累加成功计数
                if self.env.infected_map[pos[1], pos[0]] == 0:
                    self.success_count += 1
            # 4. 判断是否分化（克隆新特攻单元）
            if self.success_count >= self.clone_threshold:
                self._clone_special()
                self.success_count = 0
        else:
            # 无匹配攻击，执行普通行为（父类逻辑）
            if state.dim() == 3 and state.shape[1] == 1:
                state = state.squeeze(1)  # (1, 1, D) → (1, D)
            elif state.dim() == 1:
                state = state.unsqueeze(0)  # (D,) → (1, D)

            # 先把 state 展平成 (1, D) 的二维张量，再喂给父类
            if state.dim() > 2:
                # [C, H, W] 或 [1, C, H, W] → [1, C*H*W]
                state = state.view(1, -1)
            elif state.dim() == 1:
                # [D] → [1, D]
                state = state.unsqueeze(0)
            super().update(state)

    def _detect_attack_positions(self) -> List[Tuple[int,int]]:
        """
        扫描环境 attacks 字典，返回所有该单元关注的攻击类型位置列表
        """
        return [pos for pos, info in self.env.attacks.items() if info['type'] == self.attack_type]

    def _build_action(self, pos: Tuple[int,int]) -> Dict:
        """
        构造防御策略：在基础策略上加入目标坐标
        """
        return {"type": self.strategy["type"], "target": pos}

    def _clone_special(self):
        """
        分化（克隆）一个新的特攻单元，继承当前策略与攻击类型
        """
        # 需要外部管理器将新单元加入 CogGraph.units 列表
        new_id = self.env.step_count  # 示例：用 step_count 作为临时 id
        new_unit = SpecialEmitter(unit_id=new_id,
                                  attack_type=self.attack_type,
                                  strategy=self.strategy,
                                  env=self.env,
                                  clone_threshold=self.clone_threshold)
        # 将 new_unit 注册到 CogGraph 中（需在外部调用注册函数）
        if hasattr(self.env, 'register_special'):
            self.env.register_special(new_unit)


