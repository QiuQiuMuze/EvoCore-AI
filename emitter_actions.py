# emitter_actions.py

"""
Emitter 行为动作模块
封装了对环境(env_net.GridSecurityEnv)的各种防御动作调用，
并提供统一的接口给 CogUnit 或 SpecialEmitter 使用。
"""

from typing import Tuple, List
from env import logger

# 动作类型常量
ACTION_BLOCK = "block"
ACTION_QUARANTINE = "quarantine"
ACTION_MARK = "mark"
ACTION_ISOLATE_REGION = "isolate_region"
ACTION_TRACE_BACK  = "trace_back"
ACTION_KILL_PROCESS = "kill_process"
ACTION_ALERT_ADMIN = "alert_admin"
ACTION_DEMOTE_PRIVILEGE = "demote_privilege"
ACTION_RESTORE_VULNERABILITY = "restore_vulnerability"
ACTION_RESET_LOGIN_FAIL = "reset_login_fail"
ACTION_HACK_DEFENSE = "hack_defense"

class EmitterActions:
    def __init__(self, env):
        """
        初始化 EmitterActions
        :param env: GridSecurityEnv 实例
        """
        self.env = env

    def perform(self, action: dict):
        """
        统一执行一个动作字典
        :param action: {
            "type": 动作类型 (block/quarantine/mark/isolate_region),
            "target": (x, y) 格点坐标 或
            "region": List[(x1,y1), (x2,y2), ...] 区域列表
        }
        """
        atype = action.get("type")
        if atype == ACTION_HACK_DEFENSE:
            self.hack_defense(action["target"])
            return
        if atype == ACTION_BLOCK:
            self.block(action["target"])
        elif atype == ACTION_QUARANTINE:
            self.quarantine(action["target"])
        elif atype == ACTION_MARK:
            self.mark(action["target"])
        elif atype == ACTION_ISOLATE_REGION:
            self.isolate_region(action["region"])
        elif atype == ACTION_TRACE_BACK:
            self.trace_back(action["target"])
        elif atype == ACTION_KILL_PROCESS:
            self.kill_process(action["target"])
        elif atype == ACTION_ALERT_ADMIN:
            self.alert_admin(action.get("message", ""))
        elif atype == ACTION_DEMOTE_PRIVILEGE:
            self.demote_privilege(action["target"])
        elif atype == ACTION_RESTORE_VULNERABILITY:
            self.restore_vulnerability(action["target"])
        elif atype == ACTION_RESET_LOGIN_FAIL:
            self.reset_login_fail(action["target"])
        else:
            raise ValueError(f"未知的 emitter 动作类型：{atype}")

    def hack_defense(self, pos: Tuple[int, int]):
        """
        复合“黑客防御”动作：一次性降权、重置登录失败、修补脆弱度
        """
        x, y = pos
        # 1) 降权
        self.env.privilege_level[y, x] = 0.0
        # 2) 重置暴力破解计数
        self.env.login_failures[y, x] = 0.0
        # 3) 修补漏洞强度
        self.env.vulnerability[y, x] *= 0.5
        logger.info(f"[HACK_DEFENSE] 在 {pos} 执行黑客防御：降权 & 重置登录失败 & 修补脆弱度")

        # 4) 清 hack_strength 并从 hacks 字典删除
        self.env.hack_strength[y, x] = 0.0
        if (x, y) in self.env.hacks:
            del self.env.hacks[(x, y)]
        # 5) 清 hack_history 里最近一次记录（避免累计条目总是递增）
        self.env.hack_history[y, x] = max(0.0, self.env.hack_history[y, x] - 1.0)



    def block(self, pos: Tuple[int, int]):
        """
        block 动作：清除单个格子的感染
        """
        x, y = pos
        infected_before = self.env.infected_map[y, x].item()

        self.env.block_connection((x, y))  # 原有行为

        infected_after = self.env.infected_map[y, x].item()
        if infected_before > 0.5 and infected_after == 0.0:
            # 记录是哪个 emitter 清除的
            for u in getattr(self.env, "units", []):
                if getattr(u, "role", None) == "emitter" and hasattr(u, "position") and u.position == (x, y):
                    # 如果没初始化过，也给它一个
                    if not hasattr(u, "cleared_positions"):
                        u.cleared_positions = set()
                    u.cleared_positions.add((x, y))

    def quarantine(self, pos: Tuple[int, int]):
        """
        quarantine 动作：将格点标记为隔离状态，阻止后续传播
        :param pos: (x, y) 格点坐标
        """
        x, y = pos
        self.env.quarantine_zone((x, y))
        # 可在此触发监控或告警接口

    def mark(self, pos: Tuple[int, int]):
        """
        mark 动作：提高格点的行为异常评分，帮助后续判断
        :param pos: (x, y) 格点坐标
        """
        x, y = pos
        self.env.mark_suspicious((x, y))
        # 可在此记录标记历史

    def isolate_region(self, region: List[Tuple[int, int]]):
        """
        isolate_region 动作：对一片区域进行隔离（批量 quarantine）
        :param region: 格点坐标列表 [(x1,y1), (x2,y2), ...]
        """
        for pos in region:
            x, y = pos
            self.env.quarantine_zone((x, y))
        # 可在此做更复杂的区域封锁逻辑

    def alert_admin(self, message: str):
        """
        告警上层系统
        """
        logger.info(f"[ALERT] {message}")

    def demote_privilege(self, pos: Tuple[int, int]):
        x, y = pos
        self.env.privilege_level[y, x] = 0.0
        logger.info(f"[DEPRIV] 降权 {pos} 成功")

    def restore_vulnerability(self, pos: Tuple[int, int]):
        x, y = pos
        self.env.vulnerability[y, x] *= 0.5
        logger.info(f"[PATCH] 降低脆弱度 {pos}")

    def reset_login_fail(self, pos: Tuple[int, int]):
        x, y = pos
        self.env.login_failures[y, x] = 0.0
        logger.info(f"[RESET] 重置暴力破解计数 {pos}")

    def kill_process(self, pos: Tuple[int,int]):
        """对应 Bruteforce 破解后的隔离或重置登录失败"""
        x, y = pos
        # 重置这一节点的登录失败计数
        self.env.login_failures[y, x] = 0
        # 如果已提权，也可以降回去
        self.env.privilege_level[y, x] = 0.0

    def trace_back(self, pos: Tuple[int,int]):
        """记录一次追踪日志，供后续分析或可视化"""
        if not hasattr(self.env, "traceback_log"):
            self.env.traceback_log = []
        self.env.traceback_log.append(pos)
    def get_available_actions(self) -> List[str]:
        """
        返回当前支持的动作类型列表
        """
        return [ACTION_BLOCK, ACTION_QUARANTINE, ACTION_MARK, ACTION_ISOLATE_REGION,ACTION_DEMOTE_PRIVILEGE, ACTION_RESET_LOGIN_FAIL, ACTION_RESTORE_VULNERABILITY, ACTION_HACK_DEFENSE]

