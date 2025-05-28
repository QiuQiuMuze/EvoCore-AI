# emitter_actions.py
from typing import Tuple, List, Optional
from env_net import GridSecurityEnv  # 确保能引用到这个类型
from env import logger

# 动作类型常量
ACTION_BLOCK            = "block"
ACTION_QUARANTINE       = "quarantine"
ACTION_MARK             = "mark"
ACTION_ISOLATE_REGION   = "isolate_region"
ACTION_TRACE_BACK       = "trace_back"
ACTION_KILL_PROCESS     = "kill_process"
ACTION_ALERT_ADMIN      = "alert_admin"
ACTION_DEMOTE_PRIVILEGE = "demote_privilege"
ACTION_RESTORE_VULNERABILITY = "restore_vulnerability"
ACTION_RESET_LOGIN_FAIL = "reset_login_fail"
ACTION_HACK_DEFENSE     = "hack_defense"

class EmitterActions:
    def __init__(self, env: GridSecurityEnv):
        self.env = env

    def perform(self, action: dict, env: Optional[GridSecurityEnv]=None):
        """
        统一执行一个动作字典。
        :param action: {"type": ..., "target": (x,y) 或 "region": [...]}
        :param env: 如果传入，则在该 env 上执行；否则用 self.env
        """
        target_env = env or self.env
        atype = action.get("type")

        if   atype == ACTION_HACK_DEFENSE:      return self._hack_defense(action["target"], target_env)
        elif atype == ACTION_BLOCK:             return self._block(action["target"],     target_env)
        elif atype == ACTION_QUARANTINE:        return self._quarantine(action["target"], target_env)
        elif atype == ACTION_MARK:              return self._mark(action["target"],       target_env)
        elif atype == ACTION_ISOLATE_REGION:    return self._isolate_region(action["region"], target_env)
        elif atype == ACTION_TRACE_BACK:        return self._trace_back(action["target"], target_env)
        elif atype == ACTION_KILL_PROCESS:      return self._kill_process(action["target"], target_env)
        elif atype == ACTION_ALERT_ADMIN:       return self._alert_admin(action.get("message", ""), target_env)
        elif atype == ACTION_DEMOTE_PRIVILEGE:  return self._demote_privilege(action["target"], target_env)
        elif atype == ACTION_RESTORE_VULNERABILITY: return self._restore_vulnerability(action["target"], target_env)
        elif atype == ACTION_RESET_LOGIN_FAIL:  return self._reset_login_fail(action["target"], target_env)
        else:
            raise ValueError(f"未知的 emitter 动作类型：{atype}")

    def _hack_defense(self, pos: Tuple[int,int], env: GridSecurityEnv):
        x, y = pos
        env.privilege_level[y, x] = 0.0
        env.login_failures[y, x]  = 0.0
        env.vulnerability[y, x]  *= 0.5
        env.hack_strength[y, x]   = 0.0
        env.hacks.pop((x, y), None)
        logger.info(f"[HACK_DEFENSE] 在 {pos} 执行黑客防御")
        # 5) 清 hack_history 里最近一次记录（避免累计条目总是递增）
        env.hack_history[y, x] = max(env.hack_history[y, x] - 1.0, 0.0)

    def _block(self, pos: Tuple[int,int], env: GridSecurityEnv):
        x, y = pos
        before = env.infected_map[y, x].item()
        env.block_connection((x, y))
        after = env.infected_map[y, x].item()
        if before > 0.5 and after == 0.0:
            for u in getattr(env, "_external_units", []):
                if getattr(u, "role", None)=="emitter" and getattr(u, "position",None)==(x,y):
                    u.cleared_positions = getattr(u, "cleared_positions", set())
                    u.cleared_positions.add((x,y))

    def _quarantine(self, pos: Tuple[int,int], env: GridSecurityEnv):
        env.quarantine_zone(pos)

    def _mark(self, pos: Tuple[int,int], env: GridSecurityEnv):
        env.mark_suspicious(pos)

    def _isolate_region(self, region: List[Tuple[int,int]], env: GridSecurityEnv):
        for pos in region:
            env.quarantine_zone(pos)

    def _trace_back(self, pos: Tuple[int,int], env: GridSecurityEnv):
        env.traceback_log = getattr(env, "traceback_log", [])
        env.traceback_log.append(pos)

    def _kill_process(self, pos: Tuple[int,int], env: GridSecurityEnv):
        env.login_failures[pos[1], pos[0]] = 0
        env.privilege_level[pos[1], pos[0]] = 0.0

    def _alert_admin(self, message: str, env: GridSecurityEnv):
        logger.info(f"[ALERT] {message}")

    def _demote_privilege(self, pos: Tuple[int,int], env: GridSecurityEnv):
        env.privilege_level[pos[1], pos[0]] = 0.0

    def _restore_vulnerability(self, pos: Tuple[int,int], env: GridSecurityEnv):
        env.vulnerability[pos[1], pos[0]] *= 0.5

    def _reset_login_fail(self, pos: Tuple[int,int], env: GridSecurityEnv):
        env.login_failures[pos[1], pos[0]] = 0.0

    def get_available_actions(self) -> List[str]:
        return [
            ACTION_BLOCK, ACTION_QUARANTINE, ACTION_MARK,
            ACTION_ISOLATE_REGION, ACTION_DEMOTE_PRIVILEGE,
            ACTION_RESET_LOGIN_FAIL, ACTION_RESTORE_VULNERABILITY,
            ACTION_HACK_DEFENSE
        ]
