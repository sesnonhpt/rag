"""用户行为追踪系统 - 记录从用户请求到 Agent 执行的完整链路."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class UserAction:
    """用户行为记录."""

    action_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    action_type: str = ""  # generate_lesson, chat, feedback, etc.
    request_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecution:
    """Agent 执行记录."""

    execution_id: str = field(default_factory=lambda: uuid4().hex)
    action_id: str = ""  # 关联的用户行为 ID
    agent_name: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    elapsed_ms: float = 0.0
    status: str = "started"  # started, completed, failed
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecution:
    """工具执行记录."""

    tool_execution_id: str = field(default_factory=lambda: uuid4().hex)
    action_id: str = ""  # 关联的用户行为 ID
    execution_id: str = ""  # 关联的 Agent 执行 ID
    tool_name: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    elapsed_ms: float = 0.0
    status: str = "started"  # started, completed, failed
    params: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    degraded: bool = False


@dataclass
class ActionTrace:
    """完整的用户行为追踪链路."""

    action: UserAction
    agents: List[AgentExecution] = field(default_factory=list)
    tools: List[ToolExecution] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    final_status: str = "in_progress"  # in_progress, completed, failed


class UserActionTracker:
    """用户行为追踪器 - 记录完整的请求链路."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.log_dir / "user_action_traces.jsonl"
        self.current_traces: Dict[str, ActionTrace] = {}

    def start_action(
        self,
        action_type: str,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """开始记录用户行为."""
        action = UserAction(
            user_id=user_id,
            session_id=session_id,
            action_type=action_type,
            request_data=request_data,
            metadata=metadata or {},
        )

        trace = ActionTrace(action=action)
        self.current_traces[action.action_id] = trace

        logger.info(
            "user_action.start action_id=%s action_type=%s user_id=%s session_id=%s",
            action.action_id,
            action_type,
            user_id,
            session_id,
        )

        return action.action_id

    def start_agent(
        self,
        action_id: str,
        agent_name: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """开始记录 Agent 执行."""
        if action_id not in self.current_traces:
            logger.warning(
                "user_action.agent_start_orphan action_id=%s agent_name=%s",
                action_id,
                agent_name,
            )
            return ""

        execution = AgentExecution(
            action_id=action_id,
            agent_name=agent_name,
            input_data=input_data,
            metadata=metadata or {},
        )

        self.current_traces[action_id].agents.append(execution)

        logger.info(
            "user_action.agent_start action_id=%s execution_id=%s agent_name=%s",
            action_id,
            execution.execution_id,
            agent_name,
        )

        return execution.execution_id

    def complete_agent(
        self,
        action_id: str,
        execution_id: str,
        output_data: Dict[str, Any],
        status: str = "completed",
        error: Optional[str] = None,
    ) -> None:
        """完成 Agent 执行记录."""
        if action_id not in self.current_traces:
            return

        trace = self.current_traces[action_id]
        for execution in trace.agents:
            if execution.execution_id == execution_id:
                execution.completed_at = datetime.now(timezone.utc).isoformat()
                execution.status = status
                execution.output_data = output_data
                execution.error = error

                # 计算执行时间
                started = datetime.fromisoformat(execution.started_at)
                completed = datetime.fromisoformat(execution.completed_at)
                execution.elapsed_ms = (completed - started).total_seconds() * 1000

                logger.info(
                    "user_action.agent_complete action_id=%s execution_id=%s agent_name=%s status=%s elapsed_ms=%.1f",
                    action_id,
                    execution_id,
                    execution.agent_name,
                    status,
                    execution.elapsed_ms,
                )
                break

    def record_tool(
        self,
        action_id: str,
        execution_id: str,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        elapsed_ms: float,
        status: str = "completed",
        error: Optional[str] = None,
        degraded: bool = False,
    ) -> None:
        """记录工具执行."""
        if action_id not in self.current_traces:
            return

        tool_execution = ToolExecution(
            action_id=action_id,
            execution_id=execution_id,
            tool_name=tool_name,
            params=params,
            result=result,
            elapsed_ms=elapsed_ms,
            status=status,
            error=error,
            degraded=degraded,
        )
        tool_execution.completed_at = datetime.now(timezone.utc).isoformat()

        self.current_traces[action_id].tools.append(tool_execution)

        logger.info(
            "user_action.tool_execute action_id=%s execution_id=%s tool_name=%s status=%s elapsed_ms=%.1f degraded=%s",
            action_id,
            execution_id,
            tool_name,
            status,
            elapsed_ms,
            degraded,
        )

    def complete_action(
        self,
        action_id: str,
        final_status: str = "completed",
    ) -> None:
        """完成用户行为记录."""
        if action_id not in self.current_traces:
            return

        trace = self.current_traces[action_id]
        trace.final_status = final_status

        # 计算总执行时间
        action_started = datetime.fromisoformat(trace.action.timestamp)
        action_completed = datetime.now(timezone.utc)
        trace.total_elapsed_ms = (action_completed - action_started).total_seconds() * 1000

        # 写入日志文件
        self._write_trace(trace)

        logger.info(
            "user_action.complete action_id=%s action_type=%s final_status=%s total_elapsed_ms=%.1f agent_count=%d tool_count=%d",
            action_id,
            trace.action.action_type,
            final_status,
            trace.total_elapsed_ms,
            len(trace.agents),
            len(trace.tools),
        )

        # 清理内存
        del self.current_traces[action_id]

    def get_trace(self, action_id: str) -> Optional[ActionTrace]:
        """获取追踪记录."""
        return self.current_traces.get(action_id)

    def _write_trace(self, trace: ActionTrace) -> None:
        """写入追踪记录到文件."""
        try:
            trace_dict = {
                "action": asdict(trace.action),
                "agents": [asdict(agent) for agent in trace.agents],
                "tools": [asdict(tool) for tool in trace.tools],
                "total_elapsed_ms": trace.total_elapsed_ms,
                "final_status": trace.final_status,
            }

            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(
                "user_action.write_trace_error action_id=%s error=%s",
                trace.action.action_id,
                str(e),
                exc_info=True,
            )

    def get_statistics(self, limit: int = 100) -> Dict[str, Any]:
        """获取统计信息."""
        try:
            traces = []
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))

            # 只取最近的记录
            traces = traces[-limit:]

            if not traces:
                return {
                    "total_actions": 0,
                    "action_types": {},
                    "avg_elapsed_ms": 0,
                    "agent_usage": {},
                    "tool_usage": {},
                }

            # 统计
            action_types = {}
            agent_usage = {}
            tool_usage = {}
            total_elapsed = 0

            for trace in traces:
                # 行为类型统计
                action_type = trace["action"]["action_type"]
                action_types[action_type] = action_types.get(action_type, 0) + 1

                # Agent 使用统计
                for agent in trace["agents"]:
                    agent_name = agent["agent_name"]
                    agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1

                # 工具使用统计
                for tool in trace["tools"]:
                    tool_name = tool["tool_name"]
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

                # 总时间
                total_elapsed += trace["total_elapsed_ms"]

            return {
                "total_actions": len(traces),
                "action_types": action_types,
                "avg_elapsed_ms": total_elapsed / len(traces) if traces else 0,
                "agent_usage": agent_usage,
                "tool_usage": tool_usage,
            }

        except Exception as e:
            logger.error("user_action.get_statistics_error error=%s", str(e), exc_info=True)
            return {
                "total_actions": 0,
                "action_types": {},
                "avg_elapsed_ms": 0,
                "agent_usage": {},
                "tool_usage": {},
                "error": str(e),
            }


# 全局追踪器实例
_global_tracker: Optional[UserActionTracker] = None


def get_tracker() -> UserActionTracker:
    """获取全局追踪器实例."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = UserActionTracker()
    return _global_tracker
