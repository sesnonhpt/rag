"""用户行为追踪系统测试."""

import json
import tempfile
from pathlib import Path

import pytest

from src.observability.user_action_tracker import (
    UserAction,
    AgentExecution,
    ToolExecution,
    ActionTrace,
    UserActionTracker,
)


class TestUserActionTracker:
    """测试用户行为追踪器."""

    def test_start_action(self):
        """测试开始追踪用户行为."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UserActionTracker(log_dir=tmpdir)
            
            action_id = tracker.start_action(
                action_type="generate_lesson",
                request_data={"topic": "量子计算"},
                user_id="user_123",
                session_id="session_456",
            )
            
            assert action_id is not None
            assert action_id in tracker.current_traces
            
            trace = tracker.get_trace(action_id)
            assert trace is not None
            assert trace.action.action_type == "generate_lesson"
            assert trace.action.request_data["topic"] == "量子计算"

    def test_agent_execution(self):
        """测试 Agent 执行追踪."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UserActionTracker(log_dir=tmpdir)
            
            action_id = tracker.start_action(
                action_type="generate_lesson",
                request_data={"topic": "量子计算"},
            )
            
            execution_id = tracker.start_agent(
                action_id=action_id,
                agent_name="PlannerAgent",
                input_data={"topic": "量子计算"},
            )
            
            assert execution_id is not None
            
            trace = tracker.get_trace(action_id)
            assert len(trace.agents) == 1
            assert trace.agents[0].agent_name == "PlannerAgent"
            assert trace.agents[0].status == "started"
            
            # 完成 Agent 执行
            tracker.complete_agent(
                action_id=action_id,
                execution_id=execution_id,
                output_data={"plan_version": "v1"},
                status="completed",
            )
            
            trace = tracker.get_trace(action_id)
            assert trace.agents[0].status == "completed"
            assert trace.agents[0].elapsed_ms > 0

    def test_tool_execution(self):
        """测试工具执行追踪."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UserActionTracker(log_dir=tmpdir)
            
            action_id = tracker.start_action(
                action_type="generate_lesson",
                request_data={"topic": "量子计算"},
            )
            
            execution_id = tracker.start_agent(
                action_id=action_id,
                agent_name="PlannerAgent",
                input_data={},
            )
            
            tracker.record_tool(
                action_id=action_id,
                execution_id=execution_id,
                tool_name="web_search",
                params={"query": "量子计算"},
                result={"data": []},
                elapsed_ms=123.4,
                status="completed",
                degraded=False,
            )
            
            trace = tracker.get_trace(action_id)
            assert len(trace.tools) == 1
            assert trace.tools[0].tool_name == "web_search"
            assert trace.tools[0].elapsed_ms == 123.4
            assert trace.tools[0].degraded is False

    def test_complete_action(self):
        """测试完成用户行为追踪."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UserActionTracker(log_dir=tmpdir)
            
            action_id = tracker.start_action(
                action_type="generate_lesson",
                request_data={"topic": "量子计算"},
            )
            
            execution_id = tracker.start_agent(
                action_id=action_id,
                agent_name="PlannerAgent",
                input_data={},
            )
            
            tracker.complete_agent(
                action_id=action_id,
                execution_id=execution_id,
                output_data={},
            )
            
            tracker.complete_action(action_id=action_id, final_status="completed")
            
            # 检查是否写入文件
            trace_file = Path(tmpdir) / "user_action_traces.jsonl"
            assert trace_file.exists()
            
            # 读取并验证
            with open(trace_file, "r", encoding="utf-8") as f:
                line = f.readline()
                trace_dict = json.loads(line)
                
                assert trace_dict["action"]["action_id"] == action_id
                assert trace_dict["final_status"] == "completed"
                assert trace_dict["total_elapsed_ms"] > 0
                assert len(trace_dict["agents"]) == 1
            
            # 检查内存是否清理
            assert action_id not in tracker.current_traces

    def test_get_statistics(self):
        """测试获取统计信息."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UserActionTracker(log_dir=tmpdir)
            
            # 创建多个追踪记录
            for i in range(3):
                action_id = tracker.start_action(
                    action_type="generate_lesson",
                    request_data={"topic": f"主题{i}"},
                )
                
                execution_id = tracker.start_agent(
                    action_id=action_id,
                    agent_name="PlannerAgent",
                    input_data={},
                )
                
                tracker.complete_agent(
                    action_id=action_id,
                    execution_id=execution_id,
                    output_data={},
                )
                
                tracker.record_tool(
                    action_id=action_id,
                    execution_id=execution_id,
                    tool_name="web_search",
                    params={},
                    result={},
                    elapsed_ms=100.0,
                )
                
                tracker.complete_action(action_id=action_id)
            
            # 获取统计信息
            stats = tracker.get_statistics()
            
            assert stats["total_actions"] == 3
            assert stats["action_types"]["generate_lesson"] == 3
            assert stats["agent_usage"]["PlannerAgent"] == 3
            assert stats["tool_usage"]["web_search"] == 3
            assert stats["avg_elapsed_ms"] > 0

    def test_degraded_tool(self):
        """测试降级工具追踪."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UserActionTracker(log_dir=tmpdir)
            
            action_id = tracker.start_action(
                action_type="generate_lesson",
                request_data={"topic": "量子计算"},
            )
            
            execution_id = tracker.start_agent(
                action_id=action_id,
                agent_name="PlannerAgent",
                input_data={},
            )
            
            # 记录降级的工具
            tracker.record_tool(
                action_id=action_id,
                execution_id=execution_id,
                tool_name="web_search",
                params={"query": "量子计算"},
                result={},
                elapsed_ms=5.0,
                status="failed",
                error="API key not configured",
                degraded=True,
            )
            
            trace = tracker.get_trace(action_id)
            assert len(trace.tools) == 1
            assert trace.tools[0].degraded is True
            assert trace.tools[0].status == "failed"
            assert trace.tools[0].error == "API key not configured"

    def test_orphan_agent(self):
        """测试孤立的 Agent 执行（没有对应的 action）."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = UserActionTracker(log_dir=tmpdir)
            
            # 尝试在不存在的 action_id 上开始 Agent
            execution_id = tracker.start_agent(
                action_id="nonexistent_action",
                agent_name="PlannerAgent",
                input_data={},
            )
            
            # 应该返回空字符串
            assert execution_id == ""
