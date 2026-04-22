"""用户行为追踪系统演示脚本."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observability.user_action_tracker import get_tracker


def demo_tracking():
    """演示追踪系统."""
    print("=" * 80)
    print("用户行为追踪系统演示")
    print("=" * 80)
    print()

    tracker = get_tracker()

    # 演示 1：追踪一个完整的用户行为
    print("演示 1：追踪完整的用户行为")
    print("-" * 80)
    
    # 开始追踪
    action_id = tracker.start_action(
        action_type="generate_lesson",
        request_data={
            "topic": "量子计算最新进展",
            "template_category": "comprehensive",
        },
        session_id="demo_session_123",
    )
    print(f"✅ 开始追踪用户行为: {action_id}")
    print()

    # 追踪 PlannerAgent
    planner_exec_id = tracker.start_agent(
        action_id=action_id,
        agent_name="PlannerAgent",
        input_data={
            "topic": "量子计算最新进展",
            "template_category": "comprehensive",
        },
    )
    print(f"✅ 开始追踪 PlannerAgent: {planner_exec_id}")

    # 模拟 Agent 执行
    import time
    time.sleep(0.1)

    tracker.complete_agent(
        action_id=action_id,
        execution_id=planner_exec_id,
        output_data={
            "plan_version": "planner_v1",
            "generation_mode": "context_first",
            "need_images": True,
            "tool_calls": 3,
        },
        status="completed",
    )
    print(f"✅ 完成 PlannerAgent 追踪")
    print()

    # 追踪工具执行
    print("追踪工具执行:")
    
    # Web Search
    tracker.record_tool(
        action_id=action_id,
        execution_id=planner_exec_id,
        tool_name="web_search",
        params={"query": "量子计算 2026"},
        result={
            "success": True,
            "data": [
                {"title": "量子计算突破", "url": "https://example.com/1"},
                {"title": "量子芯片进展", "url": "https://example.com/2"},
            ],
        },
        elapsed_ms=456.7,
        status="completed",
    )
    print("  ✅ web_search: 456.7ms")

    # Image Retrieval
    tracker.record_tool(
        action_id=action_id,
        execution_id=planner_exec_id,
        tool_name="image_retrieval",
        params={"description": "量子计算 原理图"},
        result={
            "success": True,
            "data": [
                {"image_id": "img1", "caption": "量子计算原理图"},
            ],
        },
        elapsed_ms=12.3,
        status="completed",
    )
    print("  ✅ image_retrieval: 12.3ms")

    # LaTeX Renderer (降级)
    tracker.record_tool(
        action_id=action_id,
        execution_id=planner_exec_id,
        tool_name="latex_renderer",
        params={"latex_code": "E = mc^2"},
        result={
            "success": True,
            "data": {"valid": True},
        },
        elapsed_ms=5.6,
        status="completed",
        degraded=True,
    )
    print("  ⚠️  latex_renderer: 5.6ms (降级)")
    print()

    # 追踪 WriterAgent
    writer_exec_id = tracker.start_agent(
        action_id=action_id,
        agent_name="WriterReviewerAgent",
        input_data={
            "topic": "量子计算最新进展",
            "tool_results": 3,
        },
    )
    print(f"✅ 开始追踪 WriterReviewerAgent: {writer_exec_id}")

    time.sleep(0.2)

    tracker.complete_agent(
        action_id=action_id,
        execution_id=writer_exec_id,
        output_data={
            "has_content": True,
            "subject": "物理",
            "review_notes": 2,
        },
        status="completed",
    )
    print(f"✅ 完成 WriterReviewerAgent 追踪")
    print()

    # 完成追踪
    tracker.complete_action(action_id=action_id, final_status="completed")
    print(f"✅ 完成用户行为追踪: {action_id}")
    print()

    # 演示 2：查看追踪记录
    print()
    print("演示 2：查看追踪记录")
    print("-" * 80)
    
    trace = tracker.get_trace(action_id)
    if trace is None:
        # 从文件读取
        import json
        with open(tracker.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trace_dict = json.loads(line)
                    if trace_dict["action"]["action_id"] == action_id:
                        print(f"📋 Action ID: {trace_dict['action']['action_id']}")
                        print(f"📊 总耗时: {trace_dict['total_elapsed_ms']:.1f}ms")
                        print(f"✅ 状态: {trace_dict['final_status']}")
                        print()
                        print(f"🤖 Agent 执行: {len(trace_dict['agents'])} 个")
                        for agent in trace_dict['agents']:
                            print(f"  - {agent['agent_name']}: {agent['elapsed_ms']:.1f}ms")
                        print()
                        print(f"🔧 工具执行: {len(trace_dict['tools'])} 个")
                        for tool in trace_dict['tools']:
                            degraded = "⚠️" if tool.get('degraded') else ""
                            print(f"  - {tool['tool_name']}: {tool['elapsed_ms']:.1f}ms {degraded}")
                        break
    print()

    # 演示 3：统计信息
    print()
    print("演示 3：统计信息")
    print("-" * 80)
    
    stats = tracker.get_statistics()
    print(f"📊 总行为数: {stats['total_actions']}")
    print(f"⏱️  平均耗时: {stats['avg_elapsed_ms']:.1f}ms")
    print()
    
    if stats['action_types']:
        print("📋 行为类型分布:")
        for action_type, count in stats['action_types'].items():
            print(f"  - {action_type}: {count}")
        print()
    
    if stats['agent_usage']:
        print("🤖 Agent 使用统计:")
        for agent_name, count in stats['agent_usage'].items():
            print(f"  - {agent_name}: {count}")
        print()
    
    if stats['tool_usage']:
        print("🔧 工具使用统计:")
        for tool_name, count in stats['tool_usage'].items():
            print(f"  - {tool_name}: {count}")
        print()

    # 演示 4：查看日志文件
    print()
    print("演示 4：日志文件位置")
    print("-" * 80)
    print(f"📁 追踪记录保存在: {tracker.trace_file}")
    print()
    print("查看方式:")
    print(f"  1. 直接查看: cat {tracker.trace_file}")
    print(f"  2. 格式化查看: cat {tracker.trace_file} | jq .")
    print(f"  3. 使用工具: .venv/bin/python scripts/view_user_actions.py")
    print()

    print("=" * 80)
    print("演示完成！")
    print("=" * 80)
    print()
    print("提示:")
    print("  - 追踪系统已自动集成到 LessonOrchestrator")
    print("  - 每次生成教案都会自动记录追踪信息")
    print("  - 使用 view_user_actions.py 查看详细记录")
    print()


if __name__ == "__main__":
    demo_tracking()
