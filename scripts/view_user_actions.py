"""查看用户行为追踪记录的工具脚本."""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observability.user_action_tracker import get_tracker


def format_timestamp(ts: str) -> str:
    """格式化时间戳."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ts


def print_trace(trace: Dict[str, Any]) -> None:
    """打印单个追踪记录."""
    action = trace["action"]
    agents = trace["agents"]
    tools = trace["tools"]
    
    print("=" * 80)
    print(f"📋 用户行为: {action['action_type']}")
    print(f"🆔 Action ID: {action['action_id']}")
    print(f"⏰ 时间: {format_timestamp(action['timestamp'])}")
    print(f"👤 Session ID: {action.get('session_id', 'N/A')}")
    print(f"📊 总耗时: {trace['total_elapsed_ms']:.1f}ms")
    print(f"✅ 状态: {trace['final_status']}")
    print()
    
    # 请求数据
    print("📝 请求数据:")
    for key, value in action['request_data'].items():
        print(f"  - {key}: {value}")
    print()
    
    # Agent 执行
    if agents:
        print(f"🤖 Agent 执行 ({len(agents)} 个):")
        for agent in agents:
            status_icon = "✅" if agent['status'] == "completed" else "❌"
            print(f"  {status_icon} {agent['agent_name']}")
            print(f"     耗时: {agent['elapsed_ms']:.1f}ms")
            print(f"     状态: {agent['status']}")
            if agent.get('error'):
                print(f"     错误: {agent['error']}")
        print()
    
    # 工具执行
    if tools:
        print(f"🔧 工具执行 ({len(tools)} 个):")
        for tool in tools:
            status_icon = "✅" if tool['status'] == "completed" else "❌"
            degraded_icon = "⚠️" if tool.get('degraded') else ""
            print(f"  {status_icon}{degraded_icon} {tool['tool_name']}")
            print(f"     耗时: {tool['elapsed_ms']:.1f}ms")
            print(f"     状态: {tool['status']}")
            if tool.get('degraded'):
                print(f"     降级: 是")
            if tool.get('error'):
                print(f"     错误: {tool['error']}")
        print()


def print_statistics(stats: Dict[str, Any]) -> None:
    """打印统计信息."""
    print("=" * 80)
    print("📊 统计信息")
    print("=" * 80)
    print()
    
    print(f"总行为数: {stats['total_actions']}")
    print(f"平均耗时: {stats['avg_elapsed_ms']:.1f}ms")
    print()
    
    if stats['action_types']:
        print("行为类型分布:")
        for action_type, count in sorted(stats['action_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {action_type}: {count}")
        print()
    
    if stats['agent_usage']:
        print("Agent 使用统计:")
        for agent_name, count in sorted(stats['agent_usage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {agent_name}: {count}")
        print()
    
    if stats['tool_usage']:
        print("工具使用统计:")
        for tool_name, count in sorted(stats['tool_usage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {tool_name}: {count}")
        print()


def main():
    """主函数."""
    import argparse
    
    parser = argparse.ArgumentParser(description="查看用户行为追踪记录")
    parser.add_argument("--limit", type=int, default=10, help="显示最近的 N 条记录")
    parser.add_argument("--stats", action="store_true", help="只显示统计信息")
    parser.add_argument("--action-id", type=str, help="查看特定 action_id 的记录")
    
    args = parser.parse_args()
    
    tracker = get_tracker()
    trace_file = tracker.trace_file
    
    if not trace_file.exists():
        print(f"❌ 追踪文件不存在: {trace_file}")
        print("提示: 运行一次教案生成后会自动创建追踪记录")
        return
    
    # 读取所有记录
    traces = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    
    if not traces:
        print("📭 暂无追踪记录")
        return
    
    # 只显示统计信息
    if args.stats:
        stats = tracker.get_statistics(limit=len(traces))
        print_statistics(stats)
        return
    
    # 查看特定 action_id
    if args.action_id:
        found = False
        for trace in traces:
            if trace["action"]["action_id"] == args.action_id:
                print_trace(trace)
                found = True
                break
        if not found:
            print(f"❌ 未找到 action_id: {args.action_id}")
        return
    
    # 显示最近的记录
    recent_traces = traces[-args.limit:]
    print(f"📋 最近 {len(recent_traces)} 条用户行为记录:")
    print()
    
    for trace in recent_traces:
        print_trace(trace)
    
    # 显示统计信息
    print()
    stats = tracker.get_statistics(limit=len(traces))
    print_statistics(stats)


if __name__ == "__main__":
    main()
