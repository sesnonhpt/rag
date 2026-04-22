"""用户行为追踪页面 - 展示从用户请求到 Agent 执行的完整链路."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


def _load_traces(limit: int = 200) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载追踪记录."""
    trace_file = Path("logs/user_action_traces.jsonl")
    if not trace_file.exists():
        return []
    traces = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except Exception:
                    pass
    return list(reversed(traces[-limit:]))  # 最新的在前


def render() -> None:
    st.header("🔍 用户行为追踪")

    traces = _load_traces()

    if not traces:
        st.info("暂无追踪记录。运行一次教案生成后会自动记录。")
        st.code(".venv/bin/python scripts/demo_tracking.py", language="bash")
        return

    # ── 顶部统计卡片 ──────────────────────────────────────────────
    total = len(traces)
    completed = sum(1 for t in traces if t.get("final_status") == "completed")
    failed = sum(1 for t in traces if t.get("final_status") == "failed")
    avg_ms = sum(t.get("total_elapsed_ms", 0) for t in traces) / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总请求数", total)
    c2.metric("成功", completed, delta=None)
    c3.metric("失败", failed, delta=None)
    c4.metric("平均耗时", f"{avg_ms:.0f} ms")

    st.divider()

    # ── 工具 / Agent 使用统计 ─────────────────────────────────────
    col_agent, col_tool = st.columns(2)

    with col_agent:
        st.subheader("🤖 Agent 调用次数")
        agent_counts: Dict[str, int] = {}
        for t in traces:
            for a in t.get("agents", []):
                name = a.get("agent_name", "unknown")
                agent_counts[name] = agent_counts.get(name, 0) + 1
        if agent_counts:
            st.bar_chart(agent_counts)
        else:
            st.info("暂无 Agent 数据")

    with col_tool:
        st.subheader("🔧 工具调用次数")
        tool_counts: Dict[str, int] = {}
        degraded_counts: Dict[str, int] = {}
        for t in traces:
            for tool in t.get("tools", []):
                name = tool.get("tool_name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1
                if tool.get("degraded"):
                    degraded_counts[name] = degraded_counts.get(name, 0) + 1
        if tool_counts:
            st.bar_chart(tool_counts)
            if degraded_counts:
                st.caption(f"⚠️ 降级次数: { {k: v for k, v in degraded_counts.items()} }")
        else:
            st.info("暂无工具数据")

    st.divider()

    # ── 搜索过滤 ──────────────────────────────────────────────────
    keyword = st.text_input("🔎 搜索主题 / Session ID", placeholder="量子计算...")
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t for t in traces
            if kw in str(t.get("action", {}).get("request_data", {})).lower()
            or kw in str(t.get("action", {}).get("session_id", "")).lower()
        ]
        st.caption(f"找到 {len(traces)} 条匹配记录")

    # ── 追踪记录列表 ──────────────────────────────────────────────
    st.subheader(f"📋 追踪记录（{len(traces)} 条）")

    for idx, trace in enumerate(traces):
        action = trace.get("action", {})
        agents = trace.get("agents", [])
        tools = trace.get("tools", [])
        status = trace.get("final_status", "unknown")
        total_ms = trace.get("total_elapsed_ms", 0)
        action_type = action.get("action_type", "unknown")
        request_data = action.get("request_data", {})
        topic = request_data.get("topic", "—")
        session_id = action.get("session_id", "—")
        timestamp = action.get("timestamp", "")[:19].replace("T", " ")

        status_icon = "✅" if status == "completed" else "❌"
        tool_names = [t.get("tool_name", "") for t in tools]
        degraded_tools = [t.get("tool_name", "") for t in tools if t.get("degraded")]

        title = f"{status_icon} {topic[:35]}{'…' if len(topic) > 35 else ''}  ·  {total_ms:.0f}ms  ·  {timestamp}"

        with st.expander(title, expanded=(idx == 0)):
            # 基本信息
            info_col, meta_col = st.columns([3, 1])
            with info_col:
                st.markdown(f"**主题:** {topic}")
                st.markdown(f"**行为类型:** `{action_type}`")
                st.markdown(f"**Session:** `{session_id}`")
            with meta_col:
                st.markdown(f"**状态:** {status_icon} {status}")
                st.markdown(f"**总耗时:** {total_ms:.0f} ms")
                st.markdown(f"**时间:** {timestamp}")

            st.divider()

            # Agent 执行时间线
            if agents:
                st.markdown("#### 🤖 Agent 执行")
                agent_timing = {a["agent_name"]: a.get("elapsed_ms", 0) for a in agents}
                st.bar_chart(agent_timing, height=150)
                cols = st.columns(len(agents))
                for i, agent in enumerate(agents):
                    with cols[i]:
                        a_status = "✅" if agent.get("status") == "completed" else "❌"
                        st.metric(
                            label=f"{a_status} {agent['agent_name']}",
                            value=f"{agent.get('elapsed_ms', 0):.0f} ms",
                        )
                        if agent.get("output_data"):
                            with st.expander("输出详情"):
                                st.json(agent["output_data"])

            # 工具执行
            if tools:
                st.divider()
                st.markdown("#### 🔧 工具执行")
                for tool in tools:
                    t_name = tool.get("tool_name", "unknown")
                    t_ms = tool.get("elapsed_ms", 0)
                    t_status = tool.get("status", "unknown")
                    t_degraded = tool.get("degraded", False)
                    t_error = tool.get("error")

                    icon = "✅" if t_status == "completed" and not t_degraded else ("⚠️" if t_degraded else "❌")
                    label = f"{icon} **{t_name}** — {t_ms:.1f}ms"
                    if t_degraded:
                        label += " _(降级)_"

                    with st.container():
                        st.markdown(label)
                        if t_error:
                            st.caption(f"错误: {t_error}")
                        result = tool.get("result", {})
                        result_data = result.get("data")
                        if result_data and isinstance(result_data, list) and len(result_data) > 0:
                            st.caption(f"返回 {len(result_data)} 条结果")
                        elif isinstance(result_data, dict):
                            with st.expander("查看结果"):
                                st.json(result_data)
