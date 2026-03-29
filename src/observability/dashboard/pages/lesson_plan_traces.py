"""Lesson Plan Traces page – browse lesson plan generation history.

Layout:
1. Optional keyword search filter
2. Trace list (reverse-chronological, filtered to trace_type=="lesson_plan")
3. Detail view: topic, subject, generated content, and retrieval metrics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


def render() -> None:
    """Render the Lesson Plan Traces page."""
    st.header("📚 教案追踪 (Lesson Plan Traces)")

    svc = TraceService()
    traces = svc.list_traces(trace_type="lesson_plan")

    if not traces:
        st.info("暂无教案生成记录。请先生成教案！")
        return

    keyword = st.text_input(
        "搜索教案主题",
        value="",
        key="lp_keyword",
    )
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t
            for t in traces
            if kw in str(t.get("metadata", {})).lower()
            or kw in str(t.get("stages", [])).lower()
        ]

    st.subheader(f"📋 教案历史 ({len(traces)})")

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        topic = meta.get("topic", "")
        subject = meta.get("subject", "—")
        collection = meta.get("collection", "default")

        topic_preview = (
            topic[:40] + "…" if len(topic) > 40 else topic
        ) if topic else "—"
        expander_title = (
            f"📝 \"{topic_preview}\"  ·  {subject}  ·  {total_label}  ·  {started[:19]}"
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            st.markdown("#### 📝 教案信息")
            col_q, col_meta = st.columns([3, 1])
            with col_q:
                st.markdown(f"**主题:** {topic}")
                st.markdown(f"**学科:** {subject}")
            with col_meta:
                st.markdown(f"**集合:** `{collection}`")
                st.markdown(f"**耗时:** {total_label}")

            st.divider()

            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            # 从fusion阶段获取检索结果
            fusion_d = (stages_by_name.get("fusion", {}).get("data") or {})
            rerank_d = (stages_by_name.get("rerank", {}).get("data") or {})

            retrieval_count = fusion_d.get("result_count", 0)
            rerank_count = rerank_d.get("output_count", 0)

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.metric("检索结果数", retrieval_count)
            with rc2:
                st.metric("重排后结果", rerank_count if rerank_d else "—")
            with rc3:
                st.metric("总耗时", total_label)

            st.divider()

            main_stage_names = ("hybrid_search", "rerank", "llm_generation")
            main_timings = [t for t in timings if t["stage_name"] in main_stage_names]
            if main_timings:
                st.markdown("#### ⏱️ 阶段耗时")
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_timings}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        "阶段": t["stage_name"],
                        "耗时 (ms)": round(t["elapsed_ms"], 2),
                    }
                    for t in main_timings
                ])

            st.divider()

            st.markdown("#### 🔍 阶段详情")

            tab_defs = []
            if "fusion" in stages_by_name:
                tab_defs.append(("🔍 混合检索", "fusion"))
            if "rerank" in stages_by_name:
                tab_defs.append(("📊 重排序", "rerank"))
            if "llm_generation" in stages_by_name:
                tab_defs.append(("🤖 LLM生成", "llm_generation"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "fusion":
                            _render_fusion_stage(data)
                        elif key == "rerank":
                            _render_rerank_stage(data)
                        elif key == "llm_generation":
                            _render_llm_generation_stage(data)
            else:
                st.info("暂无阶段详情。")

            if retrieval_count > 0:
                st.divider()
                st.markdown("#### 📖 检索到的文档")
                chunks = fusion_d.get("chunks", [])
                if chunks:
                    _render_chunk_list(chunks)
                else:
                    st.info("未检索到相关文档（基于LLM自身知识生成）")


def _render_fusion_stage(data: Dict[str, Any]) -> None:
    """Render fusion stage details."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("融合方法", data.get("method", "rrf"))
    with c2:
        st.metric("结果数量", data.get("result_count", 0))
    with c3:
        st.metric("Top-K", data.get("top_k", "—"))

    input_lists = data.get("input_lists", 0)
    if input_lists:
        st.markdown(f"**输入列表数:** `{input_lists}`")


def _render_hybrid_search_stage(data: Dict[str, Any]) -> None:
    """Render hybrid search stage details."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("检索方法", data.get("method", "hybrid"))
    with c2:
        st.metric("结果数量", data.get("result_count", 0))
    with c3:
        st.metric("Top-K", data.get("top_k", "—"))

    query = data.get("query", "")
    if query:
        st.markdown(f"**扩展查询:** `{query}`")


def _render_rerank_stage(data: Dict[str, Any]) -> None:
    """Render rerank stage details."""
    c1, c2 = st.columns(2)
    with c1:
        st.metric("输入数量", data.get("input_count", 0))
    with c2:
        st.metric("输出数量", data.get("output_count", 0))

    if data.get("error"):
        st.error(f"重排失败: {data['error']}")


def _render_llm_generation_stage(data: Dict[str, Any]) -> None:
    """Render LLM generation stage details."""
    st.metric("模型", data.get("model", "—"))
    st.metric("Token数", data.get("tokens", "—"))

    if data.get("error"):
        st.error(f"生成失败: {data['error']}")


def _render_chunk_list(chunks: List[Dict[str, Any]]) -> None:
    """Render a list of retrieved chunks."""
    for i, chunk in enumerate(chunks[:10]):
        with st.container():
            col_idx, col_score, col_source = st.columns([1, 1, 3])
            with col_idx:
                st.markdown(f"**#{i+1}**")
            with col_score:
                score = chunk.get("score", 0)
                st.markdown(f"相关度: `{score:.4f}`")
            with col_source:
                source = chunk.get("metadata", {}).get("source_path", "unknown")
                st.markdown(f"来源: `{source}`")

            text = chunk.get("text", "")
            if text:
                with st.expander("查看内容", expanded=False):
                    st.text(text[:500] + "..." if len(text) > 500 else text)

            st.divider()
