"""
导学案深度分析服务 - 流式输出版本

使用 LLM 的 chat_stream，每个 chunk 通过 asyncio.to_thread 取出，
避免同步迭代阻塞事件循环导致所有 token 积压后一次性 flush。
"""

from __future__ import annotations

import re
import asyncio
from typing import AsyncGenerator, Dict, Any

from src.observability.logger import get_logger

logger = get_logger(__name__)


class LessonDeepAnalyzer:
    """导学案深度分析器 - 流式输出"""

    def __init__(self, llm):
        self.llm = llm

    async def analyze_stream(
        self,
        content: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式分析导学案 - 完整分析，真正逐 token 输出

        Args:
            content: 导学案文本内容

        Yields:
            流式事件，格式：
            {
                "event": "stage_start" | "content" | "stage_complete" | "complete" | "error",
                "stage": "basic" | "structure" | "design_intent" | "student_perspective" | "improvement" | "skeleton",
                "data": {...}
            }
        """
        logger.info(f"lesson_deep_analyzer.analyze_stream_start content_length={len(content)}")

        accumulated_content = ""

        try:
            from src.libs.llm.base_llm import Message

            # 构建完整的分析 prompt
            prompt = self._build_complete_analysis_prompt(content)

            # 阶段标记
            stages = [
                ("basic", "📊 基础信息"),
                ("structure", "📝 结构分析"),
                ("design_intent", "💡 设计意图"),
                ("improvement", "✨ 改进建议"),
            ]

            current_stage_index = 0
            current_stage = stages[0][0]

            # 发送第一个阶段开始事件
            yield {
                "event": "stage_start",
                "stage": current_stage,
                "data": {"title": stages[0][1]}
            }

            # 在线程中启动同步生成器，避免阻塞事件循环
            def _start_stream():
                return self.llm.chat_stream([
                    Message(role="system", content="你是资深教研员，擅长深度分析导学案。"),
                    Message(role="user", content=prompt),
                ])

            def _next_chunk(gen):
                """取下一个 chunk，返回 None 表示结束"""
                return next(gen, None)

            stream_gen = await asyncio.to_thread(_start_stream)

            # 每次 next() 都在线程池中执行，让出事件循环，实现真正的逐 token 流式
            while True:
                chunk = await asyncio.to_thread(_next_chunk, stream_gen)
                if chunk is None:
                    break

                if hasattr(chunk, 'content') and chunk.content:
                    token = chunk.content
                    accumulated_content += token

                    # 检测阶段标记，切换阶段
                    for i, (stage_name, stage_title) in enumerate(stages):
                        stage_marker = f"## {stage_title}"
                        if stage_marker in accumulated_content and i > current_stage_index:
                            # 完成当前阶段
                            yield {
                                "event": "stage_complete",
                                "stage": current_stage,
                                "data": {}
                            }
                            # 开始新阶段
                            current_stage_index = i
                            current_stage = stage_name
                            yield {
                                "event": "stage_start",
                                "stage": current_stage,
                                "data": {"title": stage_title}
                            }
                            break

                    # 发送内容 token
                    yield {
                        "event": "content",
                        "stage": current_stage,
                        "data": {"token": token}
                    }

            # 完成最后一个阶段
            yield {
                "event": "stage_complete",
                "stage": current_stage,
                "data": {}
            }

            # 解析完整内容并发送完整结果
            parsed_result = self._parse_complete_analysis(accumulated_content)
            yield {
                "event": "complete",
                "data": {
                    "full_content": accumulated_content,
                    "parsed": parsed_result
                }
            }

        except Exception as exc:
            logger.exception("lesson_deep_analyzer.analyze_stream_failed")
            yield {"event": "error", "data": {"message": str(exc)}}

    def _build_complete_analysis_prompt(self, content: str) -> str:
        """构建精简的分析 prompt，面向老师快速阅读"""
        text_preview = content[:6000] if len(content) > 6000 else content

        return f"""请对这份导学案做一个简明分析，给老师备课参考用。

要求：
- 每个部分只写关键点，不要展开解释
- 每条不超过 20 字
- 总字数控制在 400 字以内
- 用 Markdown 格式，二级标题分隔各部分

## 📊 基础信息
用 3-4 个字段列出：课题、学科年级、难度、预估时长

## 📝 结构分析
- 列出 3-5 个主要教学环节（一行一个）
- 列出 2-3 个核心重难点

## 💡 设计意图
用 2-3 句话说清楚：这节课的主线是什么、为什么这样设计

## ✨ 改进建议
列出 2-3 条最值得改进的地方，直接说问题和建议

---

导学案内容：
{text_preview}
"""

    def _parse_complete_analysis(self, content: str) -> Dict[str, Any]:
        """解析完整的分析内容，提取各个部分"""
        sections = {}

        patterns = {
            "basic": r"## 📊 基础信息(.*?)(?=## |$)",
            "structure": r"## 📝 结构分析(.*?)(?=## |$)",
            "design_intent": r"## 💡 设计意图(.*?)(?=## |$)",
            "improvement": r"## ✨ 改进建议(.*?)(?=$)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()

        return sections
