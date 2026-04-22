"""Web search tool using Tavily API."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from src.observability.logger import get_logger

from .base import Tool, ToolResult

logger = get_logger(__name__)


class WebSearchTool(Tool):
    """Search the web for latest information using Tavily API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "tavily",
        timeout: float = 10.0,
        max_results: int = 5,
    ):
        super().__init__(name="web_search", timeout=timeout)
        self.api_key = api_key or os.getenv("WEB_SEARCH_API_KEY") or os.getenv("TAVILY_API_KEY")
        self.provider = provider
        self.max_results = max_results
        self.enabled = bool(self.api_key) and os.getenv("TOOL_USING_ENABLED", "true").lower() == "true"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "搜索互联网获取最新资料、时事信息、前沿技术进展。适用于主题涉及最新动态、当前数据或时效性内容时。",
            "parameters": {
                "query": {"type": "string", "description": "搜索关键词，建议包含学科和主题，如'量子计算 物理 2026'"},
            },
            "required": ["query"],
        }

    def validate_params(self, **kwargs: Any) -> bool:
        return "query" in kwargs and isinstance(kwargs["query"], str)

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute web search."""
        if not self.enabled:
            logger.info("web_search.disabled query=%s", query[:50])
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Web search is disabled (missing API key or TOOL_USING_ENABLED=false)",
                metadata={"query": query, "degraded": True},
            )

        if not self.api_key:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Web search API key not configured",
                metadata={"query": query, "degraded": True},
            )

        try:
            results = await self._search_tavily(query)
            logger.info(
                "web_search.success query=%s result_count=%d",
                query[:50],
                len(results),
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results,
                metadata={
                    "query": query,
                    "provider": self.provider,
                    "result_count": len(results),
                },
            )
        except Exception as e:
            logger.error(
                "web_search.error query=%s error=%s",
                query[:50],
                str(e),
                exc_info=True,
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                metadata={"query": query, "degraded": True},
            )

    async def _search_tavily(self, query: str) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
            })

        return results
