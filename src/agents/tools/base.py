"""Base classes for agent tools."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "metadata": self.metadata,
        }


class Tool(ABC):
    """Base class for all agent tools."""

    def __init__(self, name: str, timeout: float = 10.0):
        self.name = name
        self.timeout = timeout

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def validate_params(self, **kwargs: Any) -> bool:
        """Validate tool parameters. Override in subclasses."""
        return True


class ToolExecutor:
    """Execute multiple tools in parallel with timeout and error handling."""

    def __init__(self, tools: List[Tool]):
        self.tools = {tool.name: tool for tool in tools}

    async def execute_tool(self, tool_name: str, **params: Any) -> ToolResult:
        """Execute a single tool."""
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found",
            )

        tool = self.tools[tool_name]
        if not tool.validate_params(**params):
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Invalid parameters",
            )

        start_time = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool.execute(**params),
                timeout=tool.timeout,
            )
            result.elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "tool_executor.execute_success tool=%s elapsed_ms=%.1f",
                tool_name,
                result.elapsed_ms,
            )
            return result
        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "tool_executor.timeout tool=%s timeout=%.1f elapsed_ms=%.1f",
                tool_name,
                tool.timeout,
                elapsed_ms,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool execution timeout after {tool.timeout}s",
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "tool_executor.error tool=%s error=%s elapsed_ms=%.1f",
                tool_name,
                str(e),
                elapsed_ms,
                exc_info=True,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                elapsed_ms=elapsed_ms,
            )

    async def execute_parallel(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        tasks = [
            self.execute_tool(call["tool_name"], **call.get("params", {}))
            for call in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all available tools (for LLM function calling)."""
        schemas = []
        for tool in self.tools.values():
            if hasattr(tool, "get_schema"):
                schemas.append(tool.get_schema())
            else:
                schemas.append({"name": tool.name, "timeout": tool.timeout})
        return schemas
