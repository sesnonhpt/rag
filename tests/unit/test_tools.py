"""Unit tests for agent tools."""

import pytest
import asyncio

from src.agents.tools.base import Tool, ToolResult, ToolExecutor
from src.agents.tools.web_search import WebSearchTool
from src.agents.tools.image_retrieval import ImageRetrievalTool
from src.agents.tools.latex_renderer import LaTeXRendererTool


class TestWebSearchTool:
    """Test web search tool."""

    @pytest.mark.asyncio
    async def test_web_search_disabled(self):
        """Test web search when disabled."""
        tool = WebSearchTool(api_key=None)
        result = await tool.execute(query="量子计算")
        
        assert result.tool_name == "web_search"
        assert not result.success
        assert "disabled" in result.error.lower()
        assert result.metadata.get("degraded") is True

    @pytest.mark.asyncio
    async def test_web_search_validation(self):
        """Test parameter validation."""
        tool = WebSearchTool(api_key="test_key")
        
        # Valid params
        assert tool.validate_params(query="test")
        
        # Invalid params
        assert not tool.validate_params()
        assert not tool.validate_params(query=123)


class TestImageRetrievalTool:
    """Test image retrieval tool."""

    @pytest.mark.asyncio
    async def test_image_retrieval_empty(self):
        """Test image retrieval with no resources."""
        tool = ImageRetrievalTool(image_resources=[])
        result = await tool.execute(description="牛顿第二定律")
        
        assert result.tool_name == "image_retrieval"
        assert result.success
        assert result.data == []
        assert result.metadata.get("result_count") == 0

    @pytest.mark.asyncio
    async def test_image_retrieval_keyword_match(self):
        """Test keyword-based image matching."""
        images = [
            {
                "image_id": "img1",
                "url": "http://example.com/newton.jpg",
                "caption": "牛顿第二定律示意图",
                "context": "力学",
            },
            {
                "image_id": "img2",
                "url": "http://example.com/cell.jpg",
                "caption": "细胞结构图",
                "context": "生物",
            },
        ]
        
        tool = ImageRetrievalTool(image_resources=images, max_results=2)
        result = await tool.execute(description="牛顿 力学")
        
        assert result.success
        assert len(result.data) > 0
        assert result.data[0]["image_id"] == "img1"

    @pytest.mark.asyncio
    async def test_image_retrieval_validation(self):
        """Test parameter validation."""
        tool = ImageRetrievalTool()
        
        # Valid params
        assert tool.validate_params(description="test")
        
        # Invalid params
        assert not tool.validate_params()
        assert not tool.validate_params(description=123)


class TestLaTeXRendererTool:
    """Test LaTeX renderer tool."""

    @pytest.mark.asyncio
    async def test_latex_valid_syntax(self):
        """Test valid LaTeX syntax."""
        tool = LaTeXRendererTool()
        result = await tool.execute(latex_code=r"F = ma")
        
        assert result.tool_name == "latex_renderer"
        assert result.success
        assert result.data["valid"] is True
        assert "latex_code" in result.data

    @pytest.mark.asyncio
    async def test_latex_invalid_braces(self):
        """Test invalid LaTeX with unbalanced braces."""
        tool = LaTeXRendererTool()
        result = await tool.execute(latex_code=r"\frac{a{b}")
        
        assert result.success  # Non-critical failure
        assert result.data["valid"] is False
        assert "braces" in result.data["error"].lower()

    @pytest.mark.asyncio
    async def test_latex_disabled(self):
        """Test LaTeX when disabled."""
        import os
        os.environ["LATEX_RENDERER_ENABLED"] = "false"
        
        tool = LaTeXRendererTool()
        result = await tool.execute(latex_code=r"E = mc^2")
        
        assert result.success
        assert result.data["valid"] is True
        assert result.metadata.get("degraded") is True
        
        # Cleanup
        os.environ["LATEX_RENDERER_ENABLED"] = "true"

    @pytest.mark.asyncio
    async def test_latex_validation(self):
        """Test parameter validation."""
        tool = LaTeXRendererTool()
        
        # Valid params
        assert tool.validate_params(latex_code="test")
        
        # Invalid params
        assert not tool.validate_params()
        assert not tool.validate_params(latex_code=123)


class TestToolExecutor:
    """Test tool executor."""

    @pytest.mark.asyncio
    async def test_execute_single_tool(self):
        """Test executing a single tool."""
        tool = LaTeXRendererTool()
        executor = ToolExecutor([tool])
        
        result = await executor.execute_tool("latex_renderer", latex_code=r"F = ma")
        
        assert result.tool_name == "latex_renderer"
        assert result.success
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_execute_parallel_tools(self):
        """Test executing multiple tools in parallel."""
        tools = [
            ImageRetrievalTool(image_resources=[]),
            LaTeXRendererTool(),
        ]
        executor = ToolExecutor(tools)
        
        tool_calls = [
            {"tool_name": "image_retrieval", "params": {"description": "test"}},
            {"tool_name": "latex_renderer", "params": {"latex_code": "F = ma"}},
        ]
        
        results = await executor.execute_parallel(tool_calls)
        
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].tool_name == "image_retrieval"
        assert results[1].tool_name == "latex_renderer"

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test executing a non-existent tool."""
        executor = ToolExecutor([])
        
        result = await executor.execute_tool("nonexistent_tool")
        
        assert result.tool_name == "nonexistent_tool"
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tool_timeout(self):
        """Test tool execution timeout."""
        class SlowTool(Tool):
            def __init__(self):
                super().__init__(name="slow_tool", timeout=0.1)
            
            async def execute(self, **kwargs):
                await asyncio.sleep(1.0)  # Sleep longer than timeout
                return ToolResult(tool_name=self.name, success=True)
        
        executor = ToolExecutor([SlowTool()])
        result = await executor.execute_tool("slow_tool")
        
        assert result.tool_name == "slow_tool"
        assert not result.success
        assert "timeout" in result.error.lower()
