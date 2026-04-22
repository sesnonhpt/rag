"""Demo script for tool-using writer agent."""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.tools import WebSearchTool, ImageRetrievalTool, LaTeXRendererTool
from src.agents.tools.base import ToolExecutor
from src.agents.tool_planner import ToolPlanner


async def demo_tools():
    """Demonstrate tool usage."""
    print("=" * 60)
    print("Tool-Using Writer Agent Demo")
    print("=" * 60)
    print()

    # Initialize tools
    web_search = WebSearchTool()
    image_retrieval = ImageRetrievalTool(
        image_resources=[
            {
                "image_id": "img1",
                "url": "http://example.com/newton.jpg",
                "caption": "牛顿第二定律示意图 F=ma",
                "context": "力学 物理",
            },
            {
                "image_id": "img2",
                "url": "http://example.com/quantum.jpg",
                "caption": "量子计算原理图",
                "context": "量子物理 计算机",
            },
        ]
    )
    latex_renderer = LaTeXRendererTool()

    executor = ToolExecutor([web_search, image_retrieval, latex_renderer])

    # Demo 1: Tool Planning
    print("Demo 1: Tool Planning")
    print("-" * 60)
    planner = ToolPlanner()
    
    topics = [
        ("量子计算最新进展", "物理"),
        ("牛顿第二定律", "物理"),
        ("细胞结构", "生物"),
    ]
    
    for topic, subject in topics:
        tool_calls = planner.plan_tools(
            topic=topic,
            subject=subject,
            template_category="comprehensive",
        )
        print(f"\nTopic: {topic} (Subject: {subject})")
        print(f"Planned tools: {[t['tool_name'] for t in tool_calls]}")
        for call in tool_calls:
            print(f"  - {call['tool_name']}: {call.get('reason', 'N/A')}")
    
    print()
    print()

    # Demo 2: Execute Tools
    print("Demo 2: Execute Tools")
    print("-" * 60)
    
    tool_calls = [
        {
            "tool_name": "image_retrieval",
            "params": {"description": "牛顿 力学 F=ma"},
        },
        {
            "tool_name": "latex_renderer",
            "params": {"latex_code": r"F = ma"},
        },
    ]
    
    print("\nExecuting tools in parallel...")
    results = await executor.execute_parallel(tool_calls)
    
    for result in results:
        print(f"\nTool: {result.tool_name}")
        print(f"Success: {result.success}")
        print(f"Elapsed: {result.elapsed_ms:.1f}ms")
        if result.success:
            if result.tool_name == "image_retrieval":
                print(f"Found {len(result.data)} images")
                for img in result.data:
                    print(f"  - {img.get('caption', 'N/A')}")
            elif result.tool_name == "latex_renderer":
                print(f"Valid: {result.data.get('valid')}")
                if result.data.get('valid'):
                    print(f"LaTeX: {result.data.get('latex_code')}")
        else:
            print(f"Error: {result.error}")
    
    print()
    print()

    # Demo 3: Degradation Strategy
    print("Demo 3: Degradation Strategy")
    print("-" * 60)
    
    print("\nTesting web search without API key (should degrade gracefully)...")
    result = await executor.execute_tool("web_search", query="量子计算")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Degraded: {result.metadata.get('degraded', False)}")
    print("✓ System continues without web search results")
    
    print()
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_tools())
