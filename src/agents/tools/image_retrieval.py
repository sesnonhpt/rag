"""Image retrieval tool using semantic search."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from src.observability.logger import get_logger

from .base import Tool, ToolResult

logger = get_logger(__name__)


class ImageRetrievalTool(Tool):
    """Retrieve relevant images from existing image library using semantic search."""

    def __init__(
        self,
        image_resources: List[Dict[str, Any]] = None,
        timeout: float = 5.0,
        max_results: int = 3,
    ):
        super().__init__(name="image_retrieval", timeout=timeout)
        self.image_resources = image_resources or []
        self.max_results = max_results
        self.enabled = os.getenv("TOOL_USING_ENABLED", "true").lower() == "true"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "从图片库中检索与主题相关的图片资源。适用于需要实验装置图、结构图、流程图、几何图形等视觉内容时。",
            "parameters": {
                "description": {"type": "string", "description": "图片描述，包含学科和视觉元素，如'物理 牛顿第二定律 受力分析图'"},
            },
            "required": ["description"],
        }

    def validate_params(self, **kwargs: Any) -> bool:
        return "description" in kwargs or "query" in kwargs

    async def execute(self, description: str = "", **kwargs: Any) -> ToolResult:
        # Accept 'query' as alias for 'description' (LLM may use either)
        if not description:
            description = kwargs.get("query", "")
        if not description:
            return ToolResult(tool_name=self.name, success=False, error="Invalid parameters")
        if not self.enabled:
            logger.info("image_retrieval.disabled description=%s", description[:50])
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=[],
                metadata={"description": description, "degraded": True},
            )

        if not self.image_resources:
            logger.info("image_retrieval.no_resources description=%s", description[:50])
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=[],
                metadata={"description": description, "result_count": 0},
            )

        try:
            # Simple keyword-based matching for now
            # In production, this should use semantic embeddings
            results = self._keyword_match(description)
            logger.info(
                "image_retrieval.success description=%s result_count=%d",
                description[:50],
                len(results),
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results,
                metadata={
                    "description": description,
                    "result_count": len(results),
                },
            )
        except Exception as e:
            logger.error(
                "image_retrieval.error description=%s error=%s",
                description[:50],
                str(e),
                exc_info=True,
            )
            return ToolResult(
                tool_name=self.name,
                success=True,  # Non-critical failure
                data=[],
                error=str(e),
                metadata={"description": description, "degraded": True},
            )

    def _keyword_match(self, description: str) -> List[Dict[str, Any]]:
        """Simple keyword-based matching."""
        desc_lower = description.lower()
        keywords = desc_lower.split()

        scored_images = []
        for img in self.image_resources:
            score = 0
            img_text = (
                str(img.get("caption", ""))
                + " "
                + str(img.get("context", ""))
                + " "
                + str(img.get("alt_text", ""))
            ).lower()

            for keyword in keywords:
                if keyword in img_text:
                    score += 1

            if score > 0:
                scored_images.append((score, img))

        # Sort by score and return top results
        scored_images.sort(key=lambda x: x[0], reverse=True)
        results = [img for _, img in scored_images[: self.max_results]]

        return results

    def update_image_resources(self, image_resources: List[Dict[str, Any]]) -> None:
        """Update the image resource library."""
        self.image_resources = image_resources
