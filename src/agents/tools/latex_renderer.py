"""LaTeX renderer tool for validating and rendering mathematical formulas."""

from __future__ import annotations

import os
import re
from typing import Any

from src.observability.logger import get_logger

from .base import Tool, ToolResult

logger = get_logger(__name__)


class LaTeXRendererTool(Tool):
    """Validate LaTeX syntax and optionally render formulas."""

    def __init__(self, timeout: float = 5.0, mode: str = "validate"):
        super().__init__(name="latex_renderer", timeout=timeout)
        self.mode = mode  # "validate" or "render"
        self.enabled = (
            os.getenv("LATEX_RENDERER_ENABLED", "true").lower() == "true"
            and os.getenv("TOOL_USING_ENABLED", "true").lower() == "true"
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "验证 LaTeX 数学公式语法。适用于学科涉及数学公式、物理方程、化学方程式时。",
            "parameters": {
                "latex_code": {"type": "string", "description": "需要验证的 LaTeX 代码，如 'F = ma' 或 '\\\\frac{d}{dx}'"},
            },
            "required": ["latex_code"],
        }

    def validate_params(self, **kwargs: Any) -> bool:
        return "latex_code" in kwargs or "expression" in kwargs or "formula" in kwargs

    async def execute(self, latex_code: str = "", **kwargs: Any) -> ToolResult:
        # Accept 'expression' or 'formula' as aliases (LLM may use either)
        if not latex_code:
            latex_code = kwargs.get("expression") or kwargs.get("formula", "")
        if not latex_code:
            return ToolResult(tool_name=self.name, success=False, error="Invalid parameters")
        if not self.enabled:
            logger.info("latex_renderer.disabled latex_code=%s", latex_code[:50])
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"valid": True, "latex_code": latex_code},
                metadata={"degraded": True},
            )

        try:
            # Basic syntax validation
            is_valid, error_msg = self._validate_syntax(latex_code)

            if not is_valid:
                logger.warning(
                    "latex_renderer.invalid latex_code=%s error=%s",
                    latex_code[:50],
                    error_msg,
                )
                return ToolResult(
                    tool_name=self.name,
                    success=True,  # Non-critical failure
                    data={
                        "valid": False,
                        "error": error_msg,
                        "latex_code": latex_code,
                    },
                    metadata={"validation_failed": True},
                )

            logger.info("latex_renderer.valid latex_code=%s", latex_code[:50])
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "valid": True,
                    "latex_code": latex_code,
                    "normalized": self._normalize_latex(latex_code),
                },
                metadata={"mode": self.mode},
            )

        except Exception as e:
            logger.error(
                "latex_renderer.error latex_code=%s error=%s",
                latex_code[:50],
                str(e),
                exc_info=True,
            )
            return ToolResult(
                tool_name=self.name,
                success=True,  # Non-critical failure
                data={"valid": True, "latex_code": latex_code},
                error=str(e),
                metadata={"degraded": True},
            )

    def _validate_syntax(self, latex_code: str) -> tuple[bool, str]:
        """Basic LaTeX syntax validation."""
        # Check for balanced braces
        if latex_code.count("{") != latex_code.count("}"):
            return False, "Unbalanced braces"

        # Check for balanced brackets
        if latex_code.count("[") != latex_code.count("]"):
            return False, "Unbalanced brackets"

        # Check for balanced parentheses in some contexts
        # (not all parentheses need to be balanced in LaTeX)

        # Check for common LaTeX commands
        common_commands = [
            r"\\frac",
            r"\\sqrt",
            r"\\sum",
            r"\\int",
            r"\\lim",
            r"\\alpha",
            r"\\beta",
            r"\\gamma",
            r"\\Delta",
            r"\\pi",
            r"\\theta",
            r"\\sin",
            r"\\cos",
            r"\\tan",
            r"\\log",
            r"\\ln",
            r"\\exp",
            r"\\cdot",
            r"\\times",
            r"\\div",
            r"\\pm",
            r"\\leq",
            r"\\geq",
            r"\\neq",
            r"\\approx",
            r"\\infty",
        ]

        # Check for invalid command patterns
        invalid_patterns = [
            r"\\\\\\",  # Triple backslash
            r"\\\s+[a-z]",  # Backslash followed by space and letter
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, latex_code):
                return False, f"Invalid pattern: {pattern}"

        return True, ""

    def _normalize_latex(self, latex_code: str) -> str:
        """Normalize LaTeX code for consistent rendering."""
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", latex_code.strip())

        # Ensure proper spacing around operators
        normalized = re.sub(r"([+\-=])", r" \1 ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized
