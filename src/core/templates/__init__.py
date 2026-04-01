"""Template manager for the active lesson-generation templates."""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.libs.llm.base_llm import Message
from .registry import (
    TEMPLATE_CATEGORY_DEFINITIONS,
    get_default_template_category,
    get_template_categories_payload,
    get_template_category_definition,
    get_template_label_by_category,
    resolve_template_type_by_category,
)


class TemplateType(Enum):
    """Template types used by the current lesson-generation product."""

    GUIDE_MASTER = "guide_master"
    TEACHING_DESIGN_MASTER = "teaching_design_master"
    COMPREHENSIVE_MASTER = "comprehensive_master"


@dataclass
class TemplateConfig:
    """Configuration for template generation."""
    template_type: TemplateType
    include_background: bool = True
    include_facts: bool = True
    include_examples: bool = True


class TemplateManager:
    """Manager for all template types."""
    
    def __init__(self):
        self.templates: Dict[TemplateType, callable] = {}
        self._register_templates()
    
    def _register_templates(self):
        """Register only the active lesson templates."""
        from .guide_templates import (
            build_guide_master_prompt,
        )
        from .comprehensive_templates import (
            build_comprehensive_master_prompt,
            build_teaching_design_prompt,
        )

        self.templates[TemplateType.GUIDE_MASTER] = build_guide_master_prompt
        self.templates[TemplateType.TEACHING_DESIGN_MASTER] = build_teaching_design_prompt
        self.templates[TemplateType.COMPREHENSIVE_MASTER] = build_comprehensive_master_prompt
    
    def build_prompt(
        self,
        config: TemplateConfig,
        topic: str,
        contexts: List[Any],
        **kwargs
    ) -> List[Message]:
        """Build prompt using specified template.
        
        Args:
            config: Template configuration
            topic: Topic or question
            contexts: Retrieved contexts
            **kwargs: Additional template-specific arguments
            
        Returns:
            List of LLM messages
        """
        template_func = self.templates.get(config.template_type)
        if not template_func:
            raise ValueError(f"Unknown template type: {config.template_type}")
        
        return template_func(
            topic=topic,
            contexts=contexts,
            config=config,
            **kwargs
        )


__all__ = [
    "TemplateType",
    "TemplateConfig",
    "TemplateManager",
    "TEMPLATE_CATEGORY_DEFINITIONS",
    "get_template_category_definition",
    "resolve_template_type_by_category",
    "get_template_label_by_category",
    "get_default_template_category",
    "get_template_categories_payload",
]
