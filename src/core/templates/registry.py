"""Central registry for lesson template categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TemplateCategoryDefinition:
    category: str
    template_type: str
    label: str
    description: str
    default: bool = False


TEMPLATE_CATEGORY_DEFINITIONS: List[TemplateCategoryDefinition] = [
    TemplateCategoryDefinition(
        category="comprehensive",
        template_type="comprehensive_master",
        label="综合模版",
        description="💡 综合模版：融合教学设计、学生任务、互动设计、分层训练与配图讲解，生成增强版成稿",
        default=True,
    ),
    TemplateCategoryDefinition(
        category="teaching_design",
        template_type="teaching_design_master",
        label="教学设计",
        description="💡 教学设计：贴近真实教师备课稿，突出教学目标、重难点、教学过程与板书设计",
    ),
    TemplateCategoryDefinition(
        category="guide",
        template_type="guide_master",
        label="导学案模板",
        description="💡 导学案模板：自动判断学科，按学校导学案体例生成固定结构成稿",
    ),
]


_BY_CATEGORY: Dict[str, TemplateCategoryDefinition] = {
    item.category: item for item in TEMPLATE_CATEGORY_DEFINITIONS
}


def get_template_category_definition(category: Optional[str]) -> Optional[TemplateCategoryDefinition]:
    if not category:
        return None
    return _BY_CATEGORY.get(str(category).strip())


def resolve_template_type_by_category(category: Optional[str]) -> Optional[str]:
    item = get_template_category_definition(category)
    return item.template_type if item else None


def get_template_label_by_category(category: Optional[str]) -> str:
    item = get_template_category_definition(category)
    if item:
        return item.label
    default_item = next((entry for entry in TEMPLATE_CATEGORY_DEFINITIONS if entry.default), None)
    return default_item.label if default_item else "导学案模板"


def get_default_template_category() -> str:
    default_item = next((entry for entry in TEMPLATE_CATEGORY_DEFINITIONS if entry.default), None)
    return default_item.category if default_item else "guide"


def get_template_categories_payload() -> List[Dict[str, object]]:
    return [
        {
            "category": item.category,
            "template_type": item.template_type,
            "label": item.label,
            "description": item.description,
            "default": item.default,
        }
        for item in TEMPLATE_CATEGORY_DEFINITIONS
    ]
