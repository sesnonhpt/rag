"""State models for lesson-generation agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LessonAgentAssets:
    """Assets retrieved for lesson generation."""

    text_results: List[Any] = field(default_factory=list)
    image_resources: List[Any] = field(default_factory=list)
    citations: List[Any] = field(default_factory=list)


@dataclass
class LessonReviewReport:
    """Structured review result for a lesson draft."""

    realism_score: int = 0
    pedagogy_score: int = 0
    structure_score: int = 0
    multimodal_score: int = 0
    strengths: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    must_fix: List[str] = field(default_factory=list)


@dataclass
class LessonAgentState:
    """Mutable state for the lesson-generation workflow."""

    topic: str
    template_category: Optional[str] = None
    template_type: Optional[str] = None
    subject: Optional[str] = None
    assets: LessonAgentAssets = field(default_factory=LessonAgentAssets)
    draft_content: Optional[str] = None
    final_content: Optional[str] = None
    review_notes: List[str] = field(default_factory=list)
    review_report: Optional[LessonReviewReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
