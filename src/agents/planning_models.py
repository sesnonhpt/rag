"""Planning models for lesson-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanConstraint:
    """A normalized constraint used by planner/executor."""

    key: str
    value: Any
    source: str = "user"


@dataclass
class ExecutionPlan:
    """Structured execution plan produced before retrieval/generation."""

    subject_guess: Optional[str] = None
    grade_hint: Optional[str] = None
    lesson_type_hint: Optional[str] = None
    need_images: bool = False
    query_focus: List[str] = field(default_factory=list)
    generation_mode: str = "context_first"  # context_first | autonomous
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    constraints: List[PlanConstraint] = field(default_factory=list)
    planner_notes: str = ""
    plan_version: str = "planner_v1"
