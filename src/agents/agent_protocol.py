"""Unified message protocol shared across lesson agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from uuid import uuid4


@dataclass
class AgentMessage:
    """Standard envelope exchanged between planner/retriever/writer agents."""

    task_id: str = field(default_factory=lambda: uuid4().hex)
    goal: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    next_action: str = "plan"

    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal[:120],
            "context_keys": sorted(self.context.keys()),
            "artifact_keys": sorted(self.artifacts.keys()),
            "constraint_count": len(self.constraints),
            "next_action": self.next_action,
        }
