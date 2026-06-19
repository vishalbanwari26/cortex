"""Cortex: a cognitive-loop orchestrator for embodied agents.

Perceive (VLM) -> plan -> orchestrate sub-agents/skills -> adapt on failure,
with real-time event streaming throughout. The orchestration layer is provider-
agnostic, so the same loop runs against a cloud model or an edge model.
"""

from .events import Event, EventBus, EventType
from .orchestrator import Orchestrator, RunResult
from .scene import Scene, SceneObject, WorldState

__all__ = [
    "Orchestrator",
    "RunResult",
    "Scene",
    "SceneObject",
    "WorldState",
    "Event",
    "EventBus",
    "EventType",
]

__version__ = "0.1.0"
