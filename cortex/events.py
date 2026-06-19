"""Real-time event streaming for the cognitive loop.

Every stage of the loop (perception, planning, execution, critique) emits
typed events. A consumer (CLI, web UI, logger, or a remote subscriber) attaches
a callback and receives feedback as it happens, rather than waiting for the run
to finish. This is what makes the orchestrator's reasoning observable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    PERCEIVING = "perceiving"
    SCENE_READY = "scene_ready"
    PLANNING = "planning"
    PLAN_READY = "plan_ready"
    STEP_STARTED = "step_started"
    STEP_RESULT = "step_result"
    CRITIQUE = "critique"
    REPLANNING = "replanning"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Event:
    type: EventType
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


Listener = Callable[[Event], None]


class EventBus:
    """Minimal synchronous pub/sub.

    Kept deliberately small: the orchestration layer should not depend on a
    particular transport. Swap this for a websocket/redis emitter without
    touching agent code.
    """

    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def emit(self, event: Event) -> None:
        for listener in self._listeners:
            listener(event)
