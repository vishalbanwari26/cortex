"""Deterministic offline client.

Lets the whole cognitive loop run with no API key and no network, which is what
makes the test suite fast and reproducible. It recognizes each agent by the
ROLE tag in its system prompt and returns scripted JSON for the built-in
kitchen scenario. The planner is intentionally stateful: its first plan is
incomplete (it skips navigation), the grasp step then fails, and the replan
returns a corrected plan. That is the adaptive loop on display without any
randomness.
"""

from __future__ import annotations

import json

from .base import ImageInput, LLMClient, LLMResponse

_SCENE = {
    "summary": "A kitchen table holding a red mug and a dirty plate; a cupboard is along the wall.",
    "objects": [
        {"name": "red mug", "location": "table", "state": "upright"},
        {"name": "dirty plate", "location": "table", "state": "dirty"},
    ],
}

_PLAN_FIRST = {
    "steps": [
        {"id": 1, "action": "grasp", "target": "red mug", "skill": "grasp",
         "rationale": "Pick up the mug to relocate it."},
        {"id": 2, "action": "place", "target": "cupboard", "skill": "place",
         "rationale": "Store the mug in the cupboard."},
    ]
}

_PLAN_REVISED = {
    "steps": [
        {"id": 1, "action": "navigate", "target": "table", "skill": "navigate",
         "rationale": "Move to the table; the mug was out of reach last attempt."},
        {"id": 2, "action": "grasp", "target": "red mug", "skill": "grasp",
         "rationale": "Pick up the mug now that the robot is at the table."},
        {"id": 3, "action": "place", "target": "cupboard", "skill": "place",
         "rationale": "Store the mug in the cupboard."},
    ]
}

_CRITIQUE = {
    "verdict": "replan",
    "reason": "Grasp failed because the robot was not at the object's location.",
    "hint": "Navigate to the target before grasping.",
}


class MockClient(LLMClient):
    def __init__(self) -> None:
        self._planner_calls = 0

    def complete(
        self,
        system: str,
        prompt: str,
        image: ImageInput | None = None,
    ) -> LLMResponse:
        role = self._role(system)
        if role == "PERCEPTION":
            return LLMResponse(text=json.dumps(_SCENE))
        if role == "PLANNER":
            self._planner_calls += 1
            plan = _PLAN_FIRST if self._planner_calls == 1 else _PLAN_REVISED
            return LLMResponse(text=json.dumps(plan))
        if role == "CRITIC":
            return LLMResponse(text=json.dumps(_CRITIQUE))
        return LLMResponse(text="{}")

    @staticmethod
    def _role(system: str) -> str:
        for tag in ("PERCEPTION", "PLANNER", "CRITIC"):
            if f"ROLE: {tag}" in system:
                return tag
        return "UNKNOWN"
