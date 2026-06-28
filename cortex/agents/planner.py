"""Planning agent.

Decomposes a natural-language goal plus the current scene into an ordered list
of steps, each bound to a registered skill. On a replan it also receives the
critic's feedback and the world state at the point of failure, so the new plan
is grounded in what actually went wrong rather than starting from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLMClient
from ..scene import Scene, WorldState
from ..skills import SkillRegistry
from .base import Agent


@dataclass
class PlanStep:
    id: int
    action: str
    target: str
    skill: str
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        return cls(
            id=int(data.get("id", 0)),
            action=data.get("action", ""),
            target=data.get("target", ""),
            skill=data.get("skill", ""),
            rationale=data.get("rationale", ""),
        )


_SYSTEM = """ROLE: PLANNER
You are the task planner of an embodied agent. Given a goal, the perceived
scene, the available skills, and (on a retry) feedback from a failed attempt,
produce a short ordered plan. Each step must use exactly one available skill.
Respond with ONLY JSON:
{"steps": [{"id": int, "action": str, "target": str, "skill": str, "rationale": str}]}
Prefer the fewest steps that reliably achieve the goal. Account for the robot's
current location and what it is holding."""


class PlannerAgent(Agent):
    role = "PLANNER"

    def __init__(self, llm: LLMClient, registry: SkillRegistry) -> None:
        super().__init__(llm)
        self.registry = registry

    def plan(
        self,
        goal: str,
        scene: Scene,
        world: WorldState,
        feedback: str | None = None,
    ) -> list[PlanStep]:
        prompt = (
            f"Goal: {goal}\n\n"
            f"Scene: {scene.summary}\n"
            f"Objects: {[ (o.name, o.location, o.state) for o in scene.objects ]}\n\n"
            f"Robot location: {world.robot_location}; holding: {world.holding}\n\n"
            f"Available skills:\n{self.registry.catalog()}\n"
        )
        if feedback:
            prompt += f"\nThe previous attempt failed. Feedback: {feedback}\nProduce a corrected plan."
        else:
            prompt += "\nProduce the plan."

        response = self.llm.complete(system=_SYSTEM, prompt=prompt)
        data = self._parse_json(response.text)
        return [PlanStep.from_dict(s) for s in data.get("steps", [])]
