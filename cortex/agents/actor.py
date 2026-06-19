"""Actor agent.

Executes a single plan step by resolving its skill in the registry and running
it against the world state. It is deliberately not an LLM call: once the planner
has chosen a skill, execution should be deterministic and auditable. The actor
is the bridge between symbolic plans and (here simulated) actuation.
"""

from __future__ import annotations

from ..scene import WorldState
from ..skills import SkillRegistry, SkillResult
from .planner import PlanStep


class ActorAgent:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def execute(self, step: PlanStep, world: WorldState) -> SkillResult:
        skill = self.registry.get(step.skill)
        if skill is None:
            return SkillResult(
                ok=False,
                observation=f"Unknown skill '{step.skill}'. Available: {self.registry.names()}",
            )
        return skill.run(step.target, world)
