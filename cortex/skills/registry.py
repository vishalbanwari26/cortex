"""Skill registry.

A skill is the smallest executable capability the robot exposes (navigate,
grasp, place, scan). Sub-agents discover skills through the registry rather
than importing them directly, so new capabilities plug in without changing the
orchestrator. This is the modular sub-agent interface the loop is built around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..scene import WorldState


@dataclass
class SkillResult:
    ok: bool
    observation: str
    payload: dict = field(default_factory=dict)


class SkillFn(Protocol):
    def __call__(self, target: str, world: WorldState) -> SkillResult: ...


@dataclass
class Skill:
    name: str
    description: str
    run: SkillFn


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def catalog(self) -> str:
        """Human-readable list injected into planner prompts."""
        return "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())
