"""Critic agent.

When a step fails, the critic inspects the failure and the world state and
decides whether the loop should replan, retry, or abort. It closes the feedback
loop: execution outcomes flow back into reasoning, which is what lets the agent
adapt in real time instead of failing on the first unexpected outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLMClient
from ..scene import WorldState
from .base import Agent

_SYSTEM = """ROLE: CRITIC
You evaluate a failed execution step for an embodied agent. Decide the next
move. Respond with ONLY JSON:
{"verdict": "replan" | "abort", "reason": str, "hint": str}
Use "replan" when a different plan could plausibly recover. Use "abort" only
when the goal is unreachable with the available skills."""


@dataclass
class Critique:
    verdict: str
    reason: str
    hint: str = ""

    @property
    def should_replan(self) -> bool:
        return self.verdict == "replan"


class CriticAgent(Agent):
    role = "CRITIC"

    def assess(self, goal: str, failed_action: str, observation: str, world: WorldState) -> Critique:
        prompt = (
            f"Goal: {goal}\n"
            f"Failed action: {failed_action}\n"
            f"Observation: {observation}\n"
            f"Robot location: {world.robot_location}; holding: {world.holding}\n"
            "Decide the next move."
        )
        response = self.llm.complete(system=_SYSTEM, prompt=prompt)
        data = self._parse_json(response.text)
        return Critique(
            verdict=data.get("verdict", "abort"),
            reason=data.get("reason", ""),
            hint=data.get("hint", ""),
        )
