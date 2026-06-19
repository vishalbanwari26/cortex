"""Perception agent.

Turns a multimodal input (an image of the scene, optionally with a text note)
into a structured `Scene`. This is the front of the cognitive loop: raw
observation in, grounded symbolic belief out, which everything downstream plans
over.
"""

from __future__ import annotations

from ..llm import ImageInput, LLMClient
from ..scene import Scene
from .base import Agent

_SYSTEM = """ROLE: PERCEPTION
You are the perception module of an embodied agent. Given an image of a scene
(and any text note), identify the objects relevant to manipulation, their
location, and their state. Respond with ONLY a JSON object:
{"summary": str, "objects": [{"name": str, "location": str, "state": str}]}
Use short, concrete object names and short location labels (for example
"table", "counter", "shelf"), and reuse the same label for objects in the same
place. Do not add commentary."""


class PerceptionAgent(Agent):
    role = "PERCEPTION"

    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm)

    def perceive(self, note: str = "", image: ImageInput | None = None) -> Scene:
        prompt = f"Scene note: {note}\nReturn the structured scene." if note else "Return the structured scene."
        response = self.llm.complete(system=_SYSTEM, prompt=prompt, image=image)
        return Scene.from_dict(self._parse_json(response.text))
