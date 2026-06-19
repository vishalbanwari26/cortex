"""Base sub-agent contract.

Every specialized agent (perception, planner, critic) is a thin, single-
responsibility wrapper around an `LLMClient`. They share one interface so the
orchestrator treats them uniformly and new agents drop in without special
casing. The `_parse_json` helper tolerates models that wrap JSON in prose or
code fences.
"""

from __future__ import annotations

import json
import re
from abc import ABC

from ..llm import LLMClient

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class Agent(ABC):
    #: Sub-classes set a ROLE tag; it goes into the system prompt and lets the
    #: offline MockClient identify the caller.
    role: str = "AGENT"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(text)
            if not match:
                raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
            return json.loads(match.group(0))
