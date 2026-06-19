"""Concrete client backed by Groq, defaulting to GPT-OSS 120B.

Groq exposes an OpenAI-style chat API and serves OpenAI's open-weight GPT-OSS
models. GPT-OSS 120B is a text-only reasoning model, so it powers the planner
and critic (and perception from a text scene description), but it cannot take
an image. The vision path stays on a VLM provider such as Anthropic. Having two
real providers behind one `LLMClient` is the point: the orchestrator does not
change when you swap the model.

The key is read from GROQ_API_KEY. Never hard-code it.
"""

from __future__ import annotations

import os

from .base import ImageInput, LLMClient, LLMResponse


class GroqClient(LLMClient):
    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
        max_completion_tokens: int = 2048,
        temperature: float = 0.2,
        reasoning_effort: str | None = "low",
        json_mode: bool = True,
    ) -> None:
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it or pass api_key=..."
            )
        try:
            from groq import Groq  # lazy: optional dependency
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the SDK first: pip install groq") from exc
        self._client = Groq(api_key=key)

    def complete(
        self,
        system: str,
        prompt: str,
        image: ImageInput | None = None,
    ) -> LLMResponse:
        if image is not None:
            raise RuntimeError(
                f"{self.model} is text-only and cannot accept an image. Run "
                "perception from a text description (--note) or use a "
                "vision-capable provider for the perception step."
            )

        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""
        return LLMResponse(text=text, raw={"id": getattr(completion, "id", None)})
