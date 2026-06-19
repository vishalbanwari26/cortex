"""Groq-backed client with vision support via Llama 4 Scout.

Uses the same GROQ_API_KEY as GroqClient. Llama 4 Scout is a multimodal
model, so this single client handles both the vision perception step and the
text-only planner/critic steps -- a fully-Groq pipeline with no Anthropic key
required. Images are passed as base64 data URLs in the OpenAI-style content
array that the Groq SDK expects.
"""

from __future__ import annotations

import os

from .base import ImageInput, LLMClient, LLMResponse


class GroqVisionClient(LLMClient):
    def __init__(
        self,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
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
            content: list[dict] = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.media_type};base64,{image.data_b64}"
                    },
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = [{"type": "text", "text": prompt}]

        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""
        return LLMResponse(text=text, raw={"id": getattr(completion, "id", None)})
