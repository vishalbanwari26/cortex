"""Concrete cloud client backed by the Anthropic API.

Imports the SDK lazily so the rest of the package stays usable offline (tests
and demos run on the MockClient with no key and no network).
"""

from __future__ import annotations

import os

from .base import ImageInput, LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key=..."
            )
        try:
            import anthropic  # lazy: optional dependency
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the SDK first: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=key)

    def complete(
        self,
        system: str,
        prompt: str,
        image: ImageInput | None = None,
    ) -> LLMResponse:
        content: list[dict] = []
        if image is not None:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": image.data_b64,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        return LLMResponse(text=text, raw={"id": getattr(message, "id", None)})
