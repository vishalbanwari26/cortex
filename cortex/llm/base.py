"""Provider-agnostic LLM / VLM interface.

The orchestration layer never imports a vendor SDK directly. It depends only on
this `LLMClient` contract, so the same cognitive loop can run against a hosted
model in the cloud or a smaller quantized model at the edge. That decoupling is
deliberate: edge-to-cloud hybrid deployment is a swap of the concrete client,
not a rewrite of the agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ImageInput:
    """A base64-encoded image passed to a vision-capable model."""

    media_type: str  # e.g. "image/jpeg"
    data_b64: str


@dataclass
class LLMResponse:
    text: str
    raw: dict = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        prompt: str,
        image: ImageInput | None = None,
    ) -> LLMResponse:
        """Return a completion. If `image` is given, the client must route it
        to a vision-capable model."""
        raise NotImplementedError
