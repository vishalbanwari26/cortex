"""Command-line runner.

Streams the cognitive loop to the terminal so you can watch perception,
planning, execution and adaptation happen live. Defaults to the offline
MockClient (no key needed); pass --live to use the real Anthropic VLM with an
image.
"""

from __future__ import annotations

import argparse
import base64
import sys

from .events import Event, EventType
from .llm import ImageInput, MockClient
from .orchestrator import Orchestrator

_ICONS = {
    EventType.PERCEIVING: "[..]",
    EventType.SCENE_READY: "[eye]",
    EventType.PLANNING: "[..]",
    EventType.PLAN_READY: "[plan]",
    EventType.STEP_STARTED: "[>>]",
    EventType.STEP_RESULT: "[ok]",
    EventType.CRITIQUE: "[!]",
    EventType.REPLANNING: "[~]",
    EventType.DONE: "[done]",
    EventType.FAILED: "[x]",
}


def _printer(event: Event) -> None:
    icon = _ICONS.get(event.type, "[--]")
    if event.type == EventType.STEP_RESULT and not event.payload.get("ok", True):
        icon = "[err]"
    print(f"{icon:>7} {event.message}")


def _load_image(path: str) -> ImageInput:
    ext = path.rsplit(".", 1)[-1].lower()
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode("ascii")
    return ImageInput(media_type=media, data_b64=data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Cortex cognitive loop.")
    parser.add_argument("goal", nargs="?", default="Put the red mug in the cupboard.",
                        help="Natural-language goal for the agent.")
    parser.add_argument("--note", default="", help="Text description of the scene (used by text-only providers).")
    parser.add_argument("--image", default=None, help="Path to a scene image (vision providers only).")
    parser.add_argument("--provider", choices=["mock", "anthropic", "groq", "groq-vision"], default="mock",
                        help="Which LLM client to use. Default: mock (offline, no key).")
    parser.add_argument("--live", action="store_true", help="Alias for --provider anthropic.")
    parser.add_argument("--model", default=None, help="Override the provider's default model.")
    args = parser.parse_args(argv)

    provider = "anthropic" if args.live else args.provider
    image = None

    if provider == "anthropic":
        from .llm.anthropic_client import AnthropicClient

        llm = AnthropicClient(model=args.model or "claude-sonnet-4-6")
        image = _load_image(args.image) if args.image else None
    elif provider == "groq-vision":
        from .llm.groq_vision_client import GroqVisionClient

        llm = GroqVisionClient(model=args.model or "meta-llama/llama-4-scout-17b-16e-instruct")
        image = _load_image(args.image) if args.image else None
    elif provider == "groq":
        from .llm.groq_client import GroqClient

        llm = GroqClient(model=args.model or "openai/gpt-oss-120b")
        if args.image:
            print("note: gpt-oss-120b is text-only; ignoring --image. Describe the scene with --note.\n")
        if not args.note:
            print("tip: text-only providers perceive from --note; pass one for a grounded scene.\n")
    else:
        llm = MockClient()
        if args.image:
            print("note: --image is only used with a vision provider; the mock ignores it.\n")

    orch = Orchestrator(llm)
    orch.bus.subscribe(_printer)

    print(f"\nGoal: {args.goal}\n" + "-" * 48)
    result = orch.run(args.goal, note=args.note, image=image)
    print("-" * 48)
    print(
        f"{'SUCCESS' if result.success else 'FAILED'} | "
        f"steps={result.steps_executed} replans={result.replans}"
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
