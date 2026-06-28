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
    parser.add_argument("--note", default="", help="Optional text note about the scene.")
    parser.add_argument("--image", default=None, help="Path to a scene image (requires --live).")
    parser.add_argument("--live", action="store_true",
                        help="Use the real Anthropic VLM instead of the offline mock.")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args(argv)

    if args.live:
        from .llm.anthropic_client import AnthropicClient

        llm = AnthropicClient(model=args.model)
    else:
        llm = MockClient()

    image = _load_image(args.image) if args.image else None

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
