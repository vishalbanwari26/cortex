"""Capture a styled terminal recording (still PNG + animated GIF + transcript)
from a real run, so a live run produces publishable portfolio assets with no
manual screen recording.

The caption is honest about provenance: a live provider is labeled
"live run | <provider> | <model>"; the offline mock is labeled
"offline deterministic demo". Requires Pillow (pip install -e ".[assets]").

Examples
--------
    # offline (no key)
    python -m cortex.capture "Put the red mug in the cupboard." --out out

    # live on Groq gpt-oss-120b (text scene)
    python -m cortex.capture "Put the red mug in the cupboard." --provider groq \\
        --note "A table with a red mug and a dirty plate; a cupboard on the wall." --out out

    # live on Anthropic VLM (real image)
    python -m cortex.capture "Tidy the table." --provider anthropic \\
        --image examples/scene.jpg --out out
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

from .events import Event, EventType
from .llm import MockClient
from .orchestrator import Orchestrator

# ---- styling ---------------------------------------------------------------

BG = (17, 19, 26)
BAR = (30, 33, 44)
TEXT = (201, 209, 217)
DIM = (125, 133, 144)

ICONS = {
    EventType.PERCEIVING: ("[..]", DIM),
    EventType.SCENE_READY: ("[eye]", (86, 212, 221)),
    EventType.PLANNING: ("[..]", DIM),
    EventType.PLAN_READY: ("[plan]", (121, 192, 255)),
    EventType.STEP_STARTED: ("[>>]", (201, 209, 217)),
    EventType.STEP_RESULT: ("[ok]", (63, 185, 80)),
    EventType.CRITIQUE: ("[!]", (210, 153, 34)),
    EventType.REPLANNING: ("[~]", (210, 168, 255)),
    EventType.DONE: ("[done]", (86, 211, 100)),
    EventType.FAILED: ("[x]", (248, 81, 73)),
}
ERR = (248, 81, 73)

_FONT_CANDIDATES = {
    "regular": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
    ],
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/Library/Fonts/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consolab.ttf",
    ],
}


def _load_font(kind, size):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES[kind]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---- run + collect ---------------------------------------------------------


def _collect(goal, provider, note, image, model):
    if provider == "anthropic":
        from .llm.anthropic_client import AnthropicClient

        llm = AnthropicClient(model=model or "claude-sonnet-4-6")
    elif provider == "groq-vision":
        from .llm.groq_vision_client import GroqVisionClient

        llm = GroqVisionClient(model=model or "meta-llama/llama-4-scout-17b-16e-instruct")
    elif provider == "groq":
        from .llm.groq_client import GroqClient

        llm = GroqClient(model=model or "openai/gpt-oss-120b")
        image = None
    else:
        llm = MockClient()
        image = None

    lines: list[tuple[str, tuple, str]] = [("goal", TEXT, f"Goal: {goal}"), ("rule", DIM, "")]

    def on_event(e: Event):
        icon, color = ICONS.get(e.type, ("[--]", DIM))
        if e.type == EventType.STEP_RESULT and not e.payload.get("ok", True):
            icon, color = "[err]", ERR
        lines.append(("icon", color, f"{icon:>7} {e.message}", icon))

    orch = Orchestrator(llm)
    orch.bus.subscribe(on_event)
    result = orch.run(goal, note=note, image=image)

    lines.append(("rule", DIM, ""))
    summary = f"{'SUCCESS' if result.success else 'FAILED'} | steps={result.steps_executed} replans={result.replans}"
    lines.append(("summary", (63, 185, 80) if result.success else ERR, summary))
    return lines, result


# ---- render ----------------------------------------------------------------


def _render(lines, n, title, caption, reveal=True):
    from PIL import Image, ImageDraw

    font = _load_font("regular", 28)
    font_b = _load_font("bold", 28)
    small = _load_font("regular", 20)
    char_w = font.getlength("M") or 16
    pad, bar_h, line_h = 40, 60, 40

    plain = []
    for ln in lines:
        text = ln[2]
        plain.append(text.strip())
    max_chars = max((len(t) for t in plain), default=40) + 4
    width = int(pad * 2 + max_chars * char_w)
    height = int(bar_h + pad * 2 + len(lines) * line_h + 30)

    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, bar_h], fill=BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([24 + i * 30, bar_h // 2 - 9, 24 + i * 30 + 18, bar_h // 2 + 9], fill=c)
    tw = small.getlength(title)
    d.text((width / 2 - tw / 2, bar_h / 2 - 12), title, font=small, fill=DIM)

    y = bar_h + pad
    for i in range(n):
        kind = lines[i][0]
        color = lines[i][1]
        text = lines[i][2]
        if kind == "icon":
            icon = lines[i][3]
            field = f"{icon:>7}"
            rest = text.strip()[len(icon):].lstrip()
            d.text((pad, y), field, font=font_b, fill=color)
            d.text((pad + 8 * char_w, y), rest, font=font, fill=TEXT)
        elif kind == "rule":
            d.text((pad, y), "\u2500" * (max_chars - 2), font=font, fill=(45, 50, 62))
        elif kind == "goal":
            d.text((pad, y), text, font=font_b, fill=(255, 255, 255))
        elif kind == "summary":
            d.text((pad, y), text, font=font_b, fill=color)
        else:
            d.text((pad, y), text, font=font, fill=TEXT)
        y += line_h

    d.text((pad, height - 34), caption, font=small, fill=(70, 76, 90))
    return img


def _render_side_by_side(lines, n, title, caption, scene_path: str):
    """Composite the scene image on the left, terminal animation on the right."""
    from PIL import Image

    terminal = _render(lines, n, title, caption)
    th = terminal.height
    tw = terminal.width

    scene_raw = Image.open(scene_path).convert("RGB")
    # scale scene to match terminal height
    ratio = th / scene_raw.height
    sw = int(scene_raw.width * ratio)
    scene = scene_raw.resize((sw, th), Image.LANCZOS)

    sep = 4  # separator width
    combined = Image.new("RGB", (sw + sep + tw, th), (40, 44, 56))
    combined.paste(scene, (0, 0))
    # thin separator line
    from PIL import ImageDraw
    d = ImageDraw.Draw(combined)
    d.rectangle([sw, 0, sw + sep, th], fill=(55, 60, 75))
    combined.paste(terminal, (sw + sep, 0))
    return combined


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Capture portfolio assets from a Cortex run.")
    p.add_argument("goal", nargs="?", default="Put the red mug in the cupboard.")
    p.add_argument("--provider", choices=["mock", "anthropic", "groq", "groq-vision"], default="mock")
    p.add_argument("--note", default="")
    p.add_argument("--image", default=None)
    p.add_argument("--scene", default=None, help="Path to scene image for side-by-side layout.")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default="out", help="Output directory for the assets.")
    args = p.parse_args(argv)

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Pillow is required: pip install -e \".[assets]\"", file=sys.stderr)
        return 2

    image = None
    if args.image and args.provider in ("anthropic", "groq-vision"):
        ext = args.image.rsplit(".", 1)[-1].lower()
        media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        from .llm import ImageInput

        with open(args.image, "rb") as fh:
            image = ImageInput(media_type=media, data_b64=base64.standard_b64encode(fh.read()).decode())

    _default_model = {
        "groq": "openai/gpt-oss-120b",
        "groq-vision": "meta-llama/llama-4-scout-17b-16e-instruct",
        "anthropic": "claude-sonnet-4-6",
    }
    model = args.model or _default_model.get(args.provider, "mock")
    lines, result = _collect(args.goal, args.provider, args.note, image, args.model)

    if args.provider == "mock":
        title = "cortex  —  cognitive loop (offline demo)"
        caption = "offline deterministic demo  ·  github.com/vishalbanwari26/cortex"
    else:
        title = f"cortex  —  live  ({args.provider} · {model})"
        caption = f"live run  ·  {args.provider}  ·  {model}"

    os.makedirs(args.out, exist_ok=True)

    scene_path = args.scene or (args.image if args.provider in ("groq-vision", "anthropic") else None)
    if scene_path and not os.path.exists(scene_path):
        scene_path = None

    def _frame(n):
        if scene_path:
            return _render_side_by_side(lines, n, title, caption, scene_path)
        return _render(lines, n, title, caption)

    still = _frame(len(lines))
    still.save(os.path.join(args.out, "demo.png"))

    frames = [_frame(n) for n in range(1, len(lines) + 1)]
    durations = [650] * len(frames)
    durations[-1] = 2600
    frames[0].save(
        os.path.join(args.out, "demo.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    with open(os.path.join(args.out, "transcript.txt"), "w") as fh:
        fh.write("\n".join(l[2] for l in lines) + "\n")

    print(f"wrote {args.out}/demo.png, demo.gif, transcript.txt  ({'success' if result.success else 'failed'})")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
