"""Generate annotated before/after images of the scene.

BEFORE: bounding boxes on each object the perception agent detected.
AFTER:  arrow from the mug's original position to the cupboard, ghost mug
        drawn inside, mug crossed out at original location.

Usage
-----
    python -m cortex.annotate --scene examples/scene.jpg --out docs
"""

from __future__ import annotations

import argparse
import os
import sys


# Approximate bounding boxes for examples/scene.jpg (1024x768).
# These are proportional (0-1) so they scale with any resize.
_OBJECTS = [
    {
        "name": "red mug",
        "location": "table",
        "state": "upright",
        "box": (0.738, 0.500, 0.826, 0.651),  # x1,y1,x2,y2 as fractions
        "color": (230, 60, 50),
    },
    {
        "name": "plate",
        "location": "table",
        "state": "on table",
        "box": (0.566, 0.579, 0.716, 0.638),
        "color": (180, 180, 200),
    },
    {
        "name": "cupboard",
        "location": "wall",
        "state": "closed",
        "box": (0.037, 0.137, 0.298, 0.750),
        "color": (160, 100, 50),
    },
]

BG_LABEL = (17, 19, 26, 210)   # dark, semi-transparent
FG_LABEL = (255, 255, 255)
ACCENT = (86, 212, 221)


def _px(frac, size):
    return int(frac * size)


def _draw_before(img, draw, font, small):
    W, H = img.size
    for obj in _OBJECTS:
        x1, y1, x2, y2 = [_px(f, W if i % 2 == 0 else H) for i, f in enumerate(obj["box"])]
        r, g, b = obj["color"]
        # box outline (2px)
        for off in range(3):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off],
                           outline=(r, g, b, 255))
        # label background
        label = f"{obj['name']}  [{obj['location']}]"
        tw = small.getlength(label)
        lx, ly = x1, max(y1 - 28, 2)
        draw.rectangle([lx, ly, lx + tw + 12, ly + 24], fill=(r // 2, g // 2, b // 2, 220))
        draw.text((lx + 6, ly + 4), label, font=small, fill=FG_LABEL)


def _draw_after(img, draw, font, small):
    W, H = img.size
    mug = _OBJECTS[0]
    cup = _OBJECTS[2]
    mx1, my1, mx2, my2 = [_px(f, W if i % 2 == 0 else H) for i, f in enumerate(mug["box"])]
    cx1, cy1, cx2, cy2 = [_px(f, W if i % 2 == 0 else H) for i, f in enumerate(cup["box"])]

    # cross out original mug position
    draw.rectangle([mx1, my1, mx2, my2], outline=(230, 60, 50, 255), width=2)
    draw.line([(mx1, my1), (mx2, my2)], fill=(230, 60, 50, 200), width=3)
    draw.line([(mx2, my1), (mx1, my2)], fill=(230, 60, 50, 200), width=3)

    # arrow from mug to cupboard (center-to-center)
    src_x = (mx1 + mx2) // 2
    src_y = (my1 + my2) // 2
    dst_x = (cx1 + cx2) // 2
    dst_y = cy1 + (cy2 - cy1) // 3
    draw.line([(src_x, src_y), (dst_x, dst_y)], fill=ACCENT, width=4)
    # arrowhead
    import math
    angle = math.atan2(dst_y - src_y, dst_x - src_x)
    al = 18
    for da in (0.4, -0.4):
        ax = int(dst_x - al * math.cos(angle + da))
        ay = int(dst_y - al * math.sin(angle + da))
        draw.line([(dst_x, dst_y), (ax, ay)], fill=ACCENT, width=4)

    # ghost mug inside cupboard
    gw = mx2 - mx1
    gh = my2 - my1
    gx = cx1 + (cx2 - cx1) // 2 - gw // 2
    gy = dst_y - gh // 2
    draw.rectangle([gx, gy, gx + gw, gy + gh], outline=(86, 212, 221, 220), width=2)
    draw.rectangle([gx + 2, gy + 2, gx + gw - 2, gy + gh - 2],
                   fill=(86, 212, 221, 45))
    draw.text((gx + 4, gy + gh + 4), "red mug", font=small, fill=ACCENT)

    # "GOAL ACHIEVED" label
    label = "GOAL ACHIEVED"
    tw = font.getlength(label)
    lx = W // 2 - int(tw) // 2
    ly = 18
    draw.rectangle([lx - 12, ly - 4, lx + tw + 12, ly + 36], fill=(30, 90, 50, 220))
    draw.text((lx, ly + 4), label, font=font, fill=(63, 185, 80))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate annotated before/after scene images.")
    p.add_argument("--scene", default="examples/scene.jpg")
    p.add_argument("--out", default="docs")
    args = p.parse_args(argv)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow required: pip install -e '.[assets]'", file=sys.stderr)
        return 2

    font_candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    for path in font_candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 22)
                small = ImageFont.truetype(path, 16)
                break
            except Exception:
                continue

    os.makedirs(args.out, exist_ok=True)

    # BEFORE
    base = Image.open(args.scene).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _draw_before(base, draw, font, small)
    before = Image.alpha_composite(base, overlay).convert("RGB")
    before.save(os.path.join(args.out, "scene-before.png"))
    print(f"wrote {args.out}/scene-before.png")

    # AFTER
    base2 = Image.open(args.scene).convert("RGBA")
    overlay2 = Image.new("RGBA", base2.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay2)
    _draw_after(base2, draw2, font, small)
    after = Image.alpha_composite(base2, overlay2).convert("RGB")
    after.save(os.path.join(args.out, "scene-after.png"))
    print(f"wrote {args.out}/scene-after.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
