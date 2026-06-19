"""Generate a simple but recognizable kitchen scene for testing the live VLM path."""

from PIL import Image, ImageDraw

W, H = 1024, 768
img = Image.new("RGB", (W, H), (236, 232, 222))  # warm wall
d = ImageDraw.Draw(img)

# Floor
d.rectangle([0, 560, W, H], fill=(196, 170, 140))

# Cupboard against the wall (left)
d.rectangle([60, 120, 300, 560], fill=(150, 110, 70), outline=(90, 64, 40), width=4)
d.line([180, 120, 180, 560], fill=(90, 64, 40), width=4)            # door split
d.ellipse([160, 330, 176, 346], fill=(40, 30, 20))                  # left handle
d.ellipse([184, 330, 200, 346], fill=(40, 30, 20))                  # right handle

# Table (right, in front)
d.rectangle([430, 470, 940, 520], fill=(140, 95, 60), outline=(95, 62, 38), width=4)  # tabletop
d.rectangle([460, 520, 482, 660], fill=(120, 80, 50))              # legs
d.rectangle([888, 520, 910, 660], fill=(120, 80, 50))

# White plate on the table
d.ellipse([560, 452, 700, 492], fill=(245, 245, 245), outline=(200, 200, 200), width=3)
d.ellipse([590, 460, 670, 484], outline=(210, 210, 210), width=2)

# Red mug on the table
mug_x, mug_y = 760, 410
d.rectangle([mug_x, mug_y, mug_x + 70, mug_y + 75], fill=(200, 50, 45), outline=(150, 30, 28), width=3)  # body
d.ellipse([mug_x, mug_y - 12, mug_x + 70, mug_y + 12], fill=(220, 70, 62), outline=(150, 30, 28), width=3)  # rim
d.arc([mug_x + 60, mug_y + 8, mug_x + 110, mug_y + 60], start=300, end=60, fill=(150, 30, 28), width=8)     # handle

img.save("examples/scene.jpg", quality=90)
print("wrote examples/scene.jpg", img.size)
