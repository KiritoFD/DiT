from PIL import Image
import numpy as np

im = Image.open("exp/purestd_posters/eval200fix_poster.png")
w, h = im.size
print(f"Poster size: {w}x{h}")

# Let's crop a small patch from top-left (e.g. 1000x1000) to see the structure and rows
crop = im.crop((0, 0, min(w, 1500), min(h, 1500)))
crop.save("exp/purestd_posters/top_left_preview.png")
print("Saved top_left_preview.png")
