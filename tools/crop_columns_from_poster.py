from PIL import Image
import numpy as np

# Load our model's eval poster (eval200fix_poster.png)
poster = Image.open("exp/purestd_posters/eval200fix_poster.png")
pw, ph = poster.size
print(f"Poster size: {pw}x{ph}")

# Let's inspect the layout of the poster:
# In eval200fix_poster.png:
# How many columns? Total is ~187-200.
# 11980 / 187 = ~64 pixels per column?
# Wait! Let's check how many columns and rows are in eval200fix_poster.png:
# Let's inspect with a script that detects column borders and row cuts!
arr = np.array(poster.convert("L"))
col_means = arr.mean(axis=0)
print(f"Col means: min={col_means.min():.1f}, max={col_means.max():.1f}")

# Let's check the height slices:
# Earlier we saw:
# y=0..100: header
# y=120..280: ~160px height
# y=340..500: ~160px height
# y=560..720: ~160px height
# y=740..900: ~160px height
# Wait! Let's check how many rows of characters are in eval200fix_poster.png!
