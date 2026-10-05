from PIL import Image
import numpy as np

im = Image.open("exp/purestd_posters/eval200fix_poster.png")
w, h = im.size
print(f"eval200fix_poster.png size: {w}x{h}")

# In top_left_preview.png earlier, we saw that:
# Each column is a sample from eval200fix.
# Let's inspect the layout of top_left_preview.png or the full poster to find column coordinates!
# Let's find vertical lines or spacing between columns.
arr = np.array(im.convert("L"))
# Look at top row of characters around y=200..400
strip = arr[150:400, :]
# Horizontal profile in this strip
profile = strip.mean(axis=0)

# Check if there are distinct columns
print("Profile min/max/mean:", profile.min(), profile.max(), profile.mean())

# Let's find columns where profile dips or peaks
# Let's save a thumbnail of the entire poster
thumb = im.resize((1200, int(1200 * h / w)))
thumb.save("exp/purestd_posters/full_thumb.png")
print("Saved full_thumb.png")
