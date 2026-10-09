import os
import json
from PIL import Image, ImageDraw, ImageFont

base_dir = "assets/historical_strict_common"
manifest_path = os.path.join(base_dir, "manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"载入 {len(manifest)} 个公共字条目:")
for item in manifest:
    print(f"  • {item['char']} ({item['highlight']}), 阶段数: {len(item['stages'])}")
