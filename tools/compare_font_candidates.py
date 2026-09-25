import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

STD_FONT_PATH = "tools/fonts/simkai.ttf"
CANDIDATE_FONTS = {
    "ZhiMangXing (毛笔行书)": "tools/fonts/candidate_fonts/ZhiMangXing-Regular.ttf",
    "MaShanZheng (毛笔楷书)": "tools/fonts/candidate_fonts/MaShanZheng-Regular.ttf",
    "LongCang (毛笔行草)": "tools/fonts/candidate_fonts/LongCang-Regular.ttf",
    "STXINWEI (华文新魏)": "tools/fonts/candidate_fonts/STXINWEI.TTF",
    "FZSTK (方正舒体)": "tools/fonts/candidate_fonts/FZSTK.TTF",
}

SIZE = 256
BOX_FRAC = 0.84

def render_char(char, font_path, size=SIZE, box_frac=BOX_FRAC):
    """Render character centered with bounding box normalization."""
    # 2x render for clean anti-aliasing
    large_size = size * 2
    img = Image.new("L", (large_size, large_size), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, int(large_size * 0.75))
    except Exception as e:
        return None
    
    # check tofu
    bbox = draw.textbbox((0, 0), char, font=font)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 2 or h <= 2:
        return None
    
    # draw centered
    x = (large_size - w) // 2 - bbox[0]
    y = (large_size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    
    # crop bbox
    arr = np.array(img)
    ink = (arr < 200)
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    ymin, ymax = ys.min(), ys.max() + 1
    xmin, xmax = xs.min(), xs.max() + 1
    crop = img.crop((xmin, ymin, xmax, ymax))
    
    # resize to box_frac * size
    cw, ch = crop.size
    target_box = int(size * box_frac)
    scale = target_box / max(cw, ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    crop_resized = crop.resize((nw, nh), Image.Resampling.LANCZOS)
    
    # paste back to white background
    out = Image.new("L", (size, size), 255)
    px = (size - nw) // 2
    py = (size - nh) // 2
    out.paste(crop_resized, (px, py))
    return out

def compute_metrics(im1, im2):
    """Compute SSIM and IoU between two grayscale PIL images."""
    arr1 = np.array(im1).astype(np.float32) / 255.0
    arr2 = np.array(im2).astype(np.float32) / 255.0
    
    s = ssim(arr1, arr2, data_range=1.0)
    
    # ink binary mask (ink is dark: < 0.7)
    ink1 = (arr1 < 0.7)
    ink2 = (arr2 < 0.7)
    intersection = (ink1 & ink2).sum()
    union = (ink1 | ink2).sum()
    iou = float(intersection) / max(1, union)
    
    return float(s), float(iou)

# Select test characters: classic benchmark + complex + rare characters
test_chars = [
    ("永", "经典永字八法"),
    ("风", "包围与飞白"),
    ("墨", "密集复杂结构"),
    ("清", "左右偏旁间架"),
    ("和", "兰亭集序典型"),
    ("龙", "气势与连贯"),
    ("德", "严谨楷书间架"),
    ("道", "走之底与长曳"),
    ("嬖", "16画生僻字(仓内仅2张)"),
    ("奂", "上紧下松(仓内仅2张)"),
]

print(f"Total test characters: {len(test_chars)}")
for ch, desc in test_chars:
    print(f"  Character: {ch} ({desc})")

# Evaluate metrics
results = []
rendered_grid = {}

for ch, desc in test_chars:
    std_img = render_char(ch, STD_FONT_PATH)
    if std_img is None:
        print(f"Warning: Std font failed on {ch}")
        continue
    
    row_images = {"std": std_img}
    for font_name, font_path in CANDIDATE_FONTS.items():
        c_img = render_char(ch, font_path)
        if c_img is None:
            print(f"Warning: {font_name} cannot render {ch}")
            continue
        s, iou = compute_metrics(std_img, c_img)
        row_images[font_name] = (c_img, s, iou)
        results.append({
            "character": ch,
            "desc": desc,
            "font": font_name,
            "ssim": s,
            "iou": iou
        })
    rendered_grid[ch] = (desc, row_images)

# Summary statistics per font
res_df = pd.DataFrame(results)
print("\n" + "="*70)
print("各候选字体与标准骨架 (simkai) 的差异度统计评测：")
print("="*70)
for font_name in CANDIDATE_FONTS.keys():
    sub = res_df[res_df['font'] == font_name]
    mean_ssim = sub['ssim'].mean()
    med_ssim = sub['ssim'].median()
    mean_iou = sub['iou'].mean()
    
    # Assessment
    if mean_ssim > 0.80:
        verdict = "⚠️ 过度相似 (形变梯度趋零，缺少风格信息)"
    elif 0.40 <= mean_ssim <= 0.72:
        verdict = "✅ 最佳黄金区间 (形态辨识明确 + 具有确定实质性风格形变)"
    elif mean_ssim < 0.35:
        verdict = "⚠️ 过度离散 (偏草书连笔，骨架拓扑差异过大)"
    else:
        verdict = "✓ 良好可用"
        
    print(f"【{font_name}】:")
    print(f"  SSIM均值: {mean_ssim:.4f} (中位 {med_ssim:.4f}) | IoU均值: {mean_iou:.4f} | 判定: {verdict}")

print("="*70)

# Build visualization poster
# Grid layout: Rows = len(rendered_grid), Cols = 1 (Std) + len(CANDIDATE_FONTS)
cell_size = 200
cols = 1 + len(CANDIDATE_FONTS)
rows = len(rendered_grid)
header_h = 70
label_w = 120
margin = 10

poster_w = label_w + cols * (cell_size + margin) + margin
poster_h = header_h + rows * (cell_size + margin) + margin

poster = Image.new("RGB", (poster_w, poster_h), (250, 250, 252))
pdraw = ImageDraw.Draw(poster)

try:
    font_title = ImageFont.truetype("tools/fonts/SimHei.ttf", 20)
    font_head = ImageFont.truetype("tools/fonts/SimHei.ttf", 15)
    font_char = ImageFont.truetype("tools/fonts/SimHei.ttf", 36)
    font_stat = ImageFont.truetype("tools/fonts/SimHei.ttf", 13)
except:
    font_title = font_head = font_char = font_stat = None

# Draw headers
col_names = ["标准骨架 (std)"] + list(CANDIDATE_FONTS.keys())
for ci, cname in enumerate(col_names):
    cx = label_w + ci * (cell_size + margin) + margin
    pdraw.rectangle([cx, 10, cx + cell_size, header_h - 10], fill=(235, 238, 245), outline=(210, 215, 225))
    clean_name = cname.split(" ")[0]
    sub_name = cname.split(" ")[1] if " " in cname else ""
    pdraw.text((cx + 10, 16), clean_name, fill=(20, 20, 30), font=font_head)
    if sub_name:
        pdraw.text((cx + 10, 38), sub_name, fill=(100, 105, 120), font=font_stat)

# Draw rows
for ri, (ch, (desc, row_imgs)) in enumerate(rendered_grid.items()):
    ry = header_h + ri * (cell_size + margin)
    
    # Draw row label
    pdraw.rectangle([margin, ry, label_w - margin, ry + cell_size], fill=(240, 242, 248), outline=(220, 225, 235))
    pdraw.text((margin + 25, ry + 30), ch, fill=(10, 20, 50), font=font_char)
    # Wrap desc
    lines = [desc[:7], desc[7:]] if len(desc) > 7 else [desc]
    for li, line in enumerate(lines):
        pdraw.text((margin + 10, ry + 120 + li * 20), line, fill=(80, 85, 100), font=font_stat)
    
    # Draw std cell
    std_img = row_imgs["std"].resize((cell_size - 10, cell_size - 10), Image.Resampling.LANCZOS)
    cx = label_w + margin
    pdraw.rectangle([cx, ry, cx + cell_size, ry + cell_size], fill=(255, 255, 255), outline=(200, 205, 215))
    poster.paste(std_img.convert("RGB"), (cx + 5, ry + 5))
    pdraw.text((cx + 10, ry + cell_size - 22), "基准: g_std", fill=(50, 100, 200), font=font_stat)
    
    # Draw candidate cells
    for ci, font_name in enumerate(CANDIDATE_FONTS.keys()):
        cx = label_w + (ci + 1) * (cell_size + margin) + margin
        pdraw.rectangle([cx, ry, cx + cell_size, ry + cell_size], fill=(255, 255, 255), outline=(200, 205, 215))
        
        if font_name in row_imgs:
            c_img, s, iou = row_imgs[font_name]
            c_resized = c_img.resize((cell_size - 10, cell_size - 10), Image.Resampling.LANCZOS)
            poster.paste(c_resized.convert("RGB"), (cx + 5, ry + 5))
            
            # Draw score tag
            col = (20, 140, 50) if 0.40 <= s <= 0.72 else ((180, 50, 20) if s > 0.80 else (180, 120, 10))
            pdraw.rectangle([cx + 5, ry + cell_size - 25, cx + cell_size - 5, ry + cell_size - 5], fill=(245, 245, 245))
            pdraw.text((cx + 10, ry + cell_size - 22), f"SSIM: {s:.3f} | IoU: {iou:.3f}", fill=col, font=font_stat)

out_png1 = "docs/04_experiments/imgs/font_candidates_preview.png"
out_png2 = "assets/results/font_candidates_preview.png"
os.makedirs(os.path.dirname(out_png1), exist_ok=True)
os.makedirs(os.path.dirname(out_png2), exist_ok=True)
poster.save(out_png1)
poster.save(out_png2)
print(f"\n✓ 抽样对照全景海报已保存至:\n  - {out_png1}\n  - {out_png2}")
