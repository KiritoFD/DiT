import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = "assets/historical_strict_common"
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
OUT_PATH = "docs/04_experiments/imgs/historical_strict_common_poster.png"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# 阶段行顺序定义
STAGE_ORDER = [
    ("01_v10b", "一阶段: v10b 初代骨架长跑", "390k步 | 12层XAttn / 41书家 / DDPM"),
    ("02_v13",  "二阶段: v13 50k高保真清洗",  "125k步 | 4层AdaLN / 45书家 / Flow"),
    ("03_v21",  "三阶段A: v21 初代SkelNet",    "155k步 | 联合形变网格 / JointLN"),
    ("04_v23",  "三阶段B: v23 独立归一化修复", "85k步 | SplitLN / 风格振幅4.24x"),
    ("05_v66",  "四阶段: v66 风格层级路由",    "150k步 | 字表驱动 / AdaLN blocks 2,4,5,6"),
    ("06_v68",  "五阶段: v68 3x对称增广+C2OT",  "200k步 | 旗舰标杆 / 条件最优传输流匹配"),
    ("07_v70",  "六阶段: v70 纯骨架解耦重新启航", "10k步 (Live) | 纯骨架拓扑+C2OT流匹配"),
    ("99_gt",   "【真迹基准】真实碑帖切片",      "Ground Truth | 历史真迹黄金对照")
]

# 字体加载
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

def get_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except:
        return ImageFont.load_default()

f_title = get_font(28)
f_sub = get_font(15)
f_col_char = get_font(32)
f_col_info = get_font(14)
f_row_name = get_font(18)
f_row_sub = get_font(13)
f_badge = get_font(12)

# 布局参数
CELL_SIZE = 196
ROW_HEADER_W = 280
COL_HEADER_H = 110
BANNER_H = 130
PAD = 12
FOOTER_H = 40

N_COLS = len(manifest)
N_ROWS = len(STAGE_ORDER)

CANVAS_W = ROW_HEADER_W + N_COLS * (CELL_SIZE + PAD) + PAD * 2
CANVAS_H = BANNER_H + COL_HEADER_H + N_ROWS * (CELL_SIZE + PAD) + FOOTER_H

img = Image.new("RGB", (CANVAS_W, CANVAS_H), "#0f1117")
draw = ImageDraw.Draw(img)

# 1. 顶部 Banner
draw.rectangle([(0, 0), (CANVAS_W, BANNER_H)], fill="#161923")
draw.line([(0, BANNER_H), (CANVAS_W, BANNER_H)], fill="#2d3748", width=2)

draw.text((PAD * 2, 22), "Callig-DiT 历史真实评测原图对比矩阵 (Historical Strict Common Poster)", font=f_title, fill="#ffffff")
banner_sub = (
    "本海报直接提取自各阶段历史原始训练评测产物 (Authentic Checkpoint Strict Eval Artifacts)  |  "
    "覆盖全周期 7 次重大版本迭代在公共测试字符上的原汁原味真实表现"
)
draw.text((PAD * 2, 62), banner_sub, font=f_sub, fill="#a0aec0")

# 统计徽章
badge_text = f"公共汉字样本: {N_COLS} 列  |  历史代际跨度: 7 个核心检查点 + 1 组真迹基准  |  无任何二次推理偏置"
draw.rectangle([(PAD * 2, 88), (PAD * 2 + 680, 114)], fill="#232936", outline="#3b4252")
draw.text((PAD * 2 + 10, 93), badge_text, font=f_badge, fill="#63b3ed")

# 2. 列标题 (汉字与名家标注)
y_col_start = BANNER_H
for col_i, col_meta in enumerate(manifest):
    x = ROW_HEADER_W + PAD + col_i * (CELL_SIZE + PAD)
    y = y_col_start
    w = CELL_SIZE
    h = COL_HEADER_H - 10
    
    # 列卡片背景
    draw.rounded_rectangle([(x, y + 8), (x + w, y + h)], radius=6, fill="#1a202c", outline="#2d3748")
    
    # 汉字
    char = col_meta["char"]
    draw.text((x + 16, y + 16), char, font=f_col_char, fill="#ecc94b")
    
    # 标注信息
    hl = col_meta["highlight"]
    lines = hl.split("(")
    main_hl = lines[0].strip()
    sub_hl = ("(" + lines[1]) if len(lines) > 1 else ""
    
    draw.text((x + 58, y + 20), main_hl, font=f_col_info, fill="#edf2f7")
    if sub_hl:
        draw.text((x + 58, y + 42), sub_hl, font=f_badge, fill="#a0aec0")
    draw.text((x + 16, y + 68), f"Col #{col_i+1}", font=f_badge, fill="#718096")

# 3. 逐行逐列渲染图像与行标签
y_row_start = BANNER_H + COL_HEADER_H

for row_i, (sid, row_title, row_sub) in enumerate(STAGE_ORDER):
    y = y_row_start + row_i * (CELL_SIZE + PAD)
    
    # 行标签卡片
    is_gt = (sid == "99_gt")
    card_bg = "#232936" if not is_gt else "#2a2118"
    outline_col = "#3182ce" if "v68" in sid or "v70" in sid else ("#d69e2e" if is_gt else "#2d3748")
    
    draw.rounded_rectangle([(PAD, y), (ROW_HEADER_W - PAD, y + CELL_SIZE)], radius=8, fill=card_bg, outline=outline_col, width=2 if is_gt else 1)
    
    # 行文字
    draw.text((PAD + 14, y + 16), row_title, font=f_row_name, fill="#ffffff" if not is_gt else "#f6ad55")
    draw.text((PAD + 14, y + 46), row_sub, font=f_row_sub, fill="#a0aec0")
    
    # 逐列填充图像
    for col_i, col_meta in enumerate(manifest):
        x = ROW_HEADER_W + PAD + col_i * (CELL_SIZE + PAD)
        
        # 寻找对应的图片文件
        ch_dir = os.path.join(BASE_DIR, f"{col_i:02d}_{col_meta['char']}")
        img_to_paste = None
        img_label = ""
        
        if is_gt:
            # 查找任何一个阶段的 gt 图 (优先 v66/v68/v70 的 gt)
            gt_cands = [f for f in os.listdir(ch_dir) if "gt" in f and f.endswith(".png")]
            if gt_cands:
                # 优先挑 e200 时代的 gt
                pref = [f for f in gt_cands if "v66" in f or "v68" in f or "v70" in f]
                chosen = pref[0] if pref else gt_cands[0]
                img_to_paste = Image.open(os.path.join(ch_dir, chosen)).convert("RGB")
                img_label = "真迹 GT"
        else:
            # 查找该 sid 的生成图
            g_files = [f for f in os.listdir(ch_dir) if f.startswith(f"{sid}_g_") and f.endswith(".png")]
            if g_files:
                img_to_paste = Image.open(os.path.join(ch_dir, g_files[0])).convert("RGB")
                stage_data = col_meta["stages"].get(sid, {})
                c_name = stage_data.get("calligrapher", "")
                s_name = stage_data.get("script", "")
                img_label = f"{c_name} · {s_name}" if c_name else sid
        
        # 贴图与绘制小标签
        cell_box = [(x, y), (x + CELL_SIZE, y + CELL_SIZE)]
        draw.rounded_rectangle(cell_box, radius=6, fill="#141822", outline="#2d3748")
        
        if img_to_paste is not None:
            # 缩放至略小一点的区域带边距
            inner_size = CELL_SIZE - 24
            im_resized = img_to_paste.resize((inner_size, inner_size), Image.Resampling.LANCZOS)
            img.paste(im_resized, (x + 12, y + 6))
            
            # 底部小标签
            draw.rectangle([(x + 2, y + CELL_SIZE - 22), (x + CELL_SIZE - 2, y + CELL_SIZE - 2)], fill="#000000bb")
            draw.text((x + 8, y + CELL_SIZE - 20), img_label, font=f_badge, fill="#edf2f7")
        else:
            # 缺失占位
            draw.text((x + 40, y + CELL_SIZE // 2 - 10), "未参与此集评测", font=f_badge, fill="#4a5568")

# 4. 底部版权与说明
y_footer = CANVAS_H - FOOTER_H + 10
draw.text((PAD * 2, y_footer), "Callig-DiT 历史真实评测归档大图 | 来源: 各版本 eval_samples_ctrl/ | 图像真实历史产物无任何重新推理", font=f_badge, fill="#718096")

# 保存
img.save(OUT_PATH, quality=95)
print(f"🎉 历史真实评测全景海报渲染成功: {OUT_PATH}")
print(f"   尺寸: {CANVAS_W}x{CANVAS_H} px | 大小: {os.path.getsize(OUT_PATH) / 1024 / 1024:.2f} MB")
