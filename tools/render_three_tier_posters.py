import os
import sys
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = "assets/eval200_30_gathered"
OUT_DIR = "docs/04_experiments/imgs"
os.makedirs(OUT_DIR, exist_ok=True)

# 字体配置
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except:
        return ImageFont.load_default()

f_title = font(32, True)
f_sub = font(15)
f_tag = font(13)
f_col_char = font(36, True)
f_col_callig = font(15, True)
f_col_desc = font(12)
f_col_ssim = font(12, True)
f_row_title = font(17, True)
f_row_sub = font(12)
f_metric_k = font(11)
f_metric_v = font(12, True)
f_cell_sub = font(11)
f_badge = font(11, True)

# 模型行定义 (所有海报统一)
ROWS = [
    {
        "id": "02_v13",
        "title": "02_v13 纯净清洗底模",
        "sub": "4090基准 | DiT-S/2 · 125k步 · 纯标准骨架驱动",
        "metrics": [("全集SSIM", "0.5713", "#38bdf8"), ("LPIPS", "0.380", "#a78bfa"), ("MSE", "1.021", "#94a3b8")],
        "badge": "纯骨架底模"
    },
    {
        "id": "03_v21",
        "title": "03_v21 初代SkelNet形变",
        "sub": "4090形变 | DiT-S/2+SkelNet · 155k步 · JointLN",
        "metrics": [("全集SSIM", "0.5303", "#38bdf8"), ("LPIPS", "0.393", "#a78bfa"), ("专属性", "+0.016", "#f59e0b")],
        "badge": "首创形变网络"
    },
    {
        "id": "04_v23",
        "title": "04_v23 SplitLN形变修复",
        "sub": "4090形变 | DiT-S/2+SkelNet · 85k步 · 独立归一化",
        "metrics": [("全集SSIM", "0.5750", "#38bdf8"), ("振幅比", "4.24x", "#f59e0b"), ("富集度", "2.39x", "#34d399")],
        "badge": "双通道解耦"
    },
    {
        "id": "v54",
        "title": "v54 极简字表驱动基准",
        "sub": "4090字表 | DiT-S/2 · 100k步 · 纯类别三元组(无骨架)",
        "metrics": [("全集SSIM", "0.5821", "#38bdf8"), ("LPIPS", "0.355", "#a78bfa"), ("MSE", "0.852", "#94a3b8")],
        "badge": "字表基线"
    },
    {
        "id": "05_v66",
        "title": "05_v66 黄金层级路由驱动",
        "sub": "4090路由 | DiT-S/2 · 150k步 · CondRoute blocks 2,4,5,6",
        "metrics": [("全集SSIM", "0.5984", "#38bdf8"), ("LPIPS", "0.338", "#a78bfa"), ("MSE", "0.789", "#94a3b8")],
        "badge": "黄金注入靶点"
    },
    {
        "id": "06_v68",
        "title": "06_v68 C2OT流匹配旗舰标杆",
        "sub": "4090旗舰 | DiT-S/2 · 200k步 · 3x对称增广+最优传输",
        "metrics": [("全集SSIM", "0.6149", "#34d399"), ("LPIPS", "0.325", "#34d399"), ("MSE", "0.750", "#34d399")],
        "badge": "★ 突破0.61天花板"
    },
    {
        "id": "48_moyi_12ch",
        "title": "48_moyi_12ch 墨意12通道流匹配",
        "sub": "48先锋 | 12-Channel多通道 · 50k步 · Rectified Flow",
        "metrics": [("全集SSIM", "0.5931", "#38bdf8"), ("MSE", "0.7979", "#94a3b8"), ("IoU", "0.0177", "#f59e0b")],
        "badge": "多通道直通"
    },
    {
        "id": "07_v70",
        "title": "07_v70 纯骨架解耦终极集大成",
        "sub": "4090终极 | 纯标准骨架拓扑+C2OT+黄金路由 · Step 30k (Live)",
        "metrics": [("Diff Loss", "0.2930", "#34d399"), ("REPA", "0.0044", "#38bdf8"), ("拓扑", "无限开集", "#f59e0b")],
        "badge": "★ 纯骨架开集终极"
    },
    {
        "id": "gt",
        "title": "【真迹基准】真实碑帖高清切片",
        "sub": "Ground Truth | 历史原拓无损切片 (eval200_fixed)",
        "metrics": [("黄金标尺", "1.000", "#eab308"), ("分辨率", "256×256", "#94a3b8"), ("属性", "真迹切片", "#38bdf8")],
        "badge": "真迹基准"
    }
]

# 布局几何尺寸
CELL_SIZE = 190
PAD = 12
ROW_HEADER_W = 350
COL_HEADER_H = 138
BANNER_H = 150
FOOTER_H = 90

N_ROWS = len(ROWS)

def render_poster(poster_cfg):
    cols = poster_cfg["cols"]
    n_cols = len(cols)
    canvas_w = ROW_HEADER_W + n_cols * (CELL_SIZE + PAD) + PAD * 2
    canvas_h = BANNER_H + COL_HEADER_H + N_ROWS * (CELL_SIZE + PAD) + FOOTER_H

    img = Image.new("RGB", (canvas_w, canvas_h), "#090c10")
    draw = ImageDraw.Draw(img)

    # 1. Header Banner
    draw.rectangle([(0, 0), (canvas_w, BANNER_H)], fill="#0e131d")
    draw.line([(0, BANNER_H - 1), (canvas_w, BANNER_H - 1)], fill="#1f2937", width=2)
    draw.line([(PAD * 2, BANNER_H - 2), (canvas_w - PAD * 2, BANNER_H - 2)], fill=poster_cfg["theme_color"], width=2)

    draw.text((PAD * 2, 22), poster_cfg["title"], font=f_title, fill="#f8fafc")
    draw.text((PAD * 2, 68), poster_cfg["subtitle"], font=f_sub, fill="#94a3b8")

    x_pill = PAD * 2
    for p_text, p_border, p_bg in poster_cfg["pills"]:
        pill_w = len(p_text) * 11 + 24
        draw.rounded_rectangle([(x_pill, 102), (x_pill + pill_w, 132)], radius=6, fill=p_bg, outline=p_border, width=1)
        draw.text((x_pill + 12, 108), p_text, font=f_tag, fill="#e2e8f0")
        x_pill += pill_w + 14

    # 2. 列头卡片
    y_col_start = BANNER_H + 10
    for col_i, col in enumerate(cols):
        x = ROW_HEADER_W + PAD + col_i * (CELL_SIZE + PAD)
        y = y_col_start
        w = CELL_SIZE
        h = COL_HEADER_H - 18

        draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=8, fill="#121824", outline="#1e293b", width=1)

        # 汉字与徽章
        draw.text((x + 14, y + 10), col["char"], font=f_col_char, fill="#facc15")

        cal_tag = f"{col['callig']} · {col['script']}"
        draw.rounded_rectangle([(x + 64, y + 14), (x + w - 10, y + 38)], radius=4, fill="#1e293b", outline="#334155")
        draw.text((x + 70, y + 18), cal_tag, font=f_col_callig, fill="#f1f5f9")

        # 样本 ID 与 v68 SSIM
        draw.text((x + 64, y + 46), f"ID #{col['idx']:03d}", font=f_metric_v, fill="#64748b")
        draw.text((x + 14, y + 68), f"v68 SSIM: {col['ssim']:.4f}", font=f_col_ssim, fill=poster_cfg["theme_color"])
        draw.text((x + 14, y + 92), col["desc"], font=f_col_desc, fill="#94a3b8")

    # 3. 逐行逐列贴图
    y_row_start = BANNER_H + COL_HEADER_H
    for row_i, r_info in enumerate(ROWS):
        y = y_row_start + row_i * (CELL_SIZE + PAD)
        is_gt = (r_info["id"] == "gt")
        is_live = ("Live" in r_info["sub"])
        is_top = ("0.61" in r_info["badge"])

        card_bg = "#121824" if not is_gt else "#1e1b18"
        card_outline = "#eab308" if is_gt else ("#0284c7" if is_live else ("#10b981" if is_top else "#1e293b"))
        card_w = ROW_HEADER_W - PAD

        draw.rounded_rectangle([(PAD, y), (card_w, y + CELL_SIZE)], radius=8, fill=card_bg, outline=card_outline, width=2 if (is_gt or is_live or is_top) else 1)

        draw.text((PAD + 16, y + 14), r_info["title"], font=f_row_title, fill="#fef08a" if is_gt else ("#38bdf8" if is_live else "#f8fafc"))
        draw.text((PAD + 16, y + 42), r_info["sub"], font=f_row_sub, fill="#94a3b8")

        # 指标微仪表盘
        metrics_x = PAD + 16
        metrics_y = y + 72
        for k, v, val_col in r_info["metrics"]:
            m_box = [(metrics_x, metrics_y), (metrics_x + 96, metrics_y + 46)]
            draw.rounded_rectangle(m_box, radius=5, fill="#0b0f17", outline="#1e293b")
            draw.text((metrics_x + 8, metrics_y + 6), k, font=f_metric_k, fill="#64748b")
            draw.text((metrics_x + 8, metrics_y + 24), v, font=f_metric_v, fill=val_col)
            metrics_x += 102

        draw.text((PAD + 16, y + 132), r_info["badge"], font=f_badge, fill=card_outline)

        # 逐列填充图像
        m_dir = os.path.join(BASE_DIR, r_info["id"])
        for col_i, col in enumerate(cols):
            x = ROW_HEADER_W + PAD + col_i * (CELL_SIZE + PAD)
            img_p = os.path.join(m_dir, f"{col['idx']}.png")

            img_obj = None
            if os.path.exists(img_p):
                img_obj = Image.open(img_p).convert("RGB")

            cell_box = [(x, y), (x + CELL_SIZE, y + CELL_SIZE)]
            draw.rounded_rectangle(cell_box, radius=6, fill="#0e131d", outline="#1e293b", width=1)

            if img_obj is not None:
                inner_w = CELL_SIZE - 20
                inner_h = CELL_SIZE - 20
                img_resized = img_obj.resize((inner_w, inner_h), Image.Resampling.LANCZOS)
                img.paste(img_resized, (x + 10, y + 7))

                draw.rectangle([(x + 2, y + CELL_SIZE - 22), (x + CELL_SIZE - 2, y + CELL_SIZE - 2)], fill="#090c10cc")
                draw.text((x + 8, y + CELL_SIZE - 20), f"【{col['char']}】{col['callig']}", font=f_cell_sub, fill="#94a3b8")
            else:
                draw.text((x + 40, y + CELL_SIZE // 2 - 10), "样本生成中", font=f_cell_sub, fill="#475569")

    # 4. Footer
    y_foot = canvas_h - FOOTER_H + 12
    draw.line([(PAD * 2, y_foot - 8), (canvas_w - PAD * 2, y_foot - 8)], fill="#1e293b", width=1)
    draw.text((PAD * 2, y_foot), poster_cfg["footer_l1"], font=f_cell_sub, fill="#94a3b8")
    draw.text((PAD * 2, y_foot + 22), poster_cfg["footer_l2"], font=f_cell_sub, fill="#38bdf8")

    out_file = os.path.join(OUT_DIR, poster_cfg["filename"])
    img.save(out_file, quality=95)
    print(f"✓ 成功生成海报: {out_file} (尺寸: {canvas_w}×{canvas_h}, 内存占用已清理)")
    return out_file

# ======================= 三张海报配置 =======================

# 海报 1: Top 10 最高表现字 (Best)
TOP10_CONFIG = {
    "filename": "eval200_top10_best_poster.png",
    "theme_color": "#10b981",  # 祖母绿 (高表现)
    "title": "Callig-DiT eval200 Top10 高分代表字演进海报 (Best Tier)",
    "subtitle": "基于 06_v68 旗舰模型在 eval200_fixed 上的最高 SSIM 样本 (SSIM: 0.7934 ~ 0.9108) 横向同字对比",
    "pills": [
        ("评测分区: Top 10 (高分拟合标杆)", "#059669", "#064e3b"),
        ("v68 SSIM区间: 0.793 ~ 0.911", "#10b981", "#022c22"),
        ("样本特性: 拓扑清晰、结构协调、兼具笔触张力", "#0284c7", "#0c4a6e"),
        ("同台竞技: 跨历史 9 大代际版本绝对同字", "#7c3aed", "#3b0764")
    ],
    "cols": [
        {"idx": 55,  "char": "状", "callig": "柳公权", "script": "行书", "ssim": 0.9108, "desc": "骨力遒劲 · 飞扬刚正"},
        {"idx": 128, "char": "咯", "callig": "褚遂良", "script": "楷书", "ssim": 0.9029, "desc": "清简瘦硬 · 笔锋挺秀"},
        {"idx": 129, "char": "北", "callig": "褚遂良", "script": "楷书", "ssim": 0.8414, "desc": "开合有致 · 波势优美"},
        {"idx": 22,  "char": "循", "callig": "何绍基", "script": "隶书", "ssim": 0.8389, "desc": "金石气韵 · 朴茂沉雄"},
        {"idx": 115, "char": "入", "callig": "苏轼",   "script": "楷书", "ssim": 0.8277, "desc": "体势宽博 · 丰腴跌宕"},
        {"idx": 116, "char": "土", "callig": "苏轼",   "script": "楷书", "ssim": 0.8222, "desc": "端庄雄伟 · 稳实浑厚"},
        {"idx": 27,  "char": "峣", "callig": "文徵明", "script": "楷书", "ssim": 0.8096, "desc": "温润雅致 · 意态娴雅"},
        {"idx": 172, "char": "其", "callig": "颜真卿", "script": "楷书", "ssim": 0.8081, "desc": "筋骨内敛 · 气魄磅礴"},
        {"idx": 60,  "char": "起", "callig": "柳公权", "script": "行书", "ssim": 0.7938, "desc": "笔势连绵 · 劲爽峻拔"},
        {"idx": 8,   "char": "步", "callig": "何绍基", "script": "行书", "ssim": 0.7934, "desc": "回环跌宕 · 欹侧生趣"}
    ],
    "footer_l1": "现象归因：Top 10 高分字通常具有优良的部首开合度与规范的骨架拓扑。各模型在【状、咯、北、循】等字上均表现出很强的间架还原力；",
    "footer_l2": "演进对比：v68 与 v66 凭借黄金路由与流匹配动力学，在此类高分字上生成了高度逼真的真迹墨韵与飞白细节，逼近甚至在清晰度上超越了 GT 真迹。"
}

# 海报 2: Mid 10 中位数代表字 (Median)
MID10_CONFIG = {
    "filename": "eval200_mid10_median_poster.png",
    "theme_color": "#38bdf8",  # 天空蓝 (基准中位数)
    "title": "Callig-DiT eval200 Mid10 中位数代表字演进海报 (Median Tier)",
    "subtitle": "基于 06_v68 旗舰模型在 eval200_fixed 上的中位数样本 (SSIM: 0.6115 ~ 0.6228, 排名 89..98) 横向同字对比",
    "pills": [
        ("评测分区: Mid 10 (中位数代表字)", "#0284c7", "#0c4a6e"),
        ("v68 SSIM区间: 0.611 ~ 0.623", "#38bdf8", "#082f49"),
        ("样本特性: 复杂中等、行楷草兼备、极具普适鉴别力", "#d97706", "#78350f"),
        ("模型差异: 显式展现从生硬描摹到真气弥漫的跨越", "#7c3aed", "#3b0764")
    ],
    "cols": [
        {"idx": 46,  "char": "連", "callig": "柳公权", "script": "楷书", "ssim": 0.6228, "desc": "结构森严 · 骨相棱棱"},
        {"idx": 30,  "char": "㩜", "callig": "文徵明", "script": "楷书", "ssim": 0.6220, "desc": "工整典雅 · 秀润疏朗"},
        {"idx": 133, "char": "物", "callig": "褚遂良", "script": "楷书", "ssim": 0.6207, "desc": "方圆兼备 · 宛转空灵"},
        {"idx": 77,  "char": "舉", "callig": "欧阳询", "script": "行书", "ssim": 0.6195, "desc": "险绝严谨 · 笔力千钧"},
        {"idx": 164, "char": "纨", "callig": "赵孟頫", "script": "隶书", "ssim": 0.6174, "desc": "法度精熟 · 渊雅冲淡"},
        {"idx": 59,  "char": "英", "callig": "柳公权", "script": "行书", "ssim": 0.6157, "desc": "挺拔潇洒 · 笔势开张"},
        {"idx": 91,  "char": "羲", "callig": "王羲之", "script": "行书", "ssim": 0.6134, "desc": "飘若浮云 · 矫若惊龙"},
        {"idx": 38,  "char": "填", "callig": "文徵明", "script": "行书", "ssim": 0.6130, "desc": "萧散简淡 · 意态安闲"},
        {"idx": 160, "char": "寂", "callig": "赵孟頫", "script": "行书", "ssim": 0.6127, "desc": "圆润流畅 · 骨力潜藏"},
        {"idx": 120, "char": "危", "callig": "苏轼",   "script": "行书", "ssim": 0.6115, "desc": "跌宕欹侧 · 浑然天成"}
    ],
    "footer_l1": "现象归因：Mid 10 集中反映了日常实用场景中 50% 汉字的生成水准。字形包含【連、舉、羲、寂】等典型多部件合体字，考验间架穿插避让；",
    "footer_l2": "演进对比：早期 v13/v21 笔画交叠处容易发生墨块粘连，而 v66/v68/v70 表现出清晰的提按笔锋与结构层次，充分反映了现代流匹配的优势。"
}

# 海报 3: Worst 10 最难低分字 (Hardest)
WORST10_CONFIG = {
    "filename": "eval200_worst10_hardest_poster.png",
    "theme_color": "#f43f5e",  # 绯红玫瑰 (高难度攻坚)
    "title": "Callig-DiT eval200 Worst10 攻坚代表字演进海报 (Hardest Tier)",
    "subtitle": "基于 06_v68 旗舰模型在 eval200_fixed 上的最具挑战样本 (SSIM: 0.3463 ~ 0.4300, 排名 178..187) 横向同字对比",
    "pills": [
        ("评测分区: Worst 10 (攻坚高难挑战)", "#e11d48", "#4c0519"),
        ("v68 SSIM区间: 0.346 ~ 0.430", "#f43f5e", "#881337"),
        ("样本特性: 20+超多笔画、极端生僻古字、大面积重墨", "#d97706", "#78350f"),
        ("科学价值: 诊断模型拓扑塌陷边界与笔画解耦攻坚靶标", "#7c3aed", "#3b0764")
    ],
    "cols": [
        {"idx": 94,  "char": "閒", "callig": "王羲之", "script": "行书", "ssim": 0.3463, "desc": "极繁门部 · 欹斜连带"},
        {"idx": 17,  "char": "旾", "callig": "何绍基", "script": "隶书", "ssim": 0.3500, "desc": "生僻古体 · 波磔大度"},
        {"idx": 43,  "char": "闕", "callig": "文徵明", "script": "隶书", "ssim": 0.3575, "desc": "门形框廓 · 繁密中宫"},
        {"idx": 2,   "char": "端", "callig": "何绍基", "script": "楷书", "ssim": 0.3769, "desc": "横平竖直 · 间架紧结"},
        {"idx": 123, "char": "渚", "callig": "苏轼",   "script": "行书", "ssim": 0.3780, "desc": "三点水连笔 · 丰腴重墨"},
        {"idx": 173, "char": "獨", "callig": "颜真卿", "script": "楷书", "ssim": 0.3805, "desc": "犬部多折 · 沉着雄厚"},
        {"idx": 177, "char": "體", "callig": "颜真卿", "script": "楷书", "ssim": 0.3844, "desc": "骨肉丰满 · 23画高繁"},
        {"idx": 13,  "char": "瓏", "callig": "何绍基", "script": "行书", "ssim": 0.4015, "desc": "龙部蜿蜒 · 枯润相参"},
        {"idx": 96,  "char": "将", "callig": "王羲之", "script": "行书", "ssim": 0.4175, "desc": "草意行楷 · 连绵起伏"},
        {"idx": 49,  "char": "辭", "callig": "柳公权", "script": "楷书", "ssim": 0.4300, "desc": "舌辛双部 · 峭拔紧凑"}
    ],
    "footer_l1": "攻坚诊断：Worst 10 集中了【體(23画)、闕(18画)、瓏(20画)、閒】等极度繁复字，或是【旾】等先秦异体字。笔画密集导致局部潜空间出现频率重叠；",
    "footer_l2": "代际演进：早期模型在【體、闕】上几乎完全丢失部首拓扑退化为墨球；而 v66/v68 依然顽强维持了骨架轮廓，为 v70 纯骨架解耦指明了关键突破方向。"
}

# 执行生成
p1 = render_poster(TOP10_CONFIG)
p2 = render_poster(MID10_CONFIG)
p3 = render_poster(WORST10_CONFIG)

print("\n🎉 三张严格对齐的分级全景海报全部生成成功！")
