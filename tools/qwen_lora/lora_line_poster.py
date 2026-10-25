"""LoRA line poster v4: wrapped text, 3-column everywhere, contamination evidence labeled."""
import json
import sys
from pathlib import Path

root = Path('/home/ds/Workspace/xy/qwen-lora-top10-20261024')
sys.path.insert(0, '/home/ds/Workspace/DiT')
sys.path.insert(0, '/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
sys.path.insert(0, str(root / 'vendor'))

from PIL import Image, ImageDraw, ImageFont

OUT = root / 'lora_line_poster'
OUT.mkdir(exist_ok=True)
FONT = '/home/ds/Workspace/DiT/tools/fonts/NotoSerifSC-VF.ttf'
data = json.loads((root / 'data.json').read_text(encoding='utf-8'))
test = data['datasets']['test']
C, G, PAD = 196, 12, 20
COLS = 3
W = PAD * 2 + COLS * C + (COLS - 1) * G          # 652
TW = W - 2 * PAD                                  # text width budget
NAVY, RED, GRAY = (23, 55, 110), (180, 40, 30), (96, 96, 96)
f_t = ImageFont.truetype(FONT, 34)
f_h = ImageFont.truetype(FONT, 22)
f_b = ImageFont.truetype(FONT, 19)
f_s = ImageFont.truetype(FONT, 15)

test_eval = json.loads((root / 'test_eval_20k' / 'metrics.json').read_text(encoding='utf-8'))
qwen = {p.name[:3]: p for p in (root / 'test_eval_20k' / 'images').glob('*.png')}
refs = {p.name[:3]: p for p in (root / 'ref_eval_20k' / 'images').glob('*.png')}


def gt(i):
    return Image.open(root / test[i]['file']).convert('RGB')


def gen(i):
    return Image.open(qwen[f'{i:03d}']).convert('RGB')


def refgen(i):
    return Image.open(refs[f'{i:03d}']).convert('RGB')


def wrap(text, font, width):
    d = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    lines, line = [], ''
    for ch in text:
        if d.textlength(line + ch, font=font) <= width:
            line += ch
        else:
            lines.append(line)
            line = ch
    if line:
        lines.append(line)
    return lines


class Sheet:
    def __init__(self):
        self.img = Image.new('RGB', (W, 6200), 'white')
        self.d = ImageDraw.Draw(self.img)
        self.y = 0

    def text_lines(self, text, font, color):
        for line in wrap(text, font, TW):
            self.d.text((PAD, self.y), line, font=font, fill=color)
            self.y += int(font.size * 1.35)

    def title(self, text):
        self.d.text((PAD, self.y), text, font=f_t, fill=NAVY)
        self.y += 48

    def section(self, text):
        self.y += 14
        self.d.line((PAD, self.y + 6, W - PAD, self.y + 6), fill=(205, 205, 205), width=1)
        self.y += 14
        self.text_lines(text, f_h, NAVY)
        self.y += 8

    def row(self, cells):
        for j, (img, label, color) in enumerate(cells):
            x = PAD + j * (C + G)
            self.img.paste(img.resize((C, C), Image.Resampling.LANCZOS), (x, self.y))
            self.d = ImageDraw.Draw(self.img)
            self.d.rectangle((x, self.y, x + C - 1, self.y + C - 1), outline=color, width=2)
            lab = label
            while self.d.textlength(lab, font=f_s) > C + G - 4 and len(lab) > 4:
                lab = lab[:-2]
            self.d.text((x + 3, self.y + C + 2), lab, font=f_s, fill=color)
        self.y += C + 24

    def rowlabel(self, text):
        self.d.text((PAD, self.y), text, font=f_b, fill=NAVY)
        self.y += 26

    def note(self, text, color=GRAY):
        self.text_lines(text, f_b, color)
        self.y += 6


def main():
    s = Sheet()
    s.title('Qwen LoRA · Top10 书法线 · 总结')
    s.text_lines('test200：SSIM .5821 / LPIPS .4365 / Ink-IoU .2827 —— LPIPS −13%、Ink +45% vs 底模，SSIM −.021', f_b, (45, 45, 45))
    s.text_lines('非 Turbo(40步) .5758/.4299/.2791 与 Turbo(8步) 打平 → 用 Turbo（1/5 推理成本）；权重 64MB×5 已存本地', f_b, (45, 45, 45))
    s.text_lines('收敛：8k 后 SSIM/Ink 平台、LPIPS 末段仅 −.005/4k；强度 0.75–1.0 最优 → 部署 scale 1.0', f_b, (45, 45, 45))
    s.y += 8

    # A
    s.section('A · 生成质量（每行同字：GT | 20k LoRA 生成，含逐图指标）')
    for idx in (0, 30, 60):
        r = test[idx]
        m = test_eval[idx]
        s.row([(gt(idx), f'GT {r["calligrapher"]}·{r["script"]}·{r["character"]}', GRAY),
               (gen(idx), f'生成 S{m["ssim"]:.2f} L{m["lpips"]:.2f} I{m["ink_iou"]:.2f}', RED)])
    s.note('全 200 对：test200_contact_sheet_20k.jpg')

    # B
    s.section('B · D1 条件信息量（每行同字：TTF 印刷体 | Qwen 生成 | GT；Qwen 携带书家墨迹）')
    from src.utils.std_soft_glyph import StdSoftGlyphStore
    store = StdSoftGlyphStore('/home/ds/Workspace/DiT/exp-std/data/std_soft_glyphs_v1',
                              is_train=False, eval_sigma=.65, preload=True)
    SID = {'楷': 0, '行': 3, '隶': 4}
    for idx in (2, 82, 162):
        r = test[idx]
        glyph = store.get(SID[r['script']], r['character'])[0]
        arr = ((1.0 - glyph.float().clamp(0, 1)) * 255).byte().numpy()
        ttf = Image.fromarray(arr, 'L').convert('RGB')
        s.row([(ttf, 'TTF(STD)：仅结构', GRAY),
               (gen(idx), 'Qwen：书家墨迹', RED),
               (gt(idx), 'GT', GRAY)])
    s.note('Qwen vs GT：Ink .283 / LPIPS .436 ≫ TTF vs GT：.197 / .681（SSIM 均 .582）→ 有 TTF 没有的笔画信息')

    # C
    s.section('C · 书家控制：同字「乘」同 seed 只换书家名（变化真实但弱）')
    sw = root / 'p0_writer_swap' / 'images'
    trio = [data['writers'].index(w) for w in ('颜真卿', '王羲之', '赵孟頫')]
    for tag, label in (('base', '底模'), ('lora', 'LoRA 20k')):
        s.rowlabel(f'{label}：')
        s.row([(Image.open(sw / f'{tag}_00_w{wi}s0_丞.png').convert('RGB'),
                data['writers'][wi], NAVY if tag == 'lora' else GRAY) for wi in trio])
    s.note('跨字识别（LPIPS 质心分类）：底模 5/30、LoRA 8/30、GT 上限 10/30；换 seed 差异 .56 > 换书家 .72 → seed 噪声主导')

    # D
    s.section('D · 参考图注入 = 负结果，且目标字被参考污染（每行同字：GT | 无参考 | +2 参考图）')
    s.row([(gt(10), 'GT 王羲之·必', GRAY), (gen(10), '无参考（基线）', GRAY),
           (refgen(10), '+2 参考图（变差）', RED)])
    s.row([(gt(70), 'GT 颜真卿·行·深', GRAY), (gen(70), '无参考（基线）', GRAY),
           (refgen(70), '+2 参考图：写成了别的字！', RED)])
    s.note('配对全量：SSIM −.012 / LPIPS +.009 / Ink −.025；底模加参考 SSIM −.044；'
           '第二行目标「深」被参考污染写成近「九」——模型未分离参考的内容与风格 → 推理注入证伪，需带参考训练', RED)

    # E
    s.section('E · 分辨率裁决（底模九字探针：256 直出 | 512 生成→缩 256，指标打平）')
    rp = root / 'res_probe'
    files = sorted((rp / 'images').glob('*_256.png'))[:2]
    for p256 in files:
        p512 = p256.parent / p256.name.replace('_256.png', '_512.png')
        s.row([(Image.open(p256).convert('RGB'), '256 直出', GRAY),
               (Image.open(p512).convert('RGB'), '512→256', GRAY)])
    s.note('九字打平 + VLM 复核无崩坏 → 训练切 256：0.101 s/样本（比 512/B8/检查点快 5.2 倍）')

    s.img = s.img.crop((0, 0, W, s.y + PAD))
    s.img.save(OUT / 'lora_line_poster.jpg', quality=80, optimize=True)
    print('POSTER_V4_READY', s.img.size)


if __name__ == '__main__':
    main()
