"""Five phase posters from archived observations/PNGs; no model or VAE inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/phase_review_20261013"
W, LEFT, CELL, GAP = 1500, 340, 256, 16
INK, MUTED, BLUE, ORANGE = "#192d3e", "#5c6f7d", "#146777", "#a35721"
SOURCES = {}


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def remember(path):
    path = ROOT / path
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def wrapped(draw, text, xy, width, size=22, fill=INK):
    x, y = xy
    line = ""
    for ch in text:
        if ch == "\n" or draw.textlength(line + ch, font=font(size)) > width:
            draw.text((x, y), line, font=font(size), fill=fill)
            y += size + 11
            line = "" if ch == "\n" else ch
        else:
            line += ch
    draw.text((x, y), line, font=font(size), fill=fill)
    return y + size + 11


def canvas(title, subtitle, h):
    im = Image.new("RGB", (W, h), "#ffffff")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 12), fill=BLUE)
    d.text((40, 30), title, font=font(36), fill=INK)
    wrapped(d, subtitle, (40, 88), W - 80, 22, MUTED)
    return im, d


def finish(im, filename, note, y):
    d = ImageDraw.Draw(im)
    d.line((40, y, W - 40, y), fill="#d8e2e6", width=2)
    end = wrapped(d, note, (40, y + 20), W - 80, 21, MUTED)
    assert end < im.height - 12, (filename, end, im.height)
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / filename)


def tile(im, path, x, y):
    with Image.open(remember(path)) as src:
        assert src.size == (256, 256), (path, src.size)
        im.paste(src.convert("RGB"), (x, y))


def row(im, y, label, paths, detail=None):
    d = ImageDraw.Draw(im)
    wrapped(d, label, (40, y + 15), LEFT - 60, 23)
    for j, p in enumerate(paths):
        x = LEFT + j * (CELL + GAP)
        tile(im, p, x, y)
        if detail:
            wrapped(d, detail[j], (x, y + CELL + 8), CELL, 17, MUTED)
    return y + CELL + (70 if detail else 36)


def stage1():
    im, d = canvas("01  两阶段 SkelNet：中间骨架成为瓶颈", "10.1—10.3 | 标准骨架 → 书家骨架 latent → 图像渲染；结构误差在接口处累积。", 1020)
    boxes = [(40, "标准骨架", "已知字形空间条件"), (540, "SkelNet / stage1", "预测真迹骨架 latent"), (1040, "stage2 渲染", "依赖预测骨架的质量")]
    for x, title, desc in boxes:
        d.rounded_rectangle((x, 180, x + 420, 310), radius=16, fill="#eef5f6", outline="#abc8cd")
        d.text((x + 20, 201), title, font=font(28), fill=BLUE)
        d.text((x + 20, 260), desc, font=font(22), fill=INK)
    for x in [474, 974]:
        d.text((x, 219), "→", font=font(34), fill=BLUE)
    wrapped(d, "latent L2 下降不等于解码后的细线与连通性正确。V45 的骨架 SSIM 已超过标准字基线，IoU 却更差。", (40, 350), 1420, 26)
    panels = [(40, "骨架 SSIM ↑", .6410, .6484, .70), (780, "骨架 IoU ↑", .1044, .0870, .12)]
    for x, title, base, pred, cap in panels:
        d.text((x, 460), title, font=font(28), fill=INK)
        for k, (name, val, color) in enumerate([("标准字基线", base, BLUE), ("V45 预测", pred, ORANGE)]):
            yy = 535 + k * 92
            d.text((x, yy), name, font=font(22), fill=INK)
            bx = x + 166
            d.rectangle((bx, yy + 4, bx + 350 * val / cap, yy + 37), fill=color)
            d.text((bx + 365, yy), f"{val:.4f}", font=font(23), fill=color)
    wrapped(d, "阶段决定：放弃“先预测一条真迹骨架，再把它当强接口渲染”的主线。换容量、注入方式与结构 loss 未形成可靠收益。", (40, 735), 1420, 26)
    remember(Path("docs/experiments/2026-10-03-twostage-archive.md"))
    finish(im, "01_skelnet.png", "数值为两阶段归档中 V45 同 checkpoint 的同口径对照；原档未明确登记 N，不与后续 N187 排名拼表。早期 strict84 存在训练重叠，oracle 高分不作泛化成绩。", 880)


def stage1_runs():
    im, d = canvas("01B  SkelNet 实际推理：预测骨架失去字形拓扑",
                   "2026-09-29 归档拼图 | 每组三列为标准骨架输入 / SkelNet 预测 / 真迹骨架目标；保留原始像素内容。", 1600)
    srcpath = Path("_review/skelnet_dit_montage.png")
    with Image.open(remember(srcpath)) as src:
        assert src.size == (776, 2048), src.size
        for part, x in enumerate([40, 780]):
            d.text((x, 165), f"样例 {part * 4 + 1}—{part * 4 + 4}", font=font(21), fill=BLUE)
            for off, label in [(10, "std 输入"), (242, "SkelNet 预测"), (476, "真迹目标")]:
                d.text((x + off, 199), label, font=font(18), fill=MUTED)
            crop = src.crop((0, part * 1024, 776, (part + 1) * 1024)).convert("RGB")
            crop = crop.resize((680, 897), Image.Resampling.NEAREST)
            im.paste(crop, (x, 235))
            d.rectangle((x, 235, x + 680, 1132), outline="#abc8cd", width=2)
    wrapped(d, "实际画面说明：输入与目标仍是稀疏可辨的字形骨架，中间预测却退化成大面积碎片和伪纹理；即使 loss 看似下降，也不能作为 stage2 的可靠结构接口。", (40, 1165), 1420, 24)
    heads = ["run", "步数", "骨架 SSIM", "结构判读"]
    xs = [40, 390, 650, 920]
    for x, t in zip(xs, heads):
        d.text((x, 1260), t, font=font(23), fill=BLUE)
    rows = [
        ("v38 · add", "2k", "0.6101", "随后退化"),
        ("v44 · concat", "4k", "0.6282", "样本效率提高，仍未过结构线"),
        ("v44 · 续训", "10k", "0.585—0.628", "震荡，无上升趋势"),
        ("v45 · Dice", "1k", "0.6484", "IoU 0.0870 < std 基线 0.1044"),
    ]
    for i, rr in enumerate(rows):
        yy = 1310 + i * 42
        for x, t in zip(xs, rr):
            d.text((x, yy), t, font=font(21), fill=INK)
    remember(Path("docs/experiments/2026-10-03-twostage-archive.md"))
    finish(im, "01_skelnet_runs.png", "拼图原文件未嵌入具体 run / step，因此只作定性失败证据；v38/v44/v45 数值来自两阶段归档，不把拼图冒充为其中某一条定量 run。", 1500)


def stage2():
    ids = [0, 1, 2, 3]
    im, d = canvas("02  Moyi 复现：12ch 没有带来结构红利", "10.3 起 | 同 50k 对照：图像 + Canny + skeleton 联合扩散，并未胜过纯图像 4ch。", 1350)
    y = 175
    base = Path("docs/106moyi/moyi/results/eval_full_metrics")
    for label, prefix in [("GT\n展示 4 张；指标不适用", "gt"),
                          ("Moyi 12ch · 50k\nSSIM 0.5935\nLPIPS 0.3724\nSkel IoU 0.0177\n全评测 N187", "moyi_12_50k"),
                          ("Moyi 4ch · 50k\nSSIM 0.5949\nLPIPS 0.3678\nSkel IoU 0.0161\n全评测 N187", "moyi_4_50k")]:
        y = row(im, y, label, [base / f"{prefix}_{i}.png" for i in ids])
    wrapped(d, "4ch 的 SSIM / LPIPS 略好；12ch 的 Skel IoU / MSE 略好。不能写成全部指标反超，但没有证据支持为骨架扩成 12ch。4ch 继续到 80k：SSIM 0.6085 / LPIPS 0.3489。", (40, y + 10), 1420, 25)
    remember(Path("docs/106moyi/moyi_experiments_summary.csv"))
    finish(im, "02_moyi.png", "所有样图直接取自归档，保持 256×256。行首为全 N187 历史指标，不是展示 4 张的均值；LPIPS 依归档口径，不与后面 alex 表直接比较。", 1190)


def stage3(metrics):
    ids = [75, 82, 158, 133]
    im, d = canvas("03  不再预测 skel：改用对比学习字符表", "10.5—10.8 左右 | 参考三因素条件：书家 / 书体 / 字符；字形由表征提供，不再作为中间骨架目标。", 1320)
    y = 175
    entries = [("GT\n固定 V68 排名选样\n指标不适用", "gt"),
               ("旧 std-skel · V70\n30k 图像批\nSSIM 0.5473\nLPIPS-alex 0.4779\nInk IoU 0.2320", "v70"),
               ("对比学习字表 · V68\n200k 图像批\nSSIM 0.6169\nLPIPS-alex 0.3911\nInk IoU 0.3044", "v68")]
    for label, key in entries:
        y = row(im, y, label, [Path("eval") / key / f"{i}.png" for i in ids])
    wrapped(d, "阶段成果：三表 + 主干路由 + 单阶段图像目标，V68 达到约 0.61。后续骨架回测同 65k：V68 0.5833 > V70 0.5535。字表路线成为主线。", (40, y), 1420, 25)
    finish(im, "03_character_table.png", "这是路线阶段展示，不是两行样图的单变量对照：图像批 step 不同。行指标由已存 N187 PNG 统一重算；LPIPS-alex。选样沿用 V68 固定排名位置。", 1165)


def stage4():
    im, d = canvas("04  VAE：DINO 丢高频，回到 LPIPS + PatchGAN", "10.8—10.11 | 像素重建分数好，不代表细线 / 飞白和高频保真；下面按真实试验顺序记录。", 1600)
    timeline = [
        (170, "① 首轮直接接 DINO", "L1 + KL + DINO_struct/style；去掉 LPIPS / GAN。PSNR / SSIM ↑，顶带高频却从 SD 0.816 降到 0.783。"),
        (270, "② 补高频并查合同", "DoG 增益很小；直接 top-band 才到约 0.828。并发现旧 shard 沿用 SD scale，存盘 latent std 约 0.20。"),
        (370, "③ 干净底座重训", "从 SD-VAE 重新开始，删除 DINO，恢复 LPIPS + PatchGAN；加入 Laplacian / top-band，GAN 5k 步后启用并自适应加权。"),
    ]
    for y, title, detail in timeline:
        d.rounded_rectangle((40, y, 1460, y + 76), radius=14, fill="#eef5f6", outline="#abc8cd")
        d.text((60, y + 20), title, font=font(24), fill=BLUE)
        wrapped(d, detail, (350, y + 16), 1080, 21)
    heads = ["同批 N187 往返", "旧 DINO 配方", "物理配方"]
    xs = [40, 590, 1050]
    for x, t in zip(xs, heads):d.text((x, 500), t, font=font(26), fill=BLUE)
    vals = [("PSNR ↑", "30.56 dB", "34.04 dB"), ("SSIM ↑", "0.9852", "0.9949"),
            ("顶带份额 / GT ↑", "0.783", "0.901"), ("重建 LPIPS ↓", "0.1281", "0.0135")]
    for i, vals_row in enumerate(vals):
        for x, t in zip(xs, vals_row):d.text((x, 555 + i * 48), t, font=font(25), fill=INK)
    wrapped(d, "最终提升来自整套像素 / 纹理配方；GAN 的自适应权重最终约 0.001，没有无 GAN 消融，不能把收益单独归给 PatchGAN。", (40, 755), 1420, 23)
    srcpath = Path("_local_ckpt/vae_vis/recon_step_19000.png")
    with Image.open(remember(srcpath)) as src:
        for r in range(2):
            label = "GT\n展示批指标不适用" if r == 0 else "物理 VAE · 19k 重建\n展示批指标未登记\n上表为最终制品验收"
            wrapped(d, label, (40, 870 + r * 278), 280, 22)
            for j, col in enumerate([0, 2, 4, 6]):
                im.paste(src.crop((col * 256, r * 256, (col + 1) * 256, (r + 1) * 256)).convert("RGB"), (LEFT + j * 272, 845 + r * 278))
    remember(Path("docs/experiments/recent_vae_20261013/vae_acceptance.json"))
    remember(Path("docs/vae_training_recipe.md"))
    finish(im, "04_vae.png", "原 19k GT / 重建图，仅裁四个 256px tile；上表是最终 N187 验收，非展示批指标。DINO 阶段的结论针对整套配方；重建改善不等于生成涨点。", 1455)


def run_wall(metrics, filename, title, subtitle, entries, note):
    ids = [75, 82, 62, 154]
    h = 205 + len(entries) * 292 + 135
    im, d = canvas(title, subtitle, h)
    y = 175
    for key, label in entries:
        if key == "gt":
            paths = [Path("eval/gt") / f"{idx}.png" for idx in ids]
            text_label = label + "\n指标不适用"
        else:
            rec = metrics[key]
            m = rec["mean"]
            paths = [Path(rec["source_pattern"].format(idx=idx)) for idx in ids]
            text_label = (f"{label}\nSSIM {m['ssim']:.4f}\nLPIPS-alex {m['lpips_alex']:.4f}\n"
                          f"Ink IoU {m['ink_iou']:.4f}\nfrag {m['frag']:.3f}\nN187")
        y = row(im, y, text_label, paths)
    finish(im, filename, note, y + 8)


def stage4_runs(metrics):
    run_wall(metrics, "04_vae_runs.png", "04B  VAE 更替后的 DiT 推理结果墙",
             "同四个 eval200fix 样例逐 run 展示；行指标按保存 PNG 统一重算。不同 step / 数据 / REPA / OT 只作阶段观察。",
             [("gt", "GT · 固定样例"),
              ("v68", "v68 · SD-VAE · 200k"),
              ("v72", "v72 · DINO-VAE · 130k"),
              ("v76", "v76 · topband VAE · 40k"),
              ("v77", "v77 · 去 REPA · 35k"),
              ("v79", "v79 · physics+C2OT\n55k"),
              ("v80_best", "v80-best · physics no-OT\n190k")],
             "每行是全 N187 指标，不是四张展示图的均值。这些 run 并非逐项单变量，v80-v79 或 v72-v68 的差值不能只归给 VAE。")


def stage5(metrics, inspect=False):
    ids = [75, 82, 62, 154]
    if inspect:
        im, d = canvas("SDE 局部退化样例核对", "诊断选样：Ink IoU 下降 / 局部碎片增加；不是总体效果估计。每列 GT | ODE | SDE。", 180 + len(ids) * 310 + 100)
        for k, idx in enumerate(ids):
            y = 170 + k * 310
            d.text((40, y + 18), f"#{idx}", font=font(25), fill=INK)
            for j, key in enumerate(["gt", "v68", "v68_sde"]):tile(im, Path("eval") / key / f"{idx}.png", 340 + j * 272, y)
        OUT.mkdir(parents=True, exist_ok=True)
        im.save(OUT / "sde_inspection.png")
        return
    im, d = canvas("05  SDE free lunch：更干净，不等于更完整", "10.8 之后 | 不重训改采样器；V68 旧 SDE 的平均 SSIM 上升，但部分字会缺笔 / 断细笔。", 1390)
    aa = {r["idx"]:r for r in metrics["v68"]["per_image"]}
    bb = {r["idx"]:r for r in metrics["v68_sde"]["per_image"]}
    y = 175
    for label, key in [("GT\n指标不适用", "gt"),
                       ("V68 ODE · 200k\nSSIM 0.6169\nLPIPS-alex 0.3911\nInk IoU 0.3044\n全评测 N187", "v68"),
                       ("V68 旧 SDE · γ0.5\nSSIM 0.6260\nLPIPS-alex 0.3989\nInk IoU 0.2961\n全评测 N187", "v68_sde")]:
        detail = [f"#{i}" if key=="gt" else f"#{i}  IoU {({'v68':aa,'v68_sde':bb}[key])[i]['ink_iou']:.3f}" for i in ids]
        y = row(im, y, label, [Path("eval") / key / f"{i}.png" for i in ids], detail)
    wrapped(d, "平均 SSIM 涨点，与局部笔画消失可以同时发生。物理 VAE 的 V80 上，后续 corrected SDE 未验证稳定额外感知收益，不能继续当统一默认。", (40, y), 1420, 24)
    finish(im, "05_sde.png", "前两列沿用原 poster 固定排名样例；后两列专门展示局部缺笔，不代表随机样本。总体指标取全 N187。旧漂移 + C2OT 不冒充 corrected SDE；tile 保持原像素。", 1240)


def stage5_runs(metrics):
    run_wall(metrics, "05_sde_runs.png", "05B  SDE 跨 run 推理结果墙",
             "同四个字纵向比较 ODE / SDE；V68、V72、V79 使用旧有偏漂移，V80 使用 corrected SDE。",
             [("gt", "GT · 固定样例"),
              ("v68", "v68 · ODE · 200k"),
              ("v68_sde", "v68 · 旧 SDE · γ0.5"),
              ("v72", "v72 · ODE · 130k"),
              ("v72_sde", "v72 · 旧 SDE · γ0.5"),
              ("v79", "v79 · ODE · 55k"),
              ("v79_sde", "v79 · 旧 SDE · γ0.5"),
              ("v80", "v80 · ODE · 50k"),
              ("v80_sde", "v80 corrected · γ0.5")],
             "全 N187 统一 PNG 指标。SDE 在 v68 改善平均 SSIM / frag，却在 v72 不复现；v79/v80 的感知或墨迹变化也不构成稳定 free lunch。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect-sde", action="store_true")
    args = p.parse_args()
    source = Path("docs/experiments/v68_v70_sde/row_metrics.json")
    metrics = json.loads(remember(source).read_text(encoding="utf-8"))
    if args.inspect_sde:
        stage5(metrics, inspect=True)
        return
    stage1();stage1_runs();stage2();stage3(metrics);stage4();stage4_runs(metrics);stage5(metrics);stage5_runs(metrics)
    remember(Path(__file__).resolve().relative_to(ROOT))
    posters = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("0[1-5]_*.png")}
    (OUT / "sources.json").write_text(json.dumps({"source_sha256":SOURCES,"poster_sha256":posters,"method":"Pillow layout only; no inference; all book/calligraphy run-wall tiles native 256px","run_wall_indices":[75,82,62,154],"sde_indices":[75,82,62,154],"sde_selection":"first two from original fixed rank poster; last two diagnostic stroke-loss cases, not random/global-effect estimate","stage3_indices":[75,82,158,133]},ensure_ascii=False,indent=2),encoding="utf-8")
    print("POSTERS_OK",OUT)


if __name__ == "__main__":
    main()
