#!/usr/bin/env python
"""额外评估指标: OCR 字准率 + KID（小样本友好的分布距离）。

## 为什么
ssim/mse/lpips 对"字形正确性"不敏感 —— 一个模糊到不可读的字 ssim 可能不低。
需要:
  - **OCR 字准率**: 生成的图能不能被识别成正确的字（最直接的"写对没写对"）
  - **KID**: 小样本友好的分布距离（FID 在 n<1000 时有偏且方差大）

## 用法
  1) 先让 batch_eval 存图:
       python tools/eval/eval_stdskel_batch.py --results-dir OUT --ckpt-override CK \
         --device cpu --sets "strict:csv:249" --save-samples
  2) 再算指标:
       python tools/eval_extra_metrics.py --dir OUT/eval_samples_ctrl \
         --csv assets/eval_v13_strict_fixed.csv --ocr rapidocr
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch

os.chdir("/root/Workspace/xy/DiT")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="含 g{i}.png / gt{i}.png 的目录")
    p.add_argument("--csv", default="assets/eval_v13_strict_fixed.csv",
                   help="对应 eval csv（按行序取 character）")
    p.add_argument("--ocr", default="rapidocr",
                   choices=["rapidocr", "qwenvl", "both", "none"])
    p.add_argument("--kid-feat", default="inception",
                   choices=["inception", "dino", "none"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="", help="结果写到哪里（csv 追加）")
    return p.parse_args()


# ── KID（polynomial kernel MMD，无偏估计）─────────────────────────────
def polynomial_mmd(a, b, degree=3, gamma=None):
    """KID 的无偏估计。a,b: (n,d) / (m,d) 特征。"""
    if gamma is None:
        gamma = 1.0 / a.shape[1]

    def _k(x, y):
        return (gamma * (x @ y.T) + 1.0) ** degree

    m, n = a.shape[0], b.shape[0]
    k_aa = _k(a, a)
    k_bb = _k(b, b)
    k_ab = _k(a, b)
    # 去掉对角线 -> 无偏
    aa = (k_aa.sum() - np.trace(k_aa)) / (m * (m - 1))
    bb = (k_bb.sum() - np.trace(k_bb)) / (n * (n - 1))
    ab = k_ab.mean()
    return float(aa + bb - 2 * ab)


def load_inception(dev):
    from torchvision.models import inception_v3, Inception_V3_Weights
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                     transform_input=False, aux_logits=True)
    m.fc = torch.nn.Identity()
    return m.to(dev).eval()


def feats_inception(paths, dev, bs=16):
    from PIL import Image
    from torchvision import transforms
    m = load_inception(dev)
    tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            xs = torch.stack([tf(Image.open(p).convert("RGB"))
                              for p in paths[i:i + bs]]).to(dev)
            out.append(m(xs).cpu().numpy())
    return np.concatenate(out, 0)


def load_dino(dev):
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                       trust_repo=True)
    return m.to(dev).eval()


def feats_dino(paths, dev, bs=16):
    from PIL import Image
    from torchvision import transforms
    m = load_dino(dev)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            xs = torch.stack([tf(Image.open(p).convert("RGB"))
                              for p in paths[i:i + bs]]).to(dev)
            out.append(m(xs).cpu().numpy())
    return np.concatenate(out, 0)


# ── OCR 字准率 ────────────────────────────────────────────────────────
def ocr_accuracy(gpaths, chars, kind):
    if kind == "none":
        return None, None
    from PIL import Image
    from opencc import OpenCC
    t2s = OpenCC("t2s")
    if kind == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

        def pad(im, p=0.4):
            w, h = im.size
            s = int(max(w, h) * (1 + p * 2))
            c = Image.new("RGB", (s, s), "white")
            c.paste(im, ((s - w) // 2, (s - h) // 2))
            return c

        preds = []
        for p in gpaths:
            try:
                res, _ = ocr(np.array(pad(Image.open(p).convert("RGB"))))
                preds.append("".join(x[1] for x in res)[:1] if res else "?")
            except Exception:
                preds.append("?")
    elif kind in ("qwenvl", "both"):
        # Qwen3-VL-4B：能输出繁体，比 rapidocr(简体字表) 更适合书法
        import torch as _t
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as _M
        except ImportError:
            from transformers import AutoModelForImageTextToText as _M
        mp = "/root/Workspace/xy/DiT/_models/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
        print(f"  [qwenvl] 加载 {mp} ...", flush=True)
        m = _M.from_pretrained(mp, dtype=_t.float32, low_cpu_mem_usage=True).eval()
        pr = AutoProcessor.from_pretrained(mp)
        PROMPT = ("这是一张中国古人书法的单字图片。请识别图片中的汉字。"
                  "如果它是繁体字或异体字，请严格按图片实际写法输出繁体/异体字符。"
                  "只输出那一个字，不要任何解释、标点或空格。")
        preds = []
        for i, p in enumerate(gpaths):
            try:
                im = Image.open(p).convert("RGB")
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": PROMPT}]}]
                tx = pr.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
                inp = pr(text=[tx], images=[im], return_tensors="pt")
                with _t.no_grad():
                    gen = m.generate(**inp, max_new_tokens=8, do_sample=False)
                n_in = inp["input_ids"].shape[1]
                preds.append((pr.batch_decode(gen[:, n_in:],
                             skip_special_tokens=True)[0] or "?").strip()[:1] or "?")
            except Exception as e:
                preds.append("?")
            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{len(gpaths)}", flush=True)
        # 存预测供核对
        try:
            with open("/tmp/_qwenvl_preds.txt", "w", encoding="utf-8") as f:
                for p, c, q in zip(gpaths, chars, preds):
                    f.write(f"{os.path.basename(p)}\tgt={c}\tvlm={q}\n")
        except Exception:
            pass
    else:
        raise NotImplementedError(kind)

    exact = sum(1 for p, c in zip(preds, chars) if p == c)
    relaxed = sum(1 for p, c in zip(preds, chars)
                  if t2s.convert(p) == t2s.convert(c))
    return exact / max(len(chars), 1), relaxed / max(len(chars), 1)


def main():
    a = get_args()
    dev = torch.device(a.device)

    # ⚠ 正则要锚定 .png 结尾：gt0.png 会被 g(\d+) 误匹配（g 后面是 t）-> None.group 崩
    gs = sorted((p for p in glob.glob(os.path.join(a.dir, "g*.png"))
                 if re.search(r"g(\d+)\.png$", p)),
                key=lambda p: int(re.search(r"g(\d+)\.png$", p).group(1)))
    gts = sorted((p for p in glob.glob(os.path.join(a.dir, "gt*.png"))
                  if re.search(r"gt(\d+)\.png$", p)),
                 key=lambda p: int(re.search(r"gt(\d+)\.png$", p).group(1)))
    print(f"  g: {len(gs)}, gt: {len(gts)}", flush=True)
    if not gs:
        raise SystemExit("没找到 g*.png —— 请给 batch_eval 加 --save-samples")

    chars = []
    if os.path.exists(a.csv):
        rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
        chars = [r["character"] for r in rows[:len(gs)]]
    print(f"  csv 取到 {len(chars)} 个 character", flush=True)

    res = {}
    # OCR
    if chars:
        ex, re_ = ocr_accuracy(gs, chars, a.ocr)
        if ex is not None:
            res["ocr_exact"] = round(ex, 4)
            res["ocr_relaxed"] = round(re_, 4)
            print(f"  OCR 字准率: exact={ex:.4f}  relaxed(简繁等价)={re_:.4f}",
                  flush=True)

    # KID
    if a.kid_feat != "none" and gts:
        print(f"  算 KID（{a.kid_feat}）...", flush=True)
        fn = feats_inception if a.kid_feat == "inception" else feats_dino
        fg = fn(gs, dev)
        fgt = fn(gts, dev)
        res["kid"] = round(polynomial_mmd(fg, fgt), 6)
        print(f"  KID = {res['kid']:.6f}", flush=True)

    print("\n  === 汇总 ===")
    for k, v in res.items():
        print(f"    {k:<14} {v}")

    if a.out:
        exists = os.path.exists(a.out)
        with open(a.out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["dir"] + list(res.keys()))
            w.writerow([a.dir] + list(res.values()))
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
