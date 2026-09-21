"""Qwen2.5-VL 批量识别书法单字 —— GPU 版，支持断点续跑。

GPU 现在空了（v15b 已停），VLM 上 GPU 比 rapidocr CPU 快得多。
输出: assets/vlm_full_scan.csv
    idx, image_path, script, calligrapher, char_csv, char_vlm, exact, relation

用法:
  ./_venv_qwenvl/bin/python tools/vlm_full_scan.py --batch 16 --device cuda
"""
import argparse
import csv
import os
import sys
from collections import Counter
from time import time

import torch
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/Workspace/xy/DiT/_models/models/"
                                   "Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master")
ap.add_argument("--out", default="assets/vlm_full_scan.csv")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max-pixels", type=int, default=512*512, dest="max_pixels")
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--csv", default="", help="指定清单 csv（含 image_path/character 列），空=全量")
ap.add_argument("--chunk", type=int, default=200)
a = ap.parse_args()

COLS = ["idx", "image_path", "script", "calligrapher", "char_csv",
        "char_vlm", "exact", "relation"]

PROMPT = ("这是一张中国古人书法的单字图片。请识别图片中的汉字。"
          "如果它是繁体字或异体字，请严格按图片实际写法输出繁体/异体字符。"
          "只输出那一个字，不要任何解释、标点或空格。")


def main():
    from opencc import OpenCC
    t2s = OpenCC("t2s")

    if a.csv:
        items = list(csv.DictReader(open(a.csv, encoding="utf-8")))
        rows = []
        for it in items:
            rows.append({"image_path": it["image_path"],
                         "script": it.get("script", ""),
                         "calligrapher": it.get("calligrapher", ""),
                         "character": it.get("char_csv") or it.get("character", "")})
        print(f"  [csv] {a.csv} -> {len(rows)} 条", flush=True)
    else:
        rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]

    done = set()
    if os.path.exists(a.out):
        try:
            for r in csv.DictReader(open(a.out, encoding="utf-8")):
                done.add(int(r["idx"]))
        except Exception:
            pass
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]
    print(f"  总 {len(rows)}, 已完成 {len(done)}, 待处理 {len(todo)}", flush=True)
    if not todo:
        print("  没有待处理项")
        return

    print(f"  加载模型 {a.model} ...", flush=True)
    t0 = time()
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _M
    except ImportError:
        from transformers import AutoModelForImageTextToText as _M
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}[a.dtype]
    # 不用 device_map（需要 accelerate），直接 .to(device)
    model = _M.from_pretrained(a.model, dtype=dt,
                               low_cpu_mem_usage=True).to(a.device).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    # ★ 限制图像 token 数 —— 否则 Qwen2.5-VL 默认 max_pixels=1280*28*28，
    #   单字小图会被 smart_resize 放大到 ~1M 像素，token 暴涨、batch 开不大。
    try:
        proc.image_processor.max_pixels = a.max_pixels
        proc.image_processor.min_pixels = 28 * 28 * 4
        print(f"  ✓ max_pixels={a.max_pixels}", flush=True)
    except Exception as e:
        print(f"  ⚠ 设置 max_pixels 失败: {e}", flush=True)
    print(f"  ✓ 加载完成 {time()-t0:.0f}s, dtype={a.dtype}", flush=True)

    fh = open(a.out, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if os.path.getsize(a.out) == 0:
        w.writerow(COLS)
        fh.flush()

    def load_im(p):
        src = p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT", p)
        if not os.path.exists(src):
            return None
        try:
            return Image.open(src).convert("RGB")
        except Exception:
            return None

    t1 = time()
    n = 0
    bs = a.batch
    for k in range(0, len(todo), bs):
        chunk = todo[k:k + bs]
        texts, imgs, metas = [], [], []
        for i, r in chunk:
            im = load_im(r["image_path"])
            if im is None:
                metas.append((i, r, "?"))
                continue
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": PROMPT}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True))
            imgs.append(im)
            metas.append((i, r, None))
        preds = {}
        if texts:
            try:
                inp = proc(text=texts, images=imgs, return_tensors="pt",
                           padding=True).to(a.device)
                with torch.no_grad():
                    gen = model.generate(**inp, max_new_tokens=8,
                                         do_sample=False)
                n_in = inp["input_ids"].shape[1]
                outs = proc.batch_decode(gen[:, n_in:], skip_special_tokens=True)
                j = 0
                for i, r, err in metas:
                    if err is None:
                        preds[i] = (outs[j] or "?").strip()[:1] or "?"
                        j += 1
            except Exception as e:
                print(f"    batch 失败: {type(e).__name__}: {str(e)[:80]}",
                      flush=True)
        for i, r, err in metas:
            pred = err or preds.get(i, "?")
            truth = r["character"]
            exact = int(pred == truth)
            rel = ("same" if exact else
                   ("简繁" if t2s.convert(pred) == t2s.convert(truth) else "other"))
            w.writerow([i, r["image_path"], r["script"], r["calligrapher"],
                        truth, pred, exact, rel])
            n += 1
        if n % a.chunk == 0:
            fh.flush()
            el = time() - t1
            print(f"    {len(done)+n}/{len(rows)}  {n/max(el,1e-9):.1f}/s  "
                  f"剩余 {(len(todo)-n)/max(n/max(el,1e-9),1e-9)/60:.0f}min",
                  flush=True)
    fh.close()
    print(f"  DONE 新增 {n} 条, {(time()-t1)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
