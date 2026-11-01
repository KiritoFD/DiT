"""Build a MoYi protocol from decoded GT chars and validate it by self-OCR.

Validation: run the same recogniser on GT images with the decoded labels as targets.
A self-consistent protocol must score near 100%; anything lower means the labels are
the bottleneck, not the images.
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
DEC = json.loads(Path("assets/metrics_index/moyi_gt_decoded.json").read_text())
PROTO = Path("assets/metrics_index/moyi12ch_decoded_protocol.csv")


def load_img(path):
    img = Image.open(path).convert("L")
    w, h = img.size
    tw = int(np.clip(round(32 * w / h), 16, 320))
    img = img.resize((tw, 32), Image.Resampling.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255
    return (x - 0.5) / 0.5


def main():
    from tools.f_20261101 import eval_arm as ea
    net, char2idx = ea.ocr_setup("cuda")
    idx2char = {v: k for k, v in char2idx.items()}
    pred = DEC["pred"]

    # 1. write the protocol
    with PROTO.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["img_id", "character", "script", "calligrapher"],
                           lineterminator="\n")
        w.writeheader()
        for i, ch in enumerate(pred):
            w.writerow({"img_id": f"moyi_gt_{i}", "character": ch,
                        "script": "", "calligrapher": ""})
    print("wrote protocol", PROTO)

    # 2. self-OCR: GT image vs its own decoded label
    arrs = [load_img(PANEL / f"gt{i}.png") for i in range(187)]
    hit = 0
    unk = 0
    with torch.no_grad():
        for i in range(0, len(arrs), 16):
            xs = torch.stack([torch.from_numpy(a)[None] for a in arrs[i:i + 16]]).to("cuda")
            prob = net(xs).cpu().softmax(-1)
            if prob.ndim > 2:
                prob = prob.mean(1)
            top = prob.argmax(-1)
            for j, t in enumerate(top):
                g = i + j
                if g >= len(pred):
                    break
                if pred[g] not in char2idx:
                    unk += 1
                elif idx2char.get(int(t)) == pred[g]:
                    hit += 1
    print(f"GT self-OCR: {hit}/187 ({(hit/187)*100:.1f}%), labels out of vocab: {unk}")

    # 3. generated images scored against the same labels
    garrs = [load_img(PANEL / f"g{i}.png") for i in range(187)]
    ghit = 0
    with torch.no_grad():
        for i in range(0, len(garrs), 16):
            xs = torch.stack([torch.from_numpy(a)[None] for a in garrs[i:i + 16]]).to("cuda")
            prob = net(xs).cpu().softmax(-1)
            if prob.ndim > 2:
                prob = prob.mean(1)
            top = prob.argmax(-1)
            for j, t in enumerate(top):
                g = i + j
                if g >= len(pred):
                    break
                if pred[g] in char2idx and idx2char.get(int(t)) == pred[g]:
                    ghit += 1
    print(f"generated vs decoded labels: {ghit}/187 ({(ghit/187)*100:.1f}%)")


if __name__ == "__main__":
    main()
