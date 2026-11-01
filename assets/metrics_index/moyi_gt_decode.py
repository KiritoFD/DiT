"""Decode characters from MoYi GT images (no targets needed, pure argmax)."""
import io
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
OUT = Path("assets/metrics_index/moyi_gt_decoded.json")


def main():
    from tools.f_20261101 import eval_arm as ea
    import numpy as np
    from PIL import Image

    def _load_img(path):
        """Exactly the preprocessing used by the project OCR (32px tall, [-1,1])."""
        img = Image.open(path).convert("L")
        w, h = img.size
        tw = int(np.clip(round(32 * w / h), 16, 320))
        img = img.resize((tw, 32), Image.Resampling.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255
        return (x - 0.5) / 0.5

    net, char2idx = ea.ocr_setup("cuda")
    idx2char = {v: k for k, v in char2idx.items()}
    imgs = [str(PANEL / f"gt{i}.png") for i in range(187)]
    arrs = [_load_img(p) for p in imgs]
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(arrs), 16):
            xs = torch.stack([torch.from_numpy(a)[None] for a in arrs[i:i + 16]]).to("cuda")
            logits = net(xs).cpu()
            prob = logits.softmax(-1)
            if prob.ndim > 2:            # (B, T, C) -> take mean over time
                prob = prob.mean(1)
            top = prob.argmax(-1)
            preds.extend(idx2char.get(int(t), "?") for t in top)
            confs.extend(float(prob[j, int(top[j])]) for j in range(len(top)))
    print("decoded", len(preds), "unique chars", len(set(preds)))
    print("mean conf", round(sum(confs) / len(confs), 3))
    print("first 20:", "".join(preds[:20]))
    print("common:", Counter(preds).most_common(8))
    OUT.write_text(json.dumps({"pred": preds, "conf": confs}, ensure_ascii=False, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
