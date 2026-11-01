"""OCR the MoYi GT images to reconstruct their character sequence.

GT renders are legible calligraphy, so the content label can be read off the image
instead of being looked up. Writer identity is NOT recoverable this way, so only the
OCR half of the evaluation becomes available for MoYi.
"""
import io
import json
import sys
from pathlib import Path

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
OUT = Path("assets/metrics_index/moyi_gt_ocr.json")


def main():
    from tools.f_20261101 import eval_arm as ea
    net, char2idx = ea.ocr_setup("cuda")
    imgs = [str(PANEL / f"gt{i}.png") for i in range(187)]
    res = ea.ocr_score(net, imgs, [""] * 187, char2idx, device="cuda")
    print("keys", list(res))
    print("top1", res.get("top1"), "n", res.get("n"))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
