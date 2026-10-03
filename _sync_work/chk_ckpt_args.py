"""看 ckpt 里存的评测参数: β 混合 / pred 目录。这决定 harness 实际喂了什么条件。"""
import torch as th

for p in ("assets/results/v32_stage2_img/20261001-062933-v32-stage2-img/checkpoints/0080000.pt",
          "assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt"):
    try:
        ck = th.load(p, map_location="cpu", weights_only=False)
        a = ck.get("args")
        d = vars(a) if a is not None else {}
        keys = [k for k in d if ("blend" in k or "pred" in k or "skel_latent" in k
                                 or "eval_steps" in k or "cfg" in k)]
        print(f"{p}\n  " + "\n  ".join(f"{k} = {d[k]!r}" for k in sorted(keys)))
    except Exception as e:                                   # noqa: BLE001
        print(f"{p}\n  读取失败: {e!r}")
