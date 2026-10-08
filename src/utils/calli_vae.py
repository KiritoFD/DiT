# -*- coding: utf-8 -*-
"""calli_vae.py — Calli-VAE 正确的 encode/decode 封装 (2026-10-09 重写)

【为什么要专门写这个文件 / 事故复盘】
阶段一 Calli-VAE (calli_vae_step_25000) 的训练脚本约定是:

    posterior = vae.encode(x).latent_dist
    z         = posterior.sample()
    x_recon   = vae.decode(z / 0.18215).sample

即: **decoder 的输入 = posterior.sample() / 0.18215** (std≈5.3),
而不是 sd-vae 的 "decoder 输入 = posterior.mode()"。
而且该 VAE 的 posterior 方差很大 (logvar≈-0.34 → std≈0.88), decoder 是
"必须带随机分量才能重建"的:

    decode(sample / 0.18215) ≈ 0.028  ✅   本文件采用的约定
    decode(mode   / 0.18215) ≈ 0.608  ❌   灰
    decode(mode)             ≈ 0.77   ❌   灰

v71 直接套用了 sd-vae 管线, 编码用 latent_dist.mode()*0.18215 (std≈0.08),
于是 eval 全灰 ([152,155,150], std≈6.7)。⇒ **不是图像预处理问题, 是 latent 约定错了。**
已发布到 ModelScope 的 Calli-VAE 权重本身没有问题。

【正确约定】(本文件 + tools/encode_calli_latents.py)
    encode:  z     = vae.encode(x).latent_dist.sample()   # (N,4,H/8,W/8), std≈0.97
    decode:  x_hat = vae.decode(z / 0.18215).sample       # (N,3,H,W) in [-1,1]
    DiT 配置:  vae_scaling_factor 仍填 0.18215 (decode 侧用);
              编码侧 (在线/离线) **不要** 再 *0.18215。

接入点:
    * 离线 latent shards   -> tools/encode_calli_latents.py
    * 在线 (无 cache) 训练 -> 不要 vae.encode(x).latent_dist.sample().mul_(sf),
                              改成 .sample() (去掉 mul_)。
    * eval / 推理 decode   -> 现有 vae.decode(lat / sf) 不用改 ✅。

自检:
    python -m src.utils.calli_vae --check \
        --vae /home/ds/Workspace/DiT/experiments/calli_vae_dino/calli_vae_step_25000 \
        --imgs /home/ds/Workspace/moyi/data/unified_393k/imgs --n 8
    期望: CORRECT L1 ≈ 0.03, WRONG L1 ≈ 0.6~0.8 (灰)。
"""
from __future__ import annotations

import argparse
import glob
import os

import torch

# 阶段一 VAE 训练脚本里的 scaling_factor (decoder 输入 = sample / 该值)
CALLI_SCALING_FACTOR = 0.18215


class CalliVAE:
    """Calli-VAE 正确的 encode/decode。

    encode(x) -> z      : z = posterior.sample()               (std≈0.97)
    decode(z) -> x_hat  : x_hat = vae.decode(z / sf).sample    (sf=0.18215)
    decode(encode(x)) ≈ x   (L1≈0.03)。

    与 sd-vae 的唯一差异: 编码器输出用 ``sample`` 而不是 ``mode * sf``;
    decode 侧 ``/sf`` 保持不变, 所以 eval/推理端代码无需改动。
    """

    def __init__(
        self,
        vae_path: str,
        device: str | None = None,
        encode_dtype: torch.dtype = torch.bfloat16,
        decode_dtype: torch.dtype = torch.float32,
        chunk: int = 64,
    ) -> None:
        from diffusers import AutoencoderKL

        self.vae_path = str(vae_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vae = AutoencoderKL.from_pretrained(self.vae_path).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.sf = float(getattr(self.vae.config, "scaling_factor", CALLI_SCALING_FACTOR) or CALLI_SCALING_FACTOR)
        self.encode_dtype = encode_dtype
        self.decode_dtype = decode_dtype
        self.chunk = int(chunk)
        self._cuda = self.device.type == "cuda"

    # ------------------------------------------------------------------ #
    # encode: 生成喂给 DiT 的 latent (raw posterior sample)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def encode(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """x:(N,3,H,W) in [-1,1] -> z:(N,4,H/8,W/8) CPU float32。

        注意: 这是 **随机** 编码 (posterior.sample)。传 ``generator`` 可复现;
        同一个图像用不同 seed 会得到不同的 z, 但 decoder 对随机分量鲁棒,
        解码质量一致 (已验证多 seed L1≈0.028)。
        """
        x = x.float()
        out = []
        for s in range(0, x.shape[0], self.chunk):
            xb = x[s:s + self.chunk].to(self.device, non_blocking=True)
            if xb.shape[1] == 1:
                xb = xb.repeat(1, 3, 1, 1)
            with torch.autocast(self.device.type, dtype=self.encode_dtype, enabled=self._cuda):
                dist = self.vae.encode(xb).latent_dist
                z = dist.sample(generator) if generator is not None else dist.sample()
            out.append(z.float().cpu())
        return torch.cat(out, 0)

    @torch.inference_mode()
    def encode_mode(self, x: torch.Tensor) -> torch.Tensor:
        """posterior mean (仅诊断用; **不要** 拿它喂 DiT 或配 /sf 解码 -> 会灰)。"""
        x = x.float()
        out = []
        for s in range(0, x.shape[0], self.chunk):
            xb = x[s:s + self.chunk].to(self.device, non_blocking=True)
            if xb.shape[1] == 1:
                xb = xb.repeat(1, 3, 1, 1)
            with torch.autocast(self.device.type, dtype=self.encode_dtype, enabled=self._cuda):
                z = self.vae.encode(xb).latent_dist.mode()
            out.append(z.float().cpu())
        return torch.cat(out, 0)

    # ------------------------------------------------------------------ #
    # decode: raw AutoencoderKL 变体, 与现有 vae.decode(lat / sf) 等价
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z:(N,4,h,w) -> x_hat:(N,3,H,W) in [-1,1]。

        等价于 ``vae.decode(z / 0.18215)`` —— 现有 eval/推理管线里
        ``vae.decode(lat / sf)`` 是直接可用的, 不用改。
        """
        z = z.float()
        out = []
        for s in range(0, z.shape[0], self.chunk):
            zb = z[s:s + self.chunk].to(self.device, non_blocking=True) / self.sf
            with torch.autocast(self.device.type, dtype=self.decode_dtype, enabled=self._cuda):
                img = self.vae.decode(zb).sample
            out.append(img.float().clamp(-1, 1).cpu())
        return torch.cat(out, 0)


# ---------------------------------------------------------------------- #
# 自检
# ---------------------------------------------------------------------- #
def _check(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image
    import torchvision.transforms.functional as TF

    vae = CalliVAE(args.vae, device=args.device, chunk=args.chunk)
    files = sorted(glob.glob(os.path.join(args.imgs, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(args.imgs, "**", "*.png"), recursive=True))
    files = files[: args.n]
    if not files:
        raise SystemExit(f"no png found under {args.imgs}")

    x = torch.stack([
        TF.to_tensor(Image.open(p).convert("RGB").resize((256, 256))) * 2 - 1 for p in files
    ])
    with torch.no_grad():
        z = vae.encode(x)                                    # 正确: sample
        xh = vae.decode(z)                                   # 正确: decode(z/sf)
        zm = vae.encode_mode(x)                              # 错误路径诊断
        xh_mode = vae.vae.decode((zm.to(vae.device) / vae.sf)).sample.float().cpu()          # decode(mode/sf)
        xh_mode_sd = vae.vae.decode((zm.to(vae.device) / vae.sf * vae.sf)).sample.float().cpu()  # decode(mode) == v71 现状

    print(f"VAE        : {vae.vae_path}")
    print(f"scaling    : {vae.sf}")
    print(f"latent z   : std={z.std():.4f} mean={z.mean():.4f}  (期望 std≈0.97)")
    print(f"CORRECT    : decode(encode(x))        L1={(xh - x).abs().mean():.4f}   <- 期望 ≈0.03")
    print(f"WRONG(灰)  : decode(mode/scale)        L1={(xh_mode - x).abs().mean():.4f}")
    print(f"WRONG(灰)  : decode(mode)  [v71 现状]  L1={(xh_mode_sd - x).abs().mean():.4f}")

    if args.out:
        rows = [x, xh, xh_mode, xh_mode_sd]

        def to_np(t):
            return ((t.cpu().clamp(-1, 1) + 1) / 2 * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)

        rows_np = [to_np(r) for r in rows]
        h, w = 256, 256
        canvas = np.ones((4 * h + 3 * 4, args.n * w + (args.n - 1) * 4, 3), np.uint8) * 255
        for ri, row in enumerate(rows_np):
            for ci in range(min(args.n, row.shape[0])):
                canvas[ri * (h + 4):ri * (h + 4) + h, ci * (w + 4):ci * (w + 4) + w] = row[ci]
        Image.fromarray(canvas).save(args.out)
        print(f"saved {args.out}  (行: GT | decode(encode) | decode(mode/sf) | decode(mode))")

    ok = (xh - x).abs().mean().item() < 0.10
    print("SELF-CHECK", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calli-VAE correct encode/decode")
    ap.add_argument("--check", action="store_true", help="跑自检 (encode->decode L1)")
    ap.add_argument("--vae", required=True, help="Calli-VAE 目录 (含 config.json + safetensors)")
    ap.add_argument("--imgs", default="", help="--check 用图片目录")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="", help="--check 对比图输出路径")
    args = ap.parse_args()
    if args.check:
        _check(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
