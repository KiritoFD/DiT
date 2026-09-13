# -*- coding: utf-8 -*-
"""vae_fast.py — 高性能 SD-VAE encode/decode 轮子.

动机: diffusers 默认 encode/decode 是小 batch、fp32、无编译的 memory-bound 实现,
4090 上 GPU util 100% 但功耗只有 ~165W (SM 没喂饱)。本模块:
  * channels_last + cudnn.benchmark
  * torch.compile (默认 max-autotune-no-cudagraphs), 产物落 TORCHINDUCTOR_CACHE_DIR
    -> 用 --precompile 预热一次, 后续进程复用编译缓存 (不必每次重编)
  * 定长分块 (remainder 用 padding 补齐) 避免 dynamic shape 反复重编
  * encode 默认 bf16 (快), decode 默认 fp32 (sd-vae fp16 decode 会掉质量)
  * 直接调 encoder/quant_conv / post_quant_conv/decoder, 精确复现 vae.encode().mode()

用法:
  # 1) 预编译 + 缓存 (跑一次; 之后冷启动也快)
  python -m src.eval.vae_fast --precompile --cache-dir ~/.cache/torchinductor_dit
  # 2) benchmark (对比默认实现)
  python -m src.eval.vae_fast --bench --batch 64 --n 1024
  # 3) 作为库
  from src.eval.vae_fast import FastVAE
  fv = FastVAE(device="cuda")
  z  = fv.encode(imgs)   # (N,3,256,256)[-1,1] -> (N,4,32,32) float32 CPU (已 ×0.18215)
  im = fv.decode(z)      # (N,4,32,32) [scaled] -> (N,3,256,256) [-1,1]
"""
import argparse
import os
import time

import torch


def _default_cache_dir():
    return os.path.expanduser("~/.cache/torchinductor_dit")


class FastVAE:
    def __init__(self, vae_path="data/pretrained/sd-vae-ft-ema", device="cuda",
                 compile=True, compile_mode="max-autotune-no-cudagraphs",
                 encode_dtype=torch.bfloat16, decode_dtype=torch.float32,
                 cache_dir=None, chunk=64):
        if cache_dir:
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
        from diffusers import AutoencoderKL
        self.device = torch.device(device)
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        self.sf = 0.18215
        self.encode_dtype = encode_dtype
        self.decode_dtype = decode_dtype
        self.chunk = int(chunk)
        self._enc = self.vae.encoder
        self._dec = self.vae.decoder
        if compile:
            try:
                self._enc = torch.compile(self._enc, mode=compile_mode)
                self._dec = torch.compile(self._dec, mode=compile_mode)
                self.compiled = True
            except Exception as e:  # noqa: BLE001
                print(f"[vae_fast] torch.compile 不可用 ({e}); 用 eager", flush=True)
                self.compiled = False
        else:
            self.compiled = False

    # ---- 精确复现 AutoencoderKL.encode(x).latent_dist.mode() * sf ----
    @torch.inference_mode()
    def encode(self, x, chunk=None):
        chunk = int(chunk or self.chunk)
        n = x.shape[0]
        outs = []
        for s in range(0, n, chunk):
            xb = x[s:s + chunk]
            pad = chunk - xb.shape[0]
            if pad:
                xb = torch.cat([xb, xb[-1:].repeat(pad, 1, 1, 1)], 0)
            xb = xb.to(self.device, non_blocking=True)
            if xb.shape[1] == 1:
                xb = xb.repeat(1, 3, 1, 1)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=self.encode_dtype,
                                enabled=(self.encode_dtype != torch.float32)):
                h = self._enc(xb)
                h = self.vae.quant_conv(h)
            h = h.float()
            mean = h[:, : h.shape[1] // 2]
            outs.append((mean * self.sf).cpu())
        return torch.cat(outs, 0)[:n]

    @torch.inference_mode()
    def decode(self, z, chunk=None):
        chunk = int(chunk or self.chunk)
        n = z.shape[0]
        outs = []
        for s in range(0, n, chunk):
            zb = z[s:s + chunk]
            pad = chunk - zb.shape[0]
            if pad:
                zb = torch.cat([zb, zb[-1:].repeat(pad, 1, 1, 1)], 0)
            zb = zb.to(self.device, non_blocking=True)
            zb = zb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=self.decode_dtype,
                                enabled=(self.decode_dtype != torch.float32)):
                h = self.vae.post_quant_conv(zb.float())
                img = self._dec(h)
            outs.append(img.float().clamp(-1, 1).cpu())
        return torch.cat(outs, 0)[:n]


def _bench(vae_path, device, batch, n, compile, chunk):
    torch.manual_seed(0)
    x = torch.randn(n, 3, 256, 256)
    z = torch.randn(n, 4, 32, 32)
    if compile:
        fv = FastVAE(vae_path, device=device, compile=True, chunk=chunk)
        e, d = fv.encode, fv.decode
        tag = "fast(compile)"
    else:
        from diffusers import AutoencoderKL
        v = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
        for p in v.parameters():
            p.requires_grad_(False)

        @torch.inference_mode()
        def e(xx):
            out = []
            for s in range(0, xx.shape[0], chunk):
                out.append((v.encode(xx[s:s + chunk].to(device)).latent_dist.mode() * 0.18215).cpu())
            return torch.cat(out, 0)

        @torch.inference_mode()
        def d(zz):
            out = []
            for s in range(0, zz.shape[0], chunk):
                out.append(v.decode(zz[s:s + chunk].to(device)).sample.float().cpu())
            return torch.cat(out, 0)
        tag = "default(diffusers)"

    # warmup (触发编译)
    t0 = time.time()
    _ = e(x[:chunk]); _ = d(z[:chunk])
    torch.cuda.synchronize()
    warm = time.time() - t0
    t0 = time.time(); _ = e(x); torch.cuda.synchronize(); te = time.time() - t0
    t0 = time.time(); _ = d(z); torch.cuda.synchronize(); td = time.time() - t0
    pk = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[{tag}] warmup(compile)={warm:.1f}s  encode {n/te:.1f} img/s ({te:.2f}s)  "
          f"decode {n/td:.1f} img/s ({td:.2f}s)  peak={pk:.2f}G", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompile", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--compare", action="store_true", help="同时跑默认实现对比")
    ap.add_argument("--vae-path", default="data/pretrained/sd-vae-ft-ema")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--cache-dir", default=_default_cache_dir())
    ap.add_argument("--compile", type=int, default=1)
    args = ap.parse_args()

    if args.precompile or args.bench:
        os.makedirs(args.cache_dir, exist_ok=True)
    if args.precompile:
        print(f"[vae_fast] precompile -> cache {args.cache_dir}", flush=True)
        fv = FastVAE(args.vae_path, device=args.device, compile=True,
                     cache_dir=args.cache_dir, chunk=args.chunk)
        fv.encode(torch.randn(args.chunk, 3, 256, 256))
        fv.decode(torch.randn(args.chunk, 4, 32, 32))
        print("[vae_fast] precompile done", flush=True)
    if args.bench:
        _bench(args.vae_path, args.device, args.batch, args.n, bool(args.compile), args.chunk)
        if args.compare:
            _bench(args.vae_path, args.device, args.batch, args.n, False, args.chunk)


if __name__ == "__main__":
    main()
