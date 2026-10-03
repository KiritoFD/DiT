# -*- coding: utf-8 -*-
"""探针 v2 (GPU 加速):
- P0 特征提取 500张 -> 缓存 _sync_work/_dino_patch_feats.npy (避免重提)
- P1' 字内 patch 相似度分布: 用绝对相似度(FG vs BG), 不用中位数
- P3' kmeans 用 GPU torch 实现 (K=512, 60000 子样本, 15 iter) 全数据分配
- P3b' 易混字对 token 差异率 + 曲面 patch cos
- P4' bag-of-tokens 检索 (重复字存在时才做)
"""
import sys, io, os, csv, time, json
try:
    if sys.stdout is not None and getattr(sys.stdout, "buffer", None) is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from PIL import Image

LOG = open("_sync_work/_patch_probe2.txt", "w", encoding="utf-8")
_t0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def stage(n): log(f"\n[{time.time()-_t0:6.1f}s] ===== {n} =====")

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_CSV = os.path.join(ROOT, "assets", "eval_fame_strict.csv")
CACHE = os.path.join(ROOT, "_sync_work", "_dino_patch_feats.npy")
DEVICE = "cuda"

rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))
chars = [r["character"] for r in rows]
paths = [os.path.join(ROOT, r["image_path"]) for r in rows]
N_all = len(rows)

# ---------- P0: 提取或缓存 ----------
if os.path.exists(CACHE):
    P2 = torch.from_numpy(np.load(CACHE))
    log(f"[P0] load cache {tuple(P2.shape)}")
else:
    stage("P0 extract")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").half().to(DEVICE).eval()
    feats = []
    B = 24
    for i in range(0, N_all, B):
        b_ = paths[i:i+B]
        imgs = [Image.open(p).convert("RGB") for p in b_]
        inp = processor(images=imgs, return_tensors="pt").to(DEVICE)
        inp = {k: (v.half() if v.dtype == torch.float else v) for k, v in inp.items()}
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(**inp)
        feats.append(out.last_hidden_state[:, 1:, :].float().cpu())
        if (i + B) % 480 == 0 or i + B >= N_all:
            log(f"  {min(i+B,N_all)}/{N_all}  gpu_mem={torch.cuda.memory_allocated()/1e9:.2f}GB")
    P2 = torch.cat(feats)
    np.save(CACHE, P2.numpy())
    log(f"[P0] extracted & cached {tuple(P2.shape)}")

N, NP, D = P2.shape
G = int(NP**0.5)
Pn = P2 / P2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
log(f"feats: {N}x{G}x{G}x{D}")

# ---------- P1': 绝对 FG/BG 二分化 ----------
stage("P1' patch cos-to-center")
fgs = []
for i in range(N):
    c = Pn[i].mean(0)
    sims = (Pn[i] @ c)
    mu, sd = sims.mean(), sims.std()
    fg_ratio = (sims > (mu + 0.5*sd)).float().mean().item()  # 显著高于均值=结构侧
    fgs.append(fg_ratio)
log(f"  >mean+0.5sd 占比: mean={np.mean(fgs):.3f} std={np.std(fgs):.3f} min={np.min(fgs):.3f} max={np.max(fgs):.3f}")

# ---------- 部首族 ----------
FAMS = {"氵":"海深流洞淡","扌":"推招挂","心":"想志忍怒","日":"晚春是晨"}
POS = {"氵":"L","扌":"L","心":"B","日":"L"}
def msk(pos):
    m = torch.zeros(G,G,dtype=torch.bool)
    if pos=="L": m[:, :G//3]=True
    elif pos=="T": m[:G//3,:]=True
    elif pos=="B": m[2*G//3:,:]=True
    else: m[:, :2]=True; m[:,-2:]=True; m[:2,:]=True; m[-2:,:]=True
    return m.reshape(-1)

# ---------- P2: 部首区跨字 cos (cached result from v1, 重算快速) ----------
stage("P2 radical cross-sim")
rng = np.random.default_rng(0)
base = []
for _ in range(1500):
    i,j = rng.integers(0,N,2)
    base.append((Pn[i]@Pn[j].T).max(1).values.mean().item())
log(f"  random best-match={np.mean(base):.4f}±{np.std(base):.4f}")
for rad,hit in FAMS.items():
    idx=[chars.index(c) for c in hit]; m=msk(POS[rad]); ss=[]
    for a in range(len(idx)):
        for b in range(a+1,len(idx)):
            ss.append((Pn[idx[a]][m]@Pn[idx[b]][m].T).max(1).values.mean().item())
    log(f"  {rad}: {np.mean(ss):.4f} (n={len(ss)})  {'<<< 分量弱' if np.mean(ss)-np.mean(base)<0.12 else ''}")

# ---------- P3': GPU kmeans ----------
stage("P3' GPU kmeans K=512")
sub_idx = rng.choice(N*NP, size=60000, replace=False) if N*NP>60000 else np.arange(N*NP)
Xsub = Pn.reshape(-1,D)[sub_idx].to(DEVICE)
K = 512; ITER = 15
# 初始化: 随机 K 个
perm = torch.randperm(len(Xsub), device=DEVICE)[:K]
cent = Xsub[perm].clone()
BCH = 8192
for it in range(ITER):
    # assign
    labs = torch.empty(len(Xsub), dtype=torch.long, device=DEVICE)
    for s in range(0, len(Xsub), BCH):
        d = Xsub[s:s+BCH] @ cent.T
        labs[s:s+BCH] = d.argmax(1)
    # update
    new = torch.zeros(K, D, device=DEVICE)
    cnt = torch.zeros(K, device=DEVICE)
    new.scatter_add_(0, labs.unsqueeze(1).expand(-1,D), Xsub)
    cnt.scatter_add_(0, labs, torch.ones_like(labs.float()))
    cnt = cnt.clamp_min(1)
    new /= cnt.unsqueeze(1)
    mv = (new-cent).norm(dim=1).mean().item()
    cent = new
    if it<3 or it%4==3: log(f"  iter{it}: move={mv:.4f}")
log("  km fit done")
# 全量分配
centn = cent / cent.norm(dim=1,keepdim=True).clamp_min(1e-9)
labs_all = torch.empty(N*NP, dtype=torch.long, device=DEVICE)
Xg = Pn.reshape(-1,D).to(DEVICE)
for s in range(0, N*NP, BCH):
    labs_all[s:s+BCH] = (Xg[s:s+BCH] @ centn.T).argmax(1)
tok = labs_all.cpu().numpy().reshape(N,NP)
log("  token ids assigned")

for rad,hit in FAMS.items():
    idx=[chars.index(c) for c in hit]; m=msk(POS[rad]).numpy(); js=[]
    for a in range(len(idx)):
        for b in range(a+1,len(idx)):
            A,B=set(tok[idx[a]][m].tolist()),set(tok[idx[b]][m].tolist())
            js.append(len(A&B)/max(1,len(A|B)))
    log(f"  {rad}: token Jaccard={np.mean(js):.3f}")

# ---------- P3b': 易混字对 ----------
stage("P3b' fine pairs token diff")
FINE=[("未","末"),("土","士"),("大","太"),("干","千"),("人","入"),("日","曰"),
      ("弋","戈"),("巳","已"),("天","夫"),("刀","力"),("申","由"),("王","玉"),
      ("辛","幸"),("戊","戌"),("牛","午"),("己","已"),("由","甲"),("士","土")]
have={}
for i,c in enumerate(chars): have.setdefault(c,[]).append(i)
cnt=0
for a,b in FINE:
    if a in have and b in have and cnt<12:
        ia,ib = have[a][0],have[b][0]
        diff = tok[ia]!=tok[ib]
        diff_r = diff.reshape(G,G)
        yy,xx = torch.where(torch.from_numpy(diff_r))
        cosb = (Pn[ia]@Pn[ib].T).max(1).values.mean().item()
        log(f"  {a}/{b}: token差异率={diff.mean():.3f}  patch-cos={cosb:.3f}  差异局部化=[{yy.float().mean():.0f},{xx.float().mean():.0f}]")
        cnt+=1
log("  (token差异率低 = 分不清; 越高越好)")

stage("done"); log(f"total {time.time()-_t0:.1f}s")