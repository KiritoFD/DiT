import os, sys, json
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(16)
sys.path.insert(0, "/root/Workspace/xy/DiT/scratch/v90_20l_gpu_20261017_01")
from src.model.dit import DiT_2Cond_models

def build_and_load(model_name, ckpt_path, cond_inject_at):
    m = DiT_2Cond_models[model_name](
        learn_sigma=False,
        callig_embed_dim=32,
        char_embed_dim=256,
        script_embed_dim=16,
        num_calligraphers=10,
        num_characters=4690,
        num_script_classes=3,
        condition_fusion="factorized_cat",
        use_script_cond=True,
        cond_inject_at=cond_inject_at,
        cond_inject_scale=True
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("ema", sd.get("model", sd))
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    m.load_state_dict(cleaned, strict=False)
    m.eval()
    return m

print("Loading models onto CPU...")
m12 = build_and_load("DiT-2Cond-Sp/2", "/root/Workspace/xy/DiT/assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/checkpoints/0200000.pt", "2,4,5,6")
m20 = build_and_load("DiT-2Cond-S384-20/2", "/root/Workspace/xy/DiT/scratch/v90_20l_gpu_20261017_01/assets/results/v90A_20L_stdstrong/20261017-150613-v90A_20L_stdstrong/checkpoints/0200000.pt", "3,6,9,12,15")

# -------------------------------------------------------------
# 1. Load REAL dataset latents and metadata
# -------------------------------------------------------------
print("Loading real dataset latents and CSV...")
csv_path = "/root/Workspace/xy/DiT/exp-std/csv/train_top10_aug_sym.csv"
df = pd.read_csv(csv_path)

# Character remap from char_index.json
ci = json.load(open("/root/Workspace/xy/DiT/assets/triple_tables_best_minimal/char_index.json"))
classes = ci.get("classes", [])
char_remap = {c: i for i, c in enumerate(classes)}

shard = np.load("/dev/shm/dit_ramdisk/exp-std/data/shards_img_aug/shard_00000.npz")
latents_all = shard["latents"]
img_ids_all = shard["img_ids"]

df_indexed = df.set_index("img_id")
valid_indices = []
for idx, iid in enumerate(img_ids_all):
    if iid in df_indexed.index:
        r = df_indexed.loc[iid]
        ch = r["character"].iloc[0] if isinstance(r["character"], pd.Series) else r["character"]
        if ch in char_remap:
            valid_indices.append((idx, iid))
    if len(valid_indices) >= 64:
        break

print(f"Matched {len(valid_indices)} real samples from shard_00000")
N = len(valid_indices)
x_1 = torch.tensor(np.stack([latents_all[v[0]] for v in valid_indices]), dtype=torch.float32)
rows = [df_indexed.loc[v[1]] for v in valid_indices]

w_true_raw = torch.tensor([r["calligrapher_id"].iloc[0] if isinstance(r["calligrapher_id"], pd.Series) else r["calligrapher_id"] for r in rows], dtype=torch.long)
y_char = torch.tensor([char_remap[r["character"].iloc[0] if isinstance(r["character"], pd.Series) else r["character"]] for r in rows], dtype=torch.long)
y_script = torch.tensor([r["script_id"].iloc[0] if isinstance(r["script_id"], pd.Series) else r["script_id"] for r in rows], dtype=torch.long)

unique_calligs = sorted(df["calligrapher_id"].unique())
c_map = {cid: i for i, cid in enumerate(unique_calligs[:10])}
w_true = torch.tensor([c_map.get(cid.item(), 0) for cid in w_true_raw], dtype=torch.long)
w_wrong = (w_true + 3) % 10

torch.manual_seed(42)
x_0 = torch.randn_like(x_1)
u_target = x_1 - x_0

# -------------------------------------------------------------
# 2. P0: 2x2 Writer Condition Swap (Backbone vs Head)
# -------------------------------------------------------------
print("\n" + "="*70)
print("P0: 主干残差流 vs 最终输出头 2x2 条件交换 (Backbone vs Head Writer Swap)")
print("="*70)

def run_2x2_swap(m, x_t, t_tensor, w_A, w_B, y_c, y_s):
    captured = {}
    def hook_fn(module, args, kwargs, output):
        captured["h"] = args[0]
        captured["c"] = args[1] if len(args) > 1 else kwargs.get("c")
        captured["kwargs"] = kwargs
        
    handle = m.final_layer.register_forward_hook(hook_fn, with_kwargs=True)
    
    # Run condition A
    _ = m(x_t, t_tensor, w_A, y_c, y_script=y_s)
    h_A = captured["h"]
    c_A = captured["c"]
    kw_A = captured["kwargs"]
    
    # Run condition B
    _ = m(x_t, t_tensor, w_B, y_c, y_script=y_s)
    h_B = captured["h"]
    c_B = captured["c"]
    kw_B = captured["kwargs"]
    
    handle.remove()
    
    # 2x2 evaluation
    v_AA = m.unpatchify(m.final_layer(h_A, c_A, **kw_A))
    v_AB = m.unpatchify(m.final_layer(h_A, c_B, **kw_B))
    v_BA = m.unpatchify(m.final_layer(h_B, c_A, **kw_A))
    v_BB = m.unpatchify(m.final_layer(h_B, c_B, **kw_B))
    
    norm_base = v_AA.flatten(1).norm(dim=1) + 1e-6
    
    dv_backbone = v_BA - v_AA
    dv_head = v_AB - v_AA
    dv_total = v_BB - v_AA
    dv_cross = v_BB - (v_AA + dv_backbone + dv_head)
    
    mag_backbone = (dv_backbone.flatten(1).norm(dim=1) / norm_base).mean().item()
    mag_head = (dv_head.flatten(1).norm(dim=1) / norm_base).mean().item()
    mag_total = (dv_total.flatten(1).norm(dim=1) / norm_base).mean().item()
    mag_cross = (dv_cross.flatten(1).norm(dim=1) / norm_base).mean().item()
    
    dot = (dv_backbone.flatten(1) * dv_head.flatten(1)).sum(dim=1)
    norm_bb = dv_backbone.flatten(1).norm(dim=1) + 1e-8
    norm_hd = dv_head.flatten(1).norm(dim=1) + 1e-8
    cosine = (dot / (norm_bb * norm_hd)).mean().item()
    
    return mag_backbone, mag_head, mag_total, mag_cross, cosine

t_test = [0.25, 0.40, 0.60, 0.80]
for name, m in [("V84A (12L)", m12), ("v90A (20L)", m20)]:
    print(f"\n--- {name} ---")
    for t_val in t_test:
        t_tensor = torch.full((N,), t_val, dtype=torch.float32)
        x_t = (1.0 - t_val) * x_0 + t_val * x_1
        mb, mh, mt, mc, cos = run_2x2_swap(m, x_t, t_tensor, w_true, w_wrong, y_char, y_script)
        print(f"  t={t_val:.2f} | Backbone 效应: {mb:.4f} | Head 效应: {mh:.4f} | 总效应: {mt:.4f} | 协同余弦: {cos:+.4f} | 非线性交叉: {mc:.4f}")

# -------------------------------------------------------------
# 3. P0: 真实样本上的 Writer 收益 G_w(t) 统计
# -------------------------------------------------------------
print("\n" + "="*70)
print("P0: 真实数据集样本上的正确书家预测收益 (Delta L_w 与 G_w on Real Latents)")
print("="*70)

t_steps = [0.10, 0.25, 0.40, 0.60, 0.80, 0.95]
for name, m in [("V84A (12L)", m12), ("v90A (20L)", m20)]:
    print(f"\n--- {name} ---")
    for t_val in t_steps:
        t_tensor = torch.full((N,), t_val, dtype=torch.float32)
        x_t = (1.0 - t_val) * x_0 + t_val * x_1
        with torch.no_grad():
            v_true = m(x_t, t_tensor, w_true, y_char, y_script=y_script)
            v_wrong = m(x_t, t_tensor, w_wrong, y_char, y_script=y_script)
            
            l_t = ((v_true - u_target) ** 2).flatten(1).mean(dim=1)
            l_w = ((v_wrong - u_target) ** 2).flatten(1).mean(dim=1)
            
            delta_l = (l_w - l_t).mean().item()
            delta_l_std = (l_w - l_t).std().item() / np.sqrt(N)
            g_w = ((l_w - l_t) / (l_w + 1e-6)).mean().item()
            g_w_std = ((l_w - l_t) / (l_w + 1e-6)).std().item() / np.sqrt(N)
        print(f"  t={t_val:.2f} | Delta L_w = {delta_l:+.4f} (±{delta_l_std:.4f}) | G_w = {g_w*100:+.2f}% (±{g_w_std*100:.2f}%)")

# -------------------------------------------------------------
# 4. P1: Frag 按真迹破损程度（完整字 vs 残破石花拓片）分层核验
# -------------------------------------------------------------
print("\n" + "="*70)
print("P1: Frag 指标分层核验：完整样本 vs 严重破损拓片")
print("="*70)

from PIL import Image
from scipy.ndimage import label
gt_dir = "/root/Workspace/xy/DiT/exp-std/runs_purestd/eval_samples_ctrl/step0020000/eval200fix"
struct = np.ones((3, 3), dtype=int)
thresh = 0.5
min_area = 5

gt_blocks = {}
for i in range(187):
    gtf = os.path.join(gt_dir, f"gt{i}.png")
    if os.path.exists(gtf):
        g_arr = np.array(Image.open(gtf).convert("RGB")) / 255.0
        bg = g_arr.mean(axis=2) < thresh
        lg, ng = label(bg, structure=struct)
        counts_g = np.bincount(lg.ravel())
        ng_clean = sum(1 for c in counts_g[1:] if c >= min_area)
        gt_blocks[i] = ng_clean

clean_indices = [i for i, ng in gt_blocks.items() if ng <= 8]
damaged_indices = [i for i, ng in gt_blocks.items() if ng > 8]
print(f"187 测试字集拆分: 完整/少破损组: {len(clean_indices)} 字 | 严重石花拓片组: {len(damaged_indices)} 字")

models_eval = [
    ("V84A (12L @ 200k)", "/root/Workspace/xy/DiT/assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix"),
    ("v90A (20L @ 160k)", "/root/Workspace/xy/DiT/scratch/v90_20l_gpu_20261017_01/assets/results/v90A_20L_stdstrong/20261017-050900-v90A_20L_stdstrong/eval_samples_ctrl/step0160000/eval200fix"),
    ("v90A (20L @ 200k)", "/root/Workspace/xy/DiT/scratch/v90_20l_gpu_20261017_01/assets/results/v90A_20L_stdstrong/20261017-150613-v90A_20L_stdstrong/eval_samples_ctrl/step0200000/eval200fix")
]

for name, d in models_eval:
    def eval_subset(indices):
        ratios = []
        gen_blks = []
        gt_blks = []
        for i in indices:
            gf = os.path.join(d, f"g{i}.png")
            gtf = os.path.join(gt_dir, f"gt{i}.png")
            if not os.path.exists(gf) or not os.path.exists(gtf):
                continue
            p_arr = np.array(Image.open(gf).convert("RGB")) / 255.0
            g_arr = np.array(Image.open(gtf).convert("RGB")) / 255.0
            lp, _ = label(p_arr.mean(axis=2) < thresh, structure=struct)
            lg, _ = label(g_arr.mean(axis=2) < thresh, structure=struct)
            np_c = sum(1 for c in np.bincount(lp.ravel())[1:] if c >= min_area)
            ng_c = sum(1 for c in np.bincount(lg.ravel())[1:] if c >= min_area)
            gen_blks.append(np_c)
            gt_blks.append(ng_c)
            ratios.append(np_c / ng_c if ng_c > 0 else 1.0)
        return np.mean(ratios), np.mean(gen_blks), np.mean(gt_blks)

    frag_clean, gen_c, gt_c = eval_subset(clean_indices)
    frag_dmg, gen_d, gt_d = eval_subset(damaged_indices)
    print(f"\n--- {name} ---")
    print(f"  [完整组 (N_GT<=8, {len(clean_indices)}字)]: Frag = {frag_clean:.4f} | 生成均值={gen_c:.1f} vs GT均值={gt_c:.1f}")
    print(f"  [残破组 (N_GT>8,  {len(damaged_indices)}字)]: Frag = {frag_dmg:.4f} | 生成均值={gen_d:.1f} vs GT均值={gt_d:.1f}")
