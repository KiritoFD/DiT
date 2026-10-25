import json, csv, os, re
from pathlib import Path
import numpy as np
from src.eval.style import bank as B
from src.eval.style.encoder import StyleEncoder
from src.eval.style.metrics import (EmbSet, build_prototypes, cross_char_retrieval, prototype_margin)

train_csv = "exp-std/csv/train_top10_aug_sym.csv"
eval_csv = "exp-std/csv/eval200_fixed.csv"

roots = [".", "data", "exp-std"]
rows = B.build_bank(train_csv, roots=roots)
rows = B.cap_balanced(rows, 3000)
ref_i, qry_i = B.split_index_by_work(rows, query_frac=0.3)
rows = B.tag_split(rows, ref_i, qry_i)

rows_ref = [rows[i] for i in ref_i]
rows_qry = [rows[i] for i in qry_i]

def _ids(rows, key):
    u = sorted({getattr(r, key) for r in rows})
    return {v: i for i, v in enumerate(u)}

def _mid(m, k):
    k = str(k)
    if k not in m: m[k] = len(m)
    return m[k]

allw, alls, allc, allk = _ids(rows, "writer"), _ids(rows, "script"), _ids(rows, "char"), _ids(rows, "work")

enc = StyleEncoder.load("pretrained_models/eval_classifier.pt", kind="callig", device="cuda")
cache_path = "assets/audits/style_eval_four_arms/emb_cache.npz"
emb_all, _ = enc.embed_paths([r.path for r in rows], cache=cache_path, batch=64)

def to_embset(rows, emb, wmap, smap, cmap, kmap, name="set"):
    return EmbSet(emb, np.array([wmap[r.writer] for r in rows]),
                  np.array([smap[r.script] for r in rows]),
                  np.array([cmap[r.char] for r in rows]),
                  np.array([kmap[r.work] for r in rows]),
                  [os.path.basename(r.path) for r in rows])

E_ref = to_embset(rows_ref, emb_all[ref_i], allw, alls, allc, allk)
E_realq = to_embset(rows_qry, emb_all[qry_i], allw, alls, allc, allk)

# Real-Real symmetric (exclude_same_char=True, exclude_same_work=True)
ret_real = cross_char_retrieval(E_realq, E_ref, topk=(1, 5), exclude_same_char=True, exclude_same_work=True)

# Other-Writer floor
wl = sorted(set(allw.values()))
wrong = np.array([wl[(wl.index(int(w)) + 1) % len(wl)] for w in E_realq.writer])
E_wrong = EmbSet(E_realq.emb, wrong, E_realq.script, E_realq.char, E_realq.work, E_realq.name)
ret_other = cross_char_retrieval(E_wrong, E_ref, topk=(1, 5), exclude_same_char=True, exclude_same_work=True)

eval_meta = list(csv.DictReader(open(eval_csv, encoding="utf-8-sig")))[:187]

models = {
    "Baseline_55k": ("/home/ds/Workspace/DiT/assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/eval_samples_ctrl/step0055000/eval200fix",
                     "assets/audits/style_eval_four_arms/emb_gen_Baseline_55k.npz"),
    "Arm1_residual_gate_75k": ("/home/ds/Workspace/DiT/assets/results/unet_dic/B_residual_gate/20261018-064328-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
                               "assets/audits/style_eval_four_arms/emb_gen_Arm1_residual_gate_75k.npz"),
    "Arm2_stage_adapter_75k": ("/home/ds/Workspace/DiT/assets/results/unet_dic/C_stage_cond_adapter/20261018-092206-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
                               "assets/audits/style_eval_four_arms/emb_gen_Arm2_stage_adapter_75k.npz"),
    "Arm3_stage_indep_75k": ("/home/ds/Workspace/DiT/assets/results/unet_dic/C_stage_cond_independent/20261018-115725-Unet-flow/eval_samples_ctrl/step0075000/eval200fix",
                             "assets/audits/style_eval_four_arms/emb_gen_Arm3_stage_indep_75k.npz"),
    "Node4090_DiC_U100_50k": ("/home/ds/Workspace/DiT/assets/results/eval_4090_b768_50k/eval_samples_ctrl/step0050000/eval200fix",
                              "assets/audits/style_eval_four_arms/emb_gen_Node4090_DiC_U100_50k.npz")
}

print("=" * 105)
print("{:32s} | {:20s} | {:10s} | {:10s} | {:8s}".format("Model", "Top-1 (Work-Excluded)", "Top-5", "mAP", "n_used"))
print("=" * 105)
print("{:32s} | {:19.2f}% | {:9.2f}% | {:10.4f} | {:6d}".format("Real-Real (Upper Bound)", ret_real['top1']*100, ret_real['topk'][5]*100, ret_real['map'], ret_real['n_used']))
print("{:32s} | {:19.2f}% | {:9.2f}% | {:10.4f} | {:6d}".format("Real-Other-Writer (Lower Bound)", ret_other['top1']*100, ret_other['topk'][5]*100, ret_other['map'], ret_other['n_used']))

for name, (g_dir, emb_p) in models.items():
    g_paths = [os.path.join(g_dir, f"g{i}.png") for i in range(187)]
    g_emb, _ = enc.embed_paths(g_paths, cache=emb_p, batch=64)
    # Pass REAL work_id to generated images!
    gm = []
    for r in eval_meta:
        w_real = B.work_id(r.get("src_image_path", ""), r.get("image_path", ""))
        gm.append(dict(writer=r.get("calligrapher").strip(),
                       script=r.get("script").strip(),
                       char=r.get("character").strip(),
                       work=w_real, path=r.get("image_path")))
    E_gen = EmbSet(g_emb, np.array([_mid(allw, g["writer"]) for g in gm]),
                   np.array([_mid(alls, g["script"]) for g in gm]),
                   np.array([_mid(allc, g["char"]) for g in gm]),
                   np.array([_mid(allk, g["work"]) for g in gm]),
                   [g["path"] for g in gm])
    ret_g = cross_char_retrieval(E_gen, E_ref, topk=(1, 5), exclude_same_char=True, exclude_same_work=True)
    print("{:32s} | {:19.2f}% | {:9.2f}% | {:10.4f} | {:6d}".format(name, ret_g['top1']*100, ret_g['topk'][5]*100, ret_g['map'], ret_g['n_used']))
print("=" * 105)
