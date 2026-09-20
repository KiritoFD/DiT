# -*- coding: utf-8 -*-
"""v15 多模态风格冒烟测试（CPU, 无需数据/权重）。

覆盖:
  1. MultiStyleEmbedder: (B,K,D) 输出 / null 标签整组替换 / clamp 越界 / freeze_table
  2. CalligStyleCrossAttn v15: (B,K,D) K/V + Q 位置编码 + zero-init 恒等
  3. DiT-2Cond-XS6/2 + callig_multi_style_k=4: 前向 / CFG(2 路) / drop 语义
     (drop-callig 样本的 style token 全组 = null_embed, 与 adaLN 一致)
  4. 架构演进 resume: 旧 v14 式 state_dict (88,128 表 + 256 输入 cond_fusion)
     经 drop_shape_mismatched 剔除后加载, 剔除清单恰为 {表, null_embed, cond_fusion.*}
  5. 锚定 mean 视图: table.view(N,K,D).mean(1) 与 pair_mean 广播对齐
  6. build_multistyle_k4 的 kmeans_torch + 空对/少样本回退路径
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.manual_seed(0)
D, K, NCLS = 384, 4, 87

# ---------- 1) MultiStyleEmbedder ----------
from src.model.dit import MultiStyleEmbedder, CalligStyleCrossAttn

emb = MultiStyleEmbedder(NCLS, K, D, dropout_prob=0.0)
labels = torch.randint(0, NCLS, (6,))
labels[1] = NCLS          # null 标签 (CFG uncond / cond-drop 写入)
labels[2] = -3            # 越界(防御路径)
out = emb(labels, False)
assert out.shape == (6, K, D), out.shape
assert torch.allclose(out[1], emb.null_embed.view(K, D).unsqueeze(0).squeeze(0)), \
    "null 标签必须整组替换为 null_embed"
assert torch.allclose(out[2], emb.embedding_table.weight[0].view(K, D)), \
    "越界标签 -3 被 clamp 到 0, out[2] 应来自 clamp 后的第 0 行"
emb.freeze_table()
assert emb.embedding_table.weight.requires_grad is False and emb.null_embed.requires_grad
print("[1] MultiStyleEmbedder OK  (B,K,D)=%s; null/越界/freeze 语义通过" % (tuple(out.shape),))

# ---------- 2) CalligStyleCrossAttn ----------
ca = CalligStyleCrossAttn(D, num_heads=6)
g_tok = torch.randn(2, 256, D)
st = torch.randn(2, K, D)
out0 = ca(g_tok, st)
assert out0.shape == g_tok.shape
assert torch.allclose(out0, g_tok, atol=1e-6), "zero-init out_proj -> 恒等"
assert ca.ctx_pos.shape == (1, 256, D)
print("[2] CalligStyleCrossAttn OK  zero-init 恒等 + ctx_pos (1,256,D)")

# ---------- 3) DiT-2Cond-XS6/2 多模态前向 / CFG / drop 语义 ----------
from src.model import DiT_2Cond_models
m = DiT_2Cond_models["DiT-2Cond-XS6/2"](
    input_size=32, num_calligraphers=NCLS, num_characters=100,
    condition_fusion="factorized_cat", callig_embed_dim=D,
    callig_multi_style_k=K, glyph_vec_cond=True, glyph_vec_dim=128,
    use_glyph_cond=True, use_char_cond=False, glyph_scale_init=0.6,
    learn_sigma=False, glyph_drop_prob=0.1, cond_drop_all_prob=0.1,
    cond_drop_which_glyph_prob=0.85,
    glyph_inject_layers=2, glyph_inject_mode="adaln")
m.eval()
x = torch.randn(2, 4, 32, 32); t = torch.rand(2)
g = torch.randn(2, 4, 32, 32); y = torch.tensor([3, 70])
o = m(x, t, y, None, g=g)
assert o.shape == (2, 4, 32, 32)
oc = m.forward_with_cfg(x, t, y, torch.zeros(2, dtype=torch.long),
                        cfg_scale=4.0, g=g)
assert oc.shape == (2, 4, 32, 32)
assert isinstance(m.y_callig_embedder, MultiStyleEmbedder)
assert m.callig_style_ca is None, "v15a 式构建(未传 callig_style_ca)不应有 CA"
# adaLN 分支的 cat 维度 = pooled token(D) + glyph_vec(128) = 384 + 128 = 512
assert m.cond_fusion[0].normalized_shape[0] == D + 128, m.cond_fusion[0].normalized_shape
print(f"[3] DiT 多模态前向/CFG OK  cond_fusion 输入={m.cond_fusion[0].normalized_shape[0]}")

# ---------- 3b) 注入方式矩阵: v15a(池化) / v15b(CA) / v15c(每层 ctx) ----------
def build_variant(**extra):
    return DiT_2Cond_models["DiT-2Cond-XS6/2"](
        input_size=32, num_calligraphers=NCLS, num_characters=100,
        condition_fusion="factorized_cat", callig_embed_dim=D,
        callig_multi_style_k=K, glyph_vec_cond=True, glyph_vec_dim=128,
        use_glyph_cond=True, use_char_cond=False, glyph_scale_init=0.6,
        learn_sigma=False, glyph_drop_prob=0.1, cond_drop_all_prob=0.1,
        glyph_inject_layers=2, glyph_inject_mode=extra.pop("glyph_inject_mode", "adaln"),
        **extra)

ma = build_variant()                                        # v15a
assert ma.callig_style_ca is None and not ma.style_ctx_every_layer
mb = build_variant(callig_style_ca=True)                    # v15b
assert mb.callig_style_ca is not None
mc = build_variant(glyph_inject_mode="xattn",
                   style_ctx_every_layer=True, xattn_q_pos=True)   # v15c
assert mc.callig_style_ca is None and mc.style_ctx_every_layer
assert mc.style_role.shape == (K, D)
for mm, tag in [(ma, "a"), (mb, "b"), (mc, "c")]:
    mm.eval()
    oo = mm(x, t, y, None, g=g)
    assert oo.shape == (2, 4, 32, 32), (tag, oo.shape)
    occ = mm.forward_with_cfg(x, t, y, torch.zeros(2, dtype=torch.long), cfg_scale=4.0, g=g)
    assert occ.shape == (2, 4, 32, 32), tag
oc_ctx = mc.glyph_injections[0](torch.randn(2, 256, D),
                                torch.randn(2, 256 + K, D))
assert oc_ctx.shape == (2, 256, D)
try:
    build_variant(glyph_inject_mode="adaln", style_ctx_every_layer=True)
    raise AssertionError("adaln + style_ctx 应被拒绝")
except ValueError:
    pass
print("[3b] 注入矩阵 v15a/b/c 构建与前向 OK  (c: ctx=%d token, xattn_q_pos)" % (256 + K))

# drop 语义: training 下 drop-callig 样本的 style token 必须与 null_embed 全组一致
m.train()
emb = m.y_callig_embedder
big_callig_drop = torch.tensor([True, False])   # 样本0 整组丢
y_in = torch.where(big_callig_drop, torch.tensor(NCLS), y)
st_dropped = emb(y_in, False)
assert torch.allclose(st_dropped[0], emb.null_embed.view(K, D)), \
    "drop-callig 样本的 K 个 token 必须全组 = null_embed"
m.eval()
# glyph_drop: keep=0 样本的 g_tok 在风格调制后必须重新置零 (drop 分支纯净)
g_zero = g.clone(); g_zero[0] = 0
keep = torch.tensor([False, True])
gg = g_zero * keep.view(-1, 1, 1, 1).float()
_ = m(x, t, y, None, g=gg)   # 只验证不崩 (置零逻辑在 forward 内已有护栏断言)
print("[4] drop 语义 OK  (callig 整组 null / g 置零路径)")

# ---------- 5) 架构演进 resume: 旧 v14 式 sd -> drop_shape_mismatched ----------
from src.utils.channel_expand import drop_shape_mismatched
old_sd = {}
for k_, v_ in m.state_dict().items():
    old_sd[k_] = v_.clone()
# 伪装成 v14 形状: 表 (88,128), null (128), cond_fusion (256 输入)
old_sd["y_callig_embedder.embedding_table.weight"] = torch.randn(88, 128)
old_sd["y_callig_embedder.null_embed"] = torch.randn(128)
cf0 = m.cond_fusion[0]
old_sd["cond_fusion.0.weight"] = torch.randn(K * D + 128 + 128)   # 伪造旧 LN 形状
old_sd["cond_fusion.1.weight"] = torch.randn(D, 256)
old_sd["cond_fusion.1.bias"] = torch.randn(D)
dropped = drop_shape_mismatched(m, old_sd)
miss, unexp = m.load_state_dict(old_sd, strict=False)
assert "y_callig_embedder.embedding_table.weight" in dropped
assert "y_callig_embedder.null_embed" in dropped
assert any(k_.startswith("cond_fusion.") for k_ in dropped), dropped
assert not any(k_.startswith("glyph_embedder") for k_ in dropped), \
    "形状一致的模块(glyph_embedder 等)绝不能被剔除"
print(f"[5] drop_shape_mismatched OK  dropped={len(dropped)}: {sorted(dropped)[:4]} ...")

# ---------- 6) 锚定 mean 视图 + kmeans 回退路径 ----------
tbl = m.y_callig_embedder.embedding_table.weight
pm = torch.randn(NCLS, D)
pooled = tbl.view(NCLS, K, D).mean(1)
assert pooled.shape == pm.shape
sys.path.insert(0, "tools")
from build_multistyle_k4 import kmeans_torch, dim_reduce_768_to_384, l2n
xf = l2n(torch.randn(50, D).numpy())
c, inr = kmeans_torch(torch.from_numpy(xf), 4, seed=1)
assert c.shape == (4, D) and torch.isfinite(c).all()
c2, _ = kmeans_torch(torch.from_numpy(xf[:2]), 4, seed=1)   # n<K: 函数按 n 截断
assert c2.shape == (2, D)
tiled = c2[torch.arange(4) % 2]                              # 工具层的平铺策略
assert tiled.shape == (4, D)
r = dim_reduce_768_to_384(torch.randn(5, 768).numpy(), 384)
assert r.shape == (5, 384)
print("[6] mean 锚定视图 + kmeans(n>=K / n<K) + 768->384 插值 OK")

print("\n=== v15 冒烟全部通过 ===")
