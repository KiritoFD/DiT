"""在**远端原地**打 A/B 补丁 (attn_tau + w_style_ortho), 事务式。

为什么不在本地改完上传: 本地工作区比远端旧 (实测 train.py 远端已有
glyph_latent_channels/flux16 改动, eval/model_io.py 差 710 行) —— 整文件覆盖会
回滚掉远端更新的工作。所以只做**精确字符串替换**。

事务性: 先校验全部锚点都在, 有一个缺失就**整体放弃**(绝不半套), 全部通过才写盘。
幂等: 已打过补丁的文件自动跳过。
"""
import io
import os
import shutil
import sys
import time

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

# ---------------------------------------------------------------- 补丁定义
# (文件, 说明, 旧串, 新串)
PATCHES = []

# ===== src/model/dit.py =====
DIT = "src/model/dit.py"
PATCHES += [
    (DIT, "ZeroCrossAttention.__init__ 加 attn_tau",
     "    def __init__(self, d_model, num_heads, grid_size=16, q_pos=False):\n"
     "        super().__init__()\n"
     "        assert d_model % num_heads == 0\n"
     "        self.num_heads = num_heads\n"
     "        self.head_dim = d_model // num_heads\n"
     "        grid_size = int(round(grid_size))",
     "    def __init__(self, d_model, num_heads, grid_size=16, q_pos=False, attn_tau=1.0):\n"
     "        super().__init__()\n"
     "        assert d_model % num_heads == 0\n"
     "        self.num_heads = num_heads\n"
     "        self.head_dim = d_model // num_heads\n"
     "        # \u2605 2026-10-03: \u6ce8\u610f\u529b\u6e29\u5ea6 \u03c4\u3002"
     "softmax(qk\u1d40/\u221ad/\u03c4) \u2261 softmax((q/\u03c4)k\u1d40/\u221ad),\n"
     "        #   \u6240\u4ee5\u53ea\u9700\u628a q \u9664\u4ee5 \u03c4, **\u96f6\u989d\u5916\u5f00\u9500**\u3002"
     "\u03c4<1 = \u9510\u5316 / hard routing\u3002\n"
     "        #   \u52a8\u673a(\u63a2\u9488\u5b9e\u6d4b tools/probe_style_routing.py): "
     "\u98ce\u683c token \u7684\u72ec\u5360\u71b5 \u03c4=1.0 \u65f6 0.974(\u5747\u5300\u5854\u7f29),\n"
     "        #   \u03c4=0.1 \u65f6\u964d\u5230 0.397(\u771f\u8def\u7531)\u3002\u6839\u56e0\u4e0d\u662f token \u5171\u7ebf, "
     "\u800c\u662f\u521d\u59cb\u5316\u65f6 q/k \u6295\u5f71\u968f\u673a -> logits \u65e0\u5dee\u5f02\u3002\n"
     "        #   \u03c4=1.0 = \u65e7\u884c\u4e3a\u9010\u4f4d\u7b49\u4ef7\u3002\n"
     "        self.attn_tau = float(attn_tau)\n"
     "        grid_size = int(round(grid_size))"),
    (DIT, "ZeroCrossAttention.forward 应用 tau",
     "        v = self.v_proj(self.norm_c(ctx_in)) \\\n"
     "            .view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)\n"
     "        out = F.scaled_dot_product_attention(q, k, v)",
     "        v = self.v_proj(self.norm_c(ctx_in)) \\\n"
     "            .view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)\n"
     "        if self.attn_tau != 1.0:\n"
     "            q = q / self.attn_tau      # = softmax(qk\u1d40/\u221ad/\u03c4)\n"
     "        out = F.scaled_dot_product_attention(q, k, v)"),
    (DIT, "GlyphStyleCrossAttn.__init__ 加 attn_tau",
     "    def __init__(self, d_model, num_heads):\n"
     "        super().__init__()\n"
     "        assert d_model % num_heads == 0\n"
     "        self.num_heads = num_heads\n"
     "        self.head_dim = d_model // num_heads\n"
     "        self.norm_x = nn.LayerNorm(d_model)",
     "    def __init__(self, d_model, num_heads, attn_tau=1.0):\n"
     "        super().__init__()\n"
     "        assert d_model % num_heads == 0\n"
     "        self.num_heads = num_heads\n"
     "        self.head_dim = d_model // num_heads\n"
     "        self.attn_tau = float(attn_tau)   # \u03c4<1 = \u9510\u5316, \u89c1 ZeroCrossAttention\n"
     "        self.norm_x = nn.LayerNorm(d_model)"),
    (DIT, "GlyphStyleCrossAttn.forward 应用 tau",
     "        v = self.v_proj(ctx).view(\n"
     "            B, Nc, self.num_heads, self.head_dim).transpose(1, 2)\n"
     "        out = F.scaled_dot_product_attention(q, k, v)",
     "        v = self.v_proj(ctx).view(\n"
     "            B, Nc, self.num_heads, self.head_dim).transpose(1, 2)\n"
     "        if self.attn_tau != 1.0:\n"
     "            q = q / self.attn_tau\n"
     "        out = F.scaled_dot_product_attention(q, k, v)"),
    (DIT, "CalligStyleCrossAttn.__init__ 加 attn_tau",
     "    def __init__(self, hidden_size, num_heads, grid_size=16):\n"
     "        super().__init__()\n"
     "        assert hidden_size % num_heads == 0\n"
     "        self.num_heads = num_heads\n"
     "        self.head_dim = hidden_size // num_heads\n"
     "        self.norm_q = nn.LayerNorm(hidden_size)",
     "    def __init__(self, hidden_size, num_heads, grid_size=16, attn_tau=1.0):\n"
     "        super().__init__()\n"
     "        assert hidden_size % num_heads == 0\n"
     "        self.num_heads = num_heads\n"
     "        self.head_dim = hidden_size // num_heads\n"
     "        self.attn_tau = float(attn_tau)   # \u03c4<1 = \u9510\u5316, \u89c1 ZeroCrossAttention\n"
     "        self.norm_q = nn.LayerNorm(hidden_size)"),
    (DIT, "CalligStyleCrossAttn.forward 应用 tau",
     "        v = self.v_proj(self.norm_kv(style_tokens)).view(\n"
     "            B, Nk, self.num_heads, self.head_dim).transpose(1, 2)\n"
     "        out = F.scaled_dot_product_attention(q, k, v)",
     "        v = self.v_proj(self.norm_kv(style_tokens)).view(\n"
     "            B, Nk, self.num_heads, self.head_dim).transpose(1, 2)\n"
     "        if self.attn_tau != 1.0:\n"
     "            q = q / self.attn_tau\n"
     "        out = F.scaled_dot_product_attention(q, k, v)"),
    (DIT, "DiT.__init__ 加 attn_tau 超参",
     "        xattn_q_pos=False,\n",
     "        xattn_q_pos=False,\n"
     "        # \u2605 2026-10-03: \u6ce8\u5165\u6a21\u5757\u7684\u6ce8\u610f\u529b\u6e29\u5ea6 \u03c4 "
     "(\u53ea\u4f5c\u7528\u4e8e 3 \u4e2a\u6ce8\u5165 cross-attn, **\u4e0d\u6539** DiT \u4e3b\u5e72\u81ea\u6ce8\u610f\u529b)\u3002\n"
     "        attn_tau=1.0,\n"),
    (DIT, "注入模块实例化传 attn_tau (GlyphStyle)",
     "                            GlyphStyleCrossAttn(hidden_size, num_heads=num_heads)",
     "                            GlyphStyleCrossAttn(hidden_size, num_heads=num_heads,\n"
     "                                                attn_tau=attn_tau)"),
    (DIT, "注入模块实例化传 attn_tau (ZeroCrossAttention)",
     "                            ZeroCrossAttention(hidden_size, num_heads=num_heads,\n"
     "                                               grid_size=self.x_embedder.num_patches ** 0.5,\n"
     "                                               q_pos=bool(xattn_q_pos))",
     "                            ZeroCrossAttention(hidden_size, num_heads=num_heads,\n"
     "                                               grid_size=self.x_embedder.num_patches ** 0.5,\n"
     "                                               q_pos=bool(xattn_q_pos),\n"
     "                                               attn_tau=attn_tau)"),
    (DIT, "CalligStyleCrossAttn 实例化传 attn_tau",
     "                self.callig_style_ca = CalligStyleCrossAttn(\n"
     "                    hidden_size, num_heads=num_heads,\n"
     "                    grid_size=int(self.x_embedder.num_patches ** 0.5))",
     "                self.callig_style_ca = CalligStyleCrossAttn(\n"
     "                    hidden_size, num_heads=num_heads,\n"
     "                    grid_size=int(self.x_embedder.num_patches ** 0.5),\n"
     "                    attn_tau=attn_tau)"),
]

# ===== src/train/train.py =====
TR = "src/train/train.py"
PATCHES += [
    (TR, "训练端把 attn_tau 传给模型",
     "            xattn_q_pos=getattr(args, 'xattn_q_pos', False),\n",
     "            xattn_q_pos=getattr(args, 'xattn_q_pos', False),\n"
     "            # \u2605 2026-10-03: \u540c xattn_q_pos \u2014\u2014 \u7eaf\u884c\u4e3a\u5f00\u5173, "
     "\u4e0d\u6539\u53d8\u53c2\u6570\u5f62\u72b6\u3002\u6f0f\u4f20 -> strict=True \u62a0\u4e0d\u5230\u3002\n"
     "            attn_tau=float(getattr(args, 'attn_tau', 1.0) or 1.0),\n"),
    (TR, "读取 w_style_ortho 并扩展开关条件",
     "    _style_anchor_mode = str(getattr(args, 'style_anchor_mode', 'row') or 'row')\n"
     "    _style_table_param = None\n"
     "    _style_anchor_target = None\n"
     "    _style_anchor_k = 1\n"
     "    if _style_anchor_w > 0:",
     "    _style_anchor_mode = str(getattr(args, 'style_anchor_mode', 'row') or 'row')\n"
     "    # \u2605 2026-10-03: K \u4e2a\u98ce\u683c token \u7684\u6b63\u4ea4\u6b63\u5219 "
     "(v15b/c \u7684\"\u4f2a\u591a\u6a21\u6001\"\u5bf9\u7b56)\u3002\n"
     "    #   \u5b9e\u6d4b K-Means \u521d\u59cb\u5316\u7684 4 \u4e2a token \u4e24\u4e24\u4f59\u5f26 0.900 = \u5df2\u5854\u7f29;\n"
     "    #   \u7528 PCA \u6b63\u4ea4\u57fa\u521d\u59cb\u5316(cos\u22480)\u540e, \u672c loss \u9632\u6b62\u8bad\u7ec3\u4e2d\u91cd\u65b0\u9760\u62e2\u3002\n"
     "    _style_ortho_w = float(getattr(args, 'w_style_ortho', 0.0) or 0.0)\n"
     "    _style_table_param = None\n"
     "    _style_anchor_target = None\n"
     "    _style_anchor_k = 1\n"
     "    if _style_anchor_w > 0 or _style_ortho_w > 0:"),
    (TR, "加 K4 正交性启动自检",
     "        elif rank == 0:\n"
     "            logger.warning(\"[style-anchor] style_anchor_weight>0 \u4f46\u65e0 "
     "callig_emb_pretrained, \u951a\u5b9a\u4e0d\u751f\u6548\")",
     "        elif rank == 0:\n"
     "            logger.warning(\"[style-anchor] style_anchor_weight>0 \u4f46\u65e0 "
     "callig_emb_pretrained, \u951a\u5b9a\u4e0d\u751f\u6548\")\n"
     "    # \u2605 2026-10-03 \u542f\u52a8\u81ea\u68c0: K \u4e2a\u98ce\u683c token \u7684\u521d\u59cb\u4e92\u4f59\u5f26\u5e73\u65b9\u3002\n"
     "    #   \u671f\u671b \u22480 (PCA \u6b63\u4ea4\u57fa); \u82e5 \u22480.8 (=cos 0.900) \u8bf4\u660e\u62ff\u7684\u8fd8\u662f\u5171\u7ebf\u7684 K-Means \u8d44\u4ea7\u3002\n"
     "    if _style_ortho_w > 0 and _style_table_param is not None and _style_anchor_k > 1 \\\n"
     "            and rank == 0:\n"
     "        with torch.no_grad():\n"
     "            _t0 = _style_table_param.detach().view(\n"
     "                _style_table_param.shape[0], _style_anchor_k, -1)\n"
     "            _n0 = torch.nn.functional.normalize(_t0, dim=-1)\n"
     "            _s0 = _n0 @ _n0.transpose(1, 2)\n"
     "            _of0 = ~torch.eye(_style_anchor_k, dtype=torch.bool, device=_s0.device)\n"
     "            _c0 = float((_s0[:, _of0] ** 2).mean())\n"
     "        logger.info(f\"[style-ortho] \u03bb={_style_ortho_w} K={_style_anchor_k} \"\n"
     "                    f\"\u8868={tuple(_style_table_param.shape)} "
     "\u521d\u59cb mean(cos\u00b2)={_c0:.6f} \u2192 mean cos={_c0 ** 0.5:.4f} \"\n"
     "                    + (\"\u2713 \u6b63\u4ea4\u57fa (\u5df2\u6253\u7834\u5bf9\u79f0)\" if _c0 < 0.05\n"
     "                       else \"\u2717\u2717 \u4ecd\u5171\u7ebf! \u68c0\u67e5 callig_emb_pretrained\"))"),
    (TR, "加正交正则 loss 项",
     "                    loss = loss + _style_anchor_w * "
     "((_Ecur - _style_anchor_target) ** 2).mean()",
     "                    loss = loss + _style_anchor_w * "
     "((_Ecur - _style_anchor_target) ** 2).mean()\n"
     "\n"
     "                # \u2605 2026-10-03 \u6b63\u4ea4\u6b63\u5219 (\u65b9\u6848\u4e09): "
     "\u60e9\u7f5a\u975e\u5bf9\u89d2\u4f59\u5f26\u5e73\u65b9 \u03bb\u00b7mean_{i\u2260j} cos\u00b2(t_i,t_j)\u3002\n"
     "                #   \u76ee\u7684: \u8ba9\"\u591a\u6a21\u6001\"\u5728\u8bad\u7ec3\u5168\u7a0b\u4fdd\u6301\u771f\u6b63\u4ea4, "
     "\u800c\u4e0d\u662f\u88ab MSE \u6324\u56de\u5171\u7ebf\u3002\n"
     "                #   \u4ec5\u5bf9\u771f\u5b9e\u7c7b\u522b\u884c\u751f\u6548 \u2014\u2014 "
     "MultiStyleEmbedder \u7684 CFG null \u662f\u72ec\u7acb Parameter, \u5929\u7136\u6392\u9664\u3002\n"
     "                if _style_ortho_w > 0 and _style_table_param is not None \\\n"
     "                        and _style_anchor_k > 1:\n"
     "                    _T = _style_table_param.view(_style_table_param.shape[0],\n"
     "                                                 _style_anchor_k, -1)\n"
     "                    _Tn = torch.nn.functional.normalize(_T, dim=-1)\n"
     "                    _S = _Tn @ _Tn.transpose(1, 2)\n"
     "                    _off = ~torch.eye(_style_anchor_k, dtype=torch.bool,\n"
     "                                      device=_S.device)\n"
     "                    loss = loss + _style_ortho_w * (_S[:, _off] ** 2).mean()"),
]

# ===== src/eval/model_io.py (评测端! 漏传 = 静默失效) =====
MIO = "src/eval/model_io.py"
PATCHES += [
    (MIO, "评测端把 attn_tau 传给模型",
     '        xattn_q_pos=bool(g("xattn_q_pos", False)),\n',
     '        xattn_q_pos=bool(g("xattn_q_pos", False)),\n'
     '        # \u2605 2026-10-03: attn_tau \u4e0e xattn_q_pos \u540c\u7c7b \u2014\u2014 '
     '\u7eaf\u884c\u4e3a\u5f00\u5173, \u4e0d\u6539\u53d8\u53c2\u6570\u5f62\u72b6\u3002\n'
     '        #   \u6f0f\u4f20 -> strict=True \u62a0\u4e0d\u5230 -> \u8bc4\u6d4b\u9759\u9ed8\u7528 \u03c4=1.0 '
     '\u53bb\u8bc4\u4e00\u4e2a \u03c4<1 \u8bad\u51fa\u6765\u7684\u6a21\u578b\u3002\n'
     '        attn_tau=float(g("attn_tau", 1.0)),\n'),
]

# ===== src/train/cli.py =====
CLI = "src/train/cli.py"
PATCHES += [
    (CLI, "加 --attn-tau / --w-style-ortho",
     '    parser.add_argument("--num-characters", type=int, default=7765)',
     '    parser.add_argument("--attn-tau", type=float, default=1.0, dest="attn_tau",\n'
     '                        help="\u2605 \u6ce8\u5165\u7528 cross-attn \u7684\u6e29\u5ea6 \u03c4 '
     '(\u4e0d\u6539 DiT \u4e3b\u5e72)\u3002\u03c4<1 = \u9510\u5316/hard routing\u3002'
     '1.0 = \u65e7\u884c\u4e3a\u9010\u4f4d\u7b49\u4ef7\u3002")\n'
     '    parser.add_argument("--w-style-ortho", type=float, default=0.0, dest="w_style_ortho",\n'
     '                        help="\u2605 K \u4e2a\u98ce\u683c token \u7684\u6b63\u4ea4\u6b63\u5219 '
     '\u03bb: \u03bb\u00b7mean_{i\u2260j} cos\u00b2(t_i,t_j)\u30020=\u5173\u95ed\u3002'
     '\u5efa\u8bae 0.01~0.1\u3002")\n'
     '    parser.add_argument("--num-characters", type=int, default=7765)'),
]


def main():
    # ---- 阶段 1: 全量校验 (事务性: 任一锚点缺失就整体放弃) ----
    print("=" * 78)
    print("阶段 1: 校验全部锚点")
    print("=" * 78)
    files = {}
    for path in sorted({p[0] for p in PATCHES}):
        if not os.path.exists(path):
            print(f"[FATAL] 文件不存在: {path}")
            return 2
        files[path] = io.open(path, encoding="utf-8").read()

    todo, skip, missing = [], [], []
    for path, desc, old, new in PATCHES:
        src = files[path]
        if old not in src:
            # 幂等: 新串已在 => 已打过
            if new in src:
                skip.append((path, desc))
            else:
                missing.append((path, desc, old.strip().splitlines()[0][:80]))
        else:
            todo.append((path, desc, old, new))

    print(f"  待打: {len(todo)}  已打过(跳过): {len(skip)}  锚点缺失: {len(missing)}")
    for path, desc in skip:
        print(f"    [skip] {path} :: {desc}")
    if missing:
        print("\n[FATAL] 以下锚点缺失 —— 远端文件与我预期不符, **不做任何修改**:")
        for path, desc, first in missing:
            print(f"    {path} :: {desc}\n        锚点首行: {first}")
        return 3
    if not todo:
        print("\n无待打补丁 (已全部就绪)。")
        return 0

    # ---- 阶段 2: 备份 ----
    ts = time.strftime("%Y%m%d-%H%M%S")
    bdir = f"_sync_work/_AB_patch/backup_{ts}"
    os.makedirs(bdir, exist_ok=True)
    for path in {p[0] for p in todo}:
        dst = os.path.join(bdir, path.replace("/", "__"))
        shutil.copy2(path, dst)
        print(f"[backup] {path} -> {dst}")

    # ---- 阶段 3: 应用 ----
    print("=" * 78)
    print("阶段 3: 应用")
    print("=" * 78)
    for path, desc, old, new in todo:
        src = files[path]
        n = src.count(old)
        files[path] = src.replace(old, new, 1)
        print(f"  ✓ {path} :: {desc}  (匹配 {n} 处, 替换第 1 处)")

    for path, content in files.items():
        io.open(path, "w", encoding="utf-8", newline="").write(content)
        print(f"[write] {path}")

    # ---- 阶段 4: 语法自检 ----
    print("=" * 78)
    print("阶段 4: 语法自检")
    import py_compile
    ok = True
    for path in files:
        try:
            py_compile.compile(path, doraise=True, cfile="/tmp/_chk.pyc")
            print(f"  ✓ {path}")
        except py_compile.PyCompileError as e:
            ok = False
            print(f"  ✗ {path}\n{e}")
    print("\n结论:", "✓ 补丁全部安装且语法通过" if ok else "✗ 语法错误, 请回滚")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
