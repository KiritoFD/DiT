# -*- coding: utf-8 -*-
"""v53 三表条件编码器 — 本地代码补丁 (dit.py / train.py / latent_dataset.py / model_io.py / cli.py)

设计 (用户指令 2026-10-04):
  书家(10 类)、书体(3 类)、汉字(4677 类) 各一张独立 1024 维 Embedding 表;
  查表后 concat -> Linear 投影到 hidden;
  分维度 Dropout: calligrapher 0.16 / font 0.08 / character 0.08
  (+ cond_drop_all_prob 5% 作 CFG uncond 底座);
  三张表用跨视图 SupCon (tools/build_triple_tables.py) 预训练作初始化。
事务式 + 锚点校验 + 备份 + 幂等。
"""
import shutil
import time
import py_compile
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def patch(path, old, new, tag, count=1, marker=None):
    src = open(path, encoding="utf-8").read()
    _mk = marker if marker is not None else new.strip().splitlines()[-1].strip()
    if _mk and _mk in src:
        print(f"[{tag}] already patched, skip")
        return
    n = src.count(old)
    assert n == count, f"[{tag}] anchor x{n} (expect {count}): {old[:70]!r}"
    bak = f"{path}.bak_triple_{int(time.time())}"
    shutil.copy2(path, bak)
    open(path, "w", encoding="utf-8", newline="").write(src.replace(old, new))
    py_compile.compile(path, doraise=True)
    print(f"[{tag}] patched (backup -> {bak}) syntax OK")


# 修复: D1 块可能被重复插入 (幂等检查缺陷), 先去重
_d = "src/model/dit.py"
_s = open(_d, encoding="utf-8").read()
_BLK = """        # ── v53 三表条件: 书家(10)/书体(3)/汉字(N) 独立 LabelEmbedder ──
        # (script_embed_dim 形参已存在于签名, S2 hier 遗产, 默认 64, config 传 1024)
        use_script_cond=False,
        num_script_classes=3,
        cond_drop_callig_prob=0.0,
        cond_drop_script_prob=0.0,
        cond_drop_char_prob=0.0,
"""
while _s.count(_BLK) > 1:
    _s = _s.replace(_BLK + _BLK, _BLK)
    print("[dedupe] collapsed duplicated D1 block")
open(_d, "w", encoding="utf-8", newline="").write(_s)
py_compile.compile(_d, doraise=True)
print(f"[dedupe] D1 block count={_s.count(_BLK)}")


# ============ dit.py ============
D = "src/model/dit.py"

patch(D, """        cond_drop_which_glyph_prob=0.5,
""", """        cond_drop_which_glyph_prob=0.5,
        # ── v53 三表条件: 书家(10)/书体(3)/汉字(N) 独立 LabelEmbedder ──
        # (script_embed_dim 形参已存在于签名, S2 hier 遗产, 默认 64, config 传 1024)
        use_script_cond=False,
        num_script_classes=3,
        cond_drop_callig_prob=0.0,
        cond_drop_script_prob=0.0,
        cond_drop_char_prob=0.0,
""", "D1-args")

patch(D, """        self.cond_drop_all_prob = float(cond_drop_all_prob)
""", """        self.cond_drop_all_prob = float(cond_drop_all_prob)
        # v53 三表
        self.use_script_cond = bool(use_script_cond)
        self.num_script_classes = int(num_script_classes)
        self.cond_drop_callig_prob = float(cond_drop_callig_prob)
        self.cond_drop_script_prob = float(cond_drop_script_prob)
        self.cond_drop_char_prob = float(cond_drop_char_prob)
        if self.use_script_cond:
            script_embed_dim = script_embed_dim or hidden_size
""", "D2-flags")

patch(D, """            else:
                self.y_char_embedder = LabelEmbedder(
                    num_characters, char_embed_dim, 0.0, use_cfg_embedding=True)
            if callig_proj_mode == "mlp":""", """            else:
                self.y_char_embedder = LabelEmbedder(
                    num_characters, char_embed_dim, 0.0, use_cfg_embedding=True)
            if self.use_script_cond:
                self.y_script_embedder = LabelEmbedder(
                    num_script_classes, script_embed_dim, 0.0, use_cfg_embedding=True)
            else:
                self.y_script_embedder = None
            if callig_proj_mode == "mlp":""", "D3-script-embedder")

patch(D, """                _cat_dim = callig_embed_dim + (char_embed_dim if self.use_char_cond else 0)""", """                            + (script_embed_dim if self.use_script_cond else 0)
                            + (char_embed_dim if self.use_char_cond else 0))""", "D4-catdim", marker="+ (script_embed_dim if self.use_script_cond else 0)")

patch(D, """                    _dims = [callig_embed_dim]
                    if self.use_char_cond:""", """                    _dims = [callig_embed_dim]
                    if self.use_script_cond:
                        _dims.append(script_embed_dim)
                    if self.use_char_cond:""", "D5-dims", marker="_dims.append(script_embed_dim)")

patch(D, """        if (self.condition_fusion in ("factorized_add", "factorized_cat", "xl_highdim")
                and self.training
                and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0)):
            r = torch.rand(y_callig.shape[0], device=y_callig.device)""", """        if getattr(self, "use_script_cond", False) and self.training:
            # v53 三表: 各因子**独立** dropout, 三者全中 = uncond;
            # cond_drop_all_prob 仍作无条件底座 (保 CFG 有足量 uncond 样本)。
            _B = y_callig.shape[0]
            _dev = y_callig.device
            _all = torch.rand(_B, device=_dev) < self.cond_drop_all_prob
            callig_drop = (torch.rand(_B, device=_dev) < self.cond_drop_callig_prob) | _all
            script_drop = (torch.rand(_B, device=_dev) < self.cond_drop_script_prob) | _all
            char_drop = (torch.rand(_B, device=_dev) < self.cond_drop_char_prob) | _all
        elif (self.condition_fusion in ("factorized_add", "factorized_cat", "xl_highdim")
                and self.training
                and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0)):
            r = torch.rand(y_callig.shape[0], device=y_callig.device)""", "D6-drop", marker="v53 三表: 各因子")

patch(D, """            _parts = [e_callig]
            if self.use_char_cond:""", """            _parts = [e_callig]
            if self.use_script_cond:
                if y_script is None:
                    raise RuntimeError("[triple] use_script_cond=True 但 forward 未收到 y_script")
                if self.training and script_drop is not None:
                    y_script = torch.where(script_drop,
                                           self.y_script_embedder.num_classes, y_script)
                _parts.append(self.y_script_embedder(y_script, False))
            if self.use_char_cond:""", "D7-cat-parts", count=1, marker="use_script_cond=True 但 forward 未收到 y_script")

# ============ train.py ============
T = "src/train/train.py"

patch(T, """            cond_drop_which_glyph_prob=getattr(args, 'cond_drop_which_glyph_prob', 0.5),""", """            cond_drop_which_glyph_prob=getattr(args, 'cond_drop_which_glyph_prob', 0.5),
            # v53 三表条件 (script_embed_dim/num_script_classes 已由 S2 的 cli 提供)
            use_script_cond=getattr(args, 'use_script_cond', False),
            cond_drop_callig_prob=float(getattr(args, 'cond_drop_callig_prob', 0.0)),
            cond_drop_script_prob=float(getattr(args, 'cond_drop_script_prob', 0.0)),
            cond_drop_char_prob=float(getattr(args, 'cond_drop_char_prob', 0.0)),""", "T1-kwargs")

patch(T, """                y_callig = batch['y_callig'].to(device, non_blocking=True)""", """                y_callig = batch['y_callig'].to(device, non_blocking=True)
                if getattr(args, 'use_script_cond', False):
                    # v53 三表: y_callig 用书家连续索引(10 类), 另带 y_script
                    y_callig = batch['y_callig_raw'].to(device, non_blocking=True)
                    y_script = batch['y_script'].to(device, non_blocking=True)""", "T2-batch")

patch(T, """                    model_kwargs = dict(y_callig=y_callig, y_char=y_char)""", """                    model_kwargs = (dict(y_callig=y_callig, y_script=y_script, y_char=y_char)
                                    if getattr(args, 'use_script_cond', False)
                                    else dict(y_callig=y_callig, y_char=y_char))""", "T3-kwargs")

# ============ latent_dataset.py ============
L = "src/utils/latent_dataset.py"

patch(L, """                 num_preload_workers=16, use_glyph_cond=False, skel_latent_shards_dir=None,""", """                 num_preload_workers=16, use_glyph_cond=False, skel_latent_shards_dir=None,
                 char_remap=None,""", "L1-ctor")

patch(L, """            'y_char': torch.tensor(
                int(row.get('glyph_id', row['character_id'])), dtype=torch.long),""", """            'y_char': torch.tensor(
                (self._char_remap[int(row.get('glyph_id', row['character_id']))]
                 if self._char_remap is not None
                 else int(row.get('glyph_id', row['character_id']))), dtype=torch.long),""", "L2-ychar")

# store remap in ctor body: anchor on use_glyph_cond store
patch(L, """        self.use_glyph_cond = bool(use_glyph_cond)""", """        self.use_glyph_cond = bool(use_glyph_cond)
        # v53: character_id(全局) -> 汉字连续索引 的重映射 (build_triple_tables 产出)
        self._char_remap = dict(char_remap) if char_remap else None""", "L3-store")

# ============ latent_dataset.py ============
L = "src/utils/latent_dataset.py"

patch(L, """                 num_preload_workers=16, use_glyph_cond=False, skel_latent_shards_dir=None,""", """                 num_preload_workers=16, use_glyph_cond=False, skel_latent_shards_dir=None,
                 callig_remap=None, font_remap=None,""", "L1-ctor", marker="callig_remap=None, font_remap=None,")

patch(L, """        self.use_glyph_cond = bool(use_glyph_cond)""", """        self.use_glyph_cond = bool(use_glyph_cond)
        # v53: 三表重映射 (build_triple_tables 产出; 保证标签空间与 SupCon 表行一致)
        self._char_remap = dict(char_remap) if char_remap else None
        self._callig_remap = dict(callig_remap) if callig_remap else None
        self._font_remap = dict(font_remap) if font_remap else None""", "L2-store")

patch(L, """            'y_callig_raw': torch.tensor(callig_idx, dtype=torch.long),""", """            'y_callig_raw': torch.tensor(
                (self._callig_remap[callig_idx] if self._callig_remap else callig_idx),
                dtype=torch.long),""", "L3a-callig", marker="self._callig_remap[callig_idx]")

patch(L, """            'y_script': torch.tensor(int(row['script_id']), dtype=torch.long),""", """            'y_script': torch.tensor(
                (self._font_remap[int(row['script_id'])] if self._font_remap
                 else int(row['script_id'])), dtype=torch.long),""", "L3b-font", marker="self._font_remap[int(row['script_id'])]")

patch(L, """            'y_char': torch.tensor(
                int(row.get('glyph_id', row['character_id'])), dtype=torch.long),""", """            'y_char': torch.tensor(
                (self._char_remap[int(row.get('glyph_id', row['character_id']))]
                 if self._char_remap is not None
                 else int(row.get('glyph_id', row['character_id']))), dtype=torch.long),""", "L4-ychar-SKIPPED-S2-already-has-it")

# ============ cli.py ============
C = "src/train/cli.py"

patch(C, """    parser.add_argument("--hier-style", type=int, default=0, dest="hier_style",""", """    # -- v53 三表条件 --
    parser.add_argument("--use-script-cond", type=_str_to_bool, default=False,
                        dest="use_script_cond",
                        help="v53 三表条件编码器: 书家/书体/汉字三个独立 LabelEmbedder, "
                             "concat 后 Linear 投影。配合 --cond-drop-callig/script/char-prob "
                             "分维度 dropout (推荐 0.16/0.08/0.08)。表用 SupCon 预训练初始化。")
    parser.add_argument("--cond-drop-callig-prob", type=float, default=0.0,
                        dest="cond_drop_callig_prob")
    parser.add_argument("--cond-drop-script-prob", type=float, default=0.0,
                        dest="cond_drop_script_prob")
    parser.add_argument("--cond-drop-char-prob", type=float, default=0.0,
                        dest="cond_drop_char_prob")
    parser.add_argument("--char-remap-json", type=str, default="",
                        dest="char_remap_json",
                        help="character_id(全局)->汉字连续索引 重映射 (build_triple_tables 产出)")
    parser.add_argument("--callig-remap-json", type=str, default="",
                        dest="callig_remap_json")
    parser.add_argument("--font-remap-json", type=str, default="",
                        dest="font_remap_json")
    parser.add_argument("--triple-table-prefix", type=str, default="",
                        dest="triple_table_prefix",
                        help="三张 SupCon 表路径前缀, 如 assets/triple_tables/")
    parser.add_argument("--hier-style", type=int, default=0, dest="hier_style",""", "C1-args", marker='"--use-script-cond", type=_str_to_bool')

# ============ model_io.py ============
M = "src/eval/model_io.py"

patch(M, """    use_char_cond=not bool(g("no_char_cond", False)),""", """    use_char_cond=not bool(g("no_char_cond", False)),
    # v53 三表条件 (ckpt args 携带; 缺省 = 旧行为)
    use_script_cond=bool(g("use_script_cond", False)),
    num_script_classes=gi("num_script_classes", 3),
    cond_drop_callig_prob=g("cond_drop_callig_prob", 0.0),
    cond_drop_script_prob=g("cond_drop_script_prob", 0.0),
    cond_drop_char_prob=g("cond_drop_char_prob", 0.0),""", "M1-kwargs")

# ============ train.py: dataset ctor + remap 加载 + 三表 init ============
patch(T, """                                    num_preload_workers=int(getattr(args, 'preload_workers', 16)),""", """                                    num_preload_workers=int(getattr(args, 'preload_workers', 16)),
                                    char_remap=(_char_remaps['char'] if _char_remaps else None),
                                    callig_remap=(_char_remaps['callig'] if _char_remaps else None),
                                    font_remap=(_char_remaps['font'] if _char_remaps else None),""", "T4-dataset")

patch(T, """    # ── DINO glyph-embedding init for y_char_embedder ───────────────────────""", """    # ── v53: 三表重映射 json + SupCon 表初始化 ──────────────────────────────
    _char_remaps = None
    if getattr(args, 'char_remap_json', ''):
        import json as _json
        _char_remaps = {}
        for _nm, _p in (('char', args.char_remap_json),
                        ('callig', getattr(args, 'callig_remap_json', '')),
                        ('font', getattr(args, 'font_remap_json', ''))):
            if _p:
                _char_remaps[_nm] = {int(k): int(v) for k, v in
                                     _json.load(open(_p, encoding="utf-8")).items()}
        logger.info(f"[triple] remap json 已加载: {list(_char_remaps.keys())}")
    _ttp = getattr(args, 'triple_table_prefix', '') or ''
    if _ttp:
        import json as _json
        import numpy as _np
        for _nm, _emb in (('callig', model.y_callig_embedder),
                          ('font', model.y_script_embedder),
                          ('char', model.y_char_embedder)):
            if _emb is None:
                continue
            _w = _np.load(f"{_ttp}{_nm}_table.npy")
            _cls = _json.load(open(f"{_ttp}{_nm}_index.json",
                                   encoding="utf-8"))["classes"]
            _tab = _emb.embedding_table.weight
            assert _w.shape[0] == len(_cls), \\
                f"{_nm}: 表行 {_w.shape[0]} != 类数 {len(_cls)}"
            assert _w.shape[1] == _tab.shape[1], \\
                f"{_nm}: 维度 {_w.shape[1]} != 表 {_tab.shape[1]}"
            with torch.no_grad():
                for _i, _c in enumerate(_cls):
                    _tab[int(_c)].copy_(torch.from_numpy(_w[_i]).float())
        logger.info(f"[triple-init] 三表 SupCon 初始化完成 (prefix={_ttp})")

    # ── DINO glyph-embedding init for y_char_embedder ───────────────────────""", "T5-init")

print("[ALL DONE]")
