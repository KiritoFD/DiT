"""修 gradio 切换 ckpt 后的"未知书家"。

两个 bug:
  1) _load_ckpt 没读 train_csv，把 raw_id（数字）当书家名 -> 与原下拉对不上
  2) 切换后书家下拉(启动时创建)不刷新 -> 选旧书家 -> 新表查不到 -> "未知书家"

修法:
  - 抽出 _build_callig_index(ckpt_args, csv_path): 统一建 name_of / raw2idx / names
    兼容两种表:
      v13: callig_id_map        {raw_callig_id -> idx}        名="怀素"
      v15: callig_script_map    {"raw:script" -> pair_idx}    名="怀素|草"
  - 启动时和切换时都用它
  - 切换时把新的书家列表返回给 gradio，刷新 Dropdown 的 choices
"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

# ── 1) 抽出统一构建函数（插在 _load_ckpt 之前）─────────────────────────
HELPER = '''
def _build_callig_index(ck_args, csv_path, map_path):
    """返回 (name_of, raw2idx, names, n_callig)。

    兼容两种书家表:
      v13: callig_id_map      -> key = 书家 raw_id,        显示名 = 书家名
      v15: callig_script_map  -> key = "书家raw:script",   显示名 = 书家名|书体
    """
    import csv as _csv
    cmap, n_callig = load_callig_id_map(map_path)
    is_pair = any((":" in str(k)) for k in list(cmap.keys())[:5])
    name_of = {}
    raw2idx = {}
    if os.path.isfile(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                raw = str(r.get("calligrapher_id", ""))
                script = str(r.get("script_id", ""))
                if not raw:
                    continue
                if is_pair:
                    key = f"{raw}:{script}"
                    if key not in cmap:
                        continue
                    label = f"{r.get('calligrapher', raw)}|{r.get('script', script)}"
                else:
                    key = raw
                    if key not in cmap:
                        continue
                    label = str(r.get("calligrapher", raw))
                raw2idx[key] = cmap[key]
                name_of[label] = key
    else:
        # 没有 csv 时退化到 raw id（保证至少能跑）
        for k, v in cmap.items():
            raw2idx[k] = v
            name_of[str(k)] = k
    return name_of, raw2idx, sorted(name_of.keys()), n_callig


'''
anchor = "def _load_ckpt(path):"
if anchor in s:
    s = s.replace(anchor, HELPER + anchor, 1)
    print("  ✓ 插入 _build_callig_index")
else:
    print("  ⚠ 没找到 _load_ckpt")

# ── 2) _load_ckpt 里改用统一函数 ───────────────────────────────────────
OLD = '''    _mp = a.get("callig_id_map") or a.get("callig_script_map") or args.callig_map
    try:
        cmap, _n = load_callig_id_map(_mp)
    except Exception as e:
        return f"✗ 书家表加载失败 {_mp}: {e}"
    raw2idx = {r: i for r, i in cmap.items()}
    name_of = {str(r): r for r in cmap}
    CALLIG_NAMES = sorted(name_of.keys())'''
NEW = '''    _mp = a.get("callig_id_map") or a.get("callig_script_map") or args.callig_map
    try:
        name_of, raw2idx, CALLIG_NAMES, _n = _build_callig_index(
            a, args.train_csv, _mp)
    except Exception as e:
        return f"✗ 书家表加载失败 {_mp}: {e}"
    if not CALLIG_NAMES:
        return f"✗ 书家表解析出 0 个书家（{_mp} + {args.train_csv} 对不上？）"'''
if OLD in s:
    s = s.replace(OLD, NEW, 1)
    print("  ✓ _load_ckpt 改用统一函数")
else:
    print("  ⚠ 没找到旧书家表代码")

# ── 3) 切换时把新书家列表返回，刷新下拉 ────────────────────────────────
OLD_SW = '''        def _on_switch(name):
            path = CKPT_MAP.get(name)
            if not path:
                return _cur_title(), f"✗ 未找到 {name}"
            try:
                m = _load_ckpt(path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return _cur_title(), f"✗ 加载失败: {e}"
            return _cur_title(), m

        ckpt_btn.click(_on_switch, [ckpt_dd], [md, ckpt_msg])'''
NEW_SW = '''        def _on_switch(name):
            path = CKPT_MAP.get(name)
            if not path:
                return _cur_title(), f"✗ 未找到 {name}", gr.update()
            try:
                m = _load_ckpt(path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return _cur_title(), f"✗ 加载失败: {e}", gr.update()
            # ★ 关键: 同时刷新书家下拉，否则下拉还是旧表的书家 -> "未知书家"
            new_names = CALLIG_NAMES
            return (_cur_title(), m,
                    gr.update(choices=new_names,
                              value=(new_names[0] if new_names else None)))

        ckpt_btn.click(_on_switch, [ckpt_dd], [md, ckpt_msg, callig_in])'''
if OLD_SW in s:
    s = s.replace(OLD_SW, NEW_SW, 1)
    print("  ✓ 切换时刷新书家下拉")
else:
    print("  ⚠ 没找到 _on_switch")

io.open(P, "w", encoding="utf-8").write(s)
import ast

ast.parse(s)
print("  SYNTAX OK")
