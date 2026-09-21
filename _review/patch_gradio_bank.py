"""修 gradio 的 bank 路径：默认是错的，导致 bank 加载为空 -> 任何字都报"不在库存"。

实测（2026-09-21）:
  默认 --bank = _sync_work/data/skel/skel_bank_std1_v8.npz   **不存在**
  实际文件   = _sync_work/skel_bank_std1_v8.npz              存在，7589 条
                                                            楷3293/行3181/隶1115, 4039 字
  '阜' 三种书体都在里面 -> 不是"不支持"，是路径写错

改动:
  1) 默认路径改成正确的那个
  2) 加回退查找: 默认路径不存在时，在若干候选位置找同名文件
  3) 加载后校验: 空 bank 时明确报错而不是静默变空
"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

OLD_DEFAULT = '_sync_work/data/skel/skel_bank_std1_v8.npz'
NEW_DEFAULT = '_sync_work/skel_bank_std1_v8.npz'
if OLD_DEFAULT in s:
    s = s.replace(OLD_DEFAULT, NEW_DEFAULT)
    print(f"  默认路径: {OLD_DEFAULT} -> {NEW_DEFAULT}")
else:
    print("  ⚠ 没找到旧默认路径（可能已改）")

OLD_LOAD = '''bank = {}
if os.path.isfile(args.bank):
    z = np.load(args.bank)
    keys = [str(k) for k in z["keys"]]
    lat = z["latents"]
    bank = {k: lat[i] for i, k in enumerate(keys)}
    print(f"[bank] {len(bank)} entries <- {args.bank}", flush=True)
else:
    print(f"[WARN] bank 缺失 {args.bank}; 将无法生成 (需先 build)", flush=True)'''

NEW_LOAD = '''# ★ [2026-09-21] 路径回退: 默认路径历史上写错过一次(多了一层 data/skel/)，
#   结果 bank 静默变空 -> 任何字都报"不在库存"。这里找不到就在候选位置里找。
def _resolve_bank(p):
    if os.path.isfile(p):
        return p
    ROOT = os.path.dirname(os.path.abspath(__file__))
    base = os.path.basename(p)
    for cand in (
        os.path.join(ROOT, "_sync_work", base),
        os.path.join(ROOT, "_sync_work", "data", "skel", base),
        os.path.join(ROOT, "data", "skel", base),
        os.path.join(ROOT, "assets", base),
    ):
        if os.path.isfile(cand):
            return cand
    return p


bank = {}
_bp = _resolve_bank(args.bank)
if os.path.isfile(_bp):
    z = np.load(_bp)
    keys = [str(k) for k in z["keys"]]
    lat = z["latents"]
    bank = {k: lat[i] for i, k in enumerate(keys)}
    print(f"[bank] {len(bank)} entries <- {_bp}", flush=True)
    if not bank:
        print(f"[bank] ✗ bank 为空! 请检查 npz 内容", flush=True)
else:
    print(f"[bank] ✗✗ 找不到 bank: {args.bank} (也试过回退位置 {_bp})",
          flush=True)
    print(f"[bank]    后果: **所有字**都会报"不在库存" —— 这不是模型不支持，"
          f"是 g 条件库没加载", flush=True)'''

if OLD_LOAD in s:
    s = s.replace(OLD_LOAD, NEW_LOAD)
    print("  已加回退查找 + 空 bank 报错")
else:
    print("  ⚠ 没找到原加载代码，检查是否已改")

io.open(P, "w", encoding="utf-8").write(s)
print("  ✓ 已修")
