"""测试 moyun 补完模块能否正常 import + 数据集能否取到数据。"""
import os
import sys

sys.path.insert(0, "/root/Workspace/xy/DiT/ref/moyi")
os.chdir("/root/Workspace/xy/DiT")

print("=== 1. import 测试 ===")
from utils.rope import VisionRotaryEmbeddingFast  # noqa: E402

print("  ✓ utils.rope")
from utils.Sampler.RF import RF  # noqa: E402

print("  ✓ utils.Sampler.RF")
from dataset_moyun import MultiLabelNestedDataset  # noqa: E402

print("  ✓ dataset_moyun")

import inspect  # noqa: E402

print("  RF:", inspect.signature(RF.__init__))

print("\n=== 2. 模型 import + 参数量 ===")
from moyun.moyun_2 import DiT_models  # noqa: E402

print("  可用模型:", list(DiT_models.keys()))
for name in ("moyun-12channel-B", "moyun-12channel", "moyun-4channel"):
    if name not in DiT_models:
        continue
    try:
        m = DiT_models[name](input_size=32, num_classes=1000,
                             learn_sigma=False, if_rope=False)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {n/1e6:.1f}M")
    except Exception as e:
        print(f"  {name}: ✗ {type(e).__name__}: {str(e)[:90]}")

print("\n=== 3. 数据集测试 ===")
ds = MultiLabelNestedDataset(
    csv_file="assets/train_50k_v2_fixed.csv",
    img_shards="data/50k/shards_img",
    edge_shards="data/50k/shards_aux_canny",
    skel_shards="data/50k/shards_std_fixed",
    num_classes=1000,
)
print(f"  len = {len(ds)}")
item = ds[0]
names = ["image", "edge", "skel", "y", "stroke", "f1", "f2", "f3"]
for n, v in zip(names, item):
    if hasattr(v, "shape"):
        print(f"    {n}: shape={tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(v, tuple):
        print(f"    {n}: tuple{tuple(tuple(x.shape) for x in v)}")
    else:
        print(f"    {n}: {v}")
