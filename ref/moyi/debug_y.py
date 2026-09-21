import os
import sys

import torch

sys.path.insert(0, "/root/Workspace/xy/DiT/ref/moyi")
os.chdir("/root/Workspace/xy/DiT")

from dataset_moyun import MultiLabelNestedDataset

ds = MultiLabelNestedDataset(
    csv_file="assets/train_50k_v2_fixed.csv",
    img_shards="data/50k/shards_img",
    edge_shards="data/50k/shards_aux_canny",
    skel_shards="data/50k/shards_std_fixed", num_classes=9100)

print("  dataset[0][3] shape:", tuple(ds[0][3].shape))
dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)
image, edge, skel, y, stroke, f1, f2, f3 = next(iter(dl))
print("  dataloader y shape:", tuple(y.shape))
print("  dataloader y:", y.flatten().tolist())
y2 = y.squeeze(-1).T.contiguous()
print("  squeeze(-1).T ->", tuple(y2.shape), " (期望 (3,4))")
print("  y2 =", y2.tolist())
