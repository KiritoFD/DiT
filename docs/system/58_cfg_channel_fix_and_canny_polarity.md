# 58 — CFG 通道数修复 + canny PNG 极性统一

> 接续 [57_model_vs_ref_deep_check_and_data_dirty.md](57_model_vs_ref_deep_check_and_data_dirty.md)
> 日期: 2026-09-14

## 一、CFG 通道数 gap 修复

### 问题

ref `moyun_2.py:648` 的 `forward_with_cfg` 只对前 3ch（RGB eps）做 CFG：

```python
eps, rest = model_out[:, :3], model_out[:, 3:]
```

我们 `dit.py` 的 `forward_with_cfg` / `forward_with_2axis_cfg` 对前 `self.in_channels` 全做 CFG：

```python
eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
# self.in_channels = 12 (image4 + canny4 + skel4)
```

**影响**: canny/skel 结构通道被 CFG 放大，推理采样质量下降（尤其 self-cond 回灌 skel）。
**不影响训练权重**（训练时不调用 `forward_with_cfg`）。

### 修复

#### 1. `src/model/dit.py`

`__init__` 新增参数：

```python
image_channels=None,    # CFG 只作用于 image latent 通道; None 时退回 in_channels (向后兼容)
```

body:

```python
self.image_channels = int(image_channels) if image_channels is not None else in_channels
```

`forward_with_cfg` (原 1088 行) 和 `forward_with_2axis_cfg` (原 1127 行):

```python
# 旧: model_out[:, :self.in_channels]
# 新: model_out[:, :self.image_channels]
eps, rest = model_out[:, :self.image_channels], model_out[:, self.image_channels:]
```

#### 2. 构建器透传

| 文件 | 行 | 改动 |
|------|-----|------|
| `src/train/train.py` | ~286 | `image_channels=getattr(args, 'latent_channels', 4)` |
| `src/eval/batch_eval.py` | ~128 | `image_channels=int(a.get("latent_channels") or 4)` |
| `src/eval/gpu_ablate_eval.py` | ~102 | `image_channels=int(a.get("latent_channels", 4)) |

#### 3. 向后兼容

- `image_channels=None` → fallback 到 `in_channels`，旧行为不变
- pixel-space 模型（`in_channels=3`）不传 → `image_channels=3`，CFG 对 3ch 做
- controlnet / pretrain_g（`in_channels=4`，无 aux）不传 → `image_channels=4`，CFG 对 4ch 做
- v11 aux 模型（`in_channels=12`）显式传 `image_channels=4` → CFG 只对 image latent 4ch 做

### 语义对照

| | ref (moyun_2.py) | 我们 (dit.py) |
|---|---|---|
| image 通道 | 3ch RGB | 4ch VAE latent |
| aux 通道 | 无 | canny4 + skel4 |
| CFG 范围 | `[:3]` | `[:4]` (image_channels) |
| 不受 CFG | — | canny4 + skel4 (结构通道) |

---

## 二、canny PNG 极性统一

### 问题

`tools/gen_base_images.py:60` 直接落盘 `cv2.Canny` 原生输出（**黑底白线**）：

```python
edges = cv2.Canny(a, 80, 180)          # 黑底 (0) 白线 (255)
Image.fromarray(edges, "L").save(...)  # 存黑底
```

而 skel 在 `:58` 存的是**白底黑线**：

```python
Image.fromarray(np.where(sk3, 0, 255).astype(np.uint8), "L").save(...)  # 白底 (255) 黑线 (0)
```

**补救**: `tools/rebuild_latents_wz.py:53` 对 canny 用 `invert=True`，在生成 wz latent 时反转：

```python
("canny", ..., True),   # invert=True: 黑底白线 -> 白底黑线
```

**隐患**: 任何绕过 wz latent、直接读 canny PNG 重编码的脚本会重踩极性不一致（canny 黑底 vs skel 白底）。

### 修复

#### 1. `tools/gen_base_images.py:60`

```python
# 旧: edges = cv2.Canny(a, 80, 180)
# 新: edges = 255 - cv2.Canny(a, 80, 180)
edges = 255 - cv2.Canny(a, 80, 180)    # 白底 (255) 黑线 (0), 与 skel 极性一致
```

#### 2. `tools/rebuild_latents_wz.py:53`

```python
# 旧: ("canny", ..., True),
# 新: ("canny", ..., False),
("canny", "data/aux/final_canny_base", "data/aux/aux_canny_latents_base_wz", False),
```

PNG 已是白底黑线，`invert=False` 不再反转。

### 注意

**此修复只影响新生成的 canny PNG**。已存在的 `data/aux/final_canny_base/*.png` 仍是黑底白线。
若要统一已有 PNG，需重跑 `gen_base_images.py` 或单独写反转脚本。
当前训练已用 `invert=True` 的 wz latent，权重不受影响；修复后新生成的 wz latent 用 `invert=False`，
两者数值等价（都最终得到白底黑线的 latent），无需重训。

---

## 三、修改文件清单

| 文件 | 改动 |
|------|------|
| `src/model/dit.py` | `__init__` 加 `image_channels` 参数; `forward_with_cfg` / `forward_with_2axis_cfg` 改用 `self.image_channels` |
| `src/train/train.py` | 模型构建处传 `image_channels=getattr(args, 'latent_channels', 4)` |
| `src/eval/batch_eval.py` | 模型构建处传 `image_channels=int(a.get("latent_channels") or 4)` |
| `src/eval/gpu_ablate_eval.py` | 模型构建处传 `image_channels=int(a.get("latent_channels", 4))` |
| `tools/gen_base_images.py` | canny 极性改成 `255 - cv2.Canny(...)` (白底黑线) |
| `tools/rebuild_latents_wz.py` | canny `invert`: True → False |