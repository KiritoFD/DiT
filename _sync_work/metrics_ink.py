"""墨迹域指标：墨迹框 SSIM + 墨迹 IoU。

## 为什么需要
全图 SSIM 被大面积白底严重抬高（实测「升」vs「陞」两个字完全不同，
全图 SSIM 仍有 0.6124，而裁到墨迹框后只有 0.3553，二值 IoU 只有 0.2333）。

原因：白底占 ~90% 且两张图完全相同 -> 贡献大量 SSIM；
      黑字只占 5-10% -> 差异被平均掉。

## 三个新指标
- `ink_ssim(pred, gt)`  先裁到**墨迹并集包围盒**再算 SSIM（消掉白底稀释）
- `ink_iou(pred, gt)`   墨迹**二值** IoU（不骨架化；直接反映"字写得对不对"）
- 已有 `skel_iou`       骨架 IoU（对线宽鲁棒，但对风格差异敏感）

输入约定：与 src/eval/metrics.py 一致 —— (H,W,C) 或 (N,H,W,C)，
          值域 [0,1] 的 float（1=白底，0=黑墨）。
"""
import numpy as np

from src.eval.metrics import ssim as _ssim_impl


def _ink_mask(a, thresh=0.5):
    """(H,W,C) float -> (H,W) bool，True=墨迹。"""
    return a.mean(axis=-1) < thresh


def ink_bbox(pred, gt, thresh=0.5, pad=2):
    """取 pred/gt 墨迹的**并集**包围盒（含少量 pad）。

    ⚠ 必须用并集：如果用 GT 的框，pred 跑到框外的部分会被裁掉，
      等于人为放宽了指标。
    """
    m = _ink_mask(pred, thresh) | _ink_mask(gt, thresh)
    if not m.any():
        return 0, pred.shape[0], 0, pred.shape[1]
    ys, xs = np.where(m)
    y0 = max(0, ys.min() - pad)
    y1 = min(pred.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(pred.shape[1], xs.max() + 1 + pad)
    return int(y0), int(y1), int(x0), int(x1)


def ink_ssim(pred, gt, thresh=0.5, win=11, sigma=1.5, pad=2):
    """裁到墨迹并集包围盒后算 SSIM。

    支持 (H,W,C) 和 (N,H,W,C)。返回 float（单张）或 list（多张）。
    """
    single = pred.ndim == 3
    if single:
        pred, gt = pred[None], gt[None]
    out = []
    for k in range(pred.shape[0]):
        y0, y1, x0, x1 = ink_bbox(pred[k], gt[k], thresh, pad)
        p, g = pred[k][y0:y1, x0:x1], gt[k][y0:y1, x0:x1]
        # SSIM 需要足够大的窗口
        if min(p.shape[0], p.shape[1]) < win:
            p = np.asarray(pred[k])
            g = np.asarray(gt[k])
        out.append(float(_ssim_impl(p, g, win=win, window="box")))
    return out[0] if single else out


def ink_iou(pred, gt, thresh=0.5):
    """墨迹二值 IoU（不骨架化）。最直接反映"字写得对不对"。

    注意与 skel_iou 的区别：
      - ink_iou : 比"墨迹区域"，对线宽/粗细敏感
      - skel_iou: 比"骨架线"，对线宽鲁棒但对风格变形敏感
    """
    single = pred.ndim == 3
    if single:
        pred, gt = pred[None], gt[None]
    vals = []
    for k in range(pred.shape[0]):
        b1, b2 = _ink_mask(pred[k], thresh), _ink_mask(gt[k], thresh)
        inter = float((b1 & b2).sum())
        union = float((b1 | b2).sum())
        if union == 0:
            vals.append(1.0)
        else:
            vals.append(inter / union)
    return vals[0] if single else vals


def ink_bbox_ssim_iou(pred, gt, thresh=0.5, win=11, sigma=1.5):
    """一次算三个：返回 (ink_ssim, ink_iou, skel_iou)。"""
    from src.eval.metrics import skel_iou as _skel_iou_impl
    return (ink_ssim(pred, gt, thresh, win, sigma),
            ink_iou(pred, gt, thresh),
            _skel_iou_impl(pred, gt, thresh=thresh))
