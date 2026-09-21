"""utils/stroke.py —— 笔画序列工具的占位实现。

⚠ moyun_2.py:31 `from utils.stroke import get_ch_strokes_tensor, max_ch_strokes, max_stroke_pts`，
   但注册表里**所有变体都是 use_stroke=False**，且 `Moyun.forward` 里
   `if self.use_stroke: stroke = self.stroke_embedder(stroke) else: stroke = None`
   —— 这条路径在 ref 的生产配置里从不执行，模块在 ref 里也缺失。

   这里给能 import 的占位（供 StrokeEmbedder 的默认参数用）。
"""
import torch

max_ch_strokes = 32      # 单字最大笔画数（占位值）
max_stroke_pts = 64      # 单笔最大点数（占位值）


def get_ch_strokes_tensor(char, **kwargs):
    """占位：返回全零的笔画张量。

    ⚠ 真实实现需要按字的笔画拆解（ref 缺失）。若要开 use_stroke 必须补真实实现。
    """
    return torch.zeros(1, max_ch_strokes, max_stroke_pts, 2)
