# -*- coding: utf-8 -*-
"""批量取数的**逐位等价**守卫（`docs/infra_1015.md` §七 ✓）

审查原文:
  "当前 `__getitem__()` 每读取一个样本，都要完成多次 Python 字段处理和 Tensor 构造 …
   对于 B=288，这意味着每个 batch 有数百次 Python 调用，以及上千个小 Tensor 对象的
   构造或处理。" ✓
  "训练时直接按整个 batch 的索引收集 … 这个设计减少 Python per-sample 逻辑，
   也避免重复创建无用的空张量。" ✓
  "**要保持原有 σ 增强分布，必须继续先采样连续 σ，再映射到现有 0.05 间隔网格，
   不能直接均匀采样网格索引后声称完全等价。**" ✓

★★ 本测试守的**两条硬不变量** ✓:
  ① **`__getitem__` 行为不变** ✓ —— 它现在走 `_labels()` ✓ 必须与旧表达式**逐位相同** ✓
  ② **`__getitems__` ≡ 逐个 `__getitem__`** ✓（**同一 RNG 种子**下 ✓）
     ⇒ 含随机的字段（字体随机 / σ / shard ✓）**序列一致** ✓
     ⇒ 否则会破坏 exact-resume ✗（`tests/test_refactor_resume.py` ✓）

★ 夹具用 `object.__new__` 绕过 `__init__` ✓ —— 与仓库既有做法一致 ✓
  （见 `src/train/strong_style.py` 的 `_ensure_luts` 注释: "兼容 `__new__` 构造的实例" ✓）
"""
import random

import numpy as np
import torch

from src.utils.latent_dataset import MCCDLatentDataset
from src.utils.std_soft_glyph import StdSoftGlyphStore


def _mk_dataset(n=6, with_std=False, is_train=True):
    """构造最小夹具 ✓（绕过 `__init__` ✓ 只设取数路径用到的属性 ✓）。"""
    d = object.__new__(MCCDLatentDataset)
    d.samples = [{'calligrapher_id': 100 + (i % 3), 'script_id': 7 + (i % 2),
                  'character_id': 200 + i, 'character': chr(0x4E00 + i),
                  'image_path': f'/nonexistent/{i}.png'}
                 for i in range(n)]
    d.preload = True
    d.load_image = False
    d._latents = np.arange(n * 4 * 2 * 2, dtype=np.float32).reshape(n, 4, 2, 2)
    d._imgs = None
    d._skel_latents = None
    d._skel_latents_list = []
    d._inst_skel_latents = None
    d._aux_latents = None
    d._skel_id_to_shard = None
    d._inst_id_to_shard = None
    d._aux_id_to_shard = []
    d._glookup = None
    d._g_hit = 0
    d._g_miss = 0
    d._oov_warned = 0
    d._char_remap = None
    d._allow_oov_char = True
    d._callig_script_map = None
    d._callig_map = None
    d._callig_remap = {100: 0, 101: 1, 102: 2}
    d._font_remap = None
    d.latent_channels, d.latent_spatial = 4, 2
    d.is_train = is_train
    d.img_root = None
    if with_std:
        st = object.__new__(StdSoftGlyphStore)
        K, N, H = 3, n, 2                    # ★ 行数 = 样本数 ✓（否则索引越界 ✗）
        st._stack = torch.arange(K * N * H * H, dtype=torch.uint8).reshape(K, N, H, H)
        st._levels = {0.4: 0, 0.5: 1, 0.6: 2}
        # ★ 夹具必须覆盖样本用到的**全部** (script_id, char) ✓
        #   （样本的 script_id = 7 + i%2 ⇒ 7 与 8 ✓；漏掉会触发 store 的**严格缺行报错** ✓
        #    这正是 PR2 的"无静默 fallback" ✓ —— 夹具写错时它会**正确**报错 ✓）
        st._key_index = {}
        for i in range(n):
            st._key_index[(7 + (i % 2), chr(0x4E00 + i))] = i
        st._size = H
        st.strict = True
        st.is_train = is_train
        st.sigma_min, st.sigma_max, st.eval_sigma = 0.4, 0.6, 0.5
        st._paths = {k: 'x' for k in st._key_index}
        d._std_soft = st
    else:
        d._std_soft = None
    d._precompute_labels()
    return d


def _fields_equal(a, b):
    assert set(a) == set(b), "字段集合不同 ✗ %s vs %s" % (sorted(a), sorted(b))
    for k in a:
        va, vb = a[k], b[k]
        if torch.is_tensor(va):
            assert torch.is_tensor(vb), "%s: 类型不同 ✗" % k
            assert va.dtype == vb.dtype, "%s: dtype 不同 ✗ %s vs %s" % (k, va.dtype, vb.dtype)
            assert va.shape == vb.shape, "%s: shape 不同 ✗ %s vs %s" % (k, va.shape, vb.shape)
            assert torch.equal(va, vb), "%s: **值不同** ✗" % k
        else:
            assert va == vb, "%s: 值不同 ✗ %r vs %r" % (k, va, vb)


def test_getitem_labels_match_precomputed():
    """① `_labels()` 逐样本路径与**预算数组**必须逐位一致 ✓。"""
    d = _mk_dataset()
    for i in range(len(d.samples)):
        row = d.samples[i]
        pre = {'y_callig': torch.from_numpy(d._arr_y_callig),
               'y_callig_raw': torch.from_numpy(d._arr_y_callig_raw),
               'y_pair': torch.from_numpy(d._arr_y_pair),
               'y_script': torch.from_numpy(d._arr_y_script),
               'y_char': torch.from_numpy(d._arr_y_char)}
        per_sample = d._labels(row, pre=None, idx=None)
        batched = d._labels(row, pre=pre, idx=i)
        for k, (x, y) in enumerate(zip(per_sample, batched)):
            assert torch.equal(x, y), "标签 #%d 不一致 ✗ %s vs %s" % (k, x, y)


def test_getitems_equals_getitem_same_seed():
    """② ★ 批量取数与逐个取数在**同一 RNG 种子**下必须逐位一致 ✓。"""
    idxs = [0, 3, 1, 5, 2]
    d = _mk_dataset()
    random.seed(1234)
    one = [d.__getitem__(i) for i in idxs]
    random.seed(1234)
    many = d.__getitems__(idxs)
    assert len(many) == len(idxs), "长度不符 ✗"
    for k, (a, b) in enumerate(zip(one, many)):
        _fields_equal(a, b)


def test_getitems_equals_getitem_with_std_soft():
    """②′ 含 `std_soft`（σ 连续采样 ✓）时同样必须逐位一致 ✓。"""
    idxs = [0, 1, 2, 3]
    d = _mk_dataset(n=4, with_std=True)
    random.seed(99)
    one = [d.__getitem__(i) for i in idxs]
    random.seed(99)
    many = d.__getitems__(idxs)
    for a, b in zip(one, many):
        _fields_equal(a, b)
    assert one[0]['std_soft'].shape == many[0]['std_soft'].shape


def test_batch_get_equals_per_sample_get():
    """★ `batch_get` 与逐个 `get`（**同样 σ** ✓）必须逐位一致 ✓。"""
    d = _mk_dataset(n=4, with_std=True)
    st = d._std_soft
    sids = [7 + (i % 2) for i in range(4)]      # ★ 与夹具的 key 一致 ✓
    chs = [chr(0x4E00 + i) for i in range(4)]
    sig = [0.41, 0.55, 0.4, 0.6]
    batched = st.batch_get(sids, chs, sig)
    per = torch.stack([st.get(s, c, sigma=g) for s, c, g in zip(sids, chs, sig)], 0)
    assert batched.shape == per.shape, "shape 不同 ✗ %s vs %s" % (batched.shape, per.shape)
    assert batched.dtype == per.dtype, "dtype 不同 ✗"
    assert torch.equal(batched, per), (
        "★ batch_get 与逐个 get **不逐位相等** ✗（infra_1015 §七 要求等价 ✓）\n"
        "  max|Δ|=%.3e" % (batched - per).abs().max())


def test_sigma_sampling_is_continuous_not_uniform_over_grid():
    """★★ 审查**明确警告**的坑 ✓: σ 必须**先连续采样再映射网格** ✗ 不得均匀采网格 ✓。

    判据: 在 `is_train=True` 时，`__getitems__` 消耗的 `random.uniform` 调用数
      必须等于**样本数** ✓（每样本一次连续采样 ✓）
      ⇒ 若改成"均匀采网格索引" ✗ 就不会有这些 `random.uniform` 调用 ✓
    """
    d = _mk_dataset(n=5, with_std=True)
    calls = []
    _orig = random.uniform

    def spy(a, b):
        calls.append((a, b))
        return _orig(a, b)

    random.seed(7)
    random.uniform = spy
    try:
        d.__getitems__([0, 1, 2, 3, 4])
    finally:
        random.uniform = _orig
    # ★ 每样本一次 ✓ 且区间是 store 的 (sigma_min, sigma_max) ✓ 不是网格索引 ✓
    assert len(calls) == 5, (
        "σ 连续采样次数 = %d ✗ 应为样本数 5 ✓\n"
        "⇒ 可能退化成了**均匀采样网格索引** ✗ —— 审查 §七 明确禁止 ✓" % len(calls))
    for lo, hi in calls:
        assert abs(lo - 0.4) < 1e-9 and abs(hi - 0.6) < 1e-9, (
            "σ 采样区间应为 (sigma_min, sigma_max)=(0.4,0.6) ✓ 实际 (%.3f,%.3f) ✗" % (lo, hi))


def test_no_shared_empty_tensor_mutation_risk():
    """★ `_EMPTY` 是模块级共享张量 ✓ 必须确实是空的且不可被误用 ✗。"""
    from src.utils.latent_dataset import _EMPTY
    assert _EMPTY.numel() == 0, "`_EMPTY` 必须为空张量 ✓"
    d = _mk_dataset(n=2)
    o = d.__getitem__(0)
    assert o['canny'] is _EMPTY and o['skeleton'] is _EMPTY, (
        "空字段应复用 `_EMPTY` ✓（审查: 避免重复创建无用的空张量 ✓）")
