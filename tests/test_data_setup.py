"""`src/train/data_setup.build_data` 的行为测试（PR4 抽取的保护网 ✓✓）

★ 为什么需要它: `tools/smoke_train.py` **不 import train.py**（它自己建模型 ✓）⇒
  抽取 `data_setup` 时**没有任何测试覆盖它的函数体** ✗ —— round 32 我就因此把一个
  ImportError 混进了提交（`MCCDDataset` 来源写错 ✗，见 tests/test_entry_imports.py）。
⇒ 本测试用**打桩**（不需要真实 shards/CSV ✓）断言 build_data 的**三条分支**与**全部关键入参** ✓:
   ① use_latent 真/假 -> MCCDLatentDataset / MCCDDataset（且 std_soft 相关 kwargs 只在需要时传 ✓）
   ② sampler 三选一: factor_balanced / LongEpoch(epoch_steps>0) / Distributed
   ③ ★ epoch_steps 与 ckpt_every 不一致 -> **SystemExit**（"单一刻度"护栏 ✓）
   ④ DataLoader 的关键 kwargs（drop_last/pin_memory/persistent_workers ✓）
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.train.data_setup as DS


class _Log:
    """假 logger **对象** ✓ —— build_data 内部用 logger.info(...) ✗ 传函数会 AttributeError ✓。"""

    def __init__(self):
        self.lines = []

    def info(self, *a, **k):
        self.lines.append(("info", a))

    def warning(self, *a, **k):
        self.lines.append(("warning", a))


_LOG = _Log()


class _Rec:
    """记录调用参数的假类 ✓。"""

    def __init__(self, name, calls, ret=None):
        self._name, self._calls, self._ret = name, calls, ret

    def __call__(self, *a, **kw):
        self._calls.append((self._name, a, kw))
        # ★ 返回**自身** ✓ —— 调用方会用 sampler.summary() / len() / inner_epoch_for_step() ✓
        #   （早先返回 SimpleNamespace 会 AttributeError ✗，是我测试的错 ✓）
        return self

    def summary(self):
        return "fake-summary"

    def __len__(self):
        return 7

    def inner_epoch_for_step(self, s):
        return 3

    def set_start_inner_epoch(self, k):
        self._calls.append(("set_start_inner_epoch", (k,), {}))


@pytest.fixture()
def patched(monkeypatch):
    calls = []

    class _DS:
        def __init__(self, *a, **kw):
            calls.append(("MCCDLatentDataset", a, kw))
            self._len = 100

        def __len__(self):
            return self._len

    class _DS2:
        def __init__(self, *a, **kw):
            calls.append(("MCCDDataset", a, kw))
            self._len = 50

        def __len__(self):
            return self._len

    class _Loader:
        def __init__(self, *a, **kw):
            calls.append(("DataLoader", a, kw))
            self.dataset = a[0] if a else None

        def __len__(self):
            return 9

    monkeypatch.setattr(DS, "MCCDLatentDataset", _DS)
    monkeypatch.setattr(DS, "MCCDDataset", _DS2)
    monkeypatch.setattr(DS, "DataLoader", _Loader)
    monkeypatch.setattr(DS, "DistributedFactorBalancedSampler",
                        _Rec("DistributedFactorBalancedSampler", calls))
    monkeypatch.setattr(DS, "LongEpochDistributedSampler",
                        _Rec("LongEpochDistributedSampler", calls))
    monkeypatch.setattr(DS, "DistributedSampler", _Rec("DistributedSampler", calls))
    return calls


def _args(**kw):
    base = dict(data_csv="c.csv", data_dir="d", image_size=256, latent_shards_dir="",
                preload=False, preload_workers=2, img_root="", w_glyph_cond=False,
                std_soft_cond=False, std_soft_root="", std_soft_sigma_min=0.4,
                std_soft_sigma_max=1.0, std_soft_eval_sigma=0.65,
                skel_as_glyph_cond=False, skel_latent_shards_dir="",
                skel_latent_shards_dirs="", skel_latent_shards_weights="",
                inst_skel_shards_dir="", w_latent_skel=0.0, w_std_mid=0.0, w_deform_skel=0.0,
                w_repa=0.0, repa_cache_dir="", repa_need_image="auto",
                global_batch_size=8, num_workers=2, global_seed=1, sampler="random",
                balance_char_alpha=1.0, balance_callig_alpha=1.0, ckpt_every=0,
                max_steps=0, epochs=1, vae_downscale=8)
    base.update(kw)
    return SimpleNamespace(**base)


def _call(calls, name):
    return [c for c in calls if c[0] == name]


def test_non_latent_branch_uses_mccddataset(patched):
    ds, sp, ld, es, ev = DS.build_data(_args(), rank=0, world_size=1, resume_start_step=0,
                                       std_soft_cond=False, char_remaps=None,
                                       aux_dirs=[], logger=_LOG)
    got = _call(patched, "MCCDDataset")
    assert len(got) == 1, "非 latent 分支必须用 MCCDDataset ✓"
    assert got[0][2]["csv_file"] == "c.csv" and got[0][2]["root_dir"] == "d"
    assert _call(patched, "MCCDLatentDataset") == [], "不该用 latent 数据集 ✗"
    assert es == 0 and ev is True, "epoch_steps/EVAL_INLINE 必须返回 ✓"


def test_latent_branch_kwargs_and_std_soft_gating(patched):
    ds, sp, ld, es, ev = DS.build_data(
        _args(latent_shards_dir="sh", std_soft_cond=True, std_soft_root="sroot",
              skel_as_glyph_cond=False),
        rank=0, world_size=1, resume_start_step=0, std_soft_cond=True,
        char_remaps={"char": {"a": 1}, "callig": {}, "font": {}}, aux_dirs=["aux1"],
        logger=_LOG)
    kw = _call(patched, "MCCDLatentDataset")[0][2]
    assert kw["latent_shards_dir"] == "sh"
    assert kw["is_train"] is True, "std_soft_cond=True 时 is_train 必须为 True ✓"
    assert kw["std_soft_root"] == "sroot", "std_soft_cond 下必须传 std_soft_root ✓"
    assert kw["char_remap"] == {"a": 1}, "char_remaps 必须传入 ✓"
    # skel_as_glyph_cond=False ⇒ 骨架目录**不应**传 ✓（这是原文的条件 ✓）
    assert kw["skel_latent_shards_dir"] is None
    assert kw["skel_latent_shards_dirs"] is None


def test_sampler_three_ways(patched):
    # ① factor_balanced
    DS.build_data(_args(latent_shards_dir="sh", sampler="factor_balanced"), rank=0, world_size=1,
                  resume_start_step=0, std_soft_cond=False, char_remaps=None, aux_dirs=[],
                  logger=_LOG)
    assert _call(patched, "DistributedFactorBalancedSampler"), "factor_balanced 分支 ✗"
    patched.clear()
    # ② epoch_steps>0 -> LongEpoch
    DS.build_data(_args(latent_shards_dir="sh", ckpt_every=5), rank=0, world_size=1,
                  resume_start_step=0, std_soft_cond=False, char_remaps=None, aux_dirs=[],
                  logger=_LOG)
    le = _call(patched, "LongEpochDistributedSampler")
    assert le and le[0][2]["steps_per_epoch"] == 5, "LongEpoch 分支 ✗"
    patched.clear()
    # ③ 否则 DistributedSampler
    DS.build_data(_args(latent_shards_dir="sh"), rank=0, world_size=1, resume_start_step=0,
                  std_soft_cond=False, char_remaps=None, aux_dirs=[], logger=_LOG)
    assert _call(patched, "DistributedSampler"), "默认 DistributedSampler 分支 ✗"


def test_epoch_steps_alias_mismatch_raises_systemexit(patched):
    """★ "单一刻度"护栏: ckpt_every 与 epoch_steps 不一致必须**拒绝启动** ✓。"""
    with pytest.raises(SystemExit):
        DS.build_data(_args(latent_shards_dir="sh", ckpt_every=10, epoch_steps=7),
                      rank=0, world_size=1, resume_start_step=0, std_soft_cond=False,
                      char_remaps=None, aux_dirs=[], logger=_LOG)


def test_dataloader_kwargs(patched):
    DS.build_data(_args(latent_shards_dir="sh", global_batch_size=8, num_workers=3),
                  rank=0, world_size=1, resume_start_step=0, std_soft_cond=False,
                  char_remaps=None, aux_dirs=[], logger=_LOG)
    kw = _call(patched, "DataLoader")[0][2]
    assert kw["batch_size"] == 8 and kw["drop_last"] is True and kw["pin_memory"] is True
    assert kw["num_workers"] == 3 and kw["persistent_workers"] is True
    assert kw["prefetch_factor"] == 4
