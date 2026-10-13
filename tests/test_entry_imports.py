"""守卫测试：训练/评测**入口模块必须可导入** ✓✓

★ 由来（真实事故，2026-10-13）: 我在 PR4 把数据装配抽成 `src/train/data_setup.py` 时，
  写错了 `MCCDDataset` 的来源（它由 `src.utils` **包**再导出，不在 `latent_dataset` 里 ✗）
  ⇒ `import src.train.train` 直接 `ImportError` ⇒ **训练脚本无法启动** ✗✗。
  而当时的 `tools/smoke_train.py` **不导入 train.py**（它自己建模型 ✓）
  ⇒ 冒烟**全绿**却漏掉了这个致命回归 ✗。
⇒ 本测试就是那次事故的**永久防线**: 只做一件事 —— 把入口模块 import 一遍 ✓。
  代价毫秒级 ✓，但能拦住"抽取/改名/搬移后 import 断链"这一整类错误 ✓。
"""
from __future__ import annotations

import importlib

import pytest

# 入口/关键模块清单（新增抽取模块时**请一并加进来** ✓）
ENTRY_MODULES = [
    "src.train.train",          # 训练入口（main() 所在）
    "src.train.data_setup",     # PR4: Dataset/sampler/DataLoader
    "src.train.setup",          # PR4: DDP/seed/目录/logging + create_logger
    "src.train.cli",            # 参数解析
    "src.eval.in_mem_eval",     # 训练内评测
    "src.eval.inference",       # 离线评测入口
    "src.eval.decoder",         # VAE/LPIPS
    "src.eval.sampler",         # 采样
    "src.eval.scoring",         # 指标
    "src.eval.report",          # 报告/身份
    "src.eval.artifacts",       # 产物写入
    "src.eval.protocol",        # eval_id / 完成判定
    "src.model.dit",            # 模型
]


@pytest.mark.parametrize("mod", ENTRY_MODULES)
def test_entry_module_importable(mod):
    """每个入口模块都必须能 import ✓（ImportError 会立刻失败 ✓）。"""
    m = importlib.import_module(mod)
    assert m is not None


def test_train_entry_exposes_main_and_create_logger():
    """训练入口必须暴露 main / create_logger ✓（PR4 搬移后由 setup 提供 create_logger ✓）。"""
    t = importlib.import_module("src.train.train")
    s = importlib.import_module("src.train.setup")
    assert callable(getattr(t, "main", None)), "train.main 缺失 ✗"
    # ★ create_logger 已**完全**搬到 setup ✓ —— train.py 里不再使用它, 故 ruff 会删掉那条
    #   未使用的 import ✓ ⇒ **不需要 shim** ✓（实测: 加 shim 反而被 ruff 判为未用而删除 ✗）。
    #   这里只断言它在 setup 里可用 ✓。
    assert callable(getattr(s, "create_logger", None)), "setup.create_logger 缺失 ✗"
    # 若将来有人为兼容又补了 shim, 则必须与 setup 里是**同一对象** ✓（不得复制实现 ✗）
    if getattr(t, "create_logger", None) is not None:
        assert t.create_logger is s.create_logger, "train.create_logger shim 必须是同一对象 ✗"
    assert callable(getattr(t, "build_data", None)), "train.build_data shim 缺失 ✗"
    d = importlib.import_module("src.train.data_setup")
    assert t.build_data is d.build_data


def test_dataset_classes_resolve_from_their_real_modules():
    """★ 事故本体: `MCCDDataset` 来自 `src.utils` 包, `MCCDLatentDataset` 来自 latent_dataset ✓。"""
    import src.utils as U
    import src.utils.latent_dataset as LD
    assert U.MCCDDataset is not None
    assert LD.MCCDLatentDataset is not None
    # data_setup 必须导入到**正确**的那两个 ✓
    ds = importlib.import_module("src.train.data_setup")
    assert ds.MCCDLatentDataset is LD.MCCDLatentDataset
    assert ds.MCCDDataset is U.MCCDDataset
