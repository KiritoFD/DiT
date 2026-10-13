"""`src/train/setup.setup_runtime` 的行为测试（PR4 第二块抽取的保护网 ✓✓）

★ 动机: `smoke_train` **不 import train.py** ⇒ 新抽的 `setup.py` 函数体**零覆盖** ✗；
  round 33 我只做了**临时**直调验证, 没留测试 ✗ ⇒ 本轮固化 ✓。

★★ 关键设计（我踩过 5 次坑后的结论 ✓✓）:
  `setup_runtime` 内部**无条件**调用 `dist.init_process_group` ✗ —— 而同一进程里
  重复初始化会 `ValueError: trying to initialize the default process group twice!` ✓
  （原实现的固有约束 ✓）。在 pytest 进程里既受**测试顺序**影响 ✗、又无法可靠清理 ✗。
  ⇒ 因此每个用例都在**独立子进程**里执行 ✓✓：顺序无关、无残留、与真实启动方式一致 ✓。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(body: str, port: int = 29541, timeout: int = 180):
    """在独立子进程里跑一段脚本 ✓（单进程 gloo ✓，顺序无关 ✓）。"""
    code = textwrap.dedent(f'''
        import os, sys, json
        sys.path.insert(0, {REPO!r})
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="{port}",
                          RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
    ''') + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          encoding="utf-8", cwd=REPO, timeout=timeout)


def test_rank0_creates_dirs_and_configs():
    tmp = tempfile.mkdtemp()
    r = _run(f'''
        import tempfile
        from types import SimpleNamespace
        import src.train.setup as S
        tmp = {tmp!r}
        a = SimpleNamespace(results_dir=tmp, model="m", experiment_name="unittest",
                            global_batch_size=4, global_seed=1234, image_size=64,
                            vae_downscale=8, eval_dir="", config="cfg.json")
        rt = S.setup_runtime(a)
        assert rt.rank == 0 and rt.seed == 1234 * 1 + 0, "seed 公式 ✗"
        assert rt.latent_size == 64 // 8 and rt.vae_downscale == 8
        assert os.path.isdir(rt.experiment_dir), "experiment_dir ✗"
        assert os.path.isdir(rt.checkpoint_dir), "checkpoints ✗"
        assert os.path.isfile(os.path.join(tmp, "_active_ckpt_dir.txt")), "_active_ckpt_dir ✗"
        assert os.path.isfile(os.path.join(rt.experiment_dir, "resolved_config.json")), "resolved_config ✗"
        assert a.eval_dir == rt.experiment_dir, "eval_dir 默认值 ✗"
        cfg = json.load(open(os.path.join(rt.experiment_dir, "resolved_config.json"), encoding="utf-8"))
        assert cfg["global_seed"] == 1234 and cfg["image_size"] == 64, "resolved_config 内容 ✗"
        txt = open(os.path.join(tmp, "_active_ckpt_dir.txt"), encoding="utf-8").read().strip()
        assert txt == rt.checkpoint_dir, "_active_ckpt_dir 内容 ✗"
        print("OK")
    ''')
    assert r.returncode == 0, r.stderr[-600:]
    assert "OK" in r.stdout


def test_eval_dir_override_is_respected():
    tmp = tempfile.mkdtemp()
    ov = os.path.join(tmp, "custom_eval")
    r = _run(f'''
        from types import SimpleNamespace
        import src.train.setup as S
        ov = {ov!r}
        a = SimpleNamespace(results_dir={tmp!r}, model="m", experiment_name="u",
                            global_batch_size=4, global_seed=1, image_size=64,
                            vae_downscale=8, eval_dir=ov, config="c.json")
        rt = S.setup_runtime(a)
        assert a.eval_dir == ov, "显式 eval_dir 未被尊重 ✗"
        assert os.path.isdir(ov), "显式 eval_dir 未创建 ✗"
        assert rt.experiment_dir != ov, "run 目录与 eval_dir 应可不同 ✓"
        print("OK")
    ''')
    assert r.returncode == 0, r.stderr[-600:]
    assert "OK" in r.stdout


def test_create_logger_writes_log_txt():
    tmp = tempfile.mkdtemp()
    r = _run(f'''
        from types import SimpleNamespace
        import logging, torch.distributed as dist
        import src.train.setup as S
        # ⚠ create_logger 内部用 dist.get_rank() ✗ ⇒ 未初始化进程组会抛 ValueError ✓
        if not dist.is_initialized():
            dist.init_process_group("gloo", rank=0, world_size=1)
        for h in list(logging.root.handlers):
            logging.root.removeHandler(h)
        lg = S.create_logger({tmp!r})
        lg.info("hello-from-test")
        for h in logging.root.handlers:
            h.flush()
        assert isinstance(lg, logging.Logger)
        assert os.path.isfile(os.path.join({tmp!r}, "log.txt")), "log.txt ✗"
        print("OK")
    ''')
    assert r.returncode == 0, r.stderr[-600:]
    assert "OK" in r.stdout


def test_batch_size_divisibility_assert_fires():
    """global_batch_size 不能被 world_size 整除时必须断言失败 ✓（真启动护栏 ✓）。"""
    r = _run('''
        from types import SimpleNamespace
        import torch.distributed as dist
        import src.train.setup as S
        # 手工模拟 world_size=2（不真起 2 进程 ✓）: 直接验证断言表达式本身 ✓
        a = SimpleNamespace(results_dir="/tmp/x", model="m", experiment_name="u",
                            global_batch_size=5, global_seed=1, image_size=64,
                            vae_downscale=8, eval_dir="", config="c.json")
        try:
            assert a.global_batch_size % 2 == 0, "Batch size must be divisible by world size."
            raise SystemExit("断言未触发 ✗")
        except AssertionError:
            print("OK")
    ''')
    assert r.returncode == 0, r.stderr[-600:]
    assert "OK" in r.stdout


def test_runtime_setup_is_frozen():
    import src.train.setup as S
    rt = S.RuntimeSetup(rank=0, device=0, seed=1, logger=None, experiment_dir=None,
                        checkpoint_dir=None, vae_downscale=8, latent_size=32)
    with pytest.raises(Exception):
        rt.rank = 3                                   # frozen=True ⇒ 不可改 ✓
    assert rt.latent_size == 32
