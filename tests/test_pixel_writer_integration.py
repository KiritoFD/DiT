"""Integration test for training loop, quartet sampler, and acceptance probe."""
import pytest
import torch
import numpy as np

from src.model.writer_style_pixel import PixelWriterV2
from src.utils.writer_quartets import WriterQuartets, build_same_script_mask
from src.loss.metric_learning import compute_metric_loss


def test_quartet_sampler_and_training_step():
    """Verify 384+128 mixed sampling and training step execution."""
    N = 100
    train = np.arange(N)
    slots = np.random.randint(0, 4, size=N)
    scripts = np.array([0, 0, 1, 1])[slots]
    chars = np.random.choice(['永', '山', '之', '大', '中', '水'], size=N)
    sources = np.random.randint(0, 3, size=N)

    # Ensure at least one valid quartet
    slots[0:2] = 0; chars[0] = '永'; chars[1] = '山'
    slots[2:4] = 1; chars[2] = '永'; chars[3] = '山'
    scripts[0:4] = 0

    sampler = WriterQuartets(train, slots, scripts, chars, sources, seed=42)
    ids, meta = sampler.sample_mixed(total_batch=32, quartet_samples=16)
    assert len(ids) == 32
    assert meta['n_quartet_samples'] == 16

    # Run forward-backward step
    model = PixelWriterV2(n_slots=4, haar_mode='2d')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.rand(32, 1, 128, 128)
    targets = torch.from_numpy(slots[ids]).long()
    slot_scripts = [0, 0, 1, 1]
    mask = build_same_script_mask(targets, slot_scripts, n_slots=4)

    out = model(x, script_mask=mask)
    ce = torch.nn.functional.cross_entropy(out['masked_logits'], targets)

    m_loss, m_stats = compute_metric_loss(
        out['embedding'], chars[ids], loss_type='contrastive', n_quartet_samples=16, contrastive_margin=1.0
    )
    total_loss = ce + 0.5 * m_loss
    total_loss.backward()
    opt.step()

    assert torch.isfinite(total_loss)
