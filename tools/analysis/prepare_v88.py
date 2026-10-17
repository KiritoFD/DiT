"""Generate v88 from the actual v84A data/VAE contract, not a post-training config."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_config(base: dict, *, steps: int = 180000, batch_size: int = 288) -> dict:
    """Return a new plan without mutating its baseline or writing any artifacts."""
    if steps <= 0 or steps % 5000 or batch_size <= 0:
        raise ValueError('steps must be a positive multiple of5000; batch_size must be positive')
    cfg = dict(base)
    cfg.update(experiment_name='v88-writer-glyph-base', results_dir='assets/results/v88_writer_glyph_base',
    writer_glyph_rank=32, writer_glyph_at='3,7', warm_start_from='',
    finetune_callig_table=True, callig_table_lr=5e-6, callig_table_anchor_weight=0.,
    callig_table_anchor_file='assets/v88_pretrained_writer_anchor.pt',
    global_batch_size=batch_size, max_steps=steps, epochs=steps,
    lr=5e-5, warmup_steps=3000, ckpt_every=5000, epoch_steps=5000,
    glyph_deform_prob=0., glyph_noise_scale=0., glyph_patch_drop=0., glyph_mask_prob=0.,
    strong_style_mode='off', w_critic_writer=0., w_critic_away=0.,
    _comment='From-scratch DiT; original v84A data/VAE/FM+REPA. Complete structural token-add + separate writer/script x glyph residual after blocks3,7. No STD deformation or Critic. Writer pretrained table lr5e-6, script frozen; CFG null writer row fixed. Budget fallback180k: measured288-batch3.53steps/s, ~14.2h train plus eval within16h; checkpoint can continue later.')
    return cfg


def main(root: Path = ROOT) -> None:
    """Explicit CLI side effects only; validate inputs before publishing a plan."""
    import numpy as np
    import torch
    base = json.loads((root/'src/train/configs/v84A_std_cond.json').read_text(encoding='utf-8'))
    cfg = build_config(base)
    table = np.load(root/'assets/triple_tables_best_minimal/callig_table.npy')
    if table.shape != (10, 32) or not np.isfinite(table).all():
        raise ValueError(f'Unexpected pretrained writer table: {table.shape}')
    torch.save({'writer_rows': torch.from_numpy(table).float()}, root/'assets/v88_pretrained_writer_anchor.pt')
    (root/'src/train/configs/v88_writer_glyph_base.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('v88 config and pretrained table contract ready')


if __name__ == '__main__':
    main()
