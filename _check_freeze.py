import models, torch
m = models.DiT_2Cond_WS_2(input_size=32, num_calligraphers=1011, num_characters=35130,
    condition_fusion='factorized_add', callig_embed_dim=128, char_embed_dim=768,
    char_proj_mode='ln_only', freeze_char_table=True,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25)
tot = sum(p.numel() for p in m.parameters())
trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"WS total={tot/1e6:.1f}M trainable={trn/1e6:.1f}M frozen={(tot-trn)/1e6:.1f}M")
# char table check
w = m.y_char_embedder.embedding_table.weight
print(f"char table shape={tuple(w.shape)} requires_grad={w.requires_grad} last_row_grad={w[-1].requires_grad}")
# callig table
wc = m.y_callig_embedder.embedding_table.weight
print(f"callig table shape={tuple(wc.shape)} requires_grad={wc.requires_grad}")