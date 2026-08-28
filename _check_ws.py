import models
m = models.DiT_2Cond_WS_2(input_size=32, num_calligraphers=1011, num_characters=35130,
    condition_fusion='factorized_add', callig_embed_dim=128, char_embed_dim=768,
    char_proj_mode='ln_only', freeze_char_table=True,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25)
print("WS params:", sum(p.numel() for p in m.parameters())/1e6, "M")
m2 = models.DiT_2Cond_S_2(input_size=32, num_calligraphers=1011, num_characters=35130,
    condition_fusion='factorized_add', callig_embed_dim=128, char_embed_dim=384,
    char_proj_mode='ln_only', freeze_char_table=True,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25)
print("S  params:", sum(p.numel() for p in m2.parameters())/1e6, "M")
