import sys, json
d = json.load(sys.stdin)
for k in ["vae","vae_path","vae_downscale","latent_channels","vae_scaling_factor",
          "char_embed_dim","data_csv","img_root","condition_fusion","model","use_lora"]:
    print(k, "=", d.get(k))
