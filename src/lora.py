import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=16, lora_alpha=16, dropout=0.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=original_linear.weight.device))
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=original_linear.weight.device))
        self.dropout = nn.Dropout(p=dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x):
        out = self.original_linear(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return out + lora_out * self.scaling

def inject_lora(model, r=16, lora_alpha=16, dropout=0.0, target="all"):
    """
    Injects LoRA into a DiT model blocks.
    Replaces qkv, proj (attn) and fc1, fc2 (mlp) with LoRALinear.
    target:
      "all"  -> qkv + proj + fc1 + fc2 (default)
      "attn" -> qkv + proj (attention only, MLP untouched)
      "mlp"  -> fc1 + fc2 (MLP only, attention untouched)
    Idempotent: already-injected layers are left untouched.
    """
    do_attn = target in ("all", "attn")
    do_mlp = target in ("all", "mlp")
    n_injected = 0
    for block in model.blocks:
        if hasattr(block, 'attn') and do_attn:
            if hasattr(block.attn, 'qkv') and not isinstance(block.attn.qkv, LoRALinear):
                block.attn.qkv = LoRALinear(block.attn.qkv, r, lora_alpha, dropout)
                n_injected += 1
            if hasattr(block.attn, 'proj') and not isinstance(block.attn.proj, LoRALinear):
                block.attn.proj = LoRALinear(block.attn.proj, r, lora_alpha, dropout)
                n_injected += 1
        if hasattr(block, 'mlp') and do_mlp:
            if hasattr(block.mlp, 'fc1') and not isinstance(block.mlp.fc1, LoRALinear):
                block.mlp.fc1 = LoRALinear(block.mlp.fc1, r, lora_alpha, dropout)
                n_injected += 1
            if hasattr(block.mlp, 'fc2') and not isinstance(block.mlp.fc2, LoRALinear):
                block.mlp.fc2 = LoRALinear(block.mlp.fc2, r, lora_alpha, dropout)
                n_injected += 1
    print(f"[inject_lora] Injected LoRA into {n_injected} linear layers "
          f"(r={r}, alpha={lora_alpha}, target={target}).")
    return model

def extract_lora_and_new_embedders(model):
    """
    Extracts only the state_dict for trainable parameters (LoRA and new embedders)
    to keep checkpoints small.
    """
    state_dict = model.state_dict()
    trainable_dict = {k: v for k, v in state_dict.items() if 'lora_' in k or 'y_' in k or 'cond_fusion' in k}
    return trainable_dict


def extract_full_inference(model):
    """
    Extracts a self-contained "inference delta" that, together with the official
    DiT-XL-2-256x256.pt pretrained body, fully reconstructs the trained model.

    Includes:
      - lora_A / lora_B (low-rank deltas)
      - y_callig / y_script / y_char embedders + cond_fusion (new condition head)
      - blocks.*.adaLN_modulation and final_layer (reset_cond_head made these
        diverge from the pretrained weights; they are frozen during training but
        MUST be captured, otherwise the model cannot be reconstructed)

    Rationale: the pretrained body (x_embedder / pos_embed / t_embedder / blocks'
    qkv/proj/mlp original weights) is frozen and identical to the pretrained
    checkpoint, so it can be loaded from disk and need not be stored per-ckpt.
    """
    state_dict = model.state_dict()
    keys = []
    for k in state_dict:
        if ('lora_' in k or k.startswith('y_') or k.startswith('cond_fusion')
                or 'adaLN' in k or 'final_layer' in k):
            keys.append(k)
    return {k: state_dict[k] for k in keys}


def build_model_from_ckpt(ckpt_path, pretrained_path=None, r=32, lora_alpha=32,
                          num_calligraphers=2021, num_scripts=12, num_characters=7765,
                          device="cpu", use_checkpoint=False, reset_cond_head=False):
    """
    Single, unified entry point for loading a trained model from a checkpoint.

    The checkpoint stores only a `delta` (the "changed" part: LoRA + condition head +
    adaLN/final_layer). This function reconstructs the full model in one call by:
      1. constructing the DiT-3Cond base model
      2. loading the official pretrained body (frozen weights shared across all ckpts)
      3. injecting LoRA
      4. loading the delta

    No caller-side extraction/filtering is needed — the pretrained body is NOT stored
    per-ckpt, and the delta loads directly.

    Returns the model (on `device`, in eval mode).
    """
    import torch as _torch
    import os as _os

    # Local import to avoid a hard circular dependency at module load time.
    from models import DiT_3Cond_XL_2

    model = DiT_3Cond_XL_2(
        num_calligraphers=num_calligraphers,
        num_scripts=num_scripts,
        num_characters=num_characters,
        use_checkpoint=use_checkpoint,
    )

    # 1) official pretrained body (frozen, shared, NOT stored per-ckpt)
    if pretrained_path is None:
        pretrained_path = "pretrained_models/DiT-XL-2-256x256.pt"
    if pretrained_path and _os.path.exists(pretrained_path):
        pre = _torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if "model" in pre:
            pre = pre["model"]
        pre = {k: v for k, v in pre.items()
               if not k.startswith(("y_embedder", "y_callig", "y_script", "y_char", "cond_fusion"))}
        model.load_state_dict(pre, strict=False)

    # 2) inject LoRA (idempotent)
    inject_lora(model, r=r, lora_alpha=lora_alpha)

    # 3) load delta (the changed part)
    ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    delta = ckpt.get("delta", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(delta, strict=False)

    # 4) optional re-init of condition head (only for fresh/legacy ckpts lacking adaLN)
    if reset_cond_head:
        import torch.nn as _nn
        for _b in model.blocks:
            _nn.init.normal_(_b.adaLN_modulation[-1].weight, std=0.02)
            _nn.init.normal_(_b.adaLN_modulation[-1].bias, std=0.02)
        _nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)
        _nn.init.normal_(model.final_layer.adaLN_modulation[-1].bias, std=0.02)
        _nn.init.normal_(model.final_layer.linear.weight, std=0.02)

    return model.to(device).eval()


# Backwards-compatible alias.
def load_full_inference(model, delta, pretrained_path=None, strict=True):
    """Legacy alias. Prefer build_model_from_ckpt for new code."""
    import torch as _torch
    import os as _os

    if pretrained_path is None:
        pretrained_path = "pretrained_models/DiT-XL-2-256x256.pt"
    if pretrained_path and _os.path.exists(pretrained_path):
        pre = _torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if "model" in pre:
            pre = pre["model"]
        pre = {k: v for k, v in pre.items()
               if not k.startswith(("y_embedder", "y_callig", "y_script", "y_char", "cond_fusion"))}
        model.load_state_dict(pre, strict=False)

    inject_lora(model, r=32, lora_alpha=32)
    return model.load_state_dict(delta, strict=strict)


def upgrade_lora_rank(model, new_r, new_alpha, old_sd, old_r, device=None):
    """
    Upgrade an already-injected LoRA model from old_r to new_r while preserving the
    learned low-rank deltas.

    Strategy: re-inject with the larger rank, copy the old A/B into the leading
    `old_r` columns/rows (so the already-learned residual is kept exactly), and leave
    the newly-added columns/rows as fresh kaiming init (A) / zeros (B). Because the new
    B rows are zero, the extra capacity contributes *zero* residual at init, so the
    model's behaviour right after upgrade is identical to before — it can then learn
    into the extra dimensions without forgetting.

    `old_sd` is a state_dict containing `lora_A` / `lora_B` keys (from a previous run).
    Returns the number of layers whose LoRA was upgraded.
    """
    import math
    device = device or next(model.parameters()).device
    # Re-inject at the new (larger) rank. inject_lora is idempotent on already-LoRA
    # layers, so it will NOT touch them — we must rebuild them explicitly instead.
    model = _rebuild_lora_at_rank(model, new_r, new_alpha, device)

    upgraded = 0
    for block in model.blocks:
        for sub in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'):
            parent_name, leaf = sub.rsplit('.', 1)
            parent = getattr(block, parent_name, None)
            layer = getattr(parent, leaf, None)
            if not isinstance(layer, LoRALinear):
                continue
            a_key = f"{sub}.lora_A"
            b_key = f"{sub}.lora_B"
            if a_key not in old_sd or b_key not in old_sd:
                continue
            old_A = old_sd[a_key].to(device)  # (in, old_r)
            old_B = old_sd[b_key].to(device)  # (old_r, out)
            with torch.no_grad():
                layer.lora_A[:, :old_r].copy_(old_A)
                layer.lora_B[:old_r, :].copy_(old_B)
                # new columns of A: kaiming init; new rows of B: stay zero
                if new_r > old_r:
                    nn.init.kaiming_uniform_(layer.lora_A[:, old_r:], a=math.sqrt(5))
            upgraded += 1
    print(f"[upgrade_lora_rank] Upgraded {upgraded} LoRA layers from r={old_r} -> r={new_r} "
          f"(alpha={new_alpha}, scaling={new_alpha/new_r:.2f}); learned deltas preserved.")
    return model


def _rebuild_lora_at_rank(model, r, lora_alpha, device):
    """Rebuild every LoRALinear at the target rank (discards any existing A/B)."""
    import math
    for block in model.blocks:
        for sub in ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'):
            parent_name, leaf = sub.rsplit('.', 1)
            parent = getattr(block, parent_name, None)
            layer = getattr(parent, leaf, None)
            if isinstance(layer, LoRALinear):
                new_layer = LoRALinear(layer.original_linear, r, lora_alpha, layer.dropout.p)
                setattr(parent, leaf, new_layer)
    return model

