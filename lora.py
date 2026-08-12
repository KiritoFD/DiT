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

def inject_lora(model, r=16, lora_alpha=16, dropout=0.0):
    """
    Injects LoRA into a DiT model blocks.
    Replaces qkv, proj, fc1, fc2 with LoRALinear.
    Idempotent: already-injected layers are left untouched.
    """
    n_injected = 0
    for block in model.blocks:
        if hasattr(block, 'attn'):
            if hasattr(block.attn, 'qkv') and not isinstance(block.attn.qkv, LoRALinear):
                block.attn.qkv = LoRALinear(block.attn.qkv, r, lora_alpha, dropout)
                n_injected += 1
            if hasattr(block.attn, 'proj') and not isinstance(block.attn.proj, LoRALinear):
                block.attn.proj = LoRALinear(block.attn.proj, r, lora_alpha, dropout)
                n_injected += 1
        if hasattr(block, 'mlp'):
            if hasattr(block.mlp, 'fc1') and not isinstance(block.mlp.fc1, LoRALinear):
                block.mlp.fc1 = LoRALinear(block.mlp.fc1, r, lora_alpha, dropout)
                n_injected += 1
            if hasattr(block.mlp, 'fc2') and not isinstance(block.mlp.fc2, LoRALinear):
                block.mlp.fc2 = LoRALinear(block.mlp.fc2, r, lora_alpha, dropout)
                n_injected += 1
    print(f"[inject_lora] Injected LoRA into {n_injected} linear layers (r={r}, alpha={lora_alpha}).")
    return model

def extract_lora_and_new_embedders(model):
    """
    Extracts only the state_dict for trainable parameters (LoRA and new embedders)
    to keep checkpoints small.
    """
    state_dict = model.state_dict()
    trainable_dict = {k: v for k, v in state_dict.items() if 'lora_' in k or 'y_' in k or 'cond_fusion' in k}
    return trainable_dict


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

