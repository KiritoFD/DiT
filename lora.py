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
