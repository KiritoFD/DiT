# -*- coding: utf-8 -*-
"""Replace FontDiffuser's hand-written matmul+softmax attention with
F.scaled_dot_product_attention (flash/mem-efficient). Numerically equivalent
(same q/k/v projections, same scale, same output layout), removes the
materialized [tokens, tokens] score matrix — the memory hog at 256px.
Idempotent. Run: python patch_sdpa.py
"""
import os

P = "/root/Workspace/xy/DiT/baseline/FontDiffuser/src/modules/attention.py"

OLD = """    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states"""

NEW = """    def _attention(self, query, key, value):
        # top10 adapter: fused SDPA (flash/mem-efficient) — numerically
        # equivalent to the original matmul+softmax, no score matrix in memory.
        # NOTE: cross-attention has key/value seq_len != query seq_len.
        bh, qseq, d = query.shape
        kseq = key.shape[1]
        heads = self.heads
        q = query.view(bh // heads, heads, qseq, d)
        k = key.view(bh // heads, heads, kseq, d)
        v = value.view(bh // heads, heads, kseq, d)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return out.reshape(bh // heads, qseq, d * heads)"""

# earlier (buggy) variant possibly present on the server
BAD = """    def _attention(self, query, key, value):
        # top10 adapter: fused SDPA (flash/mem-efficient) — numerically
        # equivalent to the original matmul+softmax, no score matrix in memory
        bh, seq, d = query.shape
        heads = self.heads
        q = query.view(bh // heads, heads, seq, d)
        k = key.view(bh // heads, heads, seq, d)
        v = value.view(bh // heads, heads, seq, d)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return out.reshape(bh // heads, seq, d * heads)"""


def main():
    src = open(P, encoding="utf-8").read()
    if "cross-attention has key/value seq_len != query seq_len" in src:
        print("already patched")
        return
    if BAD in src:
        open(P, "w", encoding="utf-8").write(src.replace(BAD, NEW))
        print("fixed cross-attention seq handling in SDPA patch")
        return
    assert OLD in src, "original _attention not found"
    open(P, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    print("patched attention.py: SDPA")


if __name__ == "__main__":
    main()
