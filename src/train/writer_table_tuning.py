"""Fine-tune real writer rows while preserving the existing table interface."""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class WriterTableTuning:
    weight: nn.Parameter
    n_writers: int
    reference: torch.Tensor
    fixed_tail: torch.Tensor
    anchor_weight: float

    def _on_device(self):
        self.reference = self.reference.to(self.weight.device, self.weight.dtype)
        self.fixed_tail = self.fixed_tail.to(self.weight.device, self.weight.dtype)

    def anchor_loss(self):
        self._on_device()
        live = F.normalize(self.weight[:self.n_writers], dim=-1)
        target = F.normalize(self.reference, dim=-1)
        return (live - target).square().sum(dim=-1).mean()

    @torch.no_grad()
    def restore_null(self, model=None):
        # A zero grad alone does not protect a row from AdamW decay or restored
        # optimizer momentum. Keep the original tail exactly even after resume.
        self._on_device()
        weight=self.weight
        if model is not None:
            module=model
            for attr in ('module','_orig_mod'):
                module=getattr(module,attr,module)
            weight=module.y_callig_embedder.embedding_table.weight
        weight[self.n_writers:].copy_(self.fixed_tail.to(weight.device,weight.dtype))


def configure_writer_table(model, args) -> Optional[WriterTableTuning]:
    if not getattr(args, 'finetune_callig_table', False):
        return None
    if not getattr(args, 'freeze_triple_tables', False):
        raise ValueError('writer-only table tuning requires freeze_triple_tables=true')
    if getattr(args, 'triple_table_lr_scale', 0.0) != 0:
        raise ValueError('writer-only table tuning cannot use triple_table_lr_scale')
    if getattr(args, 'optimizer', 'adamw') != 'adamw':
        raise ValueError('writer-only table tuning requires AdamW')
    if getattr(args, 'train_only_callig_table', False) or getattr(args, 'train_only_char_embed', False):
        raise ValueError('writer table fine-tuning keeps the trunk trainable')
    module = model
    for attr in ('module', '_orig_mod'):
        module = getattr(module, attr, module)
    emb = module.y_callig_embedder
    weight = emb.embedding_table.weight
    n = int(emb.num_classes)
    if weight.ndim != 2 or weight.shape[0] != n + 1 or emb.null_embed is not None:
        raise ValueError('expected the unchanged LabelEmbedder with one CFG null row')
    if any(p.requires_grad for p in module.y_script_embedder.parameters()):
        raise ValueError('script table must remain frozen')
    char = getattr(module, 'y_char_embedder', None)
    if char is not None and any(p.requires_grad for p in char.parameters()):
        raise ValueError('character table must remain frozen')
    lr = float(getattr(args, 'callig_table_lr', 0.0))
    aw = float(getattr(args, 'callig_table_anchor_weight', 0.0))
    if lr <= 0 or aw < 0:
        raise ValueError('callig_table_lr must be positive and anchor weight nonnegative')
    ref_path = getattr(args, 'callig_table_anchor_file', '')
    if not ref_path:
        raise ValueError('a fixed checkpoint-derived anchor file is required for resume')
    ref = torch.load(ref_path, map_location='cpu', weights_only=True)['writer_rows']
    if ref.shape != weight[:n].shape or not torch.isfinite(ref).all():
        raise ValueError('writer anchor shape/values do not match the table')
    state = WriterTableTuning(weight,n,ref.detach().clone(),weight[n:].detach().clone(),aw)
    weight.requires_grad_(True)
    mask = torch.zeros_like(weight)
    mask[:n] = 1
    weight.register_hook(lambda grad: grad * mask.to(grad.device))
    return state


def split_writer_optimizer_group(optimizer, state, lr):
    """Split before restoring optimizer state and before building the scheduler."""
    if state is None:
        return
    found = 0
    for group in optimizer.param_groups:
        found += sum(p is state.weight for p in group['params'])
        group['params'] = [p for p in group['params'] if p is not state.weight]
    if found != 1:
        raise ValueError('writer table must occur exactly once in the optimizer')
    optimizer.add_param_group({'params':[state.weight], 'lr':float(lr),
                               'weight_decay':0.0, 'group_name':'writer_table'})
