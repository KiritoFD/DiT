"""Frozen writer table -> trainable Qwen context tokens; no GT in conditions."""
import numpy as np
import torch
from torch import nn


class Conditions(nn.Module):
    def __init__(self, table, dimension=4096,num_scripts=1):
        super().__init__()
        values=torch.from_numpy(np.load(table)).float()
        self.writer=nn.Embedding.from_pretrained(values,freeze=True)
        self.writer_proj=nn.Sequential(nn.Linear(values.shape[1],256),nn.SiLU(),nn.Linear(256,dimension*4))
        self.script=nn.Embedding(num_scripts,16)
        self.script_proj=nn.Sequential(nn.Linear(16,128),nn.SiLU(),nn.Linear(128,dimension))
        self.dimension=dimension

    def forward(self, text, text_mask, writer_ids, script_ids, drop_writer=None):
        values=self.writer(writer_ids)
        if drop_writer is not None:
            values=values.masked_fill(drop_writer[:,None],0)
        writer=self.writer_proj(values).reshape(len(writer_ids),4,self.dimension)
        script=self.script_proj(self.script(script_ids))[:,None]
        tokens=torch.cat([writer,script],1).to(text.dtype)
        sequences=[torch.cat([text[i,text_mask[i].bool()],tokens[i]],0) for i in range(len(writer_ids))]
        lengths=torch.tensor([len(v) for v in sequences],device=text.device)
        padded=torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True)
        mask=torch.arange(padded.shape[1],device=text.device)[None]<lengths[:,None]
        return padded,mask
