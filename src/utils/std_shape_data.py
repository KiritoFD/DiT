"""Dataset-to-bank key alignment; workers return indices, never augment images."""
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset


class StdShapeDataset(Dataset):
    def __init__(self,base,cache_path):
        self.base=base;self.samples=base.samples
        data=torch.load(cache_path,map_location='cpu',weights_only=True,mmap=True)
        if data.get('version')!=2 or data.get('source_kind')!='precomputed_std_shape':
            raise ValueError('Expected offline STD perturbation bank')
        self.lookup={tuple(k):i for i,k in enumerate(data['keys'])}
        index=Path(base._std_soft.root)/'index.csv'
        if hashlib.sha256(index.read_bytes()).hexdigest()!=data['source_index_sha256']:
            raise ValueError('High-res font cache and STD store index disagree')
        self.positions=[]
        for row in self.samples:
            key=(int(row['script_id']),row['character'])
            if key not in self.lookup:raise ValueError(f'Missing high-resolution STD {key}')
            self.positions.append(self.lookup[key])

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        result=dict(self.base[index]);result['std_shape_index']=self.positions[index]
        return result

    def __getitems__(self,indices):
        fetch=getattr(self.base,'__getitems__',None)
        rows=fetch(indices) if fetch is not None else [self.base[i] for i in indices]
        return [dict(row,std_shape_index=self.positions[i]) for row,i in zip(rows,indices)]
