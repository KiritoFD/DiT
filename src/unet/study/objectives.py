"""Frozen latent D2 loss; GT features are training-only and never model conditions."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

NAMES={53:'何绍基',365:'文徵明',461:'柳公权',479:'欧阳询',601:'王羲之',672:'米芾',703:'苏轼',760:'褚遂良',956:'颜真卿',9001:'赵孟頫'}


class Auxiliary:
    def __init__(self,config,spec,logger):
        if spec['objective']!='global_style' or spec.get('teacher')!='D2':
            raise ValueError('Unimplemented/unaccepted auxiliary cannot silently run as FM')
        self.spec=spec;self.config=config;self.logger=logger
        from tools.three_day.contracts import checkpoint_rows,sha,read
        mapping=checkpoint_rows(config);self.writers={row:NAMES[raw] for raw,row in mapping.items()}
        fonts=json.loads(Path(config.font_remap_json).read_text());scripts={0:'楷',3:'行',4:'隶'}
        self.scripts={int(row):scripts[int(raw)] for raw,row in fonts.items()}
        bank=torch.load(spec['reference_cache'],map_location='cpu',weights_only=False)
        if bank['teacher_sha256']!=sha(spec['teacher_checkpoint']):raise ValueError('Reference/teacher mismatch')
        if (bank['vae'],bank['scale'],bank['shift'])!=(config.vae_path,config.vae_scaling_factor,config.vae_shift_factor):
            raise ValueError('Style latent contract does not match generator')
        # Training cache identity is audited against the clean source CSV, not the replay mixture.
        if bank['csv_sha256']!=spec['reference_csv_sha256']:raise ValueError('Training feature CSV changed')
        from src.model.writer_style_latent import build_writer_style_latent
        ck=torch.load(spec['teacher_checkpoint'],map_location='cpu',weights_only=False)
        self.teacher=build_writer_style_latent(len(ck['slots']),'geometry').cuda().eval().requires_grad_(False)
        self.teacher.load_state_dict(ck['model'],strict=True)
        groups=defaultdict(lambda:defaultdict(list))
        for i,(w,s,c) in enumerate(zip(bank['writers'],bank['scripts'],bank['chars'])):groups[w,s][c].append(i)
        features=bank['features'].float();self.cells={};self.totals={}
        for key,chars in groups.items():
            self.cells[key]={char:features[ids].mean(0) for char,ids in chars.items()}
            self.totals[key]=torch.stack(list(self.cells[key].values())).sum(0)
        self.row_chars={int(r['img_id']):r['character'] for r in read(config.data_csv)}
        self.keys=sorted(self.cells);self.gain=1.

    def __call__(self,pred,z,t,kwargs,batch,step):
        active=(t>=.05)&(t<=.30)&(kwargs['y_callig']!=self.config.num_writers)&(kwargs['y_script']!=self.config.num_scripts)
        ids=active.nonzero().flatten()
        if not len(ids):return pred.sum()*0
        selected=ids.detach().cpu().tolist();ws=kwargs['y_callig'].detach().cpu().tolist();ss=kwargs['y_script'].detach().cpu().tolist()
        image_ids=batch['img_id'].tolist();positive=[];negative=[];masks=[]
        for i in selected:
            char=self.row_chars[int(image_ids[i])];target=(self.writers[ws[i]],self.scripts[ss[i]])
            if target not in self.cells:raise ValueError('No style training reference for condition')
            proto=[];allowed=[]
            for key in self.keys:
                count=len(self.cells[key])-(char in self.cells[key]);valid=key[1]==target[1] and count>0
                v=(self.totals[key]-self.cells[key].get(char,0))/max(count,1)
                proto.append(F.normalize(v,dim=0));allowed.append(valid and key!=target)
            positive.append(proto[self.keys.index(target)]);negative.append(torch.stack(proto));masks.append(allowed)
        p=torch.stack(positive).to(pred.device);n=torch.stack(negative).to(pred.device);mask=torch.tensor(masks,device=pred.device)
        if not mask.any(1).all():raise ValueError('Missing same-script different-writer prototype')
        with torch.autocast('cuda',dtype=torch.bfloat16):e=self.teacher(pred[ids])['embedding'].float()
        cp=(e*p).sum(1);cn=torch.einsum('bd,bkd->bk',e,n)
        aux=(F.softplus((cn-cp[:,None]+.1)/.1)*mask).sum()/mask.sum()
        weight=float(self.spec.get('weight',.02))*min(1.,(step+1)/500)
        weighted=aux*weight
        if step%100==0:
            grad=torch.autograd.grad(weighted,pred,retain_graph=True)[0]
            aux_v=(-t[:,None,None,None]*grad).norm()
            fm_v=(2*(pred.detach()-z)/(t[:,None,None,None].clamp_min(1e-6)*pred.numel())).norm()
            ratio=float((aux_v/fm_v.clamp_min(1e-12)).detach())
            self.gain=min(1.,.15/max(ratio,1e-12))
            self.logger.info('STUDY_STYLE step=%d n=%d raw=%.6f weight=%.6g output_grad_ratio=%.4f capped=%.4f',
                step,len(ids),float(aux.detach()),weight,ratio,ratio*self.gain)
        return weighted*self.gain
