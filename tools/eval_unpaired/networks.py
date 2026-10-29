"""Strict frozen evaluators; no random fallback or generator weight updates."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import torch
from torch.nn import functional as F


def identity(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    return dict(path=str(path),size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.stat().st_size<=1<<20 else None)


def tensors(paths,size=256):
    return torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((size,size),Image.Resampling.BICUBIC)).copy()
        for p in paths])).permute(0,3,1,2).float()/255


class Evaluators:
    def __init__(self,device='cuda',iqa=True,local_checkpoint=None,iqa_metrics=None):
        from tools.eval_gen_with_disc import load_evaluator_model
        self.device=device;self.global_net,self.kind,self.size=load_evaluator_model('assets/style_disc_ce_holdout.pt',device)
        if self.kind!='resnet18':raise ValueError('Expected audited global ResNet checkpoint')
        self.global_net.eval().requires_grad_(False);self.local=None;self.iqa={}
        self.provenance={'E-G':identity('assets/style_disc_ce_holdout.pt')}
        if local_checkpoint:
            ck=torch.load(local_checkpoint,map_location='cpu',weights_only=False)
            if 'stages.0.0.body.0.weight' in ck['model']:
                import importlib.util
                source=Path(local_checkpoint).parents[2]/'src/model/writer_style_pixel.py'
                if not source.is_file():raise FileNotFoundError('Historical D1 source required: '+str(source))
                expected=ck.get('config',{}).get('source_hashes',{}).get('src/model/writer_style_pixel.py')
                if expected and identity(source)['sha256']!=expected:raise ValueError('Historical D1 source fingerprint mismatch')
                spec=importlib.util.spec_from_file_location('unpaired_historical_pixel',source)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                self.local=module.WriterStylePixel(len(ck['slots'])).to(device)
                self.local.load_state_dict(ck['model'],strict=True)
                self.provenance['E-L-source']=identity(source)
            else:
                self.local,kind,_=load_evaluator_model(local_checkpoint,device)
                if kind!='pixel_writer_v2':raise ValueError('Local E-L must be trained PixelWriterV2')
            self.local.eval().requires_grad_(False);self.provenance['E-L']=identity(local_checkpoint)
        if iqa:
            import pyiqa
            self.provenance['pyiqa_version']=pyiqa.__version__
            for name in (iqa_metrics or ['musiq-ava','musiq','topiq_nr']):
                kwargs={'backbone_pretrain':False} if name=='topiq_nr' else {}
                metric=pyiqa.create_metric(name,device=device,**kwargs).eval().requires_grad_(False)
                if name=='topiq_nr':
                    path=Path(torch.hub.get_dir())/'pyiqa/cfanet_nr_koniq_res50-9a73138b.pth'
                    # Full official checkpoint includes the semantic backbone; no second HF download.
                    state=torch.load(path,map_location='cpu',weights_only=False)['params']
                    metric.net.load_state_dict(state,strict=True)
                self.iqa[name]=metric
            self.provenance['iqa_weights']={str(p):identity(p) for p in (Path(torch.hub.get_dir())/'pyiqa').glob('*.pth')
                if p.name.startswith(('musiq_ava','musiq_koniq','cfanet_nr_koniq'))}
        self.provenance['preprocessing']='RGB PIL bicubic256; E-G bilinear160 ImageNet normalization; E-L RGB-mean bilinear128; IQA RGB[0,1]256'

    @torch.inference_mode()
    def batch(self,x):
        x=x.to(self.device);scaled=F.interpolate(x,(self.size,self.size),mode='bilinear',align_corners=False)
        normalized=(scaled-scaled.new_tensor([.485,.456,.406])[None,:,None,None])/scaled.new_tensor([.229,.224,.225])[None,:,None,None]
        result={'E-G':self.global_net.emb(normalized).cpu().numpy()}
        if self.local is not None:
            gray=x.mean(1,keepdim=True);gray=F.interpolate(gray,(128,128),mode='bilinear',align_corners=False)
            embedding=self.local.extract_features(gray) if hasattr(self.local,'extract_features') else self.local(gray)['embedding']
            result['E-L']=embedding.cpu().numpy()
        for name,net in self.iqa.items():result[name]=net(x).reshape(-1).cpu().numpy()
        return result

    def extract(self,paths,batch=8):
        result={}
        for start in range(0,len(paths),batch):
            for key,value in self.batch(tensors(paths[start:start+batch])).items():result.setdefault(key,[]).append(value)
        return {k:np.concatenate(v) for k,v in result.items()}

    def calibrate(self,paths):
        result={};raw=tensors(paths)
        variants={'real':raw,'blur':torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((256,256)).filter(ImageFilter.GaussianBlur(2))).copy() for p in paths])).permute(0,3,1,2).float()/255}
        variants['thick']=1-F.max_pool2d(1-raw,5,1,2)
        broken=raw.clone();broken[:,:,118:126,:]=1;variants['broken']=broken
        variants['blank']=torch.ones_like(raw)
        for name,x in variants.items():
            outputs={}
            for start in range(0,len(x),4):
                for key,value in self.batch(x[start:start+4]).items():
                    if not key.startswith('E-'):outputs.setdefault(key,[]).extend(value.tolist())
            result[name]=outputs
        gates={}
        for name in self.iqa:
            base=np.array(result['real'][name]);comparisons={v:float(np.mean(base>result[v][name])) for v in variants if v!='real'}
            gates[name]=dict(real_preferred_fraction=comparisons,
                usable_as_primary=False,reason='Automated perturbation screen only; requires TTF control and human calligraphy preference validation')
        return dict(scores=result,gates=gates)
