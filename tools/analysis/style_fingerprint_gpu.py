"""Batched CUDA latent statistics and train-only linear statistical probes.

No optimizer, neural-network fitting, generator forward or checkpoint mutation.
CPU reference checks are mandatory before accepting GPU features.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.style_fingerprint_features import latent_features


@torch.no_grad()
def extract(z,batch=512):
    device='cuda';size=z.shape[-1]
    freq=torch.fft.fftfreq(size,device=device,dtype=torch.float64)
    yy,xx=torch.meshgrid(freq,freq,indexing='ij');r=torch.hypot(xx,yy)
    # Match CPU six-bin radial spectrum, then combine into three bands.
    window=torch.hann_window(size,periodic=False,device=device,dtype=torch.float64)
    window=window[:,None]*window[None,:]
    masks=torch.stack([(r>0)&(r<.05),(r>=.05)&(r<.2),(r>=.2)&(r<=.71)])
    output=[]
    for start in range(0,len(z),batch):
        x=torch.as_tensor(z[start:start+batch],device=device,dtype=torch.float64)
        flat=x.flatten(2);mu=flat.mean(2);sd=flat.std(2,correction=0)
        norm=(flat-mu[:,:,None])/sd[:,:,None].clamp_min(1e-9)
        moments=torch.stack([mu,sd,(norm**3).mean(2),(norm**4).mean(2)-3],2).flatten(1)
        grad=torch.stack([torch.diff(x,dim=2).square().mean((2,3)),torch.diff(x,dim=3).square().mean((2,3))],2).flatten(1)
        power=torch.fft.fft2((x-mu[:,:,None,None])*window).abs().square()
        bands=torch.einsum('bchw,khw->bck',power,masks.double())
        bands=bands/bands.sum(2,keepdim=True).clamp_min(1e-12)
        centered=flat-mu[:,:,None];cov=centered@centered.transpose(1,2)/flat.shape[-1]
        output.append(torch.cat([moments,grad,bands.flatten(1),cov.flatten(1)],1).float().cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def probe(x,w,s,c,source,train,query):
    x=torch.as_tensor(x,device='cuda',dtype=torch.float64)
    ti=torch.as_tensor(train,device='cuda');qi=torch.as_tensor(query,device='cuda')
    mean=x[ti].mean(0);sd=x[ti].std(0,correction=0).clamp_min(1e-4)
    x=(x-mean)/sd
    # Train-only PCA whitening of the covariance, shrinkage keeps tiny modes
    # bounded. PCA is not assigned semantic latent-channel interpretations.
    covariance=x[ti].T@x[ti]/len(train)
    eigval,eigvec=torch.linalg.eigh(covariance)
    whiten=eigvec@torch.diag((eigval.clamp_min(0)+.1).rsqrt())@eigvec.T
    white=x@whiten
    per_script={};total_correct=0;n_total=0
    for script in sorted(set(s[query])):
        tr=train[s[train]==script];q=query[s[query]==script]
        writers=sorted(set(w[tr])&set(w[q]))
        q=q[np.isin(w[q],writers)]
        if len(writers)<2 or not len(q):continue
        tr=tr[np.isin(w[tr],writers)]
        prototypes=torch.stack([x[tr[w[tr]==writer]].mean(0) for writer in writers])
        residual=torch.cat([x[tr[w[tr]==writer]]-prototypes[i] for i,writer in enumerate(writers)])
        cov=residual.T@residual/max(len(residual)-len(writers),1)
        # Fixed shrinkage=0.2; no hyperparameter search on holdout.
        cov=.8*cov+.2*torch.eye(x.shape[1],device='cuda')*torch.trace(cov)/x.shape[1]+1e-6*torch.eye(x.shape[1],device='cuda')
        coef=torch.linalg.solve(cov,prototypes.T)
        scores=x[q]@coef-.5*(prototypes*coef.T).sum(1)
        truth=np.array([writers.index(writer) for writer in w[q]])
        predicted=scores.argmax(1).cpu().numpy();correct=predicted==truth
        pw=torch.stack([white[tr[w[tr]==writer]].mean(0) for writer in writers])
        pca_pred=torch.cdist(white[q],pw).argmin(1).cpu().numpy()
        per_script[script]={'n':len(q),'n_writers':len(writers),'chance':1/len(writers),
                            'shrinkage_lda_top1':float(correct.mean()),'pca_white_nearest_top1':float((pca_pred==truth).mean())}
        total_correct+=int(correct.sum());n_total+=len(q)
    return {'n':n_total,'shrinkage_lda_top1':total_correct/n_total if n_total else None,
            'per_script':per_script,'pca_eigenvalues':eigval.cpu().tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--split-root',required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.cuda.set_device(0);start=time.time()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=np.load('scratch/disc_lat_full.npz',allow_pickle=True);z=d['z'];zs=d['z_std'];gi=d['std_idx'].astype(int)
    real=extract(z);std=extract(zs);checks=[]
    for name,raw,features in [('real',z,real),('std',zs,std)]:
        for i in (0,len(raw)//2,len(raw)-1):
            expected=latent_features(raw[i]);err=float(np.max(np.abs(features[i]-expected)))
            assert np.allclose(features[i],expected,atol=2e-5,rtol=2e-5),(name,i,err)
            checks.append({'kind':name,'row':i,'max_abs_error':err})
    np.savez_compressed(out/'latent_gpu.npz',real=real,std=std,std_idx=gi)
    w,s,c,source=[d[k].astype(str) for k in ('writer','script','char','work')]
    split=np.load(Path(a.split_root)/'split_indices.npz');report={'device':torch.cuda.get_device_name(),
        'precision':'float64 accumulation; float32 export','cpu_reference_checks':checks,'results':{},
        'limitations':['Exploratory train-only linear probes, not a neural teacher.',
            'Source proxies and writer/source confounding remain.',
            'Accuracy alone does not establish usefulness of image-space style gradients.']}
    for name,features in [('raw',real),('delta_std',real-std[gi])]:
        report['results'][name]={}
        for key in ('test_char','test_source'):
            report['results'][name][key]={'all_channels':probe(features,w,s,c,source,split['train'],split[key])}
            for channel in range(4):
                # Channel-specific moments, x/y gradients and spectral bands.
                cols=list(range(channel*4,channel*4+4))+list(range(16+channel*2,18+channel*2))+list(range(24+channel*3,27+channel*3))
                report['results'][name][key]['channel_'+str(channel)]=probe(features[:,cols],w,s,c,source,split['train'],split[key])
    report['elapsed_seconds']=time.time()-start
    (out/'latent_gpu_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('GPU_LATENT_STATS_COMPLETE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
