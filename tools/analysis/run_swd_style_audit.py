"""One frozen SWD audit: equal glyph mass, exact empirical mixtures, no training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'4')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt,binary_erosion
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.allograph_stats import make_trials,l2
from tools.analysis.run_style_fingerprint_audit import interval
from tools.analysis.swd_style_stats import (projection_directions,train_normalization,mixture_quantiles,
    swd_from_quantiles,summarize_distances,global_permutation_diagnostic)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def plans(data,repeats):
    output={};coverage={};rng=np.random.default_rng(89)
    for protocol in ('test_char','test_source'):
        q=data['splits'][protocol];bank={}
        for slot in range(len(data['slots'])):
            ids=q[data['labels'][q]==slot]
            bank[slot]={c:ids[data['char'][ids]==c] for c in set(data['char'][ids])}
        for k in (1,4,16,32):
            trials,cv=make_trials(data,k,protocol=='test_source',repeats=repeats)
            for t in trials:
                w,v=t['w'],t['v'];aw,bw,av,bv=t['ids'];cs=list(map(str,data['char'][bw]));t['test_chars']=cs
                t['candidate_ids']={w:bw,v:bv}
                for slot in bank:
                    if slot in (w,v) or data['slots'][slot][1]!=data['slots'][w][1]:continue
                    if not set(cs)<=set(bank[slot]):continue
                    ids=[int(rng.choice(bank[slot][c])) for c in cs]
                    if protocol=='test_source' and set(data['source'][ids])&(set(data['source'][aw])|set(data['source'][av])):continue
                    t['candidate_ids'][slot]=ids
                assert not set(data['char'][aw])&set(cs)
                assert list(data['char'][bw])==list(data['char'][bv])
            output[protocol,k]=trials;coverage[f'{protocol}_{k}']=cv
    return output,coverage


def local_tokens(z,gray,size,patch):
    n=len(z);dim=36 if patch else 4;tokens=np.empty((n,64,dim),np.float32);fallback=0
    for start in range(0,n,128):
        zz=z[start:start+128].cuda().float()
        if size!=32:zz=F.adaptive_avg_pool2d(zz,size)
        ink=F.adaptive_avg_pool2d((gray[start:start+128].cuda().float()<140).float(),size)
        mask=F.max_pool2d((ink>=.05).float(),3,1,1).flatten(1).bool().cpu().numpy()
        xx=F.unfold(F.pad(zz,(1,1,1,1),mode='replicate'),3).transpose(1,2) if patch else zz.flatten(2).transpose(1,2)
        xx=xx.cpu().numpy()
        for j in range(len(xx)):
            available=np.flatnonzero(mask[j])
            if not len(available):available=np.arange(size*size);fallback+=1
            rng=np.random.default_rng(89+start+j+size*100000)
            ids=rng.choice(available,64,replace=len(available)<64)
            tokens[start+j]=xx[j,ids]
    return tokens,{'empty_ink_mask_fallbacks':fallback,'samples_per_glyph':64,'foreground':'gray<140; pooled occupancy>=0.05, dilate one grid cell; replicate padding for patches'}


def geometry_tokens(gray):
    n=len(gray);tokens=np.empty((n,64,5),np.float32)
    for i in range(n):
        ink=gray[i,0].numpy()<140
        contour=ink&~binary_erosion(ink)
        signed=(distance_transform_edt(ink)-distance_transform_edt(~ink))/128
        gy,gx=np.gradient(signed)
        f=np.stack((ink,contour,signed,gx,gy)).astype(np.float32)
        pooled=f.reshape(5,32,4,32,4).mean((2,4)).transpose(1,2,0)
        rng=np.random.default_rng(89+i+7000000)
        for region,(y,x) in enumerate(((0,0),(0,16),(16,0),(16,16))):
            values=pooled[y:y+16,x:x+16].reshape(-1,5)
            tokens[i,region*16:(region+1)*16]=values[rng.choice(256,16,replace=False)]
        if i%5000==0:print('GEOMETRY',i,'/',n,flush=True)
    return tokens,{'regions':'fixed 2x2; 16 uniform positions per region, INCLUDING blank geometry','features':['ink_fraction','contour_fraction','signed_distance/128','distance_dx','distance_dy']}


@torch.no_grad()
def project(tokens,train,out,name,regions):
    mean,std=train_normalization(tokens,train);dirs=projection_directions(tokens.shape[-1])
    dest=np.lib.format.open_memmap(out/f'{name}_projected.npy',mode='w+',dtype=np.float16,shape=(len(tokens),regions,128,64//regions))
    for start in range(0,len(tokens),128):
        x=(torch.from_numpy(tokens[start:start+128]).cuda()-torch.from_numpy(mean).cuda())/torch.from_numpy(std).cuda()
        y=x@torch.from_numpy(dirs).cuda()
        dest[start:start+len(x)]=y.reshape(len(x),regions,64//regions,128).permute(0,1,3,2).cpu().numpy().astype(np.float16)
    dest.flush()
    np.savez(out/f'{name}_normalization.npz',mean=mean,std=std,directions=dirs)
    return dest


@torch.no_grad()
def group_distances(q,trials,nslots):
    result=np.full((len(trials),2,nslots),np.nan,np.float32)
    if not len(trials):return result
    aa=np.array([ids for t in trials for ids in (t['ids'][0],t['ids'][2])])
    anchors=torch.cat([mixture_quantiles(q,aa[s:s+16]) for s in range(0,len(aa),16)])
    requests=[(i,int(slot),ids) for i,t in enumerate(trials) for slot,ids in t['candidate_ids'].items()]
    for start in range(0,len(requests),16):
        rr=requests[start:start+16];ids=np.array([r[2] for r in rr]);b=mixture_quantiles(q,ids)
        ii=torch.tensor([r[0]*2 for r in rr],device='cuda')
        dw=swd_from_quantiles(anchors[ii],b).cpu().numpy();dv=swd_from_quantiles(anchors[ii+1],b).cpu().numpy()
        for j,(i,slot,_) in enumerate(rr):result[i,:,slot]=[dw[j],dv[j]]
    del anchors;torch.cuda.empty_cache()
    return result


@torch.no_grad()
def image_distances(q,pairs):
    vals=[]
    for start in range(0,len(pairs),128):
        batch=pairs[start:start+128]
        a=mixture_quantiles(q,batch[:,0,None]);p=mixture_quantiles(q,batch[:,1,None]);n=mixture_quantiles(q,batch[:,2,None])
        vals.append(torch.stack((swd_from_quantiles(a,p),swd_from_quantiles(a,n)),1).cpu().numpy())
    return np.concatenate(vals)


def image_summary(dist,pairs,data):
    margin=dist[:,1]-dist[:,0];correct=(margin>0)+.5*(margin==0)
    used=set();keep=[]
    for i,ids in enumerate(pairs):
        if not used.intersection(ids):used.update(ids);keep.append(i)
    keep=np.array(keep,int)
    return {'by_char':interval(correct,data['char'][pairs[:,0]]),'by_source':interval(correct,data['source'][pairs[:,0]]),
            'image_disjoint':interval(correct[keep],data['char'][pairs[keep,0]]),'mean_margin':float(margin.mean())}


def vector_distances(features,trials,nslots,cosine=True):
    result=np.full((len(trials),2,nslots),np.nan,np.float32)
    for i,t in enumerate(trials):
        anchors=features[np.array([t['ids'][0],t['ids'][2]])].mean(1)
        if cosine:anchors=l2(anchors)
        for slot,ids in t['candidate_ids'].items():
            b=features[ids].mean(0)
            result[i,:,int(slot)]=(1-anchors@l2(b))/2 if cosine else ((anchors-b)**2).mean(1)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.12);torch.manual_seed(89);start=time.time()
    datafile=root/'assets/writer_latent_data_20261016/data.pt'
    data=torch.load(datafile,map_location='cpu',weights_only=False)
    pixelfile=root/'scratch/writer_pixel_d0d1_20261016_01/assets/data/data.pt'
    pixel=torch.load(pixelfile,map_location='cpu',weights_only=False)
    for key in ('labels','char','source'):assert np.array_equal(data[key],pixel[key])
    gray=pixel['gray'];del pixel
    tr=data['splits']['train'];pp,coverage=plans(data,a.repeats);nslots=len(data['slots'])
    (out/'trials.json').write_text(json.dumps({f'{key}_{k}':v for (key,k),v in pp.items()},ensure_ascii=False),encoding='utf-8')
    report={'config':{'seed':89,'directions':128,'patches_per_glyph':64,'repeats_per_writer_pair':a.repeats,'train_only_normalization':True,'combination':'B=0.5*(normalized patch32+patch16); C=0.5*(B+normalized geometry2x2)','scales_fitted_on':'512 random train-only within-script pairs, mean single-image squared SWD','data_sha256':digest(datafile),'pixel_data_sha256':digest(pixelfile)},'coverage':coverage,'methods':{},'elapsed_seconds':0,
        'limitations':['Source directories are proxies, not verified artworks; no permutation removes writer/source confounding.',
                      'Repeated trials reuse images; pair clusters share writers. CIs exploratory.',
                      'Finite 64 samples per glyph and 128 directions; no sweep or heldout selection.',
                      'SWD2 squared is used; numeric R is not directly interchangeable with unsquared W2 R.',
                      'Equal glyph mass. Exact pooled empirical distribution, not mean per-image quantiles.',
                      'Global permutation coverage can collapse for sparse character intersections; reported explicitly.',
                      'This is a custom statistical probe, not a reproduction or proof of a published calligraphy metric.']}
    all_dist={};single={};scales={}
    rng=np.random.default_rng(89);cal=[]
    for _ in range(512):
        i=int(rng.choice(tr));candidates=tr[(data['si'][tr]==data['si'][i])&(data['char'][tr]!=data['char'][i])]
        j=int(rng.choice(candidates));cal.append([i,j,j])
    cal=np.array(cal)
    def record(name,dists,singles):
        result={'sets':{},'single_hard_matched':{}}
        slot_scripts=[v[1] for v in data['slots']]
        for protocol in ('test_char','test_source'):
            result['sets'][protocol]={};cohort={(t['w'],t['v']) for t in pp[protocol,32]}
            for k in (1,4,16,32):
                ts=pp[protocol,k];d=dists[protocol,k]
                ar=np.arange(len(ts));w=np.array([t['w'] for t in ts],int);v=np.array([t['v'] for t in ts],int)
                within=np.stack((d[ar,0,w],d[ar,1,v]),1);between=np.stack((d[ar,0,v],d[ar,1,w]),1)
                entry=summarize_distances(within,between,ts)
                entry['global_writer_permutation']=global_permutation_diagnostic(d,ts,slot_scripts)
                keep=np.array([(t['w'],t['v']) in cohort for t in ts])
                entry['common32_cohort']=summarize_distances(within[keep],between[keep],[t for t,b in zip(ts,keep) if b])
                result['sets'][protocol][str(k)]=entry
            result['single_hard_matched'][protocol]=image_summary(singles[protocol],data['matched_pairs'][protocol],data)
        report['methods'][name]=result;report['elapsed_seconds']=time.time()-start
        report['peak_cuda_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez(out/f'{name}_distances.npz',**{f'{key}_{k}':v for (key,k),v in dists.items()},**{f'single_{key}':v for key,v in singles.items()})
        print('SWD_METHOD_COMPLETE',name,{key:result['sets'][key]['16']['R'] for key in ('test_char','test_source')},'seconds',int(time.time()-start),flush=True)
    for name,size,patch,regions in (('A_latent32',32,False,1),('patch32',32,True,1),('patch16',16,True,1),('geometry2x2',32,False,4)):
        cache=out/f'{name}_projected.npy'
        if cache.exists():q=np.load(cache,mmap_mode='r')
        else:
            tokens,detail=geometry_tokens(gray) if name=='geometry2x2' else local_tokens(data['z'],gray,size,patch)
            report.setdefault('extraction',{})[name]=detail
            q=project(tokens,tr,out,name,regions);del tokens
        scales[name]=float(image_distances(q,cal)[:,0].mean());assert scales[name]>0
        dd={};ss={}
        for key,k in pp:
            dd[key,k]=group_distances(q,pp[key,k],nslots)
            print('SWD_GROUPS',name,key,k,len(pp[key,k]),flush=True)
        for key in ('test_char','test_source'):ss[key]=image_distances(q,data['matched_pairs'][key])
        all_dist[name]=dd;single[name]=ss;report['branch_distance_scales']=scales
        record(name,dd,ss);del q
    for name,terms in (('B_patch_multiscale',[('patch32',.5),('patch16',.5)]),('C_geometry_patch',[('patch32',.25),('patch16',.25),('geometry2x2',.5)])):
        dd={key:sum(weight*all_dist[branch][key]/scales[branch] for branch,weight in terms) for key in pp}
        ss={key:sum(weight*single[branch][key]/scales[branch] for branch,weight in terms) for key in ('test_char','test_source')}
        record(name,dd,ss)
    controls={'Pixel_D1':np.load(root/'assets/d1_bridge_20261016/features.npz')['raw'],
              'Latent_D1':np.load(root/'scratch/writer_latent_d0d1_20261016_01/assets/D1/clean_features.npz')['embedding'],
              'ink_fraction_only':data['stats'][:,:1]}
    for name,x in controls.items():
        dd={key:vector_distances(x,ts,nslots,name!='ink_fraction_only') for key,ts in pp.items()};ss={}
        for key in ('test_char','test_source'):
            pairs=data['matched_pairs'][key];aa,po,ne=x[pairs[:,0]],x[pairs[:,1]],x[pairs[:,2]]
            if name=='ink_fraction_only':ss[key]=np.stack((((aa-po)**2).mean(1),((aa-ne)**2).mean(1)),1)
            else:ss[key]=np.stack(((1-(aa*po).sum(1))/2,(1-(aa*ne).sum(1))/2),1)
        record(name,dd,ss)
    print('SWD_AUDIT_COMPLETE','seconds',int(time.time()-start),'peak_GiB',report['peak_cuda_allocated_GiB'],flush=True)


if __name__=='__main__':main()
