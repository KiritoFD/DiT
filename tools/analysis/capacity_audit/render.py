"""Render GT / VAE / DiT comparisons and explicitly labelled error proxies."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt
from skimage.morphology import skeletonize
from tools.analysis.capacity_audit.common import structure, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='assets/capacity_audit_20261017')
    p.add_argument('--out',default='docs/experiments/capacity_audit_20261017');a=p.parse_args()
    root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    old=Path('assets/recent_poster_20261017')
    dirs=[Path('eval/gt'),root/'recon',old/'png/v84A',old/'png/l20_200',old/'png48_selected/l24_384']
    names=['GT','VAE posterior mean','12L x512 @200k','20L x384 @200k','24L x384 @130k']
    def load(k,i):
        filename=f'{i}.png' if k<2 else f'g{i}.png'
        return np.asarray(Image.open(dirs[k]/filename).convert('RGB'),dtype=np.float32)/255
    indices=[75,82,158,99,154,178]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    canvas=Image.new('RGB',(1470,2040),'#f6f5f1');overlay=canvas.copy()
    for page,title in [(canvas,'GT → VAE → DiT：局部细节误差分解'),(overlay,'误差代理图：不是自动书法诊断')]:
        d=ImageDraw.Draw(page);d.text((24,15),title,font=font,fill='#173b4b')
        for k,n in enumerate(names):d.text((24+k*290,62),n,font=small,fill='#173b4b')
    stats=[]
    for row,i in enumerate(indices):
        gt=load(0,i);g=gt.mean(-1)<.5
        thin=skeletonize(g)&(distance_transform_edt(g)<=2.5)
        holes=binary_fill_holes(g)&~g
        for k in range(5):
            pred=load(k,i);p=pred.mean(-1)<.5;metrics=structure(pred,gt)
            x=24+k*290;y=100+row*310
            canvas.paste(Image.fromarray((pred*255).astype('uint8')),(x,y))
            colored=pred.copy()
            colored[(p&~binary_dilation(g))]=[0,.65,.8]
            colored[holes&p]=[.85,0,.85]
            colored[thin&~binary_dilation(p)]=[1,.12,.1]
            overlay.paste(Image.fromarray((colored*255).astype('uint8')),(x,y))
            text=f"idx {i} | thin recall {metrics['thin_skeleton_recall']:.3f}"
            for page in (canvas,overlay):ImageDraw.Draw(page).text((x,y+260),text,font=small,fill='#263d43')
            stats.append({'index':i,'model':names[k],**metrics})
    for page in (canvas,overlay):
        d=ImageDraw.Draw(page)
        d.text((24,1968),'红：GT细骨架未覆盖；紫：GT内部白区被填；青：GT附近之外的墨迹。允许风格与笔形差异。',font=small,fill='#67483c')
        d.text((24,1995),'这些只是定位代理；不是缺笔、飞白损失、粘连的自动真值。24L 为较早快照，噪声未统一。',font=small,fill='#67483c')
    canvas.save(out/'vae_generation.png');overlay.save(out/'error_proxies.png')
    write(out/'selected_geometry.json',stats)
    # Independent LPIPS computation on all 187 deterministic reconstructions.
    import torch,lpips
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    net=lpips.LPIPS(net='alex',verbose=False).eval();vals=[]
    for start in range(0,187,8):
        ids=range(start,min(start+8,187))
        pred=torch.from_numpy(np.stack([load(1,i) for i in ids])).permute(0,3,1,2)*2-1
        gt=torch.from_numpy(np.stack([load(0,i) for i in ids])).permute(0,3,1,2)*2-1
        vals.extend(net(pred,gt).flatten().tolist())
    vae=json.loads((root/'vae.json').read_text(encoding='utf-8'))
    summary={k:float(np.mean([r[k] for r in vae['records'] if r[k] is not None]))
             for k in ['ssim','ink_iou','thin_skeleton_recall','ink_ratio']}
    summary.update(n=187,lpips_alex=float(np.mean(vals)))
    write(out/'vae_summary.json',summary);print(summary)
    plots(root,out)


def plots(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for name,model in report['models'].items():
        for split,style in [('train','-'),('val','--')]:
            ax.plot(report['times'],[model['buckets'][f'{split}|t={t}']['mean'] for t in report['times']],
                    style,marker='o',label=f'{name}: {split}')
    ax.set(xlabel='t (network time=1000t)',ylabel='Velocity MSE',title='Fixed GT latents / fixed noise / EMA / no updates')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(out/'fm_buckets.png',dpi=170);plt.close(fig)
    curve_paths=[root/'capacity_audit_20261017_curves/curves.json',root/'curves48/curves.json']
    grouped={}
    for path in curve_paths:
        if not path.exists():continue
        for r in json.loads(path.read_text(encoding='utf-8'))['rows']:
            name=r['model'].replace('l20_384_first','l20_384').replace('l20_384_final','l20_384')
            grouped.setdefault(name,[]).append(r)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,records in grouped.items():
        records=sorted(records,key=lambda r:r['nominal_images'])
        for ax,key in zip(axes,['ssim','lpips']):
            ax.plot([r['nominal_images']/1e6 for r in records],[r[key] for r in records],label=name)
            ax.set(xlabel='Nominal image exposure (millions)',ylabel=key)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Historical curves: batch*step, not a controlled capacity ablation')
    fig.savefig(out/'exposure.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
