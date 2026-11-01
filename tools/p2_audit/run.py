"""Fail-fast serial P2 diagnostic queue. Never launches training on a proxy score."""
import argparse
from pathlib import Path
import json
import subprocess
import sys
from .common import save,sha


def poster(root,names):
    from PIL import Image,ImageDraw
    # Include every val100 item; no cherry-picking or AI-generated visual evidence.
    n=len(json.loads((root/'cfg1.0'/'samples.json').read_text()))
    for page,start in enumerate(range(0,n,20)):
        count=min(20,n-start);grid=Image.new('RGB',(160*(len(names)+1),count*160+32),'#eeeeee');draw=ImageDraw.Draw(grid)
        for col,name in enumerate(['GT']+names):draw.text((col*160+4,8),name,fill='black')
        for row,i in enumerate(range(start,start+count)):
            files=[root/'cfg1.0'/'samples'/f'gt{i}.png']+[root/name/'samples'/f'g{i}.png' for name in names]
            for col,file in enumerate(files):grid.paste(Image.open(file).convert('RGB').resize((160,160)),(160*col,32+160*row))
        grid.save(root/f'comparison_page{page}.png')


def swap_poster(root):
    from PIL import Image,ImageDraw
    rows=json.loads((root/'samples.json').read_text());chars=list(dict.fromkeys(r['character'] for r in rows))
    grid=Image.new('RGB',(1600,len(chars)*160+32),'#dddddd');draw=ImageDraw.Draw(grid)
    for w in range(10):draw.text((160*w+4,8),f'Writer row {w}',fill='black')
    for i,r in enumerate(rows):
        col=int(r['calligrapher_id']);row=chars.index(r['character'])
        grid.paste(Image.open(root/'samples'/f'g{i}.png').convert('RGB').resize((160,160)),(160*col,32+160*row))
    grid.save(root/'writer_swap.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--csv',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);state={'status':'running','completed':[]}
    identity=dict(checkpoint_sha256=sha(a.checkpoint),input_csv_sha256=sha(a.csv))
    if (root/'source.json').exists() and json.loads((root/'source.json').read_text())!=identity:
        raise ValueError('Queue output belongs to a different checkpoint or sample set')
    save(root/'source.json',identity)
    def execute(module,args,name):
        cmd=[sys.executable,'-u','-m',module]+args
        state.update(current=name,command=cmd);save(root/'queue.json',state)
        with (root/(name+'.log')).open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(name);save(root/'queue.json',state)
    common=['--checkpoint',a.checkpoint,'--csv',a.csv]
    try:
        for scale in (1.,1.2,1.4,1.6):
            name=f'cfg{scale:.1f}';out=root/name
            if (out/'generation.json').exists() and json.loads((out/'identity.json').read_text())['checkpoint_sha256']!=identity['checkpoint_sha256']:
                raise ValueError('Existing generated images have a different checkpoint identity')
            if not (out/'generation.json').exists():
                execute('tools.p2_audit.generate',common+['--out',str(out),'--scale',str(scale)],name+'_generate')
            args=['--run',str(out)]
            if scale!=1:args+=['--baseline',str(root/'cfg1.0')]
            execute('tools.p2_audit.score',args,name+'_score')
        summaries={f'cfg{s:.1f}':json.loads((root/f'cfg{s:.1f}'/'scores.json').read_text())['summary'] for s in (1.,1.2,1.4,1.6)}
        save(root/'comparison.json',summaries)
        best=max(summaries,key=lambda k:summaries[k]['margin'])
        poster(root,list(summaries))
        b=summaries['cfg1.0'];v=summaries[best]
        promising=(best!='cfg1.0' and v['margin']>b['margin'] and v['lpips']<=b['lpips']+.005
                   and v['ink_iou']>=b['ink_iou']-.005 and v['width_w1']<=b['width_w1']*1.1)
        if promising:
            for seed in (90,91):
                for name in ('cfg1.0',best):
                    tag=f'{name}_seed{seed}';out=root/tag
                    execute('tools.p2_audit.generate',common+['--out',str(out),'--scale',name[3:],'--seed',str(seed)],tag+'_generate')
                    args=['--run',str(out)]
                    if name!='cfg1.0':args+=['--baseline',str(root/f'cfg1.0_seed{seed}')]
                    execute('tools.p2_audit.score',args,tag+'_score')
            for lo,hi in ((0.,.5),(.5,1.)):
                tag=f'{best}_window{lo}-{hi}';out=root/tag
                execute('tools.p2_audit.generate',common+['--out',str(out),'--scale',best[3:],'--window',str(lo),str(hi)],tag+'_generate')
                execute('tools.p2_audit.score',['--run',str(out),'--baseline',str(root/'cfg1.0')],tag+'_score')
        # Counterfactual generation is diagnostic, not conditional accuracy on unsupported pairs.
        execute('tools.p2_audit.generate',common+['--out',str(root/'swap'),'--scale',best[3:],'--swap'],'swap_generate')
        execute('tools.p2_audit.score',['--run',str(root/'swap')],'swap_score')
        swap_poster(root/'swap')
        state.update(status='awaiting_independent_validation',candidate=best,
                     distillation_started=False,reason='Need visual review, multi-seed replication and ink-matched/independent content validation; a single DINO probe cannot authorize distillation')
        save(root/'queue.json',state)
    except Exception as e:
        state.update(status='failed',error=repr(e));save(root/'queue.json',state);raise


if __name__=='__main__':main()
