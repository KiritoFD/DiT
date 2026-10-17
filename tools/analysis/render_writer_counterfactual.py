"""Native 256px writer counterfactual results, with per-image diagnostic scores."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eval.in_mem_eval import _get_cache


def main():
    source=ROOT/'assets/writer_counterfactual_20261014'
    result=json.loads((source/'metrics.json').read_text(encoding='utf-8'))
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    conf=json.loads((ROOT/'assets/v85_diagnostics_20261014_audit.json').read_text(encoding='utf-8'))['runs']['v84A']['config']
    args=SimpleNamespace(**conf);args.std_soft_root='exp-std/data/std_soft_glyphs_v1'
    cache=_get_cache('exp-std/csv/eval200_fixed.csv',187,None,'',args)
    out=ROOT/'ANALYSIS/writer_counterfactual_20261014';out.mkdir(parents=True,exist_ok=True)
    font=lambda s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s)
    per=result['per_image']
    for page in range(4):
        im=Image.new('RGB',(3440,1050),'white');d=ImageDraw.Draw(im)
        d.text((30,18),f'v84A 条件反事实 {page+1}/4：固定 STD / 噪声 / 书体，只换十位书家',font=font(32),fill='#192d3e')
        d.text((30,65),'seed0 · CFG1 · Heun50 · EMA · SD-VAE | M：独立 Writer-margin；B：latent Critic Brush-logit；宽度 / 墨量来自原 PNG。',font=font(20),fill='#526574')
        labels=['实际 STD-soft','该字 GT']+[str(manifest['writers'][str(w)]) for w in range(10)]
        for j,label in enumerate(labels):d.text((170+j*270,115),label,font=font(23),fill='#146777')
        for r in range(2):
            group=page*2+r;idx=manifest['selected_eval_indices'][group];y=175+r*400
            rows=per[group*10:group*10+10]
            d.text((20,y+25),f"#{idx}\n{rows[0]['character']}\n{rows[0]['script']}",font=font(24),fill='#192d3e')
            mask=cache['skels_latent'][idx,0].numpy()
            std=Image.fromarray(np.rint((1-mask)*255).astype(np.uint8)).convert('RGB').resize((256,256),Image.Resampling.NEAREST)
            im.paste(std,(170,y))
            gt=Image.open(ROOT/'assets/v85_verified_v84A_200k'/f'gt{idx}.png').convert('RGB')
            im.paste(gt,(440,y))
            for w,rec in enumerate(rows):
                path=source/f'g{group*10+w}.png'
                assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['png_sha256']
                tile=Image.open(path).convert('RGB');assert tile.size==(256,256)
                x=170+(w+2)*270;im.paste(tile,(x,y))
                caption=(f"p50 {rec['p50']:.1f}px · ink {rec['ink']*100:.1f}%\n"
                         f"M {rec['writer_margin']:+.3f} · B {rec['critic_brush_logit']:.1f}\n"
                         f"判对 {'是' if rec['prototype_top1'] else '否'} · GT字类一致 {'是' if rec['char_head_agrees_gt'] else '否'}")
                d.text((x,y+264),caption,font=font(17),fill='#526574')
        d.text((30,990),'80 图 / 8 字；GT 只对应原书家，其他书家没有配对真迹。STD 是实际64px软掩码的最近邻放大，生成图保留256px原像素。',font=font(20),fill='#526574')
        im.save(out/f'poster_{page+1:02d}.png')
    print('COUNTERFACTUAL_POSTERS_OK',out)


if __name__=='__main__':main()
