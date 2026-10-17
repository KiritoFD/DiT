"""Matched, hash-verified v85 diagnostic posters from saved inference PNGs."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eval.style.bank import resolve
from src.eval.style.brush import features_from_image

MODELS=[('v84A','v84A 200k','assets/v85_verified_v84A_200k'),
 ('v85A','v85A 30k','assets/style_eval_src/v85A_step0030000'),
 ('v85C','v85C 30k','assets/style_eval_src/v85C_step0030000'),
 ('v85E15','v85E 15k','assets/style_eval_src/v85E_step0015000'),
 ('v85E30','v85E 30k','assets/style_eval_src/v85E_step0030000'),
 ('control10','Control 10k','assets/v85_verified_control_10k')]

def main():
 out=ROOT/'ANALYSIS/v85_verified_20261014';out.mkdir(parents=True,exist_ok=True)
 audit=json.loads((ROOT/'assets/v85_diagnostics_20261014_audit.json').read_text(encoding='utf-8'))
 rows=list(csv.DictReader(open(ROOT/'exp-std/csv/eval200_fixed.csv',encoding='utf-8-sig')))
 for key,_,rel in MODELS:
  for rec in audit['runs'][key]['per_image']:
   p=ROOT/rel/f"g{rec['idx']}.png"
   assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],(key,rec['idx'])
 slots=defaultdict(list)
 for i,r in enumerate(rows):slots[(r['calligrapher'],r['script'])].append(i)
 ids=[v[0] for _,v in sorted(slots.items())]
 extras=sorted((i for i in range(len(rows)) if i not in ids),key=lambda i:-features_from_image(str(ROOT/'assets/v85_verified_v84A_200k'/f'gt{i}.png')).scalars.get('ink_frac',0))
 ids=(ids+extras)[:32]
 font=lambda s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s)
 sources={}; per=[]
 for page in range(4):
  selected=ids[page*8:(page+1)*8]
  im=Image.new('RGB',(2380,160+len(selected)*315),'white');d=ImageDraw.Draw(im)
  d.text((25,15),f'v85 同源逐字诊断 {page+1}/4 · STD | GT | v84A | A | C | E15 | E30 | Control',font=font(28),fill='#192d3e')
  d.text((25,58),'固定 seed0 / CFG1 / Heun50 / SD-VAE；所有生成 PNG 与 4090 原始文件 SHA256 逐张核对。',font=font(20),fill='#526574')
  heads=['STD 原字形','GT 真迹']+[m[1] for m in MODELS]
  for j,h in enumerate(heads):d.text((160+j*275,105),h,font=font(21),fill='#146777')
  for k,idx in enumerate(selected):
   y=150+k*315;r=rows[idx]
   d.text((15,y+25),f"#{idx} {r['character']}\n{r['calligrapher']}\n{r['script']}",font=font(20),fill='#192d3e')
   std=resolve(r.get('std_path',''),[str(ROOT),str(ROOT/'data'),str(ROOT/'exp-std')])
   assert std,(idx,'missing std')
   paths=[Path(std).resolve(),ROOT/'assets/v85_verified_v84A_200k'/f'gt{idx}.png']+[ROOT/rel/f'g{idx}.png' for _,_,rel in MODELS]
   for j,p in enumerate(paths):
    src=Image.open(p).convert('RGB');assert src.size==(256,256),(p,src.size)
    im.paste(src,(160+j*275,y))
    f=features_from_image(str(p));w=f.width_stats();ink=f.scalars.get('ink_frac',0)
    d.text((160+j*275,y+260),f"w50 {w['p50']:.1f}px · ink {ink*100:.1f}%",font=font(17),fill='#526574')
    sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    per.append({'idx':idx,'model':heads[j],'p10':w['p10'],'p50':w['p50'],'p90':w['p90'],'ink':ink})
  im.save(out/f'poster_{page+1:02d}.png')
 bad=ROOT/'assets/style_eval_src/step0200000'
 v80=ROOT/'assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/eval_samples_ctrl/step0200000/eval200fix'
 matches=sum(hashlib.sha256((bad/f'g{i}.png').read_bytes()).digest()==hashlib.sha256((v80/f'g{i}.png').read_bytes()).digest() for i in range(187))
 assert matches==187
 (out/'manifest.json').write_text(json.dumps({'selection':'first sample per writer-script slot; then largest GT ink fraction to reach32','ids':ids,'source_sha256':sources,'per_image':per,'std_display':'raw std_path, model uses sigma0.65 std-soft recorded in protocol','mislabelled_baseline':{'path':str(bad),'actual_source':str(v80),'sha256_matches':matches,'n':187}},ensure_ascii=False,indent=2),encoding='utf-8')
 print('VERIFIED_1122_PNGS_POSTERS32',out)

if __name__=='__main__':main()
