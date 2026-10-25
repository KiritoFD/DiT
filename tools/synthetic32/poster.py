"""Anonymous reference groups and every raw pilot output; no GT target exists."""
import argparse
import json
from pathlib import Path
import random
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    writers=list(manifest['writers']);random.Random(20261022).shuffle(writers)
    key=dict(zip('ABC',writers));(root/'blind_key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2),encoding='utf-8')
    candidates=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf']
    path=next(p for p in candidates if Path(p).exists());font=lambda n:ImageFont.truetype(path,n)
    canvas=Image.new('RGB',(1460,2520),'#eef2f6');draw=ImageDraw.Draw(canvas)
    draw.text((24,16),'Qwen-Image-2.1-Turbo | cross-character style pilot',font=font(27),fill='#233348')
    draw.text((24,58),'32 unverified candidates | 4 shared targets | 2 seeds | no target calligraphy supplied',font=font(19),fill='#233348')
    for j,(group,writer) in enumerate(key.items()):
        x=24+j*475;draw.text((x,98),'Reference group '+group,font=font(21),fill='#233348')
        for k,row in enumerate(manifest['reference_records'][writer]):
            im=Image.open(root/row['file']).convert('RGB').resize((130,130),Image.Resampling.LANCZOS)
            canvas.paste(im,(x+k*140,132))
    names=['Target STD','STD only','Style A','Style B','Style C'];left=100;cw=270
    for j,name in enumerate(names):draw.text((left+j*cw+10,280),name,font=font(22),fill='#233348')
    y=320
    for char in manifest['targets']:
        for seed in manifest['seeds']:
            draw.text((12,y+12),f'U+{ord(char):04X}',font=font(16),fill='#233348')
            draw.text((12,y+42),str(seed)[-4:],font=font(16),fill='#233348')
            jobs={r['writer']:r for r in manifest['jobs'] if r['character']==char and r['seed']==seed}
            std=root/jobs[None]['images'][0]
            paths=[std,*[root/'outputs'/(jobs[w]['id']+'_256.png') for w in (None,*writers)]]
            for j,path in enumerate(paths):
                im=Image.open(path).convert('RGB').resize((252,252),Image.Resampling.LANCZOS)
                canvas.paste(im,(left+j*cw,y))
            y+=265
    draw.text((24,y+8),'No automatic character/style approval. No synthetic data admitted to training.',font=font(18),fill='#233348')
    canvas.crop((0,0,1460,y+52)).save(root/'poster.png')
    print('POSTER_DONE',root/'poster.png',flush=True)


if __name__=='__main__':main()
