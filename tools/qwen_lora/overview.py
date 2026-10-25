"""Contact sheet of exact saved validation images, without alignment or width edits."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def main(root,steps=(500,1000,1500,2000),stem='overview'):
    rows=json.loads((root/'evaluations/0002000/metrics.json').read_text())
    canvas=Image.new('RGB',(1300,1790),'#eef2f6');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for i,label in enumerate(('GT',*[str(s) for s in steps])):
        draw.text((i*260+12,10),label,font=font,fill='black')
    writers={'颜真卿':0,'智永':1,'赵佶':2}
    for i,row in enumerate(rows):
        name=f'U{ord(row["character"]):04X}_w{writers[row["writer"]]}.png'
        paths=[root/row['gt']]+[root/'evaluations'/f'{s:07d}'/name for s in steps]
        y=44+i*288
        for col,path in enumerate(paths):
            image=Image.open(path).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
            canvas.paste(image,(col*260,y))
        draw.text((8,y+256),f'U+{ord(row["character"]):04X} / writer {writers[row["writer"]]}',font=font,fill='black')
        for col,step in enumerate(steps,1):
            metrics=json.loads((root/'evaluations'/f'{step:07d}'/'metrics.json').read_text())[i]
            draw.text((col*260+2,y+257),f'S {metrics["ssim"]:.3f}  L {metrics["lpips"]:.3f}  I {metrics["ink_iou"]:.3f}',font=font,fill='black')
    canvas.save(root/(stem+'.png'))
    canvas.save(root/(stem+'.jpg'),quality=65,optimize=True)
    print('OVERVIEW_READY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--steps',type=int,nargs=4,default=[500,1000,1500,2000]);p.add_argument('--stem',default='overview')
    a=p.parse_args();main(a.root,a.steps,a.stem)
