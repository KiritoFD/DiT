"""Fixed-canvas review sheets: geometry flags are proposals, never proof of dirt."""
import argparse
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

from .common import read_json,write_json,key
from .build import conflicts


def render(args):
    scan = Path(args.scan);records = read_json(scan/'images.json')
    root = Path(args.root or read_json(scan/'manifest.json')['root'])
    bad = conflicts(records)
    wanted = {key(p) for p in args.paths.split(',')} if args.paths else None
    chosen = [r for r in records if (r['image_path'] in wanted if wanted is not None else
        r['flags'] or r['hard_issues'] or r['duplicate_group'] in bad)]
    chosen.sort(key=lambda r:(not bool(r['bar_boxes'] if 'bar_boxes' in r else False),r['image_path']))
    if args.limit:chosen = chosen[:args.limit]
    output = Path(args.output);output.mkdir(parents=True,exist_ok=False)
    font = ImageFont.truetype(args.font,14)
    images = []
    cols,per_page,tile_w,tile_h = 8,48,176,220
    for start in range(0,len(chosen),per_page):
        chunk = chosen[start:start+per_page]
        canvas = Image.new('RGB',(cols*tile_w,36+((len(chunk)+cols-1)//cols)*tile_h),'#eeeeee')
        draw = ImageDraw.Draw(canvas)
        draw.text((8,8),'仅为质量候选：粗笔、飞白、独立长竖不自动视为脏污',font=font,fill='black')
        for i,r in enumerate(chunk):
            x,y=(i%cols)*tile_w,36+(i//cols)*tile_h
            path = root/r['image_path']
            if not r.get('error'):
                with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((168,168)),(x+4,y))
            meta = r['metadata'][0]
            caption = f'{Path(path).stem} {meta["calligrapher"]} {meta["script"]} {meta["character"]}'
            draw.text((x+3,y+170),caption,font=font,fill='black')
            reasons = r['flags']+r['hard_issues']
            if r['duplicate_group'] in bad:reasons.append('label conflict')
            caption = ','.join(reasons)
            draw.text((x+3,y+188),caption[:22],font=font,fill='#a00000')
            draw.text((x+3,y+204),caption[22:44],font=font,fill='#a00000')
            images.append(dict(page=start//per_page+1,cell=i,image_path=r['image_path'],file_sha256=r.get('file_sha256'),
                reasons=reasons,metadata=r['metadata']))
        canvas.save(output/f'page{start//per_page+1:03d}.jpg',quality=92)
    write_json(output/'index.json',images)
    print(f'REVIEW {len(chosen)} images, {(len(chosen)+per_page-1)//per_page} pages -> {output}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scan','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--font')
    p.add_argument('--root');p.add_argument('--paths');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--decision-action',choices=['keep','exclude'])
    p.add_argument('--reason');p.add_argument('--base-decisions')
    a = p.parse_args(argv)
    if a.decision_action:
        if not a.paths or not a.reason:p.error('Decisions need explicit --paths and --reason')
        dest = Path(a.output)
        if dest.exists():raise FileExistsError(dest)
        index = {r['image_path']:r for r in read_json(Path(a.scan)/'images.json')}
        decisions = read_json(a.base_decisions) if a.base_decisions else {}
        for path in a.paths.split(','):
            path = key(path)
            if path in decisions and decisions[path]['action'] != a.decision_action:
                raise ValueError('Conflicting existing human decision; resolve explicitly in a new manifest')
            decisions[path] = dict(action=a.decision_action,file_sha256=index[path]['file_sha256'],reason=a.reason)
        write_json(dest,decisions)
        print(f'DECISIONS {len(decisions)} -> {dest}')
    else:
        if not a.font:p.error('Rendering needs --font')
        render(a)


if __name__ == '__main__':main()
