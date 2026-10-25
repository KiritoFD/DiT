"""Read-only, fixed-canvas comparison posters with CSV-key sample matching."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .calibrate import digest, index_rows, read_image, read_rows, sample_key, write_json
from .geometry import Settings, measure


def load_font(path, size):
    candidates = [path] if path else [
        'tools/fonts/SimHei.ttf', 'C:/Windows/Fonts/msyh.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError('No CJK font; specify --font')


def wrap(draw, text, font, width):
    lines, line = [], ''
    for char in text:
        if char == '\n' or (line and draw.textlength(line+char, font=font) > width):
            lines.append(line)
            line = '' if char == '\n' else char
        else:
            line += char
    return '\n'.join(lines+[line])


def render(config_path, output, font_path=None, rows_per_page=8, cell=192):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    anchor = read_rows(config['index_csv'])
    index_rows(anchor)
    settings = Settings(threshold=config.get('threshold', .5), seed=config.get('seed', 0))
    size = int(config.get('image_size', 256))
    if rows_per_page < 1 or cell < 128:
        raise ValueError('rows_per_page >=1 and cell >=128 are required')
    columns = []
    for source in config['columns']:
        col = source.copy()
        if col['kind'] not in ('gt','std','model'):
            raise ValueError(col['kind'])
        col['rows'] = read_rows(col.get('csv', config['index_csv']))
        col['index'] = index_rows(col['rows'])
        col['dir'] = Path(col['dir'])
        if col['kind'] == 'model':
            if not all(k in col for k in ('step','state','cfg','architecture')):
                raise ValueError('Model columns require step/state/cfg/architecture; do not infer from a label')
            if col['state'] not in ('EMA','RAW','UNKNOWN'):
                raise ValueError('State must be EMA, RAW or UNKNOWN')
        calibration = col['dir']/'run.json'
        if calibration.exists():
            run = json.loads(calibration.read_text(encoding='utf-8'))
            if run.get('schema') == 'glyph-calibration-v1':
                if run['status'] != 'complete':
                    raise ValueError(f'Incomplete calibration: {calibration}')
                if run['csv_sha256'] != digest(col.get('csv', config['index_csv'])):
                    raise ValueError('Calibration CSV identity differs from poster column indexing')
                p = run['protocol']
                if (p['image_size'],p['settings']['threshold'],p['settings']['seed']) != (size,settings.threshold,settings.seed):
                    raise ValueError('Calibration measurement protocol differs from poster')
                col['calibration'] = run
                col['calibration_records'] = {r['idx']:r for r in json.loads((col['dir']/'records.json').read_text(encoding='utf-8'))}
        columns.append(col)
    if not columns:
        raise ValueError('No columns')
    selection = config.get('selection', {'policy':'all'})
    if 'indices' in selection:
        indices = list(selection['indices'])
    elif selection.get('policy') == 'intersection':
        indices = [i for i,r in enumerate(anchor) if all(sample_key(r) in c['index'] for c in columns)]
    elif selection.get('policy') == 'all':
        indices = list(range(len(anchor)))
    else:
        raise ValueError('Selection requires indices, all, or explicit intersection')
    if not indices or len(set(indices)) != len(indices) or any(not isinstance(i,int) or i < 0 or i >= len(anchor) for i in indices):
        raise ValueError('Empty, duplicate or invalid row selection')
    records = []
    images = {}
    for idx in indices:
        row = anchor[idx]
        for j, col in enumerate(columns):
            mapped = col['index'][sample_key(row)]
            other = col['rows'][mapped]
            for field in ('calligrapher','script'):
                if row[field] != other[field]:
                    raise ValueError(f'Sample metadata mismatch: {sample_key(row)} / {field}')
            path = col['dir']/f'g{mapped}.png'
            calibration_record = col.get('calibration_records', {}).get(mapped)
            if 'calibration' in col and (not calibration_record or calibration_record['output_sha256'] != digest(path)):
                raise ValueError(f'Calibrated image missing from records or changed after calibration: {path}')
            if col['kind'] == 'std':
                with Image.open(path) as im:
                    source_size = im.size
                    # STD is an explicitly labeled condition thumbnail, not a measured generated glyph.
                    image = np.asarray(im.convert('RGB').resize((size,size), Image.Resampling.BILINEAR))
                stats = None
            else:
                image = read_image(path, size)
                source_size = (size,size)
                stats = measure(image, settings)[0]
            images[idx,j] = Image.fromarray(image)
            records.append(dict(anchor_idx=idx, mapped_idx=mapped, sample_key=sample_key(row),
                                character=row.get('character',''), column=j, label=col['label'],
                                source=str(path.resolve()), sha256=digest(path), source_size=source_size,
                                stats=stats, calibration=col.get('calibration_records',{}).get(mapped)))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    label_w, gap, top, footer = 130, 8, 188, 92
    title_font, font, small = load_font(font_path, 24), load_font(font_path, 17), load_font(font_path, 14)
    row_height = cell+footer+gap
    lookup = {(r['anchor_idx'],r['column']):r for r in records}
    outputs = []
    for page, start in enumerate(range(0,len(indices),rows_per_page), 1):
        subset = indices[start:start+rows_per_page]
        width = label_w+len(columns)*(cell+gap)+gap
        canvas = Image.new('RGB', (width, top+len(subset)*row_height+32), 'white')
        draw = ImageDraw.Draw(canvas)
        title = wrap(draw, config.get('title','固定尺度模型对照'), title_font, width-24)
        if title.count('\n') > 1:
            raise ValueError('Title too long; use a concise neutral title')
        draw.multiline_text((12,8), title, font=title_font, fill='black')
        draw.text((12,68), f'第{page}页 | 本页{len(subset)}字 / 选定{len(indices)}字 | 完整画布等比例显示，不逐格裁切或缩放', font=small, fill='#444444')
        for j,col in enumerate(columns):
            x = label_w+j*(cell+gap)
            text = col['label']
            if col['kind'] == 'model':
                state = '未核验' if col['state'] == 'UNKNOWN' else col['state']
                cfg = '未知' if col['cfg'] is None else str(col['cfg'])
                text += f'\n{col["architecture"]} {col["step"]/1000:g}k\n{state} CFG {cfg}'
            if 'calibration' in col:
                run = col['calibration']
                text += '\n'+('GT-ORACLE 仅诊断' if run['oracle'] else f'校准 {run["mode"]}')
            text = wrap(draw,text,font,cell-8)
            if text.count('\n') > 3:
                raise ValueError(f'Column header too long: {col["label"]}')
            draw.multiline_text((x,96),text,font=font,fill='black',spacing=2)
        for k,idx in enumerate(subset):
            y = top+k*row_height
            row = anchor[idx]
            draw.multiline_text((8,y+12),f'{idx} {row.get("character", "")}\n{row["calligrapher"]}\n{row["script"]}',font=font,fill='black',spacing=8)
            for j,col in enumerate(columns):
                x = label_w+j*(cell+gap)
                # Same source canvas size and same resize factor for every glyph. Never recenter/crop.
                canvas.paste(images[idx,j].resize((cell,cell),Image.Resampling.LANCZOS),(x,y))
                rec = lookup[idx,j]
                if rec['stats'] is None:
                    caption = f'STD {rec["source_size"][0]}→{size} 仅显示'
                else:
                    s = rec['stats']
                    caption = f'P50 {s["p50"]:.1f}px\nP10/90 {s["p10"]:.1f}/{s["p90"]:.1f}\n墨 {s["ink_area"]} 块 {s["components"]} 孔 {s["holes"]}'
                    if rec['calibration']:
                        caption += '\n'+('已调整' if rec['calibration']['changed'] else '原图回退/无变化')
                caption = wrap(draw, caption, small, cell-8)
                if draw.multiline_textbbox((0,0), caption, font=small, spacing=2)[3] > footer-8:
                    raise ValueError('Metric caption exceeds cell; increase --cell')
                draw.multiline_text((x,y+cell+4),caption,font=small,fill='#444444',spacing=2)
                draw.rectangle((x,y,x+cell,y+cell+footer),outline='#cccccc',width=1)
        path = destination/f'page{page:02d}.png'
        canvas.save(path)
        outputs.append(dict(path=path.name,sha256=digest(path),indices=subset))
    serial_columns = [{k:v for k,v in c.items() if k not in ('rows','index','calibration_records')}
                      for c in columns]
    for c in serial_columns:
        c['dir'] = str(c['dir'].resolve())
    write_json(destination/'manifest.json',dict(schema='glyph-poster-v1', title=config.get('title'),
               config_sha256=digest(config_path), index_csv_sha256=digest(config['index_csv']),
               settings=asdict(settings),image_size=size,cell_pixels=cell,
               display='entire_canvas_uniform_scale_no_recrop', selection=selection,
               anchor_n=len(anchor),selected_n=len(indices),columns=serial_columns,
               pages=outputs,images=records))
    print(f'Poster: {len(indices)} rows, {len(columns)} columns, {len(outputs)} pages -> {destination}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--font')
    parser.add_argument('--rows-per-page', type=int, default=8)
    parser.add_argument('--cell', type=int, default=192)
    a = parser.parse_args(argv)
    render(a.config,a.output,a.font,a.rows_per_page,a.cell)


if __name__ == '__main__':
    main()
