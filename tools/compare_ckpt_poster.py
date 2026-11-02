#!/usr/bin/env python3
"""Create *image* comparison posters from existing g{idx}/gt{idx} PNGs.

Each checkpoint is a block: first half of the fixed characters, a full-width
large checkpoint/method/metrics banner, then the same panel's second half.
Never invokes a model or computes metrics from selected showcase characters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_CANDIDATES = (
    'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
)


def resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def get_font(size: int, requested: str | None) -> ImageFont.FreeTypeFont:
    for candidate in ([requested] if requested else []) + list(FONT_CANDIDATES):
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError('No Chinese-capable font. Supply --font PATH.')


def get_index(path: Path, mode: str, filters: dict[str, str] | None = None) -> dict[int, dict[str, str]]:
    if mode not in ('explicit_idx', 'row_number'):
        raise ValueError(f'Invalid index mode: {mode}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if filters:
        rows = [r for r in rows if all(str(r.get(key, "")) == str(expected) for key, expected in filters.items())]
    if not rows:
        raise ValueError(f'Empty sample index CSV: {path}')
    result = {}
    for ordinal, entry in enumerate(rows):
        raw = ordinal if mode == 'row_number' else (entry.get('idx') or entry.get('index') or entry.get('sample_idx'))
        if raw is None:
            raise ValueError(f'Missing idx in {path}; choose row_number explicitly for legacy CSV')
        idx = int(raw)
        if idx in result:
            raise ValueError(f'Duplicate sample index {idx} in {path}; filter CSV to one evaluation panel')
        result[idx] = entry
    return result


def meta(row: dict[str, str], field: str) -> str:
    variants = {
        'char': ('char', 'character', 'unicode'),
        'writer': ('calligrapher', 'writer'),
        'script': ('script', 'script_name'),
        'img_id': ('img_id', 'image_path', 'src_image_path'),
    }
    return next((str(row[k]).strip() for k in variants[field] if row.get(k) is not None and str(row[k]).strip()), '')


def validate_row_keys(master: dict, entry: dict, label: str) -> None:
    for field in ('char', 'writer', 'script'):
        expected, found = meta(master, field), meta(entry, field)
        if not expected or not found or expected != found:
            raise ValueError(f'{label}: {field} mismatch: master={expected!r}, model={found!r}')
    ma, mb = meta(master, 'img_id'), meta(entry, 'img_id')
    if ma and mb and ma != mb:
        raise ValueError(f'{label}: img_id mismatch: {ma!r} vs {mb!r}')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_and_validate(manifest: dict, base: Path):
    ids = manifest.get('sample_ids')
    if not isinstance(ids, list) or len(ids) not in (10, 20) or any(type(x) is not int or x < 0 for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('sample_ids must be exactly 10 or 20 distinct non-negative integers')
    protocol = manifest.get('protocol_id')
    if not protocol:
        raise ValueError('protocol_id is required')
    main_csv = resolve(base, manifest['sample_index_csv'])
    main_idx = get_index(main_csv, manifest.get('sample_index_mode', 'explicit_idx'), manifest.get('sample_index_filter'))
    for idx in ids:
        if idx not in main_idx:
            raise ValueError(f'Missing master idx={idx} in {main_csv}')
        for field in ('char', 'writer', 'script'):
            if not meta(main_idx[idx], field):
                raise ValueError(f'Master idx={idx} missing {field}')
    rows = manifest.get('rows')
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError('rows must contain GT/STD and at least one checkpoint')
    result = []
    for raw in rows:
        name = str(raw.get('name', '')).strip()
        method = str(raw.get('method', '')).strip()
        if not name or not method or not raw.get('image_pattern'):
            raise ValueError('Each row needs name, method and image_pattern')
        if raw.get('protocol_id') != protocol:
            raise ValueError(f'{name}: protocol_id mismatch')
        pattern = raw['image_pattern']
        if '{idx}' not in pattern:
            raise ValueError(f'{name}: image_pattern must contain {{idx}}; never glob-sort PNGs')
        metrics = raw.get('metrics', {})
        if not isinstance(metrics, dict):
            raise ValueError(f'{name}: metrics must be a {{name:value}} mapping')
        if metrics:
            if not raw.get('metrics_source') or not raw.get('metrics_protocol_id'):
                raise ValueError(f'{name}: metrics require metrics_source and metrics_protocol_id')
            if raw['metrics_protocol_id'] != protocol:
                raise ValueError(f'{name}: metric protocol is not the same panel')
            if not resolve(base, raw['metrics_source']).is_file():
                raise FileNotFoundError(f'{name}: metrics source not found: {raw["metrics_source"]}')
        csv_path = resolve(base, raw.get('sample_index_csv', manifest['sample_index_csv']))
        row_idx = get_index(csv_path, raw.get('sample_index_mode', manifest.get('sample_index_mode', 'explicit_idx')), raw.get('sample_index_filter', manifest.get('sample_index_filter')))
        match_by = raw.get('match_by', manifest.get('match_by', 'idx'))
        if match_by not in ('idx', 'img_id'):
            raise ValueError(f'{name}: unknown match_by {match_by}')
        if match_by == 'img_id':
            lookup = {}
            for k, value in row_idx.items():
                key = meta(value, 'img_id')
                if not key or key in lookup:
                    raise ValueError(f'{name}: missing or duplicate img_id {key!r} in {csv_path}')
                lookup[key] = k
        images = []
        for source_idx in ids:
            entry = main_idx[source_idx]
            if match_by == 'idx':
                other_idx = source_idx
            else:
                key = meta(entry, 'img_id')
                if not key or key not in lookup:
                    raise ValueError(f'{name}: unmatched master img_id {key!r}')
                other_idx = lookup[key]
            if other_idx not in row_idx:
                raise ValueError(f'{name}: unmatched idx {source_idx}/{other_idx}')
            validate_row_keys(entry, row_idx[other_idx], f'{name} source idx={source_idx}')
            img_path = resolve(base, pattern.format(idx=other_idx))
            if not img_path.is_file():
                raise FileNotFoundError(f'{name}: missing idx={source_idx} (model_idx={other_idx}): {img_path}')
            with Image.open(img_path) as im:
                im.verify()
            images.append({'master_idx': source_idx, 'model_idx': other_idx, 'path': str(img_path), 'sha256': digest(img_path)})
        result.append({'name': name, 'method': method, 'metrics': metrics,
                       'metrics_source': raw.get('metrics_source'), 'csv': str(csv_path),
                       'images': images})
    return main_csv, main_idx, result


def fitted_text(draw, text: str, width: int, size: int, font_path: str | None, *, minimum: int = 18):
    size = max(minimum, size)
    while size > minimum:
        f = get_font(size, font_path)
        if draw.textlength(text, font=f) <= width:
            return f
        size -= 2
    f = get_font(minimum, font_path)
    if draw.textlength(text, font=f) > width:
        raise ValueError(f'Text too long for poster at font={minimum}: {text[:90]!r}; shorten manifest label')
    return f


def make_poster(manifest: dict, base: Path, output: Path, font_path: str | None, cell: int) -> Path:
    if not 120 <= cell <= 400:
        raise ValueError('--cell must be in [120, 400]')
    index_path, index, groups = read_and_validate(manifest, base)
    ids = manifest['sample_ids']
    cols = len(ids) // 2
    gap, outer, strip, banner_h, group_gap = 9, 26, 37, 200, 23
    width = outer * 2 + cols * cell + (cols - 1) * gap
    top_h, foot_h = 145, 66
    each_h = 2 * (cell + strip) + banner_h + group_gap
    height = top_h + len(groups) * each_h + foot_h
    bg, fg, navy, accent = '#EDF2F6', '#142234', '#162D46', '#51D9C6'
    canvas = Image.new('RGB', (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    title = str(manifest.get('title', 'Calligraphy checkpoint comparison'))
    draw.text((outer, 14), title, fill=fg, font=fitted_text(draw, title, width-2*outer, 40, font_path, minimum=24))
    draw.text((outer, 78), f'Protocol: {manifest["protocol_id"]}   |   Same {len(ids)} samples / seed / decoder / sampling settings', fill='#425368', font=fitted_text(draw, f'Protocol: {manifest["protocol_id"]}   |   Same {len(ids)} samples / seed / decoder / sampling settings', width-2*outer, 19, font_path, minimum=12))
    draw.text((outer, 110), '每组：上排样本 + 中央大字 Checkpoint/方法/全量指标 + 下排样本；原始PNG等比缩放，无重绘', fill='#526273', font=fitted_text(draw, '每组：上排样本 + 中央大字 Checkpoint/方法/全量指标 + 下排样本；原始PNG等比缩放，无重绘', width-2*outer, 19, font_path, minimum=12))
    for row_i, row in enumerate(groups):
        y0 = top_h + row_i * each_h
        for half in range(2):
            strip_y = y0 if half == 0 else y0 + (cell + strip) + banner_h
            for col in range(cols):
                n = half * cols + col
                source_idx = ids[n]
                x = outer + col * (cell + gap)
                m = index[source_idx]
                draw.rounded_rectangle((x,strip_y,x+cell,strip_y+strip+cell),radius=7,fill='white',outline='#D4DFE7',width=1)
                label = f'{meta(m,"char")} / {meta(m,"writer")}·{meta(m,"script")} / #{source_idx}'
                draw.text((x+6,strip_y+4),label,fill='#34485A',font=fitted_text(draw,label,cell-12,18,font_path,minimum=10))
                img = Path(row['images'][n]['path'])
                with Image.open(img) as im:
                    tile = im.convert('RGB')
                    tile.thumbnail((cell-8,cell-8),Image.Resampling.LANCZOS)
                    canvas.paste(tile,(x+(cell-tile.width)//2,strip_y+strip+(cell-tile.height)//2))
        y = y0 + cell + strip
        draw.rounded_rectangle((outer,y, width-outer,y+banner_h),radius=8,fill=navy)
        name = row['name']
        draw.text((outer+22,y+6),name,fill='#FFFFFF',font=fitted_text(draw,name,width-2*outer-44,52,font_path,minimum=32))
        method = row['method']
        draw.text((outer+24,y+72),method,fill='#C9DFEC',font=fitted_text(draw,method,width-2*outer-48,27,font_path,minimum=17))
        pairs=[f'{k}: {v}' for k,v in row['metrics'].items()]
        metrics = '    |    '.join(pairs) if pairs else ('GT / reference, no generated-score' if row_i==0 else 'Metrics not yet available (no estimate)')
        # Reflow all metrics instead of cutting them off (old poster truncated at 48 chars).
        metric_lines, current = [], ''
        for part in (pairs or [metrics]):
            candidate = part if not current else current + '    |    ' + part
            if current and draw.textlength(candidate, font=get_font(24, font_path)) > width-2*outer-48:
                metric_lines.append(current)
                current = part
            else:
                current = candidate
        if current:
            metric_lines.append(current)
        if len(metric_lines) > 3:
            raise ValueError(f'{name}: too many metrics to display legibly; use at most ~6 concise metrics')
        for metric_i, metric_line in enumerate(metric_lines):
            draw.text((outer+24, y+110+30*metric_i), metric_line, fill=accent,
                      font=fitted_text(draw,metric_line,width-2*outer-48,24,font_path,minimum=16))
    note = 'Full-set metrics only, not average over selected showcase characters. Sample identities and original PNG SHA256: .selection.json'
    draw.text((outer,height-foot_h+14),note,fill='#425368',font=fitted_text(draw,note,width-2*outer,17,font_path,minimum=11))
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output)
    selection = {'manifest_id':manifest.get('manifest_id'), 'title':title,
                 'protocol_id':manifest['protocol_id'],'sample_index_csv':str(index_path),
                 'sample_index_sha256':digest(index_path),'sample_ids':ids,
                 'samples':[{'idx':i,'char':meta(index[i],'char'),'writer':meta(index[i],'writer'),
                            'script':meta(index[i],'script'),'img_id':meta(index[i],'img_id')} for i in ids],
                 'rows':groups,'poster_sha256':digest(output),'output':str(output)}
    frozen=output.with_suffix('.selection.json')
    frozen.write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{output} ({width}×{height}), frozen={frozen}')
    return output


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,help='JSON manifest, real per-run evaluation PNGs')
    ap.add_argument('--out',required=True,help='Output poster PNG')
    ap.add_argument('--root',default=str(ROOT),help='Repository/asset base root')
    ap.add_argument('--font',help='Local Chinese TTF/TTC (not bundled)')
    ap.add_argument('--cell',type=int,default=194,help='Image cell pixel size [120,400]')
    a=ap.parse_args()
    spec=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    make_poster(spec,Path(a.root).resolve(),Path(a.out),a.font,a.cell)


if __name__=='__main__':
    main()
