"""Fixed 32-sample pilot using the existing interpreter and task-local Diffusers."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time


ROOT=Path('/home/ds/Workspace/xy/qwen-calli-20261022')
sys.path.insert(0,str(ROOT/'vendor'))


def prompt_for(job):
    if 'prompt' in job:
        return job['prompt']
    if job['mode']=='style_only':
        return (f'请学习全部{len(job["images"])}张参考真迹的书写风格，创作一个新的楷书汉字“{job["character"]}”。'
            '所有图片都是同一位书家写的其他字，不是目标字；目标内容只由本段文字指定。'
            '任务重点是忠实迁移参考真迹的独特书风，而不是生成通用楷体或印刷字体。'
            '仔细综合各图的起笔与收笔、藏锋露锋、横竖粗细对比、提按变化、撇捺形态、钩挑转折、'
            '笔画倾角、部件比例、重心与疏密规律，并将这些共同规律用于新字。'
            '如果参考笔画瘦劲就保持瘦劲，如果厚重就保持厚重，不要把不同书风平均成商业毛笔字体。'
            f'最终只输出一个完整且可辨认的“{job["character"]}”字，不能照搬或拼贴参考字。'
            '黑色墨迹、纯白干净背景，单字居中，四周少量白边；不要纸纹、边框、印章、落款或其他文字。')
    base=(f'生成一张纯白背景、黑色毛笔书法单字“{job["character"]}”。'
          '图1只用于指定汉字内容，不要照抄它的标准字体笔法。输出必须只有这个完整汉字，'
          '不能改成参考图中的其他字，不要落款、印章、边框、纸张纹理或额外文字。')
    if job['writer'] is not None:
        return base+('图2、图3、图4是同一位书家写的其他楷书汉字，只用于迁移书写风格。'
            '结合三张真迹的起收笔、横竖撇捺、转折、粗细变化和结体规律，重新书写图1指定的字。'
            '允许符合参考书风的结体变化，不要拼贴或复制参考图的字符。单字居中，四周保留少量白边。')
    return base+'用自然真实的楷书毛笔笔法书写，单字居中，四周保留少量白边。'


def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--wait-download',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--skip-poster',action='store_true');args=p.parse_args()
    ROOT=args.root
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    state=dict(pid=os.getpid(),phase='waiting_download',completed=0,total=len(manifest['jobs']))
    def save():
        (ROOT/'inference_status.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    save()
    if args.wait_download:
        while True:
            download=json.loads((ROOT/'bootstrap_status.json').read_text())
            if download['phase']=='ready':break
            if download['phase']=='failed':raise RuntimeError(download)
            os.kill(download['pid'],0)  # Fail if the downloader died instead of waiting indefinitely.
            time.sleep(30)
    import torch
    import diffusers
    import transformers
    from diffusers import QwenImage21Pipeline
    from PIL import Image
    torch.set_num_threads(8)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Input changed: '+name)
    state.update(phase='loading',versions=dict(torch=torch.__version__,diffusers=diffusers.__version__,
        transformers=transformers.__version__),offload='model CPU offload, BF16; no quantization',resolution=1024)
    save()
    pipe=QwenImage21Pipeline.from_pretrained(str(ROOT/'model'),dtype=torch.bfloat16,local_files_only=True)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.vae,'enable_tiling'):pipe.vae.enable_tiling()
    rows=[]
    for job in manifest['jobs']:
        path=out/(job['id']+'.png');record=out/(job['id']+'.json')
        if path.exists() or record.exists():raise FileExistsError('Refusing mixed-run output: '+str(path))
        state.update(phase='generating',job=job['id']);save()
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        images=[Image.open(ROOT/name).convert('RGBA') for name in job['images']]
        prompt=prompt_for(job)
        result=pipe(prompt=prompt,image=images,height=1024,width=1024,output_resolution=1024,
            true_cfg_scale=1.,use_kv_cache=True,generator=torch.Generator('cpu').manual_seed(job['seed'])).images[0]
        actual_steps=len(pipe.scheduler.timesteps)
        if actual_steps!=8:raise ValueError('Turbo did not use the released eight-step schedule')
        result.save(path)
        rgba=result.convert('RGBA');white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba)
        white.convert('RGB').resize((256,256),Image.Resampling.LANCZOS).save(out/(job['id']+'_256.png'))
        info=dict(job,prompt=prompt,seconds=time.monotonic()-start,steps=actual_steps,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            generated_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            writer_label=None,verification='unreviewed synthetic candidate; not admitted to training')
        record.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
        rows.append(dict(id=job['id'],character=job['character'],requested_writer=job['writer'],seed=job['seed'],
            character_correct='',reference_style_match='',reject_reason=''))
        with (ROOT/'review.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        state['completed']+=1;save();print('GENERATED',state['completed'],job['id'],info['seconds'],flush=True)
    state['phase']='completed';save()
    if not args.skip_poster:
        subprocess.run([sys.executable,str(Path(__file__).with_name('poster.py')),'--root',str(ROOT)],check=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        path=ROOT/'inference_status.json';state=json.loads(path.read_text())
        state.update(phase='failed',error=repr(error));path.write_text(json.dumps(state,indent=2,ensure_ascii=False))
        raise
