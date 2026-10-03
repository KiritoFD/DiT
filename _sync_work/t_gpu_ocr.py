import csv, os, time
import numpy as np
from PIL import Image
os.chdir("/root/Workspace/xy/DiT")
from rapidocr_onnxruntime import RapidOCR

# 试三种初始化，看哪个真的用 CUDA
for kw in ({"det_use_cuda":True,"cls_use_cuda":True,"rec_use_cuda":True}, {}):
    try:
        ocr = RapidOCR(**kw)
        print(f"  init {kw} OK", flush=True)
        break
    except Exception as e:
        print(f"  init {kw} 失败: {str(e)[:80]}", flush=True)

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))[:200]
def pad(im, p=0.4):
    w,h = im.size; s=int(max(w,h)*(1+p*2))
    c=Image.new("RGB",(s,s),"white"); c.paste(im,((s-w)//2,(s-h)//2)); return c

t0=time.time(); ok=0
for r in rows:
    p=r["image_path"]; src=p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT",p)
    if not os.path.exists(src): continue
    res,_=ocr(np.array(pad(Image.open(src).convert("RGB"))))
    if res and "".join(x[1] for x in res)[:1]==r["character"]: ok+=1
el=time.time()-t0
print(f"  {len(rows)} 张 {el:.1f}s = {len(rows)/el:.1f}/s, 一致率 {ok}/{len(rows)}={ok/len(rows)*100:.0f}%")
