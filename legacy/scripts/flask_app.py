# -*- coding: utf-8 -*-
"""
书法 3-条件生成 · 简单 Flask 前端
输入汉字 + 选书法家/字体 -> DDIM 生成书法图。
用法:
  /opt/conda/bin/python flask_app.py --port 5000 --host 0.0.0.0
"""
import os, sys, io, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("XFORMERS_DISABLED", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from flask import Flask, request, send_file, render_template_string
from flask_cors import CORS

from gradio_app import Sampler  # 复用采样核心（不依赖 gradio）

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>书法生成</title>
<style>
 body{font-family:'Microsoft YaHei',sans-serif;background:#14161c;color:#e6e9ef;max-width:820px;margin:30px auto;padding:0 16px}
 h1{font-size:22px} .card{background:#1c1f28;border:1px solid #2c3140;border-radius:12px;padding:18px;margin-bottom:14px}
 label{display:block;margin:10px 0 4px;font-size:13px;color:#9aa3b5}
 input,select{width:100%;padding:8px;background:#0f1117;color:#e6e9ef;border:1px solid #333a4d;border-radius:8px;font-size:15px}
 .row{display:flex;gap:12px} .row>div{flex:1}
 input[type=range]{width:100%;padding:0} .hint{font-size:12px;color:#6c768a;margin-top:4px}
 button{margin-top:16px;width:100%;padding:12px;background:#3b82f6;color:#fff;border:0;border-radius:10px;font-size:16px;cursor:pointer}
 button:disabled{opacity:.5}
 img#out{max-width:100%;border-radius:8px;margin-top:14px;background:#000;display:none}
 #wait{display:none;color:#f59e0b;margin-top:10px}
 select#callig{height:160px}
</style></head><body>
<h1>✒️ 书法生成 · DiT-3Cond-XL + LoRA</h1>
<div class="card">
  <label>汉字（须在 7765 字训练集中）</label>
  <input id="char" value="永" maxlength="1">
  <div id="charhint" class="hint"></div>
  <div class="row">
    <div><label>书法家</label><select id="callig" size="6">{{ CALLIGS }}</select></div>
    <div><label>字体</label><select id="script">{{ SCRIPTS }}</select></div>
  </div>
  <div class="row">
    <div><label>CFG 强度 <span id="cfgv">4.0</span></label>
      <input type="range" id="cfg" min="1" max="10" step="0.5" value="4" oninput="document.getElementById('cfgv').textContent=this.value"></div>
    <div><label>DDIM 步数 <span id="stepv">50</span></label>
      <input type="range" id="steps" min="10" max="100" step="5" value="50" oninput="document.getElementById('stepv').textContent=this.value"></div>
    <div><label>种子 <span id="seedv">0</span></label>
      <input type="range" id="seed" min="0" max="9999" value="0" oninput="document.getElementById('seedv').textContent=this.value"></div>
  </div>
  <button id="go" onclick="gen()">生 成</button>
  <div id="wait">⏳ 生成中…（含 50 步 DDIM 采样，单张约几秒）</div>
  <img id="out">
</div>
<script>
 fetch('__CHARS__').then(r=>r.json()).then(ids=>{
  document.getElementById('char').addEventListener('input',e=>{
    const c=e.target.value.trim();
    const h=document.getElementById('charhint');
    if(!c){h.textContent='';return}
    h.textContent = ids[c]!==undefined ? ('✓ 该字存在 (id='+ids[c]+')') : '⚠ 该字不在 7765 字训练集中';
  });
 });
 async function gen(){
  const btn=document.getElementById('go');btn.disabled=true;
  document.getElementById('wait').style.display='block';
  const body=new URLSearchParams({
    character:document.getElementById('char').value.trim(),
    calligrapher:document.getElementById('callig').value,
    script:document.getElementById('script').value,
    cfg:document.getElementById('cfg').value,
    steps:document.getElementById('steps').value,
    seed:document.getElementById('seed').value
  });
  try{
    const r=await fetch('/generate',{method:'POST',body});
    if(!r.ok){throw new Error((await r.text()).slice(0,200))}
    const blob=await r.blob();
    const img=document.getElementById('out');
    img.src=URL.createObjectURL(blob);img.style.display='block';
  }catch(e){alert('生成失败: '+e.message)}
  document.getElementById('wait').style.display='none';
  btn.disabled=false;
 }
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="DiT-3Cond-XL/2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use-lora", type=str, default="1")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-target", default="all")
    ap.add_argument("--pretrained", default="pretrained_models/DiT-XL-2-256x256.pt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    sam = Sampler(model_name=args.model_name, ckpt_path=args.ckpt,
                  pretrained=args.pretrained,
                  use_lora=args.use_lora in ("1", "true", "True"),
                  lora_r=args.lora_r, lora_target=args.lora_target)

    app = Flask(__name__)
    CORS(app)

    # 注入下拉选项
    calligs = "".join(f'<option value="{c}">{c}</option>' for c in sam.calligraphers)
    scripts = "".join(f'<option value="{s}">{s}</option>' for s in sam.scripts)
    page = HTML.replace("{{ CALLIGS }}", calligs).replace("{{ SCRIPTS }}", scripts).replace(
        "__CHARS__", "/chars")

    @app.get("/")
    def index():
        return render_template_string(page)

    @app.get("/chars")
    def chars():
        return json.dumps(sam.name_to_id["character"], ensure_ascii=False)

    @app.post("/generate")
    def generate():
        character = request.form.get("character", "").strip()
        callig = request.form.get("calligrapher")
        script = request.form.get("script")
        cfg = float(request.form.get("cfg", 4.0))
        steps = int(request.form.get("steps", 50))
        seed = int(request.form.get("seed", 0))
        try:
            img = sam.generate(character, callig, script, cfg_scale=cfg,
                               num_steps=steps, seed=seed)
        except ValueError as e:
            from flask import Response
            return Response(str(e), status=400)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    print(f"Flask 前端启动: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
