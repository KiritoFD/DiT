import os
import sys

BASE_DIR = "/root/Workspace/xy/DiT"
if os.path.exists(BASE_DIR):
    os.chdir(BASE_DIR)

import torch
import json
from src.eval.model_io import build_model_from_args
MANIFEST = os.path.join(BASE_DIR, "exp_milestones/milestones_manifest.json")

print("=== 测试各阶段 Checkpoint 模型加载与结构对齐 ===")
with open(MANIFEST, "r", encoding="utf-8") as f:
    milestones = json.load(f)

for m in milestones:
    stage_id = m["stage_id"]
    ckpt_path = m["ckpt"]
    cfg_path = m["config"]
    resolved_cfg_path = m["resolved_cfg"]
    
    print(f"\n[{stage_id}] 正在测试 {m['name']} ...")
    if not os.path.exists(ckpt_path):
        print(f"  ❌ 权重不存在: {ckpt_path}")
        continue
    
    # 尝试读取 config
    cfg = {}
    if os.path.exists(resolved_cfg_path):
        with open(resolved_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    elif os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
    # 加载 ckpt 头部查看 keys
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"  ✓ 权重文件载入成功，根键: {list(ckpt.keys())[:6]}")
        
        # 如果 ckpt 里带 args，以 ckpt args 优先覆盖
        ckpt_args = ckpt.get("args", {})
        if ckpt_args is not None:
            if hasattr(ckpt_args, "__dict__"):
                arg_dict = vars(ckpt_args)
            elif isinstance(ckpt_args, dict):
                arg_dict = ckpt_args
            else:
                arg_dict = {}
            for k, v in arg_dict.items():
                if v is not None:
                    cfg[k] = v
                    
        # 尝试实例化模型
        model = build_model_from_args(cfg, device="cpu")
        print(f"  ✓ 模型构建成功: {model.__class__.__name__} ({m['arch']})")
        
        # 提取 weights
        sd = ckpt.get("ema") or ckpt.get("model") or ckpt
        # 移除可能的前缀
        new_sd = {}
        for k, v in sd.items():
            clean_k = k.replace("_orig_mod.", "").replace("module.", "")
            new_sd[clean_k] = v
            
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"  ✓ 权重加载成功! missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        if len(missing) > 0 and len(missing) < 10:
            print(f"    Missing sample: {missing}")
        if len(unexpected) > 0 and len(unexpected) < 10:
            print(f"    Unexpected sample: {unexpected}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
