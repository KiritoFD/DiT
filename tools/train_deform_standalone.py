#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单独训练 DeformSkel —— 不碰扩散模型，几分钟出结论。

## 为什么能单独训
形变模块是一个小网络: (g_std, 书家风格) -> g'。
目标**现成且稠密**: g' 应逼近"该书家写的那个字"的 GT 骨架（`shards_aux_skel3`，
实测逐样本比值 0.996，宽度 ~3px 最接近 std 的 ~4px）。就是一个监督回归任务。

## 三个判据
  ① **闭合率** = 1 − MSE(g',g_gt) / MSE(g_std,g_gt)
     >60% 说明形变确实把 g_std 拉近该书家的写法
  ② **style-follow（关键）**：同一个字，对每个书家 k 生成 g'(c,k)，
     再看 g'(c,k) 是否比 g'(c,k') 更接近 **k 自己的** g_gt(c,k)。
     这一项直接回答"同一个 skel 是否对不同书家产出对应 skel"。
  ③ **offset 幅值**（含风格底图那一份）: ≈0 = 退化成恒等

## ★ v2 的两处修正（v1 的问题）
  · batch 太小(512) -> 改成吃满显存
  · **同字分组采样**：每个 batch 由若干"字"组成，每个字取多个书家 ->
    同一个 g_std 在同批里对应多个不同的 g_gt，**逼模型必须用风格去区分**。
    v1 用随机采样时同批几乎不会出现同字，风格因此被忽略（correct≈shuffled）。

## 用法
  python tools/train_deform_standalone.py --steps 6000 --batch 4096 --group 32
产出 assets/deform_skel_standalone.pt
"""
import argparse
import collections
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.model.deform_skel import DeformSkel  # noqa: E402

# ★ 关掉 cuDNN TF32。本机实测: conv 走 NHWC 的 TF32 张量核之前，cuDNN 必须先做
#   NCHW->NHWC 布局转换（`nchwToNhwcKernel`，算子级 profiler 里占 13% CUDA 时间、
#   700 次调用/10 步）。关掉 TF32 后直接用 fp32 核，省掉这批转换，整体更快。
torch.backends.cudnn.allow_tf32 = False

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--group', type=int, default=32, help='每个 batch 采多少个"字"')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--std-dir', default='data/50k/shards_std_fixed')
ap.add_argument('--gt-dir', default='data/50k/shards_aux_skel3')
ap.add_argument('--out', default='assets/deform_skel_standalone.pt')
ap.add_argument('--init', default='', help='从已有头的权重续训(不换网络)')
ap.add_argument('--blur', type=int, default=0, help='输入拼一路高斯模糊(默认关)')
ap.add_argument('--ckpt', type=int, default=0, help='梯度检查点(默认关; 用朴素 batch 4096)')
ap.add_argument('--dt-ch', type=int, default=1, dest='dt_ch',
                help='距离场输入通道数(默认 1). 对 g_std 是固定量, 离线预计算一次')
ap.add_argument('--blur-sigma', type=float, default=1.5, dest='blur_sigma')
ap.add_argument('--w-tv-out', type=float, default=1e-2, dest='w_tv_out',
                help='latent 输出梯度能量惩罚(压破碎, 不需要 VAE)')
ap.add_argument('--w-tv-res', type=float, default=1e-2, dest='w_tv_res',
                help='残差梯度能量惩罚(残差是凭空写像素的唯一入口)')
ap.add_argument('--w-tv', type=float, default=1e-2, dest='w_tv',
                help='偏移场 TV 平滑正则(防把横线扭成波浪)')
ap.add_argument('--w-fold', type=float, default=1e-1, dest='w_fold',
                help='Jacobian 折叠惩罚 det(I+∇U)>0(防空间折叠/笔画翻转)')
ap.add_argument('--diag-decode', type=int, default=1, dest='diag_decode',
                help='评估时解码看墨量/破碎(只诊断, 不进训练). 1=开')
ap.add_argument('--w-img', type=float, default=0.0, dest='w_img',
                help='图像空间监督权重: 解码 g-prime 后与 g_gt 的骨架比结构。这是关键项 —— 实测 latent 余弦 0.947 但解码后碎成 28 段，latent MSE 控制不住连通性')
ap.add_argument('--img-batch', type=int, default=128, dest='img_batch',
                help='图像监督每步解码多少条(全 batch 解码太贵)')
# ★★ 监督对比学习: 治"style-follow 卡在 50%(随机)"的病根。
#   目标 MSE 被"该字的平均形变"主导, 风格特异部分只是总方差里一小块 ->
#   模型走捷径学到"字->平均形变"就能降掉大部分 loss, 风格梯度太弱 -> 躺平。
#   对比项把"风格判别"变成显式目标: 同一个字内, g'(c,k) 要比 g'(c,k') 更接近 g_gt(c,k)。
ap.add_argument('--w-contr', type=float, default=0.0, dest='w_contr',
                help='监督对比 loss 权重(0=关). 同字内 g\'(c,k) 应比 g\'(c,k\') 更接近 g_gt(c,k)')
ap.add_argument('--contr-tau', type=float, default=0.07, dest='contr_tau',
                help='对比 loss 温度. cos 模式参考 0.07(CLIP 默认); mse 模式按"超出量"量级, 参考 0.005')
ap.add_argument('--contr-mode', default='cos', dest='contr_mode',
                choices=['cos', 'mse', 'margin'],
                help='对比形式: cos=余弦InfoNCE / mse=自归一MSE的InfoNCE(与评估口径一致) '
                     '/ margin=MSE排序损失(与评估口径完全一致, 无温度要调)')
ap.add_argument('--contr-margin', type=float, default=0.005, dest='contr_margin',
                help='margin 模式的间隔(MSE 尺度; 实测正负样本差 ~0.002, 故 0.005 有推力)')
ap.add_argument('--contr-hard', type=int, default=0, dest='contr_hard',
                help='难负样本挖掘: 每个 anchor 只保留最难的 N 个负样本(0=全部)')
ap.add_argument('--vae', default='data/pretrained/pretrained_models/sd-vae-ft-ema')
ap.add_argument('--lr-min-ratio', type=float, default=0.1, dest='lr_min_ratio')
ap.add_argument('--eval-every', type=int, default=1000)
# ★ 中间 checkpoint: 之前只在最后存一次 -> 进程静默退出(无 traceback/无 OOM 记录)时整轮白跑。
#   实测踩过一次(v9c 在 step2000 后无声退出, 15 分钟产出为零), 所以加上。
ap.add_argument('--save-every', type=int, default=0, dest='save_every',
                help='每 N 步存一次中间 ckpt(0=只在最后存). 防静默退出丢整轮')
ap.add_argument('--residual', type=int, default=0, help='1=形变+加性残差')
ap.add_argument('--stroke-mod', type=int, default=0, dest='stroke_mod',
                help='1=受控笔画乘性缩放(门控调粗细, 空白背景不造墨)')
ap.add_argument('--stroke-cap', type=float, default=1.0, dest='stroke_cap')
ap.add_argument('--gate-radius', type=float, default=0.25, dest='gate_radius')
ap.add_argument('--w-tv-stroke', type=float, default=1e-2, dest='w_tv_stroke',
                help='笔画调制场 TV 正则(防边缘高频锯齿)')
ap.add_argument('--style-dim', type=int, default=128, dest='style_dim')
ap.add_argument('--style-emb', default='assets/callig_emb_pretrained_50k.pt',
                dest='style_emb',
                help='用**模型同一张**预训练书家表(冻结); 否则离线训的风格输入与'
                     '线上 _e_callig() 对不上, 训好的头接进去会失效')
ap.add_argument('--width', type=int, default=64)
ap.add_argument('--max-off', type=float, default=3.0, dest='max_off')
ap.add_argument('--res-cap', type=float, default=1.0, dest='res_cap')
ap.add_argument('--csv', default='assets/train_50k_v2_fixed.csv', help='训练数据集 CSV')
ap.add_argument('--follow-n', type=int, default=400, help='style-follow 测多少个字')
ap.add_argument('--topo-mode', type=int, default=0, dest='topo_mode',
                help='1=开启 SkelNet-V2 离散拓扑增删 (剪刀与胶水双分支)')
ap.add_argument('--w-prune-l1', type=float, default=0.05, dest='w_prune_l1',
                help='剪刀省笔掩码的 L1 稀疏惩罚权重')
ap.add_argument('--w-lig-l1', type=float, default=0.05, dest='w_lig_l1',
                help='胶水牵丝掩码的 L1 稀疏惩罚权重')
ap.add_argument('--freeze-base', type=int, default=0, dest='freeze_base',
                help='1=冻结 U-Net 主干与连续形变头，只微调新增的剪刀与胶水头')
ap.add_argument('--preserve-amp', type=int, default=0, dest='preserve_amp',
                help='1=开启墨迹保幅校准(Amplitude Calibration)')
ap.add_argument('--latent-loss', default='l2', choices=['l2', 'l1'], dest='latent_loss',
                help='[2026-09-29] 主 latent 监督用 L2(MSE) 还是 L1。'
                     '注意: 逐像素 L1 的最优解是**中位数**, 对稀疏骨架仍可能塌到背景; '
                     '且 MSE 是二次的, 对笔画像素(大误差)权重反而更高 -> L1 未必更好, 需实测。')
ap.add_argument('--w-mass', type=float, default=0.0, dest='w_mass',
                help='[2026-09-29] ★ **VAE-free 的图像域监督**。把 latent 沿 delta_ink 方向'
                     '投影得到 32x32 墨迹图(用 z_bg/delta_ink 两个物理基准向量, 解析、零开销),'
                     '再对 pred/GT 的墨迹图做空间 L1。这是 w_img 的替代品 —— w_img 需要带梯度'
                     '跑 sd-vae 解码器, 实测在 batch>=1024 时必 OOM; 本项**完全不挂 VAE**。')
ap.add_argument('--deform-prob', type=float, default=0.0, dest='deform_prob',
                help='[2026-09-29 继承 v29] 条件几何形变增强概率。直接对 g_std 施加'
                     '书法域内的随机仿射+低频弹性扰动(替代高斯白噪), 抹平两阶段分布断层。')
ap.add_argument('--deform-scale', type=float, default=1.0, dest='deform_scale',
                help='上面那个增强的幅度倍率 (rot5deg/scale0.08/shear0.06/trans1.5px/elastic1.5px 为 1.0)')
ap.add_argument('--probe', default='', dest='probe',
                help='[2026-09-29] ★ LatentSkelProbe ckpt (冻结)。给了就把**主重建损失搬到 probe '
                     '输出空间**: loss = MSE(probe(g_prime), g_gt)。'
                     '动机: 原始骨架 latent 空间里 MSE 的最小解是「数据集平均骨架」'
                     '(实测 0.265) 而非「这个字的标准骨架」(0.487) -> 退化吸引子, 字会碎/糊。'
                     'probe 输出空间的排序是正确的 (见 tools/diag_probe_loss_space.py)。')

# ══════════════════════════════════════════════════════════════════════════════
# [2026-09-29] 架构同构: 从 DiT config 读形变头的全部构造参数
# ══════════════════════════════════════════════════════════════════════════════
# ⚠ 事故复盘: 之前本脚本构造 DeformSkel 时**只传了一部分参数** —— 少了
#   style_tokens / attn_heads (风格 cross-attn 整支从未建出来, 14 个键缺失,
#   风格路径根本没被训练), 也少了 max_off (本脚本默认 3.0, DiT 侧是 6.0)。
#   max_off 不是普通超参: forward 里是 `off = tanh(r/max_off) * max_off`,
#   它同时是**归一化尺度**和**限幅**。用 3.0 训、用 6.0 推 -> 大位移被放大 ~80%,
#   直接导致笔画撕裂/碎片化。这类"架构/标定错位"必须从根上杜绝:
#   **架构参数只有一个来源 = DiT config**。CLI 显式给的仍然优先。
ap.add_argument('--from-config', default='', dest='from_config',
                help='DiT config JSON (如 src/train/configs/v29_skelnet_v2.json)。'
                     '读其中的 deform_*/residual/res_cap/stroke_*/gate_radius 作为构造参数, '
                     '保证 standalone 与主训练侧的形变头**完全同构**。CLI 显式给的优先。')
ap.add_argument('--grid', type=int, default=32, dest='grid', help='latent 网格边长')
ap.add_argument('--style-ch', type=int, default=32, dest='style_ch')
ap.add_argument('--coarse', type=int, default=8, dest='coarse')
ap.add_argument('--warp-iters', type=int, default=1, dest='warp_iters')
ap.add_argument('--style-tokens', type=int, default=0, dest='style_tokens',
                help='★ 风格 cross-attn 的 token 数 (>0 才建出该支)。DiT 侧 v29 用 16。')
ap.add_argument('--attn-heads', type=int, default=4, dest='attn_heads')
ap.add_argument('--film-mode', default='film', choices=['film', 'adaln'], dest='film_mode')

# ══════════════════════════════════════════════════════════════════════════════
# [2026-09-29] 早停验证集 + **decode 判据**
# ══════════════════════════════════════════════════════════════════════════════
# ⚠ 为什么判据必须来自 VAE 解码: 实测 v1 的 latent MSE 0.2479 是历史最优, 但把它的
#   输出 g' 解码出来, 墨量只有目标的 **5%**(近空白) —— MSE 把"输出空白"评为收敛良好。
#   latent 空间里墨迹只占极小能量, MSE 看不见它。判据必须看图。
# 验证集由 tools/build_skelnet_val.py 按**字符**留出(否则同字仍在训练集里, 指标虚高)。
ap.add_argument('--val-csv', default='', dest='val_csv',
                help='★ 早停验证集 csv。训练用 --csv 应指向剔除留出字后的 *_minusval.csv。')
ap.add_argument('--val-n', type=int, default=256, dest='val_n',
                help='每次评测解码多少条验证样本 (解码 ~0.025s/样本, 别开太大)')
ap.add_argument('--val-chunk', type=int, default=16, dest='val_chunk',
                help='验证解码的分块大小 (解码 16 条无梯度约需 6~8G)')
ap.add_argument('--es-metric', default='skel_iou', dest='es_metric',
                choices=['skel_iou', 'ink_ratio', 'frag'],
                help='早停判据(全部来自图空间): skel_iou 越大越好; ink_ratio 越接近 1 越好; '
                     'frag 越小越好。')
ap.add_argument('--es-patience', type=int, default=6, dest='es_patience',
                help='连续多少次评测没改善就停 (0=关闭早停)')
ap.add_argument('--es-min-delta', type=float, default=2e-3, dest='es_min_delta')
ap.add_argument('--no-decode-eval', action='store_true', dest='no_decode_eval',
                help='关掉解码评测(调试用)')

# ══════════════════════════════════════════════════════════════════════════════
# ★ 图空间骨架损失 (唯一与最终消费者一致的监督)
# ══════════════════════════════════════════════════════════════════════════════
# 损失 = BCE(pos_weight) + Dice( decode(g'), decode(g_gt) )
#   - 直接约束 **g' 本身解码出来是骨架**, 而不是任何 latent 距离。
#   - 为什么必须这样: latent MSE 的最优解是「平均骨架 latent」, 解码出来是模糊的灰;
#     probe 空间损失允许 g' 是糊的(只要 probe(g') 对); w-mass 只管墨量总量+边缘能量。
#     都不直接约束拓扑。DiT 消费者吃的是 g' 本身。
#   - 显存/算力实测 (tools/bench_vae_cfg.py, 不开 expandable_segments):
#       B=64, 解码 chunk=8,  bf16 -> 10.92G, 1.60s/step   ★ 推荐
#       B=64, 解码 chunk=16, bf16 -> 21.26G, 1.73s/step   (fp32 直接 OOM)
#     chunk=8 反而**更快也更省**; 关键写法是「每 chunk 独立前向 + 立即 backward」,
#     不能用 retain_graph 分块(那样解码器激活不释放, 等于白分)。
ap.add_argument('--w-px', type=float, default=0.0, dest='w_px',
                help='★ 图空间骨架损失权重 (朴素 BCE+Dice)。0=关闭。建议 1~5。')
ap.add_argument('--px-chunk', type=int, default=8, dest='px_chunk',
                help='像素损失的解码分块大小 (实测 8 比 16 又快又省)')
ap.add_argument('--px-bf16', type=int, default=1, dest='px_bf16',
                help='像素损失解码用 bf16 (省显存; chunk=16 时 fp32 会 OOM)')
ap.add_argument('--px-every', type=int, default=1, dest='px_every',
                help='每隔多少步算一次像素损失 (1=每步; 算力紧张时调大)')
ap.add_argument('--px-subset', type=int, default=0, dest='px_subset',
                help='★ 像素损失只解码前 N 条 (0=全 batch)。\n'
                     'latent 主监督管位置(全 batch, 便宜), 图像域只管连通性(少量样本, 贵)\n'
                     '-> 解码开销与训练 batch 解耦, 可以用大 batch 跑 latent + 小抽样跑像素。')
ap.add_argument('--w-lat', type=float, default=1.0, dest='w_lat',
                help='latent 重建损失的权重。⚠ 若主监督已是 --w-px, 可以把它调到 0~0.1: '
                     'latent MSE 的最优解是"平均骨架", 留着会往糊的方向拉。')
a = ap.parse_args()

# ── 架构参数同构: --from-config 提供默认值, CLI 显式给的优先 ────────────────
if a.from_config:
    import json as _json
    _cfg = _json.load(open(a.from_config, encoding='utf-8'))
    _MAP = {                       # DiT config 键 -> 本脚本 dest
        'deform_width': 'width', 'deform_grid': 'grid',
        'deform_style_ch': 'style_ch', 'deform_max_off': 'max_off',
        'deform_coarse': 'coarse', 'residual': 'residual', 'res_cap': 'res_cap',
        'stroke_mod': 'stroke_mod', 'stroke_cap': 'stroke_cap',
        'gate_radius': 'gate_radius', 'deform_dt_ch': 'dt_ch',
        'deform_topo': 'topo_mode', 'deform_warp_iters': 'warp_iters',
        'deform_film_mode': 'film_mode', 'deform_style_tokens': 'style_tokens',
        'deform_attn_heads': 'attn_heads', 'deform_preserve_amp': 'preserve_amp',
    }
    # 哪些 dest 是 CLI 显式给的 -> 不覆盖
    _given = set()
    for _act in ap._actions:
        for _os in _act.option_strings:
            if any(t == _os or t.startswith(_os + '=') for t in sys.argv[1:]):
                _given.add(_act.dest)
    _applied, _skipped = [], []
    for _ck, _dest in _MAP.items():
        if _ck not in _cfg:
            continue
        if _dest in _given:
            _skipped.append(_dest)
            continue
        setattr(a, _dest, _cfg[_ck])
        _applied.append('%s=%s' % (_dest, _cfg[_ck]))
    print('[from-config] %s -> %s' % (a.from_config, ', '.join(_applied)), flush=True)
    if _skipped:
        print('[from-config] CLI 覆盖: %s' % ', '.join(_skipped), flush=True)
    print('[from-config] ⚠ 形变头将按**主训练侧同构**构造: width=%s max_off=%s '
          'style_tokens=%s attn_heads=%s film_mode=%s warp_iters=%s coarse=%s grid=%s'
          % (a.width, a.max_off, a.style_tokens, a.attn_heads, a.film_mode,
             a.warp_iters, a.coarse, a.grid), flush=True)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)

# 加载风格表以确定 NCAL 槽位数
_emb = torch.load(a.style_emb, map_location='cpu', weights_only=False)
_tab = _emb['embedding'] if isinstance(_emb, dict) else _emb
_tab = _tab.float()
NCAL = _tab.shape[0]


def load_bank(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        L, I = z['latents'], z['img_ids']
        for j, i in enumerate(I):
            m[int(i)] = L[j].astype(np.float32)
        z.close()
    return m


print('[1] 载入 g_std 与 g_gt ...', flush=True)
A = load_bank(a.std_dir)
B = load_bank(a.gt_dir)
print(f'    g_std {len(A)} / g_gt {len(B)}', flush=True)

C2I = {}
rows = list(csv.DictReader(open(a.csv, encoding='utf-8')))
has_pair_id = 'pair_id' in rows[0] if rows else False

for r in rows:
    C2I.setdefault(str(r.get('calligrapher', '')), len(C2I))
rec = {}
for r in rows:
    try:
        i = int(os.path.basename(r.get('image_path', '')).split('.')[0])
    except Exception:
        continue
    if i in A and i in B:
        if has_pair_id:
            s_idx = int(r['pair_id'])
        else:
            s_idx = C2I.get(str(r.get('calligrapher', '')), 0)
        rec[i] = (s_idx, str(r.get('character', '')))
ids = sorted(rec)
n_slots = len({v[0] for v in rec.values()})
print(f'    有效 {len(ids)} 条, 风格槽位 {n_slots} 个 (总槽位 NCAL={NCAL})', flush=True)

G = torch.from_numpy(np.stack([A[i] for i in ids])).to(DEV)
T = torch.from_numpy(np.stack([B[i] for i in ids])).to(DEV)
Y = torch.tensor([rec[i][0] for i in ids], device=DEV)
CH = [rec[i][1] for i in ids]
N = len(ids)

# 字 -> 该字的样本下标（用于分组采样）
by_char = collections.defaultdict(list)
for k, c in enumerate(CH):
    by_char[c].append(k)
# 只保留"至少 2 个不同书家"的字（style-follow 与分组采样都需要）
groups = [v for v in by_char.values()
          if len({int(Y[i]) for i in v}) >= 2]
print(f'[1] 可用于分组/跟随测试的字 {len(groups)} 个', flush=True)
# ★ 向量化分组采样: 把 groups 拼成 (G, K) 的 int64 数组 + 掩码, 每步只用 numpy 花式索引。
#   原来每步跑 48 次 Python 循环(每次 numpy 调用), 是 step 里最大的 CPU 开销。
_K = max(len(v) for v in groups)
_GT = np.full((len(groups), _K), -1, np.int64)
_GV = np.zeros((len(groups), _K), bool)
for _gi, _v in enumerate(groups):
    _GT[_gi, :len(_v)] = _v
    _GV[_gi, :len(_v)] = True
print(f'[1] 分组表 {_GT.shape} (K={_K})', flush=True)

# ── 距离场输入（infra: 对 g_std 是固定量 -> 只算一次）──────────────────
DT = None
if a.dt_ch > 0:
    from scipy import ndimage
    _t0 = time.time()
    _mag = np.abs(G.cpu().numpy()).mean(1)               # (N,32,32) 幅度图
    _thr = np.median(_mag)
    _dt = np.empty_like(_mag)
    for _k in range(len(_mag)):
        _dt[_k] = ndimage.distance_transform_edt(_mag[_k] <= _thr)
    _dt = _dt / max(_dt.max(), 1e-6)
    DT = torch.from_numpy(_dt[:, None]).to(DEV)          # (N,1,32,32)
    print(f'[1b] 距离场预计算完成 {tuple(DT.shape)} {time.time()-_t0:.0f}s '
          f'(阈值 {_thr:.3f}, 前景占比 {float((_mag>_thr).mean()):.3f})', flush=True)

base_mse = float((G - T).pow(2).mean())
print(f'\n[2] 基线 MSE(g_std, g_gt) = {base_mse:.5f}   <- 形变要打败的就是它', flush=True)

# ── ★ 早停验证集（判据用 decode，见上方说明）────────────────────────────
Gv = Tv = Yv = None
if a.val_csv:
    _vrows = list(csv.DictReader(open(a.val_csv, encoding='utf-8')))
    _vpair = 'pair_id' in _vrows[0] if _vrows else False
    _vrec = {}
    for r in _vrows:
        try:
            i = int(os.path.basename(r.get('image_path', '')).split('.')[0])
        except Exception:
            continue
        if i in A and i in B:
            _vrec[i] = (int(r['pair_id']) if _vpair
                        else C2I.get(str(r.get('calligrapher', '')), 0))
    _vids = sorted(_vrec)
    if _vids:
        Gv = torch.from_numpy(np.stack([A[i] for i in _vids])).to(DEV)
        Tv = torch.from_numpy(np.stack([B[i] for i in _vids])).to(DEV)
        Yv = torch.tensor([_vrec[i] for i in _vids], device=DEV)
        print(f'[1c] ★ 早停验证集 {len(_vids)} 条 ({a.val_csv})  '
              f'槽位 {len(set(Yv.tolist()))} 个', flush=True)
    else:
        print(f'[1c] ⚠ {a.val_csv} 里没有可用样本，早停关闭', flush=True)

model = DeformSkel(cond_dim=a.style_dim, ch=4, grid=a.grid,
                   style_ch=a.style_ch,
                   residual=a.residual,
                   width=a.width, max_off=a.max_off, coarse=a.coarse, res_cap=a.res_cap,
                   blur=a.blur, blur_sigma=a.blur_sigma,
                   dt_ch=a.dt_ch, affine=1, ckpt=a.ckpt,
                   stroke_mod=a.stroke_mod, stroke_cap=a.stroke_cap,
                   gate_radius=a.gate_radius,
                   topo_mode=a.topo_mode,
                   warp_iters=a.warp_iters,
                   film_mode=a.film_mode,
                   style_tokens=a.style_tokens,
                   attn_heads=a.attn_heads,
                   preserve_amp=a.preserve_amp).to(DEV)
# ⚠ 打印**逐项**构造参数: 与 DiT 侧 `[deform] enabled:` 那行对齐核对,
#   一旦 width/max_off/style_tokens 对不上就能立刻发现(上次就是静默错位)。
print('[2] 形变头构造: width=%d grid=%d style_ch=%d max_off=%.2f coarse=%d '
      'residual=%d res_cap=%.2f stroke_mod=%d topo=%d warp_iters=%d film=%s '
      'style_tokens=%d attn_heads=%d preserve_amp=%d -> params=%s'
      % (a.width, a.grid, a.style_ch, a.max_off, a.coarse, a.residual, a.res_cap,
         a.stroke_mod, a.topo_mode, a.warp_iters, a.film_mode, a.style_tokens,
         a.attn_heads, a.preserve_amp,
         format(sum(p.numel() for p in model.parameters()), ',')), flush=True)
if a.init:
    _sd = torch.load(a.init, map_location='cpu', weights_only=False)
    _sd = _sd.get('deform', _sd)
    # ⚠ 加了 blur 输入通道后 d1 的 in_channels 变了 -> 形状不符的键会被跳过。
    #   注意 load_state_dict(strict=False) **仍然会因形状不符报错**(strict=False 只忽略
    #   缺失/多余键, 不忽略形状), 所以必须自己先过滤。
    _cur = model.state_dict()
    _keep = {k: v for k, v in _sd.items()
             if k in _cur and tuple(_cur[k].shape) == tuple(v.shape)}
    _skip = [k for k in _sd if k not in _keep]
    model.load_state_dict(_keep, strict=False)
    print(f'[2] 续训: 从 {a.init} 载入 {len(_keep)}/{len(_sd)} 个键'
          f'（跳过形状不符 {len(_skip)} 个: {_skip[:3]}）', flush=True)

if a.freeze_base:
    print("[2] 冻结主干 U-Net 与连续形变头，只微调 head_prune 与 head_ligature")
    for name, p in model.named_parameters():
        if "head_prune" not in name and "head_ligature" not in name:
            p.requires_grad = False
# ★ 风格源必须与模型 _e_callig() 一致
print(f'[2] 用预训练书家表 {a.style_emb} {tuple(_tab.shape)} (冻结)', flush=True)
style = nn.Embedding.from_pretrained(_tab, freeze=True).to(DEV)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
print(f'[2] 形变模块参数 {sum(p.numel() for p in model.parameters()):,}  '
      f'(含风格底图 {sum(p.numel() for p in model.style_off.parameters()):,})', flush=True)

n_val = max(1, int(N * 0.03))
perm = np.random.permutation(N)
val_i = torch.tensor(perm[:n_val], device=DEV)
tr_i = torch.tensor(perm[n_val:], device=DEV)

# ── ★ step0 零初始化自检 ───────────────────────────────────────────────────
# 本仓库历史上被 `_basic_init` 冲掉过新模块的 zero-init(docs/01_architecture/05 §1),
# 冲毁后形变头会从乱码状态开始, 而 loss 曲线看不出异常。
#
# 判据**不是** "g' 是否等于 std": preserve_amp 是无可学参数的保幅变换,
# 位移为 0 时也会按 3x3 局部峰值把墨迹放大 1.0~2.5x, 所以 g' ≠ std 是设计行为
# (实测相对偏差 ~1.4e-1, 与 xavier 冲毁无关)。
# 该守的是**可学零初始化项**在 step0 必须为 0:
#   off(位移场) / stroke_mod(笔画调制) / head_prune(剪刀掩码, bias=-5 -> 0.0067)
with torch.no_grad():
    _g0 = model(G[val_i[:64]], style(Y[val_i[:64]]),
                dt=(DT[val_i[:64]] if DT is not None else None))
    _id_dev = float((_g0 - G[val_i[:64]]).abs().mean())
    _id_ref = float(G[val_i[:64]].abs().mean())
    _st0 = model.offset_stats() or {}
    _pr0 = (float(model.last_mask_prune.mean())
            if getattr(model, 'last_mask_prune', None) is not None else 0.0)
_rel0 = _id_dev / max(_id_ref, 1e-9)
_off0 = float(_st0.get('mean_abs', 0.0))
_sm0 = float(_st0.get('stroke_mod', 0.0))
_sty0 = float(_st0.get('style_part', 0.0))
print(f'[2a] step0 自检: |g\'(std)-std| = {_id_dev:.3e} (std 尺度 {_id_ref:.3f}, '
      f'相对 {_rel0:.2e}  <- preserve_amp 的保幅放大, 设计行为非故障)', flush=True)
print(f'     可学零初始化项: off={_off0:.3e}  stroke_mod={_sm0:.3e}  '
      f'style_part={_sty0:.3e}  prune_mask={_pr0:.4e}', flush=True)
if _off0 > 1e-4 or _sm0 > 1e-4 or _sty0 > 1e-4:
    raise SystemExit(
        f'[FATAL] step0 位移场/笔画调制/风格分量不为 0 '
        f'(off={_off0:.3e}, stroke_mod={_sm0:.3e}, style_part={_sty0:.3e})。'
        f'新模块 zero-init 被 _basic_init 冲掉, 形变头会从乱码状态开始 —— 不要开训。')
if abs(_pr0 - 0.0067) > 5e-3:
    raise SystemExit(
        f'[FATAL] step0 剪刀掩码 {_pr0:.4f} 偏离设计值 sigmoid(-5)=0.0067 —— '
        f'head_prune 初始化被冲掉。')
print('     -> 零初始化项 OK: 位移/笔画调制/风格分量全为 0, 剪刀掩码 = sigmoid(-5)', flush=True)

# style-follow 评测集：每个字取最多 6 个不同书家的样本
follow = []
for v in groups:
    byk = {}
    for i in v:
        byk.setdefault(int(Y[i]), i)
    if len(byk) >= 2:
        follow.append(list(byk.values())[:6])
    if len(follow) >= a.follow_n:
        break
print(f'[2] style-follow 用 {len(follow)} 个字', flush=True)


@torch.no_grad()
def style_follow():
    """同一个字、不同书家: g'(c,k) 是否比 g'(c,k') 更接近 k 自己的 g_gt(c,k)。"""
    model.eval()
    good = tot = 0
    same_g = []
    for v in follow:
        gs = G[v]                                  # 同字的多个样本（g_std 基本同图）
        same_g.append(float((gs - gs[0]).abs().mean()))
        for i in v:
            others = [j for j in v if int(Y[j]) != int(Y[i])]
            if not others:
                continue
            _d = DT[i:i + 1] if DT is not None else None
            gp_self = model(G[i:i + 1], style(Y[i:i + 1]), dt=_d)[0]
            gp_oth = model(G[i:i + 1], style(Y[others[0]:others[0] + 1]), dt=_d)[0]
            d_self = float((gp_self - T[i]).pow(2).mean())
            d_oth = float((gp_oth - T[i]).pow(2).mean())
            good += int(d_self < d_oth)
            tot += 1
    model.train()
    return (good / tot if tot else float('nan')), (np.mean(same_g) if same_g else float('nan'))


@torch.no_grad()
def evaluate(tag):
    model.eval()
    vi = val_i
    g2 = model(G[vi], style(Y[vi]), dt=(DT[vi] if DT is not None else None))
    # ★ probe 模式下所有 MSE 都算在 probe 输出空间, 与训练损失同口径
    #   (否则打印出来的 MSE 是原始空间, 和 loss 对不上, 会误判"没收敛")
    _P = (lambda x: _probe_apply(x)) if probe is not None else (lambda x: x)
    mse = float((_P(g2) - T[vi]).pow(2).mean())
    # ★ [2026-09-29] 基线必须与 mse **同子集**。原来 mse 算在随机 3% 的 val 子集、
    #   base_mse 算在全集上, 于是 step0(恒等, 输出就是 g_std)会报出**负的**闭合率
    #   (-41.8% / -7.6%), 把"还没开始学"误读成"初始化坏了"。
    base_vi = float((G[vi] - T[vi]).pow(2).mean())
    ysh = (Y[vi] + 3) % NCAL
    g2s = model(G[vi], style(ysh), dt=(DT[vi] if DT is not None else None))
    off = model.offset_stats() or {}
    fr, sg = style_follow()
    # 不可约下界(同字同书家组内方差) —— 判"收敛没收敛"要看残差/下界, 而不是看残差绝对值。
    # 实测下界 = 0.06505 (tools/diag_skelnet_floor.py), 闭合率天花板 = 87.1%。
    _FLOOR = 0.06505
    _sm = off.get('stroke_mod')
    _sm_str = (' | stroke_mod %.4f' % _sm) if _sm is not None else ''
    print('  %-10s MSE=%.5f (子集基线 %.5f, 闭合 %5.1f%%) [全集基线 %.5f] | '
          'correct %.5f / shuffled %.5f '
          '| 输出变化 %.4f | off %.4f (风格底图 %.4f)%s | **style-follow %.1f%%**'
          % (tag, mse, base_vi, 100 * (1 - mse / base_vi), base_mse,
             float((_P(g2) - T[vi]).pow(2).mean()), float((_P(g2s) - T[vi]).pow(2).mean()),
             float((g2 - g2s).abs().mean()), off.get('mean_abs', 0),
             off.get('style_part', 0), _sm_str, 100 * fr), flush=True)
    # 判"收敛没收敛"要看 残差/下界，而不是残差绝对值。
    # 下界 = 0.06505（同字同书家组内方差，tools/diag_skelnet_floor.py 实测），天花板 = 87.1%。
    _ex = ''
    if vae is not None and a.diag_decode > 0:   # decode 只在评估里当诊断, 不进训练
        with torch.no_grad():
            gp = model(G[vi[:64]], style(Y[vi[:64]]),
                       dt=(DT[vi[:64]] if DT is not None else None))
            pr = _decode_gray(gp)
            tg = _decode_gray(T[vi[:64]])
            ink_p = float((pr < 0.5).float().mean())
            ink_t = float((tg < 0.5).float().mean())
            _ex = ' | **图空间**: 墨量 %.4f / 目标 %.4f (比 %.2f)' % (
                ink_p, ink_t, ink_p / max(ink_t, 1e-9))
    print('             -> 残差/下界 = %.2fx | 闭合 %.1f%% / 天花板 87.1%%%s'
          % (mse / 0.06505, 100 * (1 - mse / base_vi), _ex), flush=True)
    # ★ 早停判据: decode 图空间 (MSE 会把"输出空白"评为收敛良好, 见 decode_eval 说明)
    _dm = {}
    if not a.no_decode_eval and Gv is not None:
        _dm = decode_eval()
        if _dm:
            print('             ★[decode] 墨量 %.4f / 目标 %.4f (比 %.2f) | '
                  '骨架IoU %.4f | 连通分量 %.1f / 目标 %.1f (frag %.2f)'
                  % (_dm['ink'], _dm['ink_t'], _dm['ink_ratio'], _dm['skel_iou'],
                     _dm['ncomp'], _dm['ncomp_t'], _dm['frag']), flush=True)
    model.train()
    return dict(mse=mse, closed=100 * (1 - mse / base_vi), **_dm)


# ── 图像空间监督: 需要 VAE ─────────────────────────────────────────
vae = None
_vae_on_gpu = False
_sc = 0.18215
if a.w_img > 0 or a.diag_decode > 0 or a.val_csv or a.w_px > 0:
    from diffusers.models import AutoencoderKL
    # ★ VAE 只在 eval 诊断时用一次 -> 常驻 CPU, 解码前再搬上 GPU(省 ~1.5G 常驻显存)
    #   但 w_img>0 时每步都要解码 -> 常驻 GPU, 否则每步来回 PCIe 搬 160MB 太慢。
    vae = AutoencoderKL.from_pretrained(a.vae, local_files_only=True).eval()
    for _p in vae.parameters():
        _p.requires_grad_(False)
    _sc = float(getattr(vae.config, 'scaling_factor', 0.18215))
    _vae_on_gpu = bool(a.w_img > 0 or a.w_px > 0)
    if _vae_on_gpu:
        vae.to(DEV)
    print('[2b] VAE 就绪 (scaling=%.5f, 常驻 %s)' % (_sc, 'GPU' if _vae_on_gpu else 'CPU'),
          flush=True)
    print('[2b] 目标图**不预解码**: 全量 50786 张 256x256 float32 = 13.3GB, 会 OOM。'
          '改成每步只解码当前 batch 的 %d 条。' % a.img_batch, flush=True)


def _decode_gray(lat, grad=False):
    """latent -> 灰度图 (n,256,256) in [0,1]。

    ⚠ 训练用的那条路**必须**开梯度: 旧版把整个函数包了 @torch.no_grad(),
      于是 img_loss 返回的 tensor requires_grad=False -> 加进 loss 也**不回传**,
      `--w-img` 就是个静默空操作。诊断/评估那条路仍然 no_grad。
    """
    if _vae_on_gpu:
        pass
    else:
        vae.to(DEV)
    try:
        with (torch.enable_grad() if grad else torch.no_grad()):
            _im = vae.decode(lat / _sc).sample
            return ((_im.clamp(-1, 1) + 1) / 2).mean(1).float()
    finally:
        if not _vae_on_gpu:
            vae.to('cpu')
            if DEV != 'cpu':
                torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# ★ 早停判据: **全部来自 VAE 解码后的图空间**
# ══════════════════════════════════════════════════════════════════════════════
# 为什么不能用 MSE: 实测 v1 的 latent MSE 0.2479 是历史最优, 但把它的输出 g' 解码出来,
#   墨量只有目标的 **5%**(近空白) —— MSE 会把"输出空白"评为收敛良好。
#   latent 空间里墨迹只占极小能量, MSE 看不见它。判据必须看图。
# 三个同口径量:
#   skel_iou  输出墨迹掩膜 与 GT 墨迹掩膜 的 IoU   (越大越好; 输出空白 -> 0)
#   ink_ratio 输出墨量 / GT 墨量                    (越接近 1 越好)
#   frag      输出连通分量数 / GT 连通分量数          (越小越好; 碎 -> 大)
_val_idx = (torch.arange(min(a.val_n, Gv.shape[0]), device=DEV)
            if Gv is not None else torch.arange(0, 0, device=DEV))
if Gv is not None:
    print(f'[2e] ★ 早停判据 = decode 图空间 / {a.es_metric}  '
          f'patience={a.es_patience}  每次解码 {len(_val_idx)} 条', flush=True)


@torch.no_grad()
def decode_eval():
    """在验证集上解码评测。返回 dict(可为空)。"""
    global _vae_on_gpu
    if Gv is None or len(_val_idx) == 0 or vae is None:
        return {}
    from scipy.ndimage import label, generate_binary_structure
    _ST = generate_binary_structure(2, 2)
    # ⚠ 分块解码时别每块都来回搬 VAE（每次 160MB PCIe）。整段常驻 GPU。
    _moved = not _vae_on_gpu
    if _moved:
        vae.to(DEV)
        _vae_on_gpu = True
    try:
        g2 = model(Gv[_val_idx], style(Yv[_val_idx]))
        _p, _t = [], []
        for i in range(0, len(_val_idx), max(1, a.val_chunk)):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(DEV == 'cuda')):
                _p.append(_decode_gray(g2[i:i + a.val_chunk]).float().cpu())
                _t.append(_decode_gray(Tv[_val_idx][i:i + a.val_chunk]).float().cpu())
        pr = torch.cat(_p, 0) < 0.5
        tg = torch.cat(_t, 0) < 0.5
        inter = (pr & tg).sum().float()
        union = (pr | tg).sum().clamp_min(1).float()
        _pn = float(np.mean([label(x.numpy(), _ST)[1] for x in pr]))
        _tn = float(np.mean([label(x.numpy(), _ST)[1] for x in tg]))
        _pi = float(pr.float().mean())
        _ti = float(tg.float().mean())
        return dict(skel_iou=float(inter / union), ink=_pi, ink_t=_ti,
                    ink_ratio=_pi / max(_ti, 1e-9), ncomp=_pn, ncomp_t=_tn,
                    frag=_pn / max(_tn, 1e-9))
    finally:
        if _moved:
            vae.to('cpu')
            _vae_on_gpu = False
            if DEV != 'cpu':
                torch.cuda.empty_cache()


def _es_score(m):
    """早停打分: 统一成"越大越好"。"""
    if not m:
        return None
    if a.es_metric == 'skel_iou':
        return m['skel_iou']
    if a.es_metric == 'ink_ratio':
        return -abs(m['ink_ratio'] - 1.0)
    return -m['frag']


def _balanced_bce_dice_gray(pr, tg, thr=0.5):
    """图空间稀疏监督: **朴素** BCE + Dice。pr/tg 都是 (n,256,256) 灰度 [0,1]。

    ⚠ **极性**: 骨架图是**白底黑线**, 灰度里 **墨 = 暗(接近 0)**, 背景 = 亮(接近 1)。
       所以墨迹掩膜是 `tg < thr`, 墨迹概率是 `1 - pr`。

    ★ [2026-09-29] **删掉 pos_weight 重加权**(原来是 (1-f)/f 并 cap 到 10)。
      它不是"中性"的平衡项, 而是把**最优解搬走**了:
          加权 BCE 对常数预测 p 的最优解 = pw·f / (pw·f + 1 - f)
          取 pw=10, 前景 f≈0.05 -> p* = 0.345, 即**约 7x 的墨量**。
      实测 (v6 首次运行, lr 1e-4): 墨量 1.32x -> 5.01x, 连通分量 2.9 -> 24.9
      (frag 8.55), latent MSE 反而变差(0.689 -> 0.923)。与上述公式同量级。
      pw=1 时最优解回到真实前景比例 f, 类别不平衡交给 Dice 处理 —— 这也是
      "朴素 BCE + Dice" 的本来用法。
    """
    t = (tg < thr).float()                       # 墨迹掩膜 (墨是暗的!)
    logit = (0.5 - pr) * 12.0                    # "该像素是墨" 的 logit
    bce = F.binary_cross_entropy_with_logits(logit, t)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return bce + dice


def _pixel_loss_pass(Gin, bi, dt_in):
    """★ 图空间骨架损失: 分块前向 + **立即 backward**（不 retain_graph）。

    ⚠ 关键: 不能"整批前向一次、再分块 backward(retain_graph=True)" ——
      retain_graph 会让**解码器激活不被释放**, 分块等于白分(实测全 OOM)。
      正确写法是每个 chunk 独立前向、立即反传, 峰值 = deform(chunk)+decoder(chunk),
      与总 batch 解耦。代价是 deform 前向做 N 次。
    """
    # ★ [2026-09-29] --px-subset: 只解码前 N 条。
    #   动机: latent 墨迹 L1 排序正确且便宜(0.65s/step @ batch2048), 但它**管不了连通性**
    #   —— 32x32 上一根骨架线只有约 0.4 个格子宽, 连通性在 latent 里不可表示
    #   (实测 200 步就把 frag 从 1.69 推到 21.47)。
    #   所以分工: latent 管位置(全 batch, 便宜), 图像域只管连通性(少量样本, 贵)。
    #   解码开销因此与训练 batch 解耦。
    _n = Gin.shape[0]
    if int(getattr(a, 'px_subset', 0)) > 0:
        _n = min(int(a.px_subset), _n)
        Gin = Gin[:_n]
        bi = bi[:_n]
        dt_in = None if dt_in is None else dt_in[:_n]
    _ch = max(1, a.px_chunk)
    _nb = max(1, (_n + _ch - 1) // _ch)
    _acc, _last = 0.0, None
    for i in range(0, _n, _ch):
        _sl = slice(i, min(i + _ch, _n))
        _d = None if dt_in is None else dt_in[_sl]
        g2c = model(Gin[_sl], style(Y[bi][_sl]), dt=_d)
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=bool(a.px_bf16) and DEV == 'cuda'):
            pr = _decode_gray(g2c, grad=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=bool(a.px_bf16) and DEV == 'cuda'):
                tg = _decode_gray(T[bi][_sl], grad=False).float()
        _last = _balanced_bce_dice_gray(pr.float(), tg)
        (_last * (a.w_px / _nb)).backward()
        _acc += float(_last)
        del g2c, pr, tg
    return _acc / _nb


def img_loss(gp, idx):
    """解码 g-prime 与 g_gt 的**图像**比结构。

    为什么必须加: 实测 latent 余弦 0.947 / MSE 0.129 看着很好, 但解码出来墨量只剩 40%、
    连通分量从 1.7 涨到 28.4 —— latent MSE 完全控制不住图像空间的连通性。
    结构项: L1(管位置) + 边缘梯度差(管"线还在不在")。
    """
    _n = min(a.img_batch, gp.shape[0])
    _pr = _decode_gray(gp[:_n], grad=True)
    _tg = _decode_gray(T[idx[:_n]], grad=False)
    _l1 = (_pr - _tg).abs().mean()
    _gx = lambda x: (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    _gy = lambda x: (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return _l1 + (_gx(_pr) - _gx(_tg)).abs() + (_gy(_pr) - _gy(_tg)).abs()


def _pair_mse(gp, gt):
    """(B,B) 逐对 MSE: d[i,j] = mean((gp_i - gt_j)^2)，用 Gram 矩阵一次算完（O(B²D) 但不显式广播）。"""
    a = gp.flatten(1).float()
    p = gt.flatten(1).float()
    d = (a * a).sum(1)[:, None] + (p * p).sum(1)[None, :] - 2.0 * (a @ p.t())
    return (d / a.shape[1]).clamp_min(0.0)          # clamp: 浮点误差可能给负


def _hard_neg(sel, score, k):
    """在每个 anchor 的负样本里只保留 score 最"难"的 k 个。

    score: (B,B), **越大越难**(越像 / 越近)。sel: (B,B) bool, 当前可选负样本。
    """
    masked = score.masked_fill(~sel, float('-inf'))
    kk = max(1, min(int(k), masked.shape[1]))
    idx = masked.topk(kk, dim=1).indices                     # (B,kk)
    out = torch.zeros_like(sel)
    out.scatter_(1, idx, True)
    return out & sel


def contrastive_loss(gp, gt, gid, sid, tau=0.07, mode='cos', margin=0.005, hard=0):
    """★ 监督对比: 同一个字内, g'(c,k) 要比 g'(c,k') 更接近 g_gt(c,k)。

        anchor    = g'(c,k)        —— 模型对"字 c + 书家 k"的形变输出
        positive  = g_gt(c,k)      —— k 自己写的那个字的骨架
        negatives = {g_gt(c,k')}   —— 同字、但**别的书家**写的骨架

    ## 为什么必须加这个
    目标函数 `MSE(g', g_gt)` **被"该字的平均形变"主导**: 风格特异的那部分只是
    总方差里的一小块。于是模型走捷径 —— 学到"字 -> 平均形变"就能降掉大部分 loss,
    风格特异那部分的梯度太弱 -> 直接躺平。实测: 闭合率 32.3%->34.4% 在涨(平均形变在变准),
    但 **style-follow 51.4%->51.7% 一动没动**(= 与随机无异)。
    对比项把"风格判别"从"隐含的弱梯度"变成**显式目标**, 直接优化
    `d(g'(c,k), g_gt(c,k)) < d(g'(c,k), g_gt(c,k'))`。

    ## 三种 mode（可切换做 A/B）
    | mode | 形式 | 特点 |
    |---|---|---|
    | `cos`    | InfoNCE, 相似度 = 余弦/tau | 标准 SupCon; 数值 O(1) 好调; 但**余弦与评估口径(MSE)不一致**, 可能降了 cos 却不涨 style-follow |
    | `mse`    | InfoNCE, 相似度 = -(d - d_self)/tau | **与评估口径一致**; 先减去 d_self 自归一 -> 尺度无关、不必猜 d 的量级 |
    | `margin` | 排序损失 mean relu(margin + d_self - d_neg) | **与评估口径完全一致**, 且**没有温度要调**; 直接就是"要拉开多少" |

    `hard`>0 时每个 anchor 只保留最难的 N 个负样本（难负样本挖掘）——
    组内负样本最多可达 ~63 个, 但真正有区分度的只是那几个形近书家, 全用会被稀释。

    ## 为什么 `mse` 要先减 d_self（而不是直接 -(d/tau)）
    直接 `-(d/tau)` 时 d≈0.33 会让 logits 整体落在 −33 附近 —— softmax 是平移不变的,
    所以**绝对值本身无害**, 真正的问题是**正负样本之间只差 ~0.002**, 温度若按绝对尺度
    去猜就会要么全平要么全尖。先减 d_self 让对角线恒为 0, 温度只需覆盖"超出量"的量级。

    ## 负样本只取"同字且不同书家"
    同一个(字,书家)可能有多张样本(组内重复), 它们**不能**当负样本, 否则等于把
    positive 当 negative。
    """
    dev = gp.device
    B = gp.shape[0]
    if B < 2:
        return gp.sum() * 0.0
    eye = torch.eye(B, dtype=torch.bool, device=dev)          # 对角 = positive 自身
    neg = (gid[:, None] == gid[None, :]) & (sid[:, None] != sid[None, :])
    valid = neg.any(1)                                        # 该行至少要有一个负样本
    if not bool(valid.any()):
        return gp.sum() * 0.0
    vmask = valid[:, None]

    if mode == 'cos':
        a = F.normalize(gp.flatten(1).float(), dim=1)
        p = F.normalize(gt.flatten(1).float(), dim=1)
        sim = (a @ p.t()) / max(float(tau), 1e-6)             # 越大越像
    else:
        d = _pair_mse(gp, gt)
        if mode == 'margin':
            r = (float(margin) + d.diagonal()[:, None] - d).clamp_min(0.0)
            sel = neg & vmask
            if hard and hard > 0:
                sel = _hard_neg(sel, -d, hard)                # d 越小越难 -> 用 -d 当 score
            if not bool(sel.any()):
                return gp.sum() * 0.0
            return r[sel].mean()
        if mode != 'mse':
            raise ValueError('unknown contr mode: %r' % mode)
        sim = -(d - d.diagonal()[:, None]) / max(float(tau), 1e-6)

    if hard and hard > 0:
        neg = _hard_neg(neg & vmask, sim, hard)
    logits = sim.masked_fill(~(eye | neg), -1e4)              # 其余位置屏蔽
    rows = valid.nonzero(as_tuple=True)[0]
    return F.cross_entropy(logits[rows], rows)                # 目标 = 对角线


if a.w_contr > 0:
    print('[2c] 监督对比: mode=%s  w=%.3g  tau=%.4g  margin=%.4g  hard=%d'
          % (a.contr_mode, a.w_contr, a.contr_tau, a.contr_margin, a.contr_hard), flush=True)

# ── [2026-09-29] probe 空间的主损失 ──────────────────────────────────────
# probe 是**冻结**的 (0 新增可训练参数), 只作为损失空间。它把 latent 映射到"骨架结构"
# 这个子空间, 该子空间里 MSE 的排序是正确的(平均骨架不再占优) —— 这正是原始 latent
# 空间缺失的性质。
probe = None


def _probe_apply(x, chunk=512):
    """probe 前向, **分块**做。

    为什么要分块: probe 虽然冻结, 但它的激活要留着回传到 g'。batch=4096 时
    每层 (4096,64,32,32) float32 = 1.07GB, 7 层就是 7.5GB -> 直接 OOM(实测)。
    分块后显存与 batch 解耦, 上限只由 chunk 决定; 梯度仍然完整(逐块 cat 后求均值)。
    """
    if probe is None:
        return x
    if x.shape[0] <= chunk:
        return probe(x.float())
    return torch.cat([probe(x[i:i + chunk].float())
                      for i in range(0, x.shape[0], chunk)], 0)


if a.probe:
    from src.train.latent_structure import LatentSkelProbe
    _pk = torch.load(a.probe, map_location='cpu', weights_only=False)
    _pa = _pk.get('args', {}) if isinstance(_pk, dict) else {}
    probe = LatentSkelProbe(
        in_channels=int(_pa.get('in_channels', 4)), out_channels=int(_pa.get('out_channels', 4)),
        width=int(_pa.get('width', 64)), depth=int(_pa.get('depth', 3)))
    _ms, _us = probe.load_state_dict(_pk.get('model', _pk), strict=False)
    if _ms or _us:
        raise SystemExit('probe ckpt 不匹配: missing=%s unexpected=%s' % (_ms, _us))
    probe = probe.to(DEV).eval()
    for _p in probe.parameters():
        _p.requires_grad_(False)
    print('[2d] ★ probe 空间主损失: %s (params=%d, 冻结)  loss = %s(probe(g\'), g_gt)'
          % (a.probe, sum(p.numel() for p in probe.parameters()),
             'L1' if a.latent_loss == 'l1' else 'MSE'), flush=True)
    # ⚠ 基线要跟着换空间, 否则"闭合率"是拿 probe 空间残差去比原始空间基线, 无意义。
    #   (evaluate() 里 `残差/下界 = mse/0.06505` 那行的 0.06505 是**原始空间**的下界,
    #    probe 模式下该参考已失效, 只看 mse 与闭合率即可。)
    _bm_raw = base_mse
    # ⚠ 必须分块: 全量 38583 条过 probe 会产生 38583x64x32x32 的激活 -> 单层 10GB, 必 OOM
    with torch.no_grad():
        _acc, _n = 0.0, 0
        for _s in range(0, G.shape[0], 2048):
            _g = G[_s:_s + 2048]
            _acc += float((_probe_apply(_g).float() - T[_s:_s + 2048]).pow(2).sum())
            _n += _g.shape[0] * _g[0].numel()
        base_mse = _acc / max(_n, 1)
    print('    [probe 自检] 基线: 原始空间 MSE(g_std,g_gt)=%.5f -> probe 空间 %.5f'
          % (_bm_raw, base_mse), flush=True)
    # 一次性自检: probe 作用在**骨架 latent**上还保不保持骨架(自己对自己) —— 若崩了, 损失空间就是废的
    with torch.no_grad():
        _q = probe(T[:64].to(DEV)).float().cpu()
        _t = T[:64].float().cpu()
        _mz = float((T - T.mean(0, keepdim=True)).pow(2).mean())
        print('    [probe 自检] MSE(probe(g_gt), g_gt) = %.5f | MSE(g_gt, mean) = %.5f | '
              'probe 输出 std = %.4f (GT std %.4f)'
              % (float((_q - _t).pow(2).mean()), _mz, float(_q.std()), float(_t.std())),
              flush=True)
        if vae is not None:
            _dg = _decode_gray(_q.to(DEV))
            _dt = _decode_gray(_t.to(DEV))
            _mg, _mt = (_dg < 0.5), (_dt < 0.5)
            print('    [probe 自检] 解码墨量 probe %.4f / GT %.4f | IoU %.4f'
                  % (float(_mg.float().mean()), float(_mt.float().mean()),
                     float((_mg & _mt).sum().float() / (_mg | _mt).sum().clamp_min(1).float())),
                  flush=True)

print('\n[3] 训练 %d 步 (batch %d, 每批 %d 个字 分组采样)' % (a.steps, a.batch, a.group),
      flush=True)
_es0 = _es_score(evaluate('step0'))
_es_best = _es0 if _es0 is not None else -1e9
_es_bad = 0
if _es0 is not None:
    print('    [es] 初始 %s = %.5f (patience=%d, min_delta=%g)'
          % (a.es_metric, _es0, a.es_patience, a.es_min_delta), flush=True)
t0 = time.time()
per = max(1, a.batch // a.group)
for step in range(a.steps):
    _gi = np.random.randint(0, len(groups), a.group)          # 选 a.group 个字
    _pos = np.random.randint(0, _K, (a.group, per))           # 每字随机取 per 个位置
    _sel = _GT[_gi[:, None], _pos]                            # (a.group, per) 花式索引
    _bad = _sel < 0                                           # 落到填充位
    if _bad.any():
        _sel[_bad] = _GT[_gi[np.nonzero(_bad)[0]], 0]         # 回退到该组的第 0 个
    bi = torch.from_numpy(_sel.reshape(-1)).to(DEV)
    # ★ [2026-09-29 继承 v29] 条件几何形变增强: 对 **输入** g_std 施加书法域内的
    #   随机仿射 + 低频弹性扰动（替代高斯白噪）。目的: 让 SkelNet 对"不完美的骨架"
    #   鲁棒, 抹平 standalone(干净 g_std) 与主训练管线 之间的分布断层。
    _Gin = G[bi]
    _dt_in = (DT[bi] if DT is not None else None)
    if a.deform_prob > 0:
        from src.utils.deform_aug import random_skeleton_deformation
        _s = a.deform_scale
        _Gin = random_skeleton_deformation(
            _Gin, prob=a.deform_prob,
            max_rot_deg=5.0 * _s, max_scale=0.08 * _s, max_shear=0.06 * _s,
            max_trans_px=1.5 * _s, max_elastic_px=1.5 * _s)
        # ⚠ 输入被扰动后, 预计算的 DT 不再对应 -> 交回模型自动按新输入算 DT
        #   (主模型 dit.py 里也是 dt=None 自动算, 这样反而更一致)
        _dt_in = None
    g2 = model(_Gin, style(Y[bi]), dt=_dt_in)
    _tgt = T[bi]
    # ★ [2026-09-29] 主重建损失 —— probe 空间(若给了 --probe) vs 原始 latent 空间。
    #   原始空间: MSE 的最小解是「数据集平均骨架」(0.265) 而非「这个字的标准骨架」(0.487)
    #   -> 退化吸引子, 训练必然塌成糊团。probe 空间里排序正确, 故搬过去。
    _g2s = _probe_apply(g2) if probe is not None else g2
    if a.latent_loss == 'l1':
        loss = a.w_lat * (_g2s - _tgt).abs().mean()
    else:
        loss = a.w_lat * (_g2s - _tgt).pow(2).mean()
    # ★ [2026-09-29] VAE-free 的图像域监督: 沿 delta_ink 投影出墨迹图再比空间 L1。
    #   动机同 img_loss(latent MSE 控制不住解码后的墨量/连通性), 但**不需要 VAE** ->
    #   省掉带梯度解码的巨大显存(batch>=1024 时必 OOM)。z_bg/delta_ink 是 DeformSkel
    #   里无条件注册的物理基准向量(SD-VAE 白背景均值 / 墨迹-背景差分)。
    if a.w_mass > 0:
        _zbg = model.z_bg.to(g2.dtype)
        _dink = model.delta_ink.to(g2.dtype)
        _dd = (_dink * _dink).sum().clamp_min(1e-6)
        _ip = ((g2 - _zbg) * _dink).sum(1, keepdim=True) / _dd      # (B,1,32,32) pred 墨迹图
        _it = ((T[bi] - _zbg) * _dink).sum(1, keepdim=True) / _dd   # (B,1,32,32) GT 墨迹图
        # ⚠⚠ 关键: 只做 L1 是不够的 —— latent MSE 有个**退化吸引子**: 输出"空白/背景"
        #   反而 MSE 更低(实测 blank 0.297 < g_std 基线 0.487), 模型会塌到墨量 0.0001。
        #   必须加上 **边缘梯度差** 项(空白图没有边缘, 会被重罚) —— 这正是 img_loss 里
        #   `|gx(pr)-gx(tg)| + |gy(pr)-gy(tg)|` 的作用, 这里搬到 latent 投影域做 VAE-free 版。
        _gx = lambda x: (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        _gy = lambda x: (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        _lm = ((_ip - _it).abs().mean()
               + (_gx(_ip) - _gx(_it)).abs() + (_gy(_ip) - _gy(_it)).abs())
        loss = loss + a.w_mass * _lm
    # ★ TV + Jacobian: 防撕裂/防折叠（纯几何量, 作用在偏移场）
    _r = model.regularizers()
    if _r is not None:
        loss = loss + a.w_tv * _r['tv'] + a.w_fold * _r['fold']
        loss = loss + a.w_tv_out * _r.get('tv_out', 0.0) + a.w_tv_res * _r.get('tv_res', 0.0)
        if 'tv_stroke' in _r and a.w_tv_stroke > 0:
            loss = loss + a.w_tv_stroke * _r['tv_stroke']
        if a.topo_mode:
            if 'l1_prune' in _r and a.w_prune_l1 > 0:
                loss = loss + a.w_prune_l1 * _r['l1_prune']
            if 'l1_lig' in _r and a.w_lig_l1 > 0:
                loss = loss + a.w_lig_l1 * _r['l1_lig']
    # ★ 图像空间监督: latent MSE 控制不住解码后的连通性(实测墨量 40%/28 段)。
    #   w_img>0 时才解码当前 batch 的前 img_batch 条, 走带梯度的 _decode_gray。
    if a.w_img > 0 and vae is not None:
        loss = loss + a.w_img * img_loss(g2, bi)
    # ★ 监督对比: 直接优化"同字内 g'(c,k) 比 g'(c,k') 更接近 g_gt(c,k)"
    _cl = None
    if a.w_contr > 0:
        _gid = torch.arange(a.group, device=DEV).repeat_interleave(per)
        _cl = contrastive_loss(g2, T[bi], _gid, Y[bi], a.contr_tau,
                               mode=a.contr_mode, margin=a.contr_margin, hard=a.contr_hard)
        loss = loss + a.w_contr * _cl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    # ★ pass 2: 图空间骨架损失（分块前向 + 立即反传，见 _pixel_loss_pass）
    _pxv = None
    if a.w_px > 0 and vae is not None and (step + 1) % max(1, a.px_every) == 0:
        _pxv = _pixel_loss_pass(_Gin, bi, _dt_in)
    opt.step()
    sched.step()
    if (step + 1) % 50 == 0:
        _cs = ('  contr %.4f' % float(_cl)) if _cl is not None else ''
        _ps = ('  px %.4f' % _pxv) if _pxv is not None else ''
        print('    step %5d  loss %.5f%s%s  %.0fs'
              % (step + 1, float(loss), _cs, _ps, time.time() - t0), flush=True)
    if (step + 1) % a.eval_every == 0:
        _ev = evaluate('step%d' % (step + 1))
        _s = _es_score(_ev)
        if _s is not None:
            if _s > _es_best + a.es_min_delta:
                _es_best, _es_bad = _s, 0
                torch.save(dict(deform=model.state_dict(), style_emb=a.style_emb,
                                step=step + 1, es_metric=a.es_metric,
                                es_score=_s, es_metrics=_ev), a.out + '.best')
                print('    [es] ★ 新最佳 %s=%.5f -> %s.best'
                      % (a.es_metric, _s, a.out), flush=True)
            else:
                _es_bad += 1
                print('    [es] 未改善 %d/%d (best %.5f, 当前 %.5f)'
                      % (_es_bad, a.es_patience, _es_best, _s), flush=True)
                if a.es_patience > 0 and _es_bad >= a.es_patience:
                    print('    [es] ★★ 触发早停 @ step %d (best %s=%.5f, 已存 %s.best)'
                          % (step + 1, a.es_metric, _es_best, a.out), flush=True)
                    break
    if a.save_every > 0 and (step + 1) % a.save_every == 0:
        _sd = dict(deform=model.state_dict(), style_emb=a.style_emb, step=step + 1)
        torch.save(_sd, a.out)
        # ★ 带步号的快照: 原来只写 a.out, 每 save_every 步互相覆盖 -> 无法回看历史。
        #   500 步一档 ≈ 26MB/档, 8000 步共 16 档 ≈ 0.4GB。
        _snap = '%s.step%06d' % (a.out, step + 1)
        torch.save(_sd, _snap)
        print('    [ckpt] %s @ step %d  (+ %s)' % (a.out, step + 1, _snap), flush=True)

print('\n[4] 最终', flush=True)
evaluate('final')
print()
print('=== 判读 ===')
print('  闭合率: >60% = 形变确实把 g_std 拉近该书家的写法')
print('  style-follow: 50% = 与随机无异(风格没驱动); >75% = 同一个 skel 对不同书家产出了对应 skel')
print('  off 的"风格底图"那一项: >0 说明风格专属的全局形变在起作用')
torch.save(dict(deform=model.state_dict(), style_emb=a.style_emb), a.out)
print('  已存', a.out, flush=True)
