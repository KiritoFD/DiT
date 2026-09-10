# 45. 书家词表收紧 + 语义书家表 + Cross-Attention 骨架注入 (c41 系列)

日期: 2026-09-08
状态: 进行中 (c41x 训练)
前置: 42, 43, 44 号文档

## 0. 环境事实 (纠正记录)

- **训练/评测统一用 cu121 环境** (`/opt/conda/envs/cu121/bin/python`, torch 2.x,
  F.scaled_dot_product_attention 原生可用)。
- base 环境 (`/opt/conda/bin/python`, torch 1.13) 仅历史脚本残留: 其 xformers
  0.0.16 在 CPU fp32 上无可用 attention kernel (任意头数都 NotImplementedError)。
  **daemon 启动必须用 cu121 python** —— 曾因用错环境误判为"8 头模型 CPU 不兼容"。
- 配置键 `gpu_eval_n` 不是 train.py 的 argparse 参数, config 里的值会被静默丢弃;
  cpu_eval_daemon 的评测样本数改为**按 eval csv 行数**确定 (2026-09-08 修)。

## 1. 诊断链 (为什么走到这里)

1. **S 容量欠拟合** (44 号文档): train/strict velocity loss 全 t 桶高且平,
   低 t 桶连训练原图都重建不动 → 扩容 Sp/2 (h=512, d=12, 65.6M)。
   效果: strict 0.506→0.518, follow-IoU3 0.205→0.289 ✓
2. **Sp 后期记忆化**: seen10 后期 0.56 穿越 strict 0.52 —— (g,callig)→图 查表化。
   反记忆化: callig cond_drop 0.5 (sp2/c41d01)。
3. **风格塌缩顽固**: 生成统一浓墨, callig 轴学不出书家差异。
   根因链: (a) 词表 1013 行 96% 死行 + 端到端下 41 embedding 塌缩 (pairwise
   cos 0.323); (b) callig 只能全局 adaLN 调制, 表达不了结体差异; (c) g 注入
   (ZeroAdaLN 固定 1:1 位置调制) 表达力不足。

## 2. 本轮三件套

### 2.1 干净 41 词表 (非重映射补丁)

- `src/utils/callig_map.py`: raw calligrapher_id → 连续 0..40, 持久化 json
  `5script/callig_id_map.json` (train/eval/inference 三端共用)。
- 数据层 (`latent_dataset.py` / `dataset.py`) 查表映射; `make_eval_cache` 加
  `callig_id_map` 参数 (worker / 批量评测器都传)。
- `num_calligraphers` 自动收紧为词表长度 (41), 表 42 行 = 41 + CFG null。
- audit 脚本 `_ot_scratch/audit_callig41.py`: 三 csv 覆盖/行序/forward 边界全 PASS。
  注: "不同书家输出相同" 在未训练模型上是预期 (adaLN 零初始化, c 未接入输出)。

### 2.2 对比预训练书家表 (方案 i)

- `tools/pretrain_callig_emb.py`: SupCon (同书家正对, temp 0.07, logsumexp
  稳定实现) + DINO CLS 质心锚定 (A·E[y] ↔ centroid[y], 让 embedding 继承
  真实风格语义)。
- DINO 特征: `5script/dino_cls_train.npz` (51321×384, 含 raw id, 覆盖 41/41
  书家 min 268 样本)。
- 结果: pairwise cos 0.323 → mean 0.024 / |0.073| / range [-0.24, 0.29] ✓
- 接入: `--callig-emb-pretrained` 加载前 41 行 + `--freeze-callig-table`
  (null token 拆独立 Parameter 保持可训练, LabelEmbedder.freeze_table())。
- 教训: E init std=0.02 会被 P 的 bias 量级淹没 → 全部 z 同向 → SupCon 退化。
  E 用 std=1.0, 投影头去 bias; 手写 exp/log SupCon 在该环境数值异常,
  用 `torch.logsumexp` 标准实现。

### 2.3 Cross-Attention 骨架注入 (c41x)

- `src/model/dit.py` 新增 `ZeroCrossAttention`:
  Q = x token, K/V = g_tok (+ 与 x 同网格 16×16 的固定 sincos 位置嵌入 ——
  g_tok 是 conv 特征图展平, 无绝对位置, 不补则 2D 绑定不成立);
  out_proj 零初始化 → 初始恒等。
- `glyph_inject_mode`: "adaln" (旧默认, 旧 ckpt 兼容) | "xattn" (新)。
  eval 构建器 (worker/批量评测/follow) 按 ckpt args 的 mode 选类,
  否则 state_dict 键不匹配 assert 崩。
- c41x: glyph_inject_layers=12 (全层), 其余 = c41d01 配方。

## 3. drop / null / CFG 的理论立场 (2026-09-08 用户裁定)

- 部署是**闭集 41 书家**, 无未见书家兜底需求。
- CFG 理论值 **s = 1.0** (条件模型本身即答案); s>1 = 放大弱条件信号,
  s<1 = 往风格均值稀释 (现 eval cfg 0.7 对风格分化是反方向, 待重估)。
- callig cond_drop 0.5 → **0.1** (c41d01 起): 冻结表已免疫塌缩, 任务闭集
  不需要泛化性, 先学会。null 行保留但降级为低权重正则。
- 校准公式 (备查): s* ≈ r_data / r_model (数据风格分离度 / 模型条件响应度)。

## 4. 实验记录

| 实验 | 配方 | 结果 |
|---|---|---|
| fame3 (S) | 33M, drop.25 | strict 0.506@57.5k, IoU3 0.205 |
| Sp | 65.6M, drop.1 | strict 0.518@80k, IoU3 0.289, seen 记忆化 0.56 |
| sp2 | Sp+repa.03+drop.5 | (被 c41 系列取代) |
| c41 (drop.5) | +干净词表+冻结表 | 3.8k 步止, 无 eval |
| **c41d01** | c41+drop.1 | 17:44 起跑, 被 c41x 取代 |
| **c41x** | c41d01+xattn×12 层 | 进行中 |

## 5. 待验证

- xattn 注入是否提升 follow-IoU3 (0.289 → 0.3+) 与结构精度
- 书家风格分化 (海报列间差异; callig-cfg 扫描 s∈{0.7,1.0,1.5} 备选)
- 记忆化分叉 (seen-strict) 是否被 drop 0.1 恶化
