# 18 · 模型层：ControlNet 设计与条件域理论

## 1. 架构

```
DiT-2Cond-S/2 (v2 架构)
  hidden=384, depth=12, heads=6, patch=2, latent 4×32×32
  RMSNorm / SwiGLU / QK-norm / 2D axial RoPE
  条件: y_callig(1013类,128d) + y_char(35130类,DINO 384d→ln/mlp)
  → factorized_add / sqrt(2) → adaLN 调制

ControlNet 注入分支
  ctrl_encoder: PatchEmbed(4ch→384, p=2) → 12×DiTBlock(与主模型同架构)
  injections: ZeroAdaLNInjection × 12 (逐层, 零初始化)
    out = x * (1 + s) + t    (s,t = zero_init_linear(feat))
  语义: init 时 s=t=0 → 恒等 → warm-start 完美继承主模型
  梯度种子: d(out)/d(W) = feat(非零) → W 立即有梯度
           d(out)/d(feat) = W(=0) → feat 后学
```

### 条件流
```
y_callig ──→ embedding ──→ proj ──┐
                                  ├──→ /sqrt(2) ──→ c = t_emb + y_emb ──→ adaLN(主模型) + adaLN(ctrl_encoder)
y_char   ──→ DINO+embedding ──→ proj ──┘
骨架latent ──→ ctrl_encoder ──→ injections[i](x, feat) ──→ 主模型逐层
```

## 2. 条件域三定理（实验实证，见 17_training_lineage §3）

### 定理 1: 条件域匹配
训练时条件风格与推理时必须一致。GT 书法骨架训练的模型对标准字体
骨架响应 IoU 0.33（近乎忽略）；反之 std-skel 训练的模型对 GT 骨架
ΔSSIM -0.005。跨域条件 = 分布外输入 → 注入分支无法激活。

### 定理 2: CFG 上限
骨架条件下 CFG > 1 单调有害：cfg 1.7→0.7 使 SSIM 0.683→0.752、
失败率 8%→2%。原因：骨架条件已完全确定结构，CFG 的"无条件→有条件"
外推把样本推出数据流形。最优区间 0.7-0.85。

### 定理 3: 冗余条件门控
当骨架信息可由 char ID（DINO 嵌入）推出时，注入分支被梯度优化
门控掉——std-skel 训练 30k 步后跟随 IoU 仅 0.09-0.19。
对策：char dropout 大幅提高（迫使依赖骨架），或骨架携带
char ID 不可推断的实例信息（GT 骨架天然满足）。

## 3. 失败模式清单

| 模式 | 案例 | 根因 |
|---|---|---|
| 密集字崩坏 | 聽(20+笔) 笔画粘连 | S/2 容量 + 每字中位 5 张样本 |
| 彩色伪影 | 稷 (color=13.4) | CFG 外推把样本推出 VAE 流形 |
| 黑白反转 | 祭 (黑底白字) | 训练数据含反色拓片（已修 896+97+39 张） |
| 骨架跟随失败 | std-skel ctrl IoU 0.33 | 条件域不匹配（定理 1） |

## 4. 注入方式对比

| 方式 | 语义 | 适用 |
|---|---|---|
| zero-conv add (旧) | x += W·feat | 简单，但注入幅度受线性限制 |
| **ZeroAdaLN modulate (现行)** | x*(1+s)+t, s/t=zero_init_linear(feat) | 门控更灵活，与主模型 adaLN 对称 |
| cross-attention | feat 作 KV | 未采用（token 数不匹配） |

## 5. 零样本推理管线（已验证）

```
用户输入字 → 标准字体渲染(256×256 白底黑字) → 骨架化(1px) → 膨胀3px
  → VAE encode → 骨架 latent → fame-ctrl / v8b → 书法字
```

- 字库覆盖：楷 simkai / 行 STXINGKA / 隶 SIMLI（草/篆/六体无开源标准字体）
- 跟随 IoU 0.82-0.87（对标准字库骨架）
- 字形正确率 100%（8/8 测试样本目检）

## 6. 可选 v2：字→骨架小网络（NLP 先验）

### 动机
标准字体只覆盖规范字形；书法骨架有笔势变化。小网络从字预测
"书法合理的骨架" = 桥接标准字体骨架与书法骨架。

### 架构（10-20M 参数，latent 域）
```
输入:
  ① 标准字体骨架 latent (4,32,32) — 结构锚
  ② 部件分解序列 (IDS 表达式 → Transformer 编码) — 结构先验
  ③ DINO char embedding (384d) — 语义/字形先验（与主模型共享冻结表）
  ④ 可选: 字音/字义 embedding (BERT/Glyce)
输出: 书法骨架 latent (4,32,32)
```

### 训练数据
fame 训练集每字多位书家骨架（51k 条 (字,书体,书家,骨架) 对）
→ 一字多骨架 = 天然 one-to-many → 模型学分布而非单一答案。

### 评估
跟随 IoU（对 GT 骨架）+ 字形分类器准确率。
