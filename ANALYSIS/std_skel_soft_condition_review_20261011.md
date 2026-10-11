# Soft std-skel 方案评审（2026-10-11）

## 结论

首版应选择：**中性 TTF 字面直接做 Area/BOX 下采样，再做轻度 Gaussian blur**。

不要在首版把字面先 skeletonize，再做固定宽度膨胀。后者可作为严格消融臂保留，但不应作为默认方案。这里的“直接 blur”仅指**标准字体渲染图**，绝不能用对应训练真迹图；对真迹直接模糊仍会泄漏书家、书体、笔画粗细、连笔和墨色。

推荐载体：512×512 固定字面渲染（不按每字 bbox 拉满）→ 64×64 Area/BOX → 训练时 `sigma ~ U(0.4, 1.0)`，评测固定 `sigma=0.65` → 单通道 `[0,1]`，墨迹为 1。

## 为什么直接 soft mask 更适合

### 1. 我们要保留的是“部件占据区域”，不是一条唯一中心线

直接字面保留偏旁比例、包围关系、留白、闭环和部件间距。对“體、獨”等密集字，这些信息比每一笔的中心线更稳定。Skeletonize 是多对一的拓扑压缩：在交叉、粘连、钩挑和密集平行笔画处容易产生毛刺、错误分叉或断点；随后膨胀无法恢复已经丢失的轮廓关系，还可能把小间隙合并。

### 2. 固定标准字体不会泄漏样本书家，只会带来可控的字体先验

同一个字符的条件图对所有书家样本都相同，因此不会泄漏目标真迹的实例级笔触。它可能带入中性 TTF 的结构偏置，但这可以通过以下方式控制：

- 首轮每种书体固定一个主字体，并把逐字 fallback 写进索引；另保留“三体共用中性字体”作为后续消融，区分书体先验收益与字体风格泄漏；
- 低分辨率 Area 下采样和轻模糊削弱衬线、收笔和边缘细节；
- 条件 dropout 和随机 sigma 防止模型逐像素照抄；
- 第二轮再做多中性字体随机化，学习对字体细节不变，而不是先破坏拓扑。

### 3. “去笔画特性”不等于“去掉笔画宽度全部信息”

字的空间布局不仅由中心线决定，也包括部件占据范围、字腔大小和疏密。把所有笔画规范为固定宽度会把这些内容信号一起抹掉。书法的真实粗细和墨色仍由书家/书体条件决定；soft mask 只提供一个低频占据先验，并不需要成为目标轮廓。

### 4. 新方案应绕开 VAE 条件载体

历史结论“3px 骨架优于 1px”来自稀疏线条经过连续 VAE 后的失真：1px 往返 IoU 0.861，3px 为 0.960。它说明的是 **VAE 不适合搬运细线条件**，不说明模型需要固定宽度骨架。

新的 1×64×64 soft mask 应直接进入小型 CNN，再变成 16×16 token；这样既避开 VAE，也避免条件表示随主生成 VAE（SD/Calli/physics）切换而漂移。

## 为什么 v70 的失败不能直接否定本方案

v70 实际同时改变了多个关键因素：

1. `no_char_cond=true`：删掉了全局字符表；
2. 条件载体是 `std skel -> 1px skeleton -> 约 7px dilation -> SD-VAE 4×32×32 latent`；
3. 再由现有 `glyph_embedder` 编成 16×16 token；
4. 因而它同时测试了“去字符表”“中心线载体”“VAE 稀疏条件瓶颈”，不是只测试 soft std glyph。

v70 在 65k 的 SSIM 为 0.5535，且 std-skel 贡献探针约 2.7%；但 GT-skel 是 10.4%，说明主干并非完全不会使用空间条件，更像是标准条件的信息形态和全局内容源不足。历史上“表+骨架”与“仅表”终点相同，也只说明旧载体在闭集表条件上冗余，不能说明一个独立的开集 soft-glyph 编码器没有价值。

## 两种候选的直接对比

| 维度 | TTF 字面直接下采样+blur | skeletonize→dilate→blur |
|---|---|---|
| 复杂字部件/字腔 | 保留较好 | 容易在密集区合并或断裂 |
| 笔画宽度/字体细节 | 保留一部分，但可用低分辨率与随机 blur 抑制 | 抑制更强 |
| 拓扑稳定性 | 高，主要误差来自字体缺字 | 依赖阈值和细化算法，易生毛刺/分叉 |
| 条件连续性 | 灰度软边天然连续 | 硬细化与膨胀后再软化，中间步骤不连续 |
| 单笔字/简单字比例 | 固定字面大小时自然 | 细化后容易显得过细、过空 |
| 实现与复现 | 简单 | 参数多：阈值、连通域、细化算法、宽度、膨胀结构元 |
| 首版建议 | **主方案** | 仅做消融臂 |

## 接入现有 DiT 的建议

### 数据侧

- 新增独立 `std_soft` 通路，不覆盖现有 `skel_latent`；
- 以 `(font_hash, Unicode)` 缓存 uint8 64×64 图，几十 MB 即可覆盖数千汉字；
- 缺字、方框 glyph、空图必须硬报错或进入明确的 `missing.txt`，禁止静默回落为全零；
- 保持固定字体字面大小和字体 metrics 对齐，不按每个字的紧 bbox 拉伸。

### 模型侧

建议单独增加 raw-mask encoder：

```text
1×64×64
  -> Conv 1→32, 3×3, stride 2 + SiLU
  -> Conv 32→64, 3×3, stride 2 + SiLU
  -> Conv 64→hidden, 1×1
  -> hidden×16×16 -> 256 spatial tokens
```

不要把 1×64×64 软图误送给当前面向 4×32×32 VAE latent 的 `glyph_embedder`。如果从现有强 checkpoint 热启动，最后一层或注入残差必须零初始化，使 step 0 与原模型完全等价；条件 dropout 后也必须同时清零空间 token 和其池化出的全局向量。

### 全局内容与空间内容必须分开看

首轮为了回答“soft glyph 作为空间先验是否有效”，应保留现有全局字符条件，避免重演 v70 的混杂。真正开集时，再让同一 soft-glyph encoder 同时产生：

- 16×16 空间 token：告诉模型部件大致在哪里；
- pooled global glyph embedding：告诉模型这是哪个字。

OOV 路径不能再强制查询不存在的 `character_id` 表项；CFG/dropout 也必须把表条件和 soft glyph 同步处理，避免字符身份从另一条支路泄漏。

## 最小、可判定的实验

从同一个主线 checkpoint、同一随机种子和同一数据/调度启动三臂：

| 臂 | 空间条件 | 全局字符条件 | 目的 |
|---|---|---|---|
| C0 | 无 | 保留 | 当前基线 |
| C1 | **直接 soft mask**，`sigma~U(0.4,1.0)` | 保留 | 推荐方案 |
| C2 | skeletonize→匹配墨迹覆盖率的 dilation→同分布 blur | 保留 | 回答“去宽度特征是否值得拓扑损失” |

三臂必须匹配输入的平均能量/墨迹覆盖率，否则模型可能只是在利用条件强度差。评测除 SSIM/LPIPS/frag/HF 外，至少加入：

- 同 seed 配对的字符识别或 glyph retrieval；
- 组合未见/OOV 字子集；
- 条件置零与 batch 内错配后的性能下降（直接测模型是否真的使用 g）；
- 书家内多样性，防止空间条件把不同书家拉回同一个印刷体。

若 C1 与 C2 的结构正确率相当，优先 C1；只有当 C2 在 OOV 字符正确率上稳定领先、且没有造成 LPIPS/HF/风格多样性下降时，才值得采用细化—膨胀链。

## 代码证据位置

- `src/train/configs/v70_aug_sp_stdskel_c2ot.json`：v70 同时设置 `no_char_cond=true`、`skel_as_glyph_cond=true`，条件来自 `shards_std_w7`。
- `tools/build_std_skel_widths.py`：现有宽骨架流程明确为二值化→skeletonize→dilation→SD-VAE encode。
- `src/model/dit.py`：生产训练入口实际使用的 DiT；旧 `glyph_embedder` 面向 4×32×32 latent，并以 token-add 和逐层 AdaLN 注入。
- `src/utils/latent_dataset.py`：当前标准条件按 latent shard 读取；增强样本通过 `std_path` 映射到基础字。
- `README.md` 与 `docs/system/76_experiments_20261005.md`：记录 v70、表/骨架对照和 1px VAE 损失证据。

## 已实现链路（2026-10-11）

这次只完成代码和确定性缓存构建，没有启动训练实验。

### 字体选择与实测覆盖

覆盖统计针对 `exp-std/csv/train.csv + eval200_fixed.csv + seen20.csv` 的实际 `(script_id, character)` 词表；不是根据“GBK/GB2312”标签推测。

| 书体 | 当前主 TTF | 主字体命中 | 明确 fallback | 最终覆盖 |
|---|---|---:|---|---:|
| 楷 | `simkai.ttf` | 3561/3564 | `simsunb.ttf` 3 字 | 3564/3564 |
| 行 | `STXINGKA.TTF` | 2400/3571 | `NotoSerifSC-VF.ttf` 1168 字；`simsunb.ttf` 3 字 | 3571/3571 |
| 隶 | `SIMLI.TTF` | 1298/1316 | `NotoSerifSC-VF.ttf` 16 字；`simsunb.ttf` 2 字 | 1316/1316 |

`ZhiMangXing-Regular.ttf` 已加入行书级联作为可移植的 OFL 备选；它在当前行书词表同样是 2400/3571，和华文行楷覆盖集合相同，所以在华文行楷可用时不会改变逐字选择。

方正王羲之行书是有价值的高质量候选，但当前机器和仓库中没有获得授权的实际 TTF，因此没有把一个未经 cmap 审计的字体名伪装成可用主字体。方正官网确认它是 2023 年行书、字符集标为简繁扩展 GBK，并明确要求按场景获取授权：<https://www.foundertype.com/index.php/FontInfo/index/id/6716.html>。拿到合法 TTF 后，只需把真实文件放到字体目录并置于 `configs/std_soft_fonts.json` 行书列表首位，再运行同一 strict audit；模型和数据代码无需改动。

### 数据与模型接口

- `configs/std_soft_fonts.json`：三书体字体级联、搜索目录和选择理由。
- `tools/build_std_soft_glyphs.py`：逐字体 cmap 审计；固定 em 字面渲染；512→64 BOX 下采样；输出 `index.csv`、`missing.txt`、字体/CSV SHA256 provenance 和限宽预览。
- `src/utils/std_soft_glyph.py`：训练期随机 blur、评测固定 blur、严格 `(script_id, char)` 查找；缺字直接报错，不返回静默零条件。
- `src/utils/latent_dataset.py`：新增独立 `std_soft` batch 字段，不覆盖旧 `g`/`skel_latent`。
- `src/model/dit.py`：新增 `1×64×64 → 32×32 → 64×16×16 → hidden×16×16` 编码器；末层可 zero-init；token 数与图像 latent 不一致时硬报错。
- `src/train/train.py`：新旧三种 g 来源互斥校验、训练数据接线、raw mask 对应的零背景增强语义。
- `src/eval/inference.py`、`src/eval/in_mem_eval.py`：评测固定 sigma，同一路径查表；输入预览直接保存 raw mask，不经过 VAE。
- `src/train/configs/v80_aug_sp_stdsoft_c2ot.json`：已配好的代码路径配置，仅用于后续启动，不代表已经运行实验。

本地已生成忽略版本控制的数据缓存 `exp-std/data/std_soft_glyphs_v1`：8451 个条目、strict 缺失数 0。每个 PNG 保存的是**未 blur** 的 uint8 base mask，训练/评测 blur 在读取时完成，因此无需为不同 sigma 重建缓存。

### 验证

- 所有改动 Python 文件通过 `py_compile`。
- raw store 读取、Gaussian 软边、评测 cache `(1,1,64,64)`、新编码器 `(N,1,64,64)→(N,D,16,16)`、完整 forward、旧 4 通道编码器兼容和错误尺寸 guard 均通过 CPU smoke。
- 新增 `tests/test_std_soft_glyph.py` 覆盖上述契约；当前系统 Python 未安装 `pytest`，因此未执行 pytest runner，本轮没有擅自安装依赖。
