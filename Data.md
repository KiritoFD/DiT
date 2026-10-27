# 数据资产与重建

2026-10-27 15:18复核：两机仍在使用matrix64既有S/J数据配置，本次进度检查没有新增、传输或重建数据；下方配方与身份约束继续适用。

更新：2026-10-27。原则见[AGENTS.md](AGENTS.md)：**数据缺了先搜索本机、再本地重建；跨机传输几乎不考虑**（本文件登记每个资产的重建配方与配对校验）。

## 机器与仓库

| 机器 | SSH别名 | 仓库路径 | Python |
|---|---|---|---|
| 4090 24GB | `4090` | `/root/Workspace/xy/DiT` | `/opt/conda/envs/cu121/bin/python` |
| 48GB | `48` | `/home/ds/Workspace/DiT` | `/home/ds/miniconda3/envs/vcp-embryo/bin/python` |

## S组协议资产（Top10，两机一致）

| 资产 | 路径（相对仓库） | 校验 | 缺失时 |
|---|---|---|---|
| 训练CSV（23,485行，canonical 0–9） | `assets/u100_p1p2_20261025/p2_train.csv` | 两机SHA一致 `976b573d…` | 重建：从原始标注CSV按Top10筛选（见u100_pretrain记录） |
| latent shards（26片） | `assets/generic48_kxl_top10_20261022/latents` | 每机封存时记path+size | 重建：VAE编码管线，见`PROVENANCE.json` |
| STD store（15,000键 64px） | `assets/generic48_kxl_top10_20261022/std_bbox64` | `index.csv`两机SHA一致 `c02cef37…` | 重建：字体渲染管线 |
| **std_shape bank（strong形变）** | 4090/48：`assets/u100_pretrain_20261024/std_shape_bank.pt`（约2.8GB） | 配对校验=bank内`source_index_sha256`须等于本机store的`index.csv` SHA | **本地重建（见下）** |
| tables（char/font/callig remap） | `assets/u100_pretrain_20261024/tables/` | KB级JSON | 从4090复制（映射文件随数据定义，KB级直传可接受）或按store索引重建 |
| 评测CSV（S组过程筛选100张） | `assets/u100_pretrain_20261024/validation_val100.csv` | 行数=100 | 重建：从eval池抽样并固定 |
| 187字终评CSV | 48：`exp-std/csv/eval200_fixed.csv`（187行） | — | 重建：字符面板生成脚本 |

### std_shape bank 重建配方（2026-10-27在48实测通过）

store 的 `index.csv` 两机一致 ⇒ 48 本地重建即同协议，无需传 2.8GB：

```bash
# 1) 确定性重渲256px raster；逐位BOX下采样校验store 64px图，任何不一致直接失败
python -m tools.u100_pretrain.build_rasters \
  --store assets/generic48_kxl_top10_20261022/std_bbox64 --out /tmp/rasters_top10.pt
# 2) GPU strong形变bank（seed=90，8变体×3档，FP16），source_index_sha256自动继承
python -m tools.u100_pretrain.precompute_bank \
  --raw /tmp/rasters_top10.pt \
  --store assets/generic48_kxl_top10_20261022/std_bbox64 \
  --out assets/u100_pretrain_20261024/std_shape_bank.pt
```

GPU占用分钟级；build_rasters为CPU渲染约万五千键。**不要**用别的store（如48的`exp-std/data/std_soft_glyphs_v1`）构建的bank配本store——`StdShapeDataset`的配对校验会正确拒绝（2026-10-27实测拦截过一次）。

## J组协议资产（Deep85，仅48）

| 资产 | 路径 | 说明 |
|---|---|---|
| Deep85 U-Net 85k EMA | `assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0085000.pt` | SHA `f80304d7…`；J组统一起点 |
| 未增强26,002行CSV | `assets/style_matrix_20261027/deep85_decoder/48_v2/train_unaugmented.csv` | 由原始77,823行CSV滤除`aug`生成（`joint/prepare.py`），SHA `935cc552…` |
| 原始Writer remap | `assets/triple_tables_best_minimal/callig_remap.json`（53→0…9001→9） | 不可用P2 canonical读Deep85书家表 |
| 图像损失校准 | `…/48_v2/calibration.json` | image_weight=0.0212917472；复用不重算 |
| VAE | `pretrained_models/sd-vae-ft-ema`（两机） | scale=0.18215、shift=0；冻结部分指纹`01a66068…` |

## 历史教训（登记为规则）

- 2026-10-27：为对齐S组bank曾启动4090→48的2.8GB中转（4090上行~440KB/s，需1h+）——错误做法。正确做法是上节本地重建，分钟级完成且`source_index_sha256`自动一致。
- 传输仅接受的例外：KB级映射/清单文件，且确认无重建脚本时。
