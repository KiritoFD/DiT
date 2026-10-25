# Qwen 字形双通道条件实验（A/B/C 臂）

目标：验证 Qwen-LoRA 生成的书家风格字形作为 U-Net 第二条件通道是否带来增量（用户方案C）。
底座：48 机 Deep-Sym4 114M `0085000.pt`（60M 曝光收敛点）。

## 数据与条件

- 子集：`train_top10_aug_sym.csv` 确定性抽样 5,000 行（seed90）→ `train_subset5k.csv`；
  分片按 img_id 查表，过滤安全。评测 `eval200_fixed.csv`（187 行）。
- Qwen 字形存储：`assets/qwen_glyph_store_20261025`，4,493 键（书家ID×书体ID×字），
  Qwen Turbo LoRA 20k 生成（256/8步/seed90）→ 灰度反转 → BOX 64×64 → 白墨黑底 uint8，
  加载时 σ0.65 soften、/255——与 TTF std_soft 存储逐位同协议（D1 诊断的依据）。
- 三臂（同数据/同 seed/同 5k 步/同 warm-start，仅条件不同）：
  - A `arm_a.json`：TTF-only（对照组，std_channels=1, qwen off）
  - B `arm_b.json`：Qwen 替换 TTF（std_channels=1, qwen replace）
  - C `arm_c.json`：双通道（std_channels=2, qwen dual, dropout 0.15）
- 调度：项目 A/B/C 协议"全局曝光时钟"——lr 6e-5、lr_decay_steps 117200（原值）、warmup 1500、
  85000→90000 段 LR 自然落 2.0e-5→1.3e-5；ckpt/eval 每 2500（87500/90000）。

## 模型与加载

`std_stem[0]` 从 Conv2d(1,64) 扩到 Conv2d(2,64)：通道0=85k 原权重逐位拷贝、通道1全零
（第0步输出与原模型严格等价，`verify_warmstart.py` 验证 334 键严格加载、0 个未批准新键）。
warm-start 经 `load_expansion` 审计路径，optimizer/EMA 重置，全 RNG 恢复——三臂完全同起点。

## 48 仓库补丁（`apply_patches.py` + 运行期修复，备份 `.bak_qwencond`，diff 留档）

- `src/unet/model.py`：std_channels 参数化 + forward 形状检查跟随 stem
- `src/unet/io.py`：Config 增 std_channels/qwen_mode/qwen_dropout/qwen_soft_root；build 传参；
  training_args 补演进缺失默认值（std_shape_log_every 等 11 项）
- `src/train/data_setup.py`：向数据集传 qwen_soft_root
- `src/utils/latent_dataset.py`：QwenGlyphStore 加载 + batch['qwen_soft']
- `src/train/trainer.py`：`_std_shape` 后拼/换 Qwen 通道（含训练期 dropout）；`_inj_probe` None 保护
- `src/eval/dataset.py` + `in_mem_eval.py`：评测缓存带 qwen_latent、缓存键补 qwen 字段、
  按模式拼接/替换 skels_latent；save_input_g 支持 2 通道
- `src/unet/variants/transfer.py`：std_stem.0 1→2 通道扩展（零初始化）
- 兼容性修复（老 unet 入口 vs 演进后 shared trainer）：补回 `build_cosine_scheduler`
  （src/train/step.py）、unet/train.py 去掉已废弃 condition_transform、单进程 gloo 组初始化、
  style anchor 字段默认值、std_shape 探针 hasattr 保护（DiT 专用）

## 入口

```
python -m src.unet.train --config tools/qwen_condition/arm_X.json \
  --warm-start .../0085000.pt          # X∈{a,b,c}
```

冒烟：30 步全链路通过（--max-steps 85030）。产物：`assets/results/qwen_cond_arm_{a,b,c}/时间戳-Unet-flow/`。
判定口径（用户定义）：C>A 且 B<A ⇒ Qwen 有增量但不可替代 TTF；B>C 则考虑替换；
A≈B≈C ⇒ Qwen 对 U-Net 冗余。字符正确率与独立 Writer 指标另行评测。
