# U-Net 与同期 DiT（2026-10-18）

只评估，不训练、不修复在跑训练。U-Net已停止，最后日志21700，最新完整checkpoint20000。

`evaluate.py`：读取原始U-Net完整checkpoint，核验`.done`/size/SHA；从`args.architecture_config`
恢复真实架构（兼容这批缺顶层元数据的checkpoint），严格加载raw，明确传`eval_vae_path`。
只在该评测进程中将EvalRequest标记raw，使用原统一评测器与FM求解器，生成DONE。

```bash
python tools/analysis/unet_compare/evaluate.py --run assets/results/Unet/20261017-233321-Unet-flow \
  --out assets/audits/unet_vs_dit_20261018 --steps 15000 20000
```

DiT 20×384历史PNG从4090 run `20261017-050900-v90A_20L_stdstrong` 的40k/50k/60k取回；
不在4090运行新推理，不影响v92训练。全部原PNG/protocol/metrics/DONE保留。
用现有`tools/eval_gen_with_disc.py`统一评分，保存`writer_per_image.json`。
独立holdout CE权重`assets/style_disc_ce_holdout.pt`；书家原型由同一训练库构建，GT ceiling71.66%。
这不是人工风格偏好或完全独立来源的保证，缺STD原型时不报告STD margin。

```bash
python tools/analysis/unet_compare/compare.py --root assets/audits/unet_vs_dit_20261018
```

`compare.py`核验5组DONE和GT逐像素对齐，然后统一对PNG重算SSIM/LPIPS-VGG/Ink-IoU/Frag、
薄笔画召回、缺失/粘连组件，并合入独立Writer逐样本结果。按字符分组bootstrap5000次。
PNG重评SSIM不是原训练日志的浮点GPU SSIM，不混用两者。
输出comparison.json与固定8例poster。子集由实际训练CSV字符集合重新推导：13个全局未见字符，
不能未经固定索引证据就套用历史“62 unseen”标签。

主对照15k×512=40k×192=768万张；20k×512=1024万，DiT50k/60k分别960万/1152万。
U-Net72.09M/raw vs DiT55.03M/EMA、LR/条件注入/batch/EMA interval不同，不是纯架构因果消融。
单采样seed区间不包含训练seed不确定性。报告见`ANALYSIS/unet_vs_dit_20261018.md`。
