# Built with Qwen：32张跨字风格迁移验证

这是研究/评估小样，不是已通过验证的合成训练集。未修改冻结Top20数据或主模型训练。

## 环境与模型

48主机工作目录 `/home/ds/Workspace/xy/qwen-calli-20261022`。
复用 `/home/ds/miniconda3/envs/pytorch/bin/python`：torch2.14.0+cu126、transformers5.19.0、accelerate1.15.0。
仅在任务 `vendor/` 安装diffusers0.41.0，运行时加入模块搜索路径，原环境diffusers0.40.0不覆盖；此前临时venv不再使用。
官方Qwen-Image-2.1-Turbo约32.45GB，ModelScope下载；下载前核对其权重SHA与HF revision fc13ba1986fcb905854dc8de2f52eee395f49b5b一致，完成后全文件校验。
原研究许可随模型保留。输出若用于训练并公开提供模型，需按许可保留显著Qwen归因；商业使用另行核查授权。

来源：[模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)、[许可](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE)。

## 固定设计

颜真卿、智永、赵佶，同为楷书；目标泉/梅/岳/霞，seed20261022/20261023。
目标字符不出现在这三位书家的任何书体训练集，也排除冻结Top20 validation/test及旧187评测字符。
每书家3张其他字符训练真迹；无目标字真迹输入。精确CSV、图像SHA及参考来源在manifest.json中。
源source标签不是已核验作品ID，不声称严格同作品参考；无法排除教师预训练见过历史真迹。
24张带参考生成+8张STD-only，使用相同目标/seed；不在prompt里写书家名，以便检验参考图而非名字先验。
STD用新bbox256图，黑字白底；参考图只调整画幅，不改变笔宽或对比度。优先剔除独立扫描边框。

## 执行与产物

`bootstrap.py`下载/校验，`infer.py --wait-download`等待下载完成后自动推理，随后`poster.py`生成匿名A/B/C对照。
1024方图、BF16、model CPU offload、CFG1、官方保存的8步sigmas，prefix KV cache开启。不做量化，不覆盖已有结果。
bootstrap_status.json/inference_status.json记录阶段、PID、完成数与错误；raw1024、白底256、逐图prompt/seed/耗时/峰值显存/SHA齐备。
poster.png展示全部32张；blind_key.json单独保存书家映射，review.csv由人工检查字符、风格、失败原因。
所有生成图writer_label均为空、状态unreviewed；不自动贴书家标签，不加入训练，不计算不存在的目标GT SSIM。

只有字形正确、跨字风格可辨、且不是STD轻微改写或参考字符复制，才考虑后续400张。

## 2026-10-23：无STD与配对GT评测

旧STD版本在15张完成后停止。`prepare.py --refs 6 --style-only`构造6张泉/梅小样，
每书家6个异字训练真迹，不传STD图片；`infer.py --root <目录> --skip-poster`独立保存结果。

泉/梅没有冻结数据中的同书家GT，不能以STD冒充真迹计算质量指标。
`paired.py prepare --source <无STD小样目录> --root <新目录>`从冻结Top20 test中，
为每位书家选2个不同于参考字的楷书目标，GT仅存在`gt_file`，不进入`images`。
`paired.py score --root <目录>`复用`src.eval.metrics`、`metrics_ink`计算SSIM、Alex LPIPS、
历史口径MSE、Ink-IoU、Ink-SSIM、Frag、SoftInk，逐图CSV和总体/分书家JSON保留。
GT必须原生256；生成图1024→256 LANCZOS，不做bbox重对齐或粗细校准。
6张只用于初筛，不能与完整187字评测均值直接比较，也不代表独立书家风格识别结果。

配对6张首轮结果（每书家2字）：SSIM 0.52277，Alex LPIPS 0.55255，Ink-IoU 0.17273，
Ink-SSIM 0.42750，Frag 1.42284，SoftInk比0.64122。
本地产物：`assets/synthetic/qwen_paired6_20261023/metrics.csv`、`metrics.json`；
冻结GT和输入清单在同目录`manifest.json`、`gt/`、`inputs/`。
