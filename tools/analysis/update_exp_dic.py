with open('exp.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update table in line 1
old_table = """| 实验组别 | 详细架构与变动 | 参数量 | 优化与调度 | 样本曝光量 | 当前状态与进度 |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **U100-Base**<br>(`baseline_resume`) | 原始 72M CNN U-Net：channels [192, 384, 576]，Encoder/Decoder 各 2 ResBlocks/Stage，14 ResBlocks + 5 SDPA Attention。保留旧 time/writer/script 共享融合。 | 72,088,676 | Global Batch 512，LR 6e-5（cosine 调度至 100k，min_ratio 0.1，warmup 1500），WD 0.02，EMA decay 0.999（interval 2）。 | 新增 35.84M<br>累计 51.20M | **在训**：已达 ~50,300 步（50.3% 过半），速度 ~2.76 step/s，显存 13.80G。50k 完整 checkpoint/评测已落盘通过。 |
| **U100-Writer**<br>(`G_writer_only`) | 相同主干，仅在每个 ResBlock 新增 32→16→2C 的 Writer-only 残差 AdaGN 调制支路。使用 `--resume-expanded` 继承原基线优化器 slots。 | 72,291,908 | 同上；新增独立调制参数从头初始化。 | 0 (已取消) | **已取消**：依据用户指令于 05:32 从队列安全移除，不占用机器资源，让位给 DiC 条件门控与分尺度条件实验。 |"""

new_table = """| 实验组别 | 详细架构与变动 | 参数量 | 优化与调度 | 样本曝光量 | 当前状态与进度 |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **U100-DiC-768**<br>(`DiC_U100_b768`) | 75.8M 全功能 DiC 架构 U-Net：全开 `residual_cond_gate` 有界条件残差门控、`stage_specific_cond` 多尺度独立条件（adapter 模式）、`act_fn='gelu'` 全网 GELU 激活函数。 | 75,831,524 | Global Batch 768，LR 7e-5（cosine 调度至 100k，warmup 1500，min_ratio 0.1），WD 0.02，EMA decay 0.999（interval 2）。 | 目标 76.80M | **在训 (新启动)**：已推进至 200+ 步，速度 ~1.80 step/s（~1,382 samples/s），显存占用 21.36 GiB / 48 GiB。PID 3458877。 |
| **U100-Base**<br>(`baseline_resume`) | 原始 72M CNN U-Net：channels [192, 384, 576]，Encoder/Decoder 各 2 ResBlocks/Stage，14 ResBlocks + 5 SDPA Attention。保留旧 time/writer/script 共享融合。 | 72,088,676 | Global Batch 512，LR 6e-5。 | 累计 28.16M | **已正常收尾结案**：在 55,000 步（55%）达成收敛，SSIM 达到 0.6017，LPIPS 0.3317，IoU 0.3192，完整权重已归档。 |
| **U100-Writer**<br>(`G_writer_only`) | 相同主干，仅在每个 ResBlock 新增 32→16→2C 的 Writer-only 残差 AdaGN 调制支路。 | 72,291,908 | 同上。 | 0 (已取消) | **已取消**：依据用户指令于 05:32 从队列安全移除，不占用机器资源。 |"""

if old_table in text:
    text = text.replace(old_table, new_table)
    print("Replaced table successfully.")
else:
    print("Warning: old_table not found exactly, check diff.")

# Append 55k milestone line to U100-Base table
target = "| **50k** | 25.60M | **0.59926** | **0.33352** | **0.32017** | **2.5848** |"
line_55k = "\n| **55k (终点)** | 28.16M | **0.60167** | **0.33167** | **0.31921** | **2.6791** | `eval_auto_55000.json`、`0055000.pt.done`（收敛结案终点，SSIM 站上 0.6017） |"

idx = text.find(target)
if idx != -1:
    end_of_line = text.find("\n", idx)
    text = text[:end_of_line] + line_55k + text[end_of_line:]
    print("Appended 55k milestone successfully.")

with open('exp.md', 'w', encoding='utf-8') as f:
    f.write(text)

print('exp.md update complete.')
