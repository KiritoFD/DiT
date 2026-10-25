# Built with Qwen：冻结书家表条件LoRA

## 2026-10-24：Top10全量纯Prompt条件LoRA

独立目录`/home/ds/Workspace/xy/qwen-lora-top10-20261024`，直接train.py（无runner）。
数据：Top10全量train23485/val100/test200（楷9363/行12285/隶1837，10书家真实组合，无阈值清洗）。
条件=纯文字prompt：`书法家：{名}。书体：{书}书。请用该书法家的{书}书风格书写汉字「{字}」。纯白背景…`。
不再使用32D冻结书家表→4token、书体embedding→1token链路；writer/script标签只用于均衡采样与评测选图。
只训Attention q/k/v/out rank16 LoRA（16.78M）；文本缓存键=（字,书家,书体）17,453键。

分辨率裁决实验（res_probe.py，底模9字×256/512对照GT）：SSIM .588/.597、LPIPS .507/.501、
Ink-IoU .184/.192，256无结构崩坏→正式训练切256（GT原生分辨率，不再512上采）。
加速路径：关梯度检查点（train.py改为config开关`gradient_checkpointing`）+降分辨率。
OOM实测：512/B8无ckpt 46.5GiB爆、256/B16无ckpt 46.9GiB爆（文本token不随分辨率缩，
每样本序列≈80文本+64图像≈144 token，激活≈1.7GiB/样本）；最终**256/B12/无ckpt，峰值43.9GiB稳定**，
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。1.215s/步=0.101s/样本，比512/B8/ckpt快5.2倍。
配置top10_turbo.json：256/B12/rank16/LR1e-4余弦至1e-5/warmup100/10,000步（120k样本≈5.1epoch），
每1000步保存+评测（每书家×书体1张，原生8步，预计总~4h）。数据构建`data.py --full-top10`。

## 2026-10-24：非Turbo同预算对照

独立目录`/home/ds/Workspace/xy/qwen-lora-base-20261024`，runner1642156。
使用已完整校验的非Turbo基模，从零初始化LoRA/条件投影，重新编码VAE后验和文本，不加载Turbo适配器或缓存。
base6000.json复现Turbo首轮2k+续训4k的逐步LR：前2k峰值1e-4余弦至1e-5，
再100步升至5e-5，余下余弦至1e-5；B8/rank16/512/数据/冻结表/seed/保存间隔相同。
非Turbo用原生40步验证，Turbo对照为8步；质量和推理墙钟必须分开比较，不声称等采样算力。
只训练LoRA与条件适配器，原生主干权重/VAE/文本编码器冻结，初始权重不复用Turbo。
每500步生成同6字及统一指标；当前仍应以远程run_status和train_status区分编码与实际优化。

## 最新：先训练Turbo（2026-10-23）

**最新追加**：用户要求继续训练，已从2k完整resume至6k（trainer927137）。
train.py --resume <checkpoints/0002000> --config <turbo_extend6000.json>，精确核对LoRA张量与冻结表，
恢复条件投影、AdamW状态与RNG。LR从原1e-5用100步升至5e-5再余弦下降；不重做数据/文本编码。
2500至6000每500保存/验证，原500–2000产物保留。run_status.json链接活动train_status.json。

用户指定后已实际切换到`/home/ds/Workspace/xy/qwen-lora-turbo-20261023`。
正式trainer614522，完整2786图VAE后验及1694字符文本缓存；60/2000步时实测B8约4.10秒/步，峰值18.58GiB。
采用turbo.json，rank16、冻结表、LR1e-4、warmup100、2000步不变，每500步保存/验证。
验证使用原生8步，记录并断言实际采样步数，不能把请求40步但实际8步误标成40步。
run.py负责校验全部Turbo文件SHA、独立缓存，再串接训练；不是两步probe。
旧非Turbobootstrap/caches已停止，重启bootstrap.py --download-only仅续下载，不会再启动GPU任务。
以下非Turbo方案作为保留路线说明，不是当前活动训练。普通FM能否保持Turbo蒸馏优势仍待生成评估。

48任务：`/home/ds/Workspace/xy/qwen-lora-20261023`。
复用pytorch Python3.12环境和原任务diffusers0.41；PEFT0.21.2仅装在任务vendor，不覆盖全局依赖。

## 任务与数据

正式底模为非蒸馏`Qwen/Qwen-Image-2.1`，ModelScope官方来源，组件逐文件SHA256校验。
Turbo不是不能训练；本任务另做两步真实反向预检，但不保证普通FM微调后保留Turbo八步质量，
不把Turbo测试权重或缓存冒充非Turbo基模。

数据沿用冻结Top20划分，只取颜真卿/智永/赵佶楷书。去空白/过黑/明显外侧扫描线后：
train2730、validation18、test38；剔除444/3/4，逐图路径和SHA、源CSV SHA在data.json。
这项简单阈值过滤不是完整人工清洗，也可能排除合法的长边缘横画，不能声称全部GT无损坏。
书家表来自writer_tables_supcon_v1，按raw-ID取3行32D，冻结且逐步检查不变。
本轮只有楷书，不能据此证明跨书体控制。

## 条件和训练

字符文字经冻结Qwen3-VL编码；不把书家名写入文字。
冻结32D书家表→可训练MLP→4个4096D上下文token；书体embedding→MLP→1个token。
拼接到文本条件序列，正确扩展padding mask和图像slot mask；GT不进入条件。
推理依赖字符、writer-ID、script-ID、噪声，以及保存的条件适配器与LoRA，不需STD或参考图。

冻结文本编码器、VAE、生成主干原参数，只训练Attention q/k/v/out的rank16 LoRA和条件投影。
GT RGBA、256→512 LANCZOS，不做翻转、随机裁字或粗细处理。
缓存原生VAE后验mean/std，每次训练重新采样并按该VAE latents_mean/std标准化；
不复用旧4通道latent，也不交叉使用Turbo与非Turbo缓存。文本特征按字符缓存。
FM均匀抽原生scheduler timestep，target=noise-clean；输入无图像条件，图像slots按2×2 latent分组。

配置见config.json：2000更新，microbatch8、累积1、书家均衡采样，LoRA/投影LR1e-4，
warmup100、余弦至10%，writer-null dropout10%，AdamW FP32状态，BF16前向和梯度检查点。
每500步保存LoRA+条件适配器+优化器/RNG，生成6个验证字并计算统一SSIM/LPIPS/Ink-IoU。
不加Critic/REPA/像素辅助Loss。训练不触碰test指标来选权重。

## 执行链路

- data.py：冻结CSV、GT、书家表与来源身份；不改变原数据。
- bootstrap.py：并行下载VAE/文本编码器/主干；组件验证后启动相应cache进程。
- cache.py：基模VAE完成即编码GT；文本编码器完成即缓存文字；用文件锁避免GPU阶段叠加OOM。
- train.py：两步Turbo probe或正式2000步训练；失败记录后退出，不静默换模型/batch。
- condition.py：书家与书体条件投影。

bootstrap_status.json=下载/编码/正式训练链路；cache_*_status.json=具体编码进度；
probe_status.json=Turbo两步检查；train_status.json=正式优化步数、梯度和显存。
只有train_status中step>0才能称正式训练已开始，后台runner启动不等于训练启动。
GPU编码与训练等待下载，不能在完整权重到齐前确认正式底模的峰值显存和吞吐。

实测Turbo预检两步通过：batch1峰值14.17GiB；batch8峰值18.56GiB，含优化器更新。
batch8的LoRA/条件投影初始梯度范数0.521/1.653，LoRA权重改变、冻结表不变。
LoRA16,777,216参数，条件适配器4,749,712参数。两步短测约4.22秒/步，
不是正式基模长训吞吐或收敛证据。预检cache单列`*_probe.pt`；正式缓存检查完成状态、模型路径、分辨率、
完整图像键与字符集合，编码时逐图核对冻结GT SHA，禁止混用或用部分预检缓存训练。

参考训练接口：[Diffusers Qwen2.1训练例子](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_qwenimage21.py)。
遵循模型原有Qwen Research License，不因示例代码许可而改变权重/输出用途限制。
