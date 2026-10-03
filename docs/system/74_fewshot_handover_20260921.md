# 74 — Few-shot 交接文档（2026-09-21 凌晨）

> 目标读者：接手 few-shot 实验的人。本文自包含：数据在哪、怎么准备、怎么训、
> 已经踩过/修过哪些坑、结果怎么判读。
> 环境：远端 `ssh 4090`，仓库 `/root/Workspace/xy/DiT`，
> `PY=/opt/conda/envs/cu121/bin/python`，`export PYTHONPATH=/root/Workspace/xy/DiT`。
> GPU 当前空闲（18MiB）。

---

## 1. 实验目标与判据

**问题**：模型的风格条件够不够"装下"一个全新书家？
- v13 base = 45 书家 × 1 个 128 维**冻结**向量；v15a = 87 个 (书家×书体) pair ×
  **K=4 个子风格 token × 384 维 = 1536 维/行**（DINO K-Means 质心初始化）。
- few-shot 做法：给 v15a@150k 的表**新增 1 行（第 88 行, 1536 维）**，
  **冻结全部主干，只训这一行**，用新书家的 50 张图。
- **判据**：
  1. fewshot eval ssim（n=100 留出集，cfg=0.7）显著高于"不训练"的 baseline；
  2. Diff loss 从 ~1.6 降到明显低于 baseline 水平并**趋平**（v13 版趟平在 1.05，
     全量训练的书家在 0.24–0.31 —— 这是参照系）。
- **结论预设**：v13 (K=10, 128 参数) 已实测**失败**（loss 趟平 1.05、eval ±0.02 噪声）。
  v15 的 1536 维是关键未测项。

## 2. 数据从哪里来（wild.zip）

- 原始数据：`/root/Workspace/xy/HCSU/wild.zip`（+ .001~.006 分卷），
  已解压在 `/root/Workspace/xy/HCSU/wild_extract/`：**78 个 `书家-书体` 文件夹**，
  每个文件夹内文件名 = `字.png`。
- 50k 训练集（`assets/train_50k_v2.csv`，50,786 行）已经用了其中 **18,444** 张
  （看 CSV 的 `src_image_path` 列是否含 `wild/<文件夹>/`）；**还剩 13,843 张完全没用**。
- **29 个文件夹完全没用过**。规律：同一书家两个文件夹，白底的进了 50k，
  黑底的（拓片/反相扫描）被剔——**被剔的原因是质量（背景极性），不是数量**。
- **书体限制（重要）**：v15a 的 87-pair 词表只含 楷(script_id=0) / 行(3) / 隶(4)。
  没用过的干净文件夹大多是 草/篆 —— **分布外，不能用**（v13 版用怀素·草书已被
  判定为混淆因素）。

### 2.1 质量体检结论（`_sync_work/wild_quality.py`，抽样 80 张/夹）

| 文件夹 | 灰度均值 | 暗像素占比(ink<128) | 判定 |
|---|---|---|---|
| 沈周-行 | 155.6 | 0.242 | ✅ 白底干净 |
| 伊秉绶-行 | 174.7 | 0.211 | ✅ 白底干净 |
| 傅山-行 | 136.9 | 0.310 | ✅ 可用（仅 194 张） |
| 伊秉绶-隶 | 93.7 | **0.611** | ❌ 黑底 |
| 宋高宗-楷 | 118.6 | **0.650** | ❌ 黑底 |
| 徐渭-行 | 103.6 | **0.564** | ❌ 黑底 |

⚠ **首批实验（伊秉绶-隶/沈周-行/徐渭-行/宋高宗-楷）里 3 个是黑底**。
模型是白底墨字训练的 → baseline ssim 排序 = 脏污程度排序
（沈周 0.395 > 宋高宗 0.282 > 徐渭 0.263 > 伊秉绶 0.185）——
之前测的"few-shot baseline"大半是**背景极性错配**，不是风格容量。

### 2.2 可用主题清单

- **A 组（立刻可用）**：完全没用 ∩ 白底(ink<0.35) ∩ 楷/行/隶 ——
  只有 **沈周-行(500) / 伊秉绶-行(500) / 傅山-行(194)** 三个。
- **B 组（需极性矫正后可用）**：黑底但书体在分布内的完全未用文件夹，
  以楷为主：王羲之-楷(500), 钟繇-楷(459), 欧阳询-楷(500), 虞世南-楷(500),
  董其昌-楷(500), 柳公权-行(326), 王宠-行(500), 朱熹-行(459), 王献之-行(316),
  鲜于枢-行(404), 黄庭坚-楷(500), 颜真卿-楷(260), 赵孟頫-楷(411), 祝允明-楷(406),
  蔡襄-楷(500), 米芾-草/苏轼-行/文征明-楷 等。矫正方法见 §5 Step 7。

## 3. 管线：数据怎么准备、模型怎么训

参考脚本（都在 `_sync_work/`，旧主题已跑通一遍）：
`prepare_fs50.py`（合并/同步 id/出配置）→ `resize_fs_v15.py`（统一 256）→
`prepare_fs_v15.py`（v15 的 88-pair 词表 + 配置）。编码 latent 的命令**没有落盘**，
需要按 §5 Step 4 现写（10 行 torch，模式抄 `_sync_work/_build_clean_latents.py`）。

**标准流程**（每个新书家 `<cal>`，书体 `<sc>`）：
1. 从 `wild_extract/<书家>-<书体>/` 选 **50 张训练 + 100 张评测**，
   **按"字"划分：eval 的字与 train 的字绝不重叠**（否则 eval 是 seen，不是泛化）。
2. 写 `assets/fs50_<cal>_train.csv / _eval.csv`，列格式照抄
   `assets/fs50_沈周_train.csv`：`calligrapher_id=9999`（新书家原始 id）、
   `script_id` 用 50k 里该书体的 id（楷 0 / 行 3 / 隶 4）、
   `std_path` 借用 50k 里**同一个字**的 std 骨架图（`data/50k/std/<6位id>.png`，
   用 character 在 `train_50k_v2.csv` 里反查；同字多张任选其一）。
3. 编码 **img latent** → `data/50k/shards_fs50_<cal>/shard_00000.npz`
   （键：`latents (N,4,32,32) fp16` + `img_ids (N,) 字符串`；VAE 用
   `data/pretrained/sd-vae-ft-ema`，模式抄 `_sync_work/_build_clean_latents.py`）。
4. 编码 **skel latent** → `data/50k/shards_fs50_<cal>_std/`（对每行的 `std_path`
   图做同样的 VAE encode，img_id 用**同一个键**）。
   ⚠ 这一步旧管线**没做**，是 §4-④ 的 bug 来源，必须补。
5. resize 原图到 256 → `data/50k/fs50_imgs_<cal>/`，CSV `image_path` 指向副本
   （`resize_fs_v15.py` 干的事）。
6. 运行 `_sync_work/prepare_fs_v15.py`（改 `CALS` 列表）：生成
   `assets/callig_script_id_map_fs15_<cal>.json`（88-pair 词表，新 pair = 9999:<sid>）
   和 `src/train/configs/v15_fs_<cal>.json`。
7. 训练：
   ```
   tmux new-session -d -s v15fs 'bash _sync_work/launch_fs_v15.sh 2>&1 | tee logs/_v15fsN.log'
   ```
   `launch_fs_v15.sh` 里 `for c in 伊秉绶 沈周 徐渭 宋高宗` 改成新主题列表；
   每个书家先 `--eval-only` 测 baseline，再
   `--resume-full <v15a@150k ckpt> --train-only-new-callig --init-new-callig mean_scaled`。
   v15a ckpt：`assets/results/v15a_multistyle_k4/20260919-223715-*/checkpoints/0150000.pt`。

### 3.1 训练超参（当前 `v15_fs_*.json` 的状态）

- `lr 3e-3, lr_schedule constant, warmup_steps 100, global_batch_size 256,
  max_steps 200000(=+50k 步), gpu_eval_every 2000, ckpt_every=epoch_steps 5000,
  use_ema false, cond_drop_all_prob 0, glyph_drop_prob 0.1`。
- ⚠ 3e-3@b256 实测 Diff 仍卡 1.37（但那是黑底脏数据跑的，不能下结论）。
  **换干净数据后如果 Diff 不降，优先怀疑 lr/数据，不要加步数硬磨。**
- 监控：`bash _sync_work/fscheck.sh 沈周`（打印 eval 轨迹 + 最新两行 loss）。

## 4. 问题清单（5 个；①已修，②③④必须先修再跑，⑤是背景结论）

**① `--train-only-new-callig` 的 mask bug（已修，2026-09-21 00:30 部署）**
`src/train/train.py:793-835`。旧代码写死"表末行 = CFG null 行"→ 可训范围
`[_n_old, _n_all-1)`。v13 的 LabelEmbedder 表确实是 N+1 行含 null；但 v15 的
`MultiStyleEmbedder` 表是 N 行**不含 null**（null_embed 是独立参数）→ v15_fs
新行(=第 87 行)恰好被排除 → **`[87:87) -> 0 参数`，训练完全空转**（首轮 4 书家
150100/150200 的 eval 与 baseline 逐位相同）。已改为按 `isinstance(..., MultiStyleEmbedder)`
分支取 `_n_new_end`，备份在 `src/train/train.py.bak_20260920`。
**验证方法**：起训后日志必须出现 `[87:88) -> 1536 参数`，且 Diff 必须动。

**② LR/warmup/步数**：原配置 `max_steps=150200`（只训 200 步）+ `warmup_steps=3000`
→ LR 永远爬不起来（实际 ≈6.7e-6）。已改 warmup=100。步数给到了 +50k，
但**别信"步数够就行"**——盯 Diff 曲线，不动就调 lr（batch 放大要按线性缩放调 lr）。

**③ g（骨架条件）错位 —— 静默、最危险、必须先修**
`fs50_*_train/eval.csv` 的 `img_id = 0..149`（`prepare_fs50.py: combine()` 重新编号），
而 config 的 `skel_latent_shards_dir = data/50k/shards_std`（52,457 条，键=50k 的 img_id）。
`0..149` **全部命中** shards_std 的前 150 个 50k 图 → 训练和评测用的 g 是
**无关字的骨架**（50k 000000–000149 = 智永/颜真卿等人的字），不报错、指标照出。
修法（二选一，推荐 a）：
- a) 每个 `<cal>` 把 `std_path` 指到的 std 图 encode 成
  `data/50k/shards_fs50_<cal>_std/`（键=img_id 同名），config 的
  `skel_latent_shards_dir` 指过去；
- b) 干脆让 fs 行的 `img_id` 直接**沿用所借 std 图的 50k id**（如 032217），
  则 shards_std 天然对上；但 img latent shard 的键必须用同一套 id
  （且这些 id 与 50k 训练集重叠没关系，目录是独立的）。
⚠ 这也意味着**此前所有 few-shot 数字（baseline 和训练后）都带病**，
只能看相对趋势，绝对值作废。

**④ 数据质量（背景极性）**：见 §2.1。黑底主题**不能直接用**：
模型白底训练，黑底 GT 使 eval ssim 天花板 ≈0.2-0.3、Diff 卡 ~1.37（lr 1e-3/3e-3 都试过）。
要用黑底文件夹（B 组）必须先极性矫正：检测 `ink<128 占比 > 0.5` → 整图反相 →
必要时二值化/去灰边，使灰度统计落到 50k 白底图的域（参考 50k 图：ink≈0.04-0.25，
gray_mean>150，near_white>0）。矫正后**先肉眼看几张**再入库。

**⑤ v13 版（K=10）的结论（背景，不需要重跑）**：
正确实现（128 参数可训），Diff 1.33→~1.05 后**趟平 1800 步不动**，
fewshot eval ±0.02 噪声、个别下降（10 张图过拟合）。
→ **冻结主干 + 128 维单向量学不会新书家**。这就是 v15 (1536 维) 的动机。

## 5. 给接手人的执行清单（建议顺序）

1. 修 ③（g 错位）：写 encode-std-shard 小脚本 + 改 4 个 `v15_fs_*.json` 的
   `skel_latent_shards_dir`。这是**硬前置**，不修一切数字无效。
2. 按 §3 流程重建 **A 组三个主题**：沈周-行 / 伊秉绶-行 / 傅山-行
   （50 train + 100 eval，按字划分；傅山-行只有 194 张，eval 可降到 94）。
3. 训练（先单主题 沈周-行 冒烟）：盯两件事——
   日志出现 `[87:88) -> 1536 参数`；Diff 从 ~1.6 持续下降。
   若 Diff 卡住：先查 g（打印一个 batch 的 g 与 GT 字是否一致），再调 lr。
4. 三主题跑完 → 与各自 baseline 对比 fewshot ssim（判读见 §6）。
5. （可选）B 组极性矫正 → 扩到楷书主题。
6. **别忘了**：主线 v15b 在 118k/150k 被中断、v15c 未跑
   （`logs/v15_series/v15b_train_20260919-223712.log`，tmux `v15` 已死）。
   few-shot 结束后要不要续跑需用户拍板。

## 6. 结果判读

| 现象 | 结论 |
|---|---|
| Diff 降到 ≤0.5 且 fewshot ssim 比 baseline 高 ≥0.05（n=100，SE≈0.02） | 1536 维容量能捕获新书家 → 方向成立，扩大主题数 |
| Diff 趋平在 ~1.0+ 且 eval 不动 | 冻结主干根本读不进新行 → 条件注入机制是瓶颈（呼应 ratio_style 1.21 vs 1.48 的结论），下一步不在表容量 |
| Diff 降但 eval 反而降 | 50 张过拟合（v13 沈周先平后掉就是这样）→ 早停 / 减步 |
| eval 高但生成肉眼不是该书家风格 | 只对 ssim 优化了，必须放大看 poster（`assets/results/v15_fs_<cal>/posters/`） |

**历史参照**：全量书家 Diff≈0.24–0.31；strict 饱和 0.57；v15a 的 ratio_style=1.21
（同口径 baseline v13_base=1.48，见 `logs/v15_series/v15a_ratio_150k.log`）。

## 7. 相关产物位置

- 少量-shot 历史日志：`logs/_fs*.log`、`logs/_v15fs*.log`、
  `logs/v15_series/v15_fewshot/<书家>_{base,train}_*.log`
- 结果目录：`assets/results/v13_fs_*`、`v13_fs50_*`、`v15_fs_*`
- 质量体检/用量分析：`_sync_work/wild_quality.py`、`_sync_work/wild_unused.py`
- 检查脚本：`_sync_work/fscheck.sh <书家>`（注意：Windows 编辑后需 `sed -i 's/\r//'`）
- 旧配置生成器：`_sync_work/prepare_fs50.py` / `prepare_fs_v15.py` / `resize_fs_v15.py`
  （CALS 列表要改成新主题；两者都硬编码了旧主题名）


