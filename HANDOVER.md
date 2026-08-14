# MCCD 数据集问题排查交接文档

> 状态：**调查已完成，重组织方案已落地（2026-08-14 核对）**。详细现状见 `DOCUMENTATION.md`。
> 目标已达成：确认官方 329715 张图互不重合；新切分 train/test/eval = 318715/10000/1000（`final_manifest_split.json`），新映射 `_id_maps.json`（书家 1873 / 字体 12 / 字 7765）。
> 注意：本文档 §3~§6 描述的「旧 80/10/10 脏 csv」与「待执行方案」**仅作为历史背景保留，均已过时**。新版数据管道请以 `DOCUMENTATION.md` §3 为准。

---

## 1. 核心结论速览

- 官方图片**不是重复的**，`MCCD_Character` 真实图片 **329715 张**（此前误判为 15 万，是解析错误，已纠正）。
- **可用主数据源 = 本地 `MCCD/MCCD/MCCD_Character/trainset_dataset` 的 329715 张图**，已含全部三条件（字/字体/书家）。
- ~~远程 `dataset/` 有 372862 张唯一图 + 80/10/10 脏切分~~（历史状态，已废弃，见下方 §3 标注）。当前远程正在**按新管道重建数据**：`train/test/val.csv` 均为 0 行，重建完成后即可训练。
- 泄漏根源有二：① 两个 prepare 脚本都 `random.shuffle(records)` 后 80/10/10 切分，无视官方 train/test 划分；② label 字段解析错位导致 vocab 脏。

---

## 2. 官方数据真实结构

### 2.1 目录结构
```
MCCD/MCCD/MCCD_Character/trainset_dataset/
├── train/    (234255 png，按 <字>/ 分目录)
├── test/     (95460 png，按 <字>/ 分目录)
├── train_label.txt   (234255 行)
├── test_label.txt    (95460 行)
```

图片命名：`trainset_dataset/{split}/{字}/{字}-{字体}-{朝代}-{出处}-{样本id}.png`

**无 validation 目录**（官方只有 train/test）。

### 2.2 label 格式（关键，每行 5 段）
`字 - 字体 - 朝代 - 出处 - 样本id`

例：`竝-篆-宋-集古文韵上声韵第三-66275`

- `parts[0]` = **字**（7765 类）
- `parts[1]` = **字体**（12 类：篆/楷/甲骨/草/行/隶/金/简/印/石/六体/其他）
- `parts[2]` = 朝代
- `parts[3]` = **出处/书家**（2243 类，这才是真正的"书家/风格"条件）
- `parts[-1]` = **样本 id**（同字内序号，**不是全局图片 id！**）

### 2.3 重要澄清：`parts[-1]` 是"同字内序号"，不是全局 id
- 数字 `164` 对应 `一-简-楚-信阳-164.png`、`丘-篆-null-説文部首-164.png`……完全不同的字、书家，末段都是 164。
- **这是此前所有"去重/重叠统计错误"的根源**——拿 `parts[-1]` 当全局图片 id 去重，会错误地得到 ~15 万。
- **正确的图片标识 = 完整文件名**。按完整文件名去重：train ∩ test = **0**，train+test = **329715 张互不重复图**。

### 2.4 四子集关系（完整文件名核对）
| 子集 | 图片数 | 与 Character 关系 |
|---|---|---|
| MCCD_Character | 329715 | 基准（本地真实有图） |
| MCCD-Style | 258830 | **100% ⊂ Character**（同图换风格标签） |
| MCCD-Dynasty | 258830 | 217909 ⊂ Character，40921 独有（本地无图） |
| MCCD-Calligrapher | 92122 | 89996 ⊂ Character，2126 独有（本地无图） |

> 注意：除 `MCCD_Character` 外的三个子集，**本地磁盘未下载图片**（`trainset_dataset` 目录只有 label 没有 png）。

**对 3Cond 模型（书家/字体/字）的结论：只需用 `MCCD_Character` 的 329715 张**，其他子集不增加可用新图。

---

## 3. 远程 `dataset/` 真实情况（`/root/Workspace/xy/DiT`）（2026-08-12 排查时快照，**已过时**）

### 3.1 结构
```
dataset/
├── images/    (372862 png，按 书家/字体/字/样本id.png 组织，如 images/春秋僖公石经/篆/楚/317289.png)
├── canny/     (335567 png，比 images 少 ~3.7 万，有缺失)
├── skeleton/  (372862 png)
train.csv         298281 行 (unique 298213)
test.csv          37286 行  (unique 37285)
val.csv           37285 行  (unique 37283)
eval_small.csv    0 行  (空！)
mccd_mapping.json (177KB)
labels_map.json
```

### 3.2 csv 列格式
```
image_path, canny_path, skeleton_path, calligrapher, script, character, calligrapher_id, script_id, character_id
```
- csv 里的路径是**相对 `dataset/` 目录**的（如 `images/...`），物理文件在 `dataset/images/...`。
- csv 的 `calligrapher` 字段 = `parts[3]`（如"春秋僖公石经""篆刻""null"）。

### 3.3 当前切分是脏的
- **train/test/val = 298281 / 37286 / 37285，即典型的 80/10/10 shuffle**，无视官方划分。
- 存在微量样本重叠：train/test 重叠 9、train/val 重叠 17、test/val 重叠 3。
- `eval_small.csv` 为空（0 行）——eval 从未真正生成。

---

## 4. 泄漏与脏数据的根源

1. **shuffle 泄漏**：`prepare_mccd_dataset.py` 与 `prepare_mccd_dataset_fast.py` 都用 `random.shuffle(records)` 后 80/10/10 切分，导致同一图片可能跨 train/test（或三元组跨集合）。
2. **字段解析错位**：旧脚本用 `parts[3]` 当 calligrapher，但把部分错位内容混入，导致 `mccd_mapping.json` 里出现 `喜`/`數`/`永` 等**字**被当字体（script=22，实际应为 12）。正确解析应为 `parts[0]=字, parts[1]=字体, parts[-1]... `，而书家取 `parts[3]`。
3. **vocab 规模矛盾**：当前训练 config `num_calligraphers=2021`，但官方全集 `parts[3]` 去重是 **2243 类**（差 56 为含连字符的错位变体）。

---

## 5. 关键文件清单

| 文件 | 说明 |
|---|---|
| `prepare_mccd_dataset.py` | 旧脚本，shuffle 泄漏 |
| `prepare_mccd_dataset_fast.py` | 旧脚本，shuffle 泄漏 + 字段解析错位 |
| `mccd_mapping.json` | calligrapher 2299 / script 22(错) / character 8633(错) |
| `train_full_3cond.json` | 训练配置（`num_calligraphers=2021` 等） |
| `dataset/`（本地） | 混乱的旧派生数据（images 375339 / canny 338941 不等） |
| `MCCD/`（本地） | 官方原始数据，`MCCD_Character` 可用 |
| 临时探针 `_probe_*.py`、`_source_list.txt` | 排查用的临时脚本，需清理 |

---

## 6. 重组织方案（已落地，保留原方案文档说明过程）

**数据源**：官方 `MCCD_Character` 的 329715 张图（train 234255 + test 95460）。
**远程已有**这些图的 images/canny/skeleton 派生，**只需重新生成 csv + mapping，不需要传图/重新派生**（canny 缺失的 3.7 万需补）。

**已执行的目标切分**（图片级不重合）：
- **eval**：1000 张，从官方 test 固定抽（seed 锁定，独立未见）
- **test**：10000 张
- **train**：官方 train(234255) + 官方 test 剩余(94460-10000=84460) = **318715 张**
- （val 并入 train，不单独留）

**字段映射**：character←parts[0](7765)、script←parts[1](12)、calligrapher←parts[3]（清洗后 1873 类含 `others` 兜底）。

**已落地产物**：
1. `final_manifest_split.json`：329715 行，每行带 `final_split` 字段（train 318715 / test 10000 / eval 1000）。
2. `gen_split_csv.py`：由此生成 `final_train.csv / final_test.csv / final_eval.csv`。
3. `build_id_maps.py` + `_id_maps.json`：清洗后的映射（书家 1873，含 `others`；统计见 `DOCUMENTATION.md` §3.3）。
4. `train_full_3cond.json` / `train_full_3cond_skel0.json`：当前训练配置（详见 `DOCUMENTATION.md` §5.1）。
5. 远程重建：当前 `train/test/val.csv` 0 行，重建完成后即可直接训练；远程活跃训练为 tmux `skel0`（`new_data_skel0/results_full_3cond`，从 0 训练，见 `DOCUMENTATION.md` §2）。

**远程执行注意**：
- 远程 python（正确）：`/opt/conda/bin/python`（conda 环境，含 numpy/torch1.13.1+cu117/cv2/diffusers）。
  - **注意**：`/root/Workspace/xy/DiT/.venv/bin/python` 缺 numpy，**不可用**（此前文档误记）。
  - `python` / `python3` 也不在 PATH（仅 `/usr/bin/python3` 但缺 torch 等，不可用）。
- 远程 SSH 偶发连接关闭（可能重负载），需重试。

---

## 7. 待确认/待决策点（均已确认，保留记录）

1. **eval 从官方 test 切 1000 张**——已确认（用户原话"eval 是快速的1000"）。
2. **test 留 10000 张**——已确认。
3. **canny 缺失的 ~3.7 万张**：需要确认是否需要补派生，还是训练可以接受 canny 缺失（若训练 3 条件都需要 canny，必须补）。
4. **书家字段**：确认用 `parts[3]` 全部 2243 类（含典籍/地名/类别，如"篆刻""null""广韵"），不做人名白名单过滤（当前模型就用这套）。
5. **是否从头重训**：因 vocab 从 2021 调整为 2243，且切分全变，**现有 ckpt 大概率不可复用，需从头重训**。

---

## 8. 操作备忘（SSH 等）

- 远程主机：`root@10.176.54.17`，端口 `36430`。
- 远程项目：`/root/Workspace/xy/DiT`。
- 远程 python：`/opt/conda/bin/python`（不要用 `.venv/bin/python`，缺 numpy）。
- 本地官方数据：`g:/GitHub/DiT/MCCD/MCCD/MCCD_Character/trainset_dataset`。
- 本地 Shell 是 PowerShell：**不要用 `cd /d`（cmd 语法）**，用 `cd g:\...` 或 `cd g:/GitHub/DiT`。
- 中文路径/引号在命令行易乱码，探查逻辑建议写成 `.py` 文件执行，避免内联 `-c`。
