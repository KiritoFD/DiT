# 108-05 · 本地隔离记录（2026-10-09）

> 与远程同规格：**只移动、不删除**，保留原名，留 MANIFEST，可逐行回滚。
> 远程侧见 [`01_remote_inventory.md`](01_remote_inventory.md) §3。

## 1. 隔离区

- 位置：`_quarantine_20261009/`（仓库根，已加入 `.gitignore`）
- 清单：[`_quarantine_20261009/MANIFEST.csv`](../../_quarantine_20261009/MANIFEST.csv)
- 回滚：`Move-Item <quarantine_path> <original_path>`
- 脚本：`scratch/local_quarantine.ps1`

## 2. 移入内容（19 个根目录一次性产物 / 961.4 KB）

| 文件 | 大小 | 为什么是死件 |
|---|---|---|
| `tmp_col_0_出.png` … `tmp_col_5_陞.png`（6） | 13,394–21,764 B | 同一时间戳 `2026-09-28 12:34:31` 的一次性取字渲染，6 个不同汉字列 |
| `tmp_strict_poster_first8_full.png` | 630,108 B | 根目录最大松散图，一次性拼接海报 |
| `test_shards_img_0.png` | 44,645 B | 分片检查用图 |
| `poster_sample0_row0..3.png`（4） | 1,828–36,953 B | 同一 sample 的旧管线拼图（`2026-02-10 22:14`） |
| `render_sample0_row0..2.png`（3） | 17,476–28,180 B | 同一 sample 的 render 管线拼图 |
| `v34_sample0_row0..2.png`（3） | 1,828–30,131 B | 同一 sample 的 v34 管线拼图 |
| `paseo.json` | **3 B** | 写入被中断的残骸 |

判定依据：全部**未被 git 跟踪**（被 `/*.png` 与 `paseo.json` 规则忽略），且三组 `*_sample0_row*` 是**同一张图在三条管线下的过时产物**（当前主线为 v54/v66/v68/v72，v34 已归档）。

## 3. 本轮**未做**（需你确认，均为不可回滚操作）

| 项 | 预期收益 | 命令 | 风险 |
|---|---|---|---|
| `git gc --prune=now` | **≈26 GB**（26.72 GiB loose 对象，其中全历史可达仅 12,589 个） | 见 [04](04_local_inventory.md) §8-P0 | 不可回滚；但证据充分（最大 9 个 loose 对象经 `git rev-list --objects --all` 核对全部不可达） |
| 删 `.git/lfs/objects` | 638 MB | — | 孤儿对象（无 `.gitattributes`、无 `filter.lfs`、HEAD 无引用） |
| `archive/`（18.5 GB）移远端 | 18.5 GB | — | 本地 0 入库，但含 `dataset.zip` 13.2 GB 等；需确认远端有副本 |
| 根目录 4 个大包（626 MB）移远端 | 626 MB | — | 原始数据集/评测产物 |
| 26 个根目录 `run_v*.sh` → `tools/launchers/` | 0（结构收益） | — | 已入库，移动会产生一次 rename 提交；但能让 `run_*` 可按版本检索 |
| `.gitignore` 补 `assets/` `docs/archive/` `logs/` `exp-std/` 等 | 防未来误提交 | 见 [04](04_local_inventory.md) §6 | 策略性选择（只影响未来 add，不会 untrack 现有文件） |

## 4. 已做的唯一 `.gitignore` 改动

追加一条，避免隔离区被误提交：
```gitignore
_quarantine_*/
```

其余 `.gitignore` 建议（`assets/`、`docs/archive/`、`logs/`、`exp-std/`、`src/utils/std_glyph_latent_v2/`、IDE 残留目录）**保持为建议**，未擅自改动——它们涉及"什么该进版本控制"的策略选择。

## 5. ⚠ 本轮踩到的真实坑（值得写进运维纪律）

**PowerShell 5.1 会按 ANSI(GBK) 读取 UTF-8 无 BOM 文件**，任何"读入—改—写回"都会**双重编码中文**。

- 本轮第一次给 `.gitignore` 追加规则时用了 `Get-Content -Raw` + `Set-Content -Encoding UTF8`，结果**全部中文注释变成乱码**（`# DiT 书法仓库` → `# DiT 涔︽硶浠撳簱`），还多出 BOM。所幸提交后立即用 `git diff --ignore-cr-at-eol` 发现并 `git checkout` 复原。
- 同样原因，隔离脚本里 6 个中文名文件（`tmp_col_*_{出,禮,簿,靜,紹,陞}.png`）首次"未找到"，改用 `-Filter` 通配后才移成功。

**纪律**：
1. 改含非 ASCII 的文件，**只用 `edit` 工具或 `[System.IO.File]::AppendAllText(path, text, UTF8Encoding($false))`**；不要用 `Get-Content -Raw` + `Set-Content` 往返。
2. 提交后**必须**用 `git diff HEAD~1 --ignore-cr-at-eol` 复核非 ASCII 文件，否则乱码会静默入库。
3. 同理适用于远程：本项目 `logs/` 里 253 个文件名含汉字（见 [04](04_local_inventory.md) §7.6），跨机 `rsync/scp/tar` 需显式指定 UTF-8。

