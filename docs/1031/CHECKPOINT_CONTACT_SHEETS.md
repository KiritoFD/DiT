# 真实PNG多Checkpoint对照 Poster

**这才是用于读图的 Poster**：同一批10/20个字，每个 checkpoint 展示上半部样本图，中央大横幅标注 **名称 / 主要方法 / 全量指标**，下半部展示剩余样本。默认包含GT。不是T1–T4数值信息图，也不生成书法图像。

## 直接使用

```bash
# 1. 脚本只拼合4090已经生成的PNG；对每个row填写实际eval路径和协议。
cp tools/compare_ckpt_poster.example.json /tmp/my_poster.json

# 2. 手工填入真实的protocol_id、评测目录、指标来源；需要10字时sample_ids只留10个。
python tools/compare_ckpt_poster.py \
  --manifest /tmp/my_poster.json \
  --root /root/Workspace/xy/DiT \
  --out assets/f_20261101/posters/f0_f1_compare20.png \
  --cell 194

# 3. 不用GPU，单纯读PNG；输出同时保存f0_f1_compare20.selection.json。
python -m unittest discover -s tools/tests -p test_compare_ckpt_poster.py -v
```

**示例配置只提供路径形状**：仓库目前没有承诺已同步`f`的s1–s8产物，所有 `REPLACE` 必须替换。脚本缺图、错字符、错书家/书体、重复idx、不同protocol会直接失败，**不绘制虚假的比较**。

## 关键设计

- `sample_ids`仅支持10或20个互不重复的idx。GT和各checkpoint必须是**同字符/书家/书体**；如eval顺序不同，给每个row指定其自己的`sample_index_csv`，并设置`match_by: "img_id"`按唯一 `img_id` 交叉映射。若两边PNG本就同IDX，用`match_by: "idx"`。
- 每份`sample_index_csv`必须是该面板的权威源，可提供`sample_index_filter`（例如 `{"set":"heldout187","step":"25000"}`）从多面板CSV筛选后再检查重复idx。
- 原生图片保持宽高比、白底居中；不做增粗、裁剪、结构变形或生成。图的顺序由manifest明确锁定；禁止字符串glob排序。
- `metrics`是**整个对应评测集**的数值，不是入选10/20个字的挑优均值。每条有指标的checkpoint必须填`metrics_source`（真实存在的JSON/CSV）与同名 `metrics_protocol_id`，禁止猜测缺失数值和跨面板拼分。指标正文直接来自manifest，源文件路径只作为存在性约束，请人工核对字段和数值。
- 如需要同时对照方法沿革，可单独做另一个相同eval协议的Poster；MoYi、Deep/S、P125协议不同，不应该在同一个“同协议排行”海报中直接排成绩。
- 输出的`.selection.json`包含每个字符在每个模型里的idx、实际图片绝对路径和SHA256，便于回溯/拉回本地。不会自动搬运模型权重或动运行中tmux。

## `f`队列的推荐对比

**首张：** `GT | P3A@20k | 关两个site（无训练） | F0@5k | F1@5k | 最终s8`。如果某个点没有完整同面板PNG，则不画该行，不借旧面板拼字。

**第二张（专门诊断）：** `GT | P3A原始 | off_32 | off_16 | off_8 | off_32+16`，同30字Writer-swap可先选20字；必须保证不同site变体的Writer-ID/seed/采样设置一致。

**数量指标**：每行建议至少 `SSIM / LPIPS-VGG / InkIoU / OCR Top1 / Writer direction / amplitude`，但 direction/amplitude 必须与同一 Writer-swap 面板匹配；没有数据就省略，别写0。

## 与旧脚本的关系

- `tools/milestone_poster.py`：保留原20列 strict archive；
- `tools/make_ckpt_poster.py`：保留原单ckpt预览；
- **`tools/compare_ckpt_poster.py`**：多ckpt、10/20字、每行**中央大banner**、允许跨评测CSV显式按身份配对、每张图带原始SHA和测试。

本PR只增加后处理工具与测试，**不会启动或阻断4090无人值守训练**。
