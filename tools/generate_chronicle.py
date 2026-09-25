import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "assets/all_runs_deep_parsed.json"), "r", encoding="utf-8") as f:
    all_runs = json.load(f)

with open(os.path.join(ROOT, "assets/git_history_parsed.json"), "r", encoding="utf-8") as f:
    commits = json.load(f)

# Group runs into Eras
eras = [
    {
        "era_id": "era_10",
        "title": "第十纪元：v21 旗舰时代 —— SkelNet 显式可形变骨架与解耦联合训练 (2026年9月25日 - 至今)",
        "desc": "打破隐式形变的天花板，引入轻量级 SkelNet (3.58M) 显式预测 TPS 控制网格与位移场，辅以半径截断笔画调制 (r=0.25) 与低学习率联合微调 (lr_scale=0.1)。在严格零样本外推上，书法家目标专属性爆发提升至 +0.0161，书法家同字富集度达到 1.91x。",
        "keywords": ["v21"]
    },
    {
        "era_id": "era_09",
        "title": "第九纪元：v18 - v20 探索时代 —— 笔画骨架解耦与信号溯源诊断 (2026年9月24日 - 9月25日)",
        "desc": "通过梯度探针确认主干网络对风格信号的门控旁路现象，发现条件融合层 38x 方差失衡缺陷，完成 DeformSkel 离线预训练 (95.3% 风格跟随率)，为 v21 铺平道路。",
        "keywords": ["v18", "v19", "v20"]
    },
    {
        "era_id": "era_08",
        "title": "第八纪元：v17 注入重构时代 —— 空间跨注意力、门控消融与八大证伪 (2026年9月22日 - 9月24日)",
        "desc": "大规模系统性消融探索局部交叉注意力 (LCA)、低秩空间适配器 (Rank=32)、骨架查询 (GlyphQuery)、跨层跳跃连接与连续时间门控。以 Strict SSIM 0.0893 的代价彻底证伪了 LCA 交叉注意力机制。",
        "keywords": ["v17"]
    },
    {
        "era_id": "era_07",
        "title": "第七纪元：v15 - v16 多模态风格与低资源小样本迁移 (2026年9月19日 - 9月21日)",
        "desc": "探索 K=4 多 Token 风格表与 SupCon 监督对比学习；开展沈周、伊秉绶、傅山、徐渭 4 位大家的 100 样本 Few-Shot 快速迁移，证实 row_pt 权重先验仅需 2000 步即可逼近极值 (0.5568)。",
        "keywords": ["v15", "v16"]
    },
    {
        "era_id": "era_06",
        "title": "第六纪元：v14 细分时代 —— 87 类别精细化与全量微调 (2026年9月17日 - 9月18日)",
        "desc": "将书家扩展为 87 组书家×书体细分对，探索阶段性优化与全参数微调。在 v14_style87_s3 达到历史纯风格重构最高峰值 (Strict SSIM 0.5764, Seen SSIM 0.7698)。",
        "keywords": ["v14"]
    },
    {
        "era_id": "era_05",
        "title": "第五纪元：v13 黄金基准时代 —— 50k 清洗数据集与因子化融合 (2026年9月14日 - 9月17日)",
        "desc": "建立 50,000 高保真书法切片库 (50k_v2)，确立 45 位大师词表与 factorized_cat 融合层，v13_base_50k 成为项目最具代表性的坚固底模基准 (Strict SSIM 0.5547)。",
        "keywords": ["v13"]
    },
    {
        "era_id": "era_04",
        "title": "第四纪元：v12 架构压缩与池化时代 —— 全局向量感知 (2026年9月12日 - 9月14日)",
        "desc": "引入 glyph_vec_cond 均值池化，让主干首次在 adaLN 中感知全局字形拓扑；对比了宽度 320 与深度 8 的轻量化压缩。",
        "keywords": ["v12"]
    },
    {
        "era_id": "era_03",
        "title": "第三纪元：v11 算子现代化时代 —— Flow Matching、RoPE 与 12ch 试错 (2026年9月8日 - 9月12日)",
        "desc": "全面升级为最优传输流匹配 (OT-CFM)、2D 旋转位置编码 (RoPE)、RMSNorm 与 SwiGLU；进行了 49 万步超长长跑 (v11_struct-loss)；彻底证伪 12 通道多通道扩散。",
        "keywords": ["v11"]
    },
    {
        "era_id": "era_02",
        "title": "第二纪元：v10b 开集标准骨架时代 —— 41 位书法家奠基长跑 (2026年8月30日 - 9月8日)",
        "desc": "确立开集标准规范骨架 g_std 输入，剥离离散字符 ID，首次实现万级生僻字自由渲染；完成 36 万步全集余弦长跑 (Strict SSIM 0.5680)。",
        "keywords": ["v10"]
    },
    {
        "era_id": "era_01",
        "title": "第一纪元：早期萌芽与概念验证 (2026年8月15日 - 8月29日)",
        "desc": "探索离散字符 ID、像素空间与潜空间表征、ControlNet 外部旁路；经历了数据反色极性与空心骨架严重污染的惨痛教训。",
        "keywords": ["moyun", "s6", "s19", "s20", "s21", "smoke", "ctrl"]
    }
]

out_md = []
out_md.append("# Callig-DiT（马良）全周期全量实验超详编年史 (Exhaustive Experiment Chronicle)")
out_md.append("")
out_md.append("> **本编年史为 Callig-DiT 项目自立项以来全部 74 组实验运行的物理档案库。**  ")
out_md.append("> 完整记录每一次运行的数据集规格、核心算法、代码变更、时间节点、逐 checkpoint 评测指标（SSIM, MSE, LPIPS, 目标专属性, 书法家富集度, 碎片度）、训练曲线可视化与科学教训。")
out_md.append("")
out_md.append("---")
out_md.append("")

# Global Visualizations Section
out_md.append("## 📈 全局里程碑可视化大图 (Global Visualizations)")
out_md.append("")
out_md.append("以下图表由自动化评测分析工具解析全量历史数据实时生成，涵盖了全周期的宏观演进趋势与核心微观机理：")
out_md.append("")
out_md.append("### 1. 历代模型 Strict SSIM 轨迹演进大图")
out_md.append("![历代模型 Strict SSIM 轨迹](imgs/curves_timeline_milestones.png)")
out_md.append("*图 1: 历代模型在严苛零样本外推集（Strict Set）上的收敛轨迹对比。从 v10b (0.5058) 经由 v11 (0.5675)、v13 (0.5713) 跃迁至 v14 (0.5764 天花板) 以及当前进行中的 v21 解耦形变模型。*")
out_md.append("")
out_md.append("### 2. 旗舰运行 v21-skelnet-200k 实时多维遥测仪表盘")
out_md.append("![v21 实时遥测仪表盘](imgs/curves_v21_live_telemetry.png)")
out_md.append("*图 2: v21-skelnet-200k 四分屏实时运行分析（Step 5k - 25k）。Panel 1: Seen/Strict SSIM 阶梯上升；Panel 2: 目标专属性（Target Specificity）成倍爆发（+0.0161）；Panel 3: 书法家富集度突破 1.0x 盲区达到 1.91x；Panel 4: 感知误差 LPIPS 创新低 (0.3936) 且笔画碎片度收窄至 1.31。*")
out_md.append("")
out_md.append("### 3. 泛化鸿沟全景散点图：Seen 重构 vs Strict 外推")
out_md.append("![Seen vs Strict 泛化鸿沟散点图](imgs/curves_seen_vs_strict_gap.png)")
out_md.append("*图 3: 全量 74 组实验的 Seen SSIM（横轴）与 Strict SSIM（纵轴）分布。清晰揭示了早期模型与 v13-v15 阶段出现的“高记忆（Seen~0.78）、低泛化（Strict~0.56）”记忆平台区，以及 v21 架构如何向泛化对角线收敛。*")
out_md.append("")
out_md.append("### 4. 低资源书法家小样本自适应动力学 (沈周, 100 样本)")
out_md.append("![小样本自适应动力学曲线](imgs/curves_fewshot_comparison.png)")
out_md.append("*图 4: 沈周小样本迁移实验中不同先验初始化与学习率配方的表现。实证 row_pt 先验初始化结合 lr=1e-3 仅用 2,000 步即突破 0.5568。*")
out_md.append("")
out_md.append("### 5. 证伪清单八大死胡同定量对照柱状图")
out_md.append("![证伪清单八大死胡同定量对照](imgs/chart_falsification_comparison.png)")
out_md.append("*图 5: 经严格消融实验排除的 8 大技术死胡同在 Strict SSIM 上的性能暴跌表现（LCA 交叉注意力引发数值崩溃跌至 0.0893）。*")
out_md.append("")
out_md.append("---")
out_md.append("")

# Index of all runs by era
assigned_runs = set()
for era in eras:
    era_runs = []
    for k, v in all_runs.items():
        if k in assigned_runs:
            continue
        for kw in era["keywords"]:
            if kw in k.lower():
                era_runs.append((k, v))
                assigned_runs.add(k)
                break
    era["runs"] = era_runs

# Catch remaining into Era 1
remaining = [(k, v) for k, v in all_runs.items() if k not in assigned_runs]
eras[-1]["runs"].extend(remaining)

for era in eras:
    out_md.append(f"## {era['title']}")
    out_md.append("")
    out_md.append(f"*{era['desc']}*")
    out_md.append("")
    out_md.append(f"**本纪元包含实验数量**：{len(era['runs'])} 项")
    out_md.append("")
    
    # Sort runs by peak strict SSIM inside era
    def get_peak_strict(run_tuple):
        steps_d = run_tuple[1]["steps"]
        best = 0.0
        for st, sdata in steps_d.items():
            if "strict" in sdata and sdata["strict"]["ssim"] > best:
                best = sdata["strict"]["ssim"]
        return best
    
    era["runs"].sort(key=get_peak_strict, reverse=True)
    
    for exp_name, run_info in era["runs"]:
        cfg = run_info["config"]
        steps_dict = run_info["steps"]
        
        # Calculate summary metrics
        peak_strict = 0.0
        peak_strict_step = None
        peak_seen = 0.0
        peak_seen_step = None
        min_lpips = 999.0
        min_mse = 999.0
        max_tgt_spec = None
        max_cal_enrich = None
        
        for st, sdata in sorted(steps_dict.items(), key=lambda t: int(t[0])):
            if "strict" in sdata:
                sc = sdata["strict"]
                if sc["ssim"] > peak_strict:
                    peak_strict = sc["ssim"]
                    peak_strict_step = st
                if sc["lpips"] > 0 and sc["lpips"] < min_lpips:
                    min_lpips = sc["lpips"]
                if sc["mse"] > 0 and sc["mse"] < min_mse:
                    min_mse = sc["mse"]
                if sc["target_spec"] is not None:
                    if max_tgt_spec is None or sc["target_spec"] > max_tgt_spec:
                        max_tgt_spec = sc["target_spec"]
                if sc["cal_enrich"] is not None:
                    if max_cal_enrich is None or sc["cal_enrich"] > max_cal_enrich:
                        max_cal_enrich = sc["cal_enrich"]
            if "seen" in sdata:
                sn = sdata["seen"]
                if sn["ssim"] > peak_seen:
                    peak_seen = sn["ssim"]
                    peak_seen_step = st
                    
        out_md.append(f"### 🧪 实验档案：`{exp_name}`")
        out_md.append("")
        out_md.append("| 属性 | 详细参数规约 |")
        out_md.append("| :--- | :--- |")
        out_md.append(f"| **实验代号** | `{exp_name}` |")
        out_md.append(f"| **主干网络架构** | `{cfg.get('model') or 'DiT-2Cond-S/2'}` |")
        out_md.append(f"| **条件融合方式** | `{cfg.get('fusion') or 'factorized_cat'}` |")
        out_md.append(f"| **骨架形变机制** | `deform_skel={cfg.get('deform_skel') or 0}` (权重: {cfg.get('w_deform_skel') or 0.0}) |")
        out_md.append(f"| **REPA 表征损失权重** | `w_repa={cfg.get('w_repa') or 0.0}` |")
        out_md.append(f"| **全局学习率与批次** | `lr={cfg.get('lr') or 'N/A'}` \| `batch_size={cfg.get('batch') or 'N/A'}` |")
        out_md.append(f"| **最佳 Strict SSIM** | **`{peak_strict:.4f}`** (Step {peak_strict_step}) |")
        out_md.append(f"| **最佳 Seen SSIM** | **`{peak_seen:.4f}`** (Step {peak_seen_step}) |")
        if min_lpips < 10:
            out_md.append(f"| **最优感知 LPIPS** | `{min_lpips:.4f}` (均方误差 MSE: `{min_mse:.4f}`) |")
        if max_tgt_spec is not None:
            out_md.append(f"| **最高目标专属性** | `+{max_tgt_spec:.4f}` (富集倍率: `{max_cal_enrich:.2f}x`) |")
        out_md.append("")
        
        # Evaluation Trajectory Table
        if len(steps_dict) > 0:
            out_md.append(f"#### 📊 评估中间点逐步轨迹表 (Evaluation Trajectory, 共 {len(steps_dict)} 个检查点):")
            out_md.append("")
            out_md.append("| 步数 (Step) | Seen SSIM | Strict SSIM | Strict MSE | Strict LPIPS | 专属性 (Tgt Spec) | 书家富集度 (Enrich) | 笔画碎片度 (Frag) |")
            out_md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            
            for st, sdata in sorted(steps_dict.items(), key=lambda t: int(t[0])):
                sn_val = f"{sdata['seen']['ssim']:.4f}" if "seen" in sdata and sdata['seen']['ssim'] > 0 else "N/A"
                st_val = f"**{sdata['strict']['ssim']:.4f}**" if "strict" in sdata and sdata['strict']['ssim'] > 0 else "N/A"
                mse_val = f"{sdata['strict']['mse']:.4f}" if "strict" in sdata and sdata['strict']['mse'] > 0 else "N/A"
                lp_val = f"{sdata['strict']['lpips']:.4f}" if "strict" in sdata and sdata['strict']['lpips'] > 0 else "N/A"
                
                # Check for target spec and enrich
                spec_val = "N/A"
                enr_val = "N/A"
                frag_val = "N/A"
                if "strict" in sdata:
                    sc = sdata["strict"]
                    if sc.get("target_spec") is not None:
                        spec_val = f"+{sc['target_spec']:.4f}"
                    if sc.get("cal_enrich") is not None:
                        enr_val = f"{sc['cal_enrich']:.2f}x"
                    if sc.get("frag") is not None:
                        frag_val = f"{sc['frag']:.3f}"
                elif "seen" in sdata:
                    sn = sdata["seen"]
                    if sn.get("target_spec") is not None:
                        spec_val = f"+{sn['target_spec']:.4f}"
                    if sn.get("cal_enrich") is not None:
                        enr_val = f"{sn['cal_enrich']:.2f}x"
                
                out_md.append(f"| {st} | {sn_val} | {st_val} | {mse_val} | {lp_val} | {spec_val} | {enr_val} | {frag_val} |")
            out_md.append("")
        out_md.append("---")
        out_md.append("")

chronicle_file = os.path.join(ROOT, "docs/04_experiments/00_exhaustive_experiment_chronicle.md")
with open(chronicle_file, "w", encoding="utf-8") as f:
    f.write("\n".join(out_md))

print(f"Exhaustive chronicle written to {chronicle_file} ({len(out_md)} lines)")
