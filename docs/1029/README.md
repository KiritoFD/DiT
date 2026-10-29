# 10-29评测与匹配续训

当前已完成24个历史模型端点、30个协议面板的HWD及独立HWD特征跨字核验；已有E-G/E-L指标保留，未将指标加权成总排名。S0/S4/S5到85k的训练仍在运行，不把55k checkpoint说成终评结果。

| 主题 | 状态和结论 | 入口 |
|---|---|---|
| E-G/E-L身份、分布 | AUC/EER/mAP、Style-FID/KID、Density/Coverage已完成；S5为E-G风格候选 | [领域指标](domain_metrics/REPORT.md) |
| 配对区间、匹配续训 | S5−S0逐查询宏AUC +5.73pp [1.65,9.84]；S5−S4区间跨零，仍单seed | [85k计划](style_focus_85k/README.md) |
| 官方权重HWD | 30面板完成；S5低于S0，但特征GT跨字AUC仅51–54%，不能独立验证书家身份 | [HWD](hwd/REPORT.md) |
| CHAED审美基线 | 990张可用源图、嵌套按字留出；MAE17.97对18.32，扰动校准失败 | [审美负结果](chaed/REPORT.md) |
| 无参考摄影IQA | MUSIQ/TOPIQ此前未通过书法校准，不升为主指标 | [原评测报告](eval_unpaired/REPORT.md) |

当前可保留的判断：S0为内容基线、S4为局部融合对照、S5为风格候选；复杂Decoder条件仍缺乏稳定收益。新增评测没有提供足够可靠的独立中文书家验证或审美分数，不据此启动S5++或复杂新Loss。

实现完成：可恢复的HWD缓存评测、额外跨字核验、输入身份记录、CHAED探索基线和分组留出验证。未完成边界：原CHAED ANN/TabNet全量复现、可靠OCR、历史书法人类审美校准，以及尚在训练中的85k三seed终评。这些项目继续明确记为缺失，不能用代理分数替代。
