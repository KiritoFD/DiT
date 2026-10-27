# 48关键结果本地备份

目标目录：`G:\DiT-48-artifacts-20261027`。清单见[manifest.json](manifest.json)，下载器见[`tools/artifact_transfer/download_48_manifest.ps1`](../../tools/artifact_transfer/download_48_manifest.ps1)。

## 纳入范围

- Deep-Sym4：50k、75k、85k三个关键checkpoint；
- U100-Base：55k；
- 当前J0/J1/J2/J3：各5k；
- Qwen-condition A/C：各90k；
- 上述运行的配置、训练日志、指标、poster和完成身份，打入69MB metadata bundle。

合计约18.65GiB。checkpoint属于不可重建实验结果，因此尽管链路慢，仍按用户要求下载。数据缓存、latent shards、STD bank、逐样本PNG副本和普通中间checkpoint均不传。

## 延后项

- S2仍在训练，完成后只补50k，不拉当前25k；
- J组后续计划延长到12k，届时补四个12k权重；
- 下载器逐文件写`.part`，完成后核对字节数再原子改名；重复运行会跳过大小一致的完成文件。

48到本地的实测链路较慢，下载采用单流后台串行，优先顺序为metadata → Deep85 → U100 Base55 → Deep75/50 → J2/J3/J0/J1 → Qwen A/C。
