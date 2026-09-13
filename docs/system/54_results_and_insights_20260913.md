# 54. 实验结果总表与 Insights（2026-09-13）

> 全量机器可读版本：`assets/registry/runs.csv` / `runs.json`（38 runs，含每 run 的 ckpt 数、
> last_step、best seen/strict、关键配置）。本文是"人读版"：全结果 + 结论 + **哪些有用**。
> 评测口径：strict n=237 / cfg 0.7（可部署，字条件=g）；seen n=10 / cfg 0.7；GPU in-mem。
> 历史数值均已按该口径重评（26-run 复评），除非另注"非可部署"。

## 1. 最佳结果排行（可部署）

| 排名 | run | 模型/配方 | strict | seen |
|---|---|---|---|---|
| 🥇 | `v10b_stdskel_fame3_c41x_cos_e@390k` | Sp/2 + xattn12 + 4ch + REPA0.03 + 增强 113k + resume | **0.5680** | 0.763 |
| 🥈 | `v11_struct-loss@490k` | 同 c41x_cos_e + latent 结构 loss(skel0.02/canny0.1, t≤0.3) | 0.5675 | 0.769 |
| 🥉 | `v11_pretrain_M432_v8_sk3@150k` | M/2(49M) + 3px skel + 12ch(canny0.3/skel0.8) + 非增强 | 0.5108 | 0.571 |
| 4 | `v11_pretrain_S2_v8_aux02@120k` | S/2(30M) + 1px + 12ch(0.2) | 0.5091 | 0.495 |
| 5 | `v11_pretrain_M432_v8_auxc03s08@30k` | M/2 + 1px + 12ch(0.3/0.8) | 0.4855 | 0.486 |
| — | `v11_pretrain_M432_adaln4_sym`（进行中） | M/2 + adaLN4 + REPA + 对称增强 + 12ch | 0.4912@20k | 0.4745@20k |
| — | `v10b_stdskel_fame3_scratch_sty32@197k` | Sp/2 + style + xattn（归档） | 0.5518@190k | **0.669** |
| — | `v10b_stdskel_fame3_sp@77.5k` | Sp/2 + adaLN4（归档） | 0.5179 | 0.5636 |
| — | `v9a_repa_pretrain@130k` | S/2 + **无 skel，仅 char 条件** | 0.3469 | 0.520 |

**非可部署（推理时喂 GT 实例骨架，仅供归因上限参考）**：v10b_skel_only 0.7326 / v10adino(skel+dino) 0.7295 / v10a 0.7261 — 说明"骨架信息"本身极有价值（+0.16 vs 标准骨架），瓶颈在**标准字形骨架的信息量/精度**。

## 2. 全量结果（按系列，38 runs）

- **s 系列（s20–s32，旧 3cond/ControlNet/REPA）**：v8 链 ctrl 0.7425–0.7674（同协议复评），
  s30(DINO 字表) 库内检索 84% / 库外 **4%**（`docs/system/14`）→ DINO 字表**编码"图长什么样"而非"什么字"**。
- **v9**：v9a（char-only）0.3469；v9c（skel joint）无同口径 strict。
- **v10a / v10b（实例 skel 条件）**：v10a 0.7261 / v10adino 0.7295 / v10b_skel_only 0.7326（非可部署）。
- **v10b_stdskel（标准骨架 g，主线）**：
  - base S/2 0.5059@57.5k → sp(Sp/2+adaLN4) 0.5179@77.5k → c41x(xattn12) 0.5316@180k
  - c41x_cos(cosine) 0.5354@170k → **cos_e(增强 113k+长训) 0.5680@360k**
  - scratch 线：sty32(Sp/2) seen 0.669 / sty32_s(S/2) seen 0.524（容量分水岭）
- **v11（12ch 联合 + 新数据线）**：
  - S2_ref12ch（等权）0.5226@80k；S2_v8_aux02（w0.2）0.5091@120k；M432_sk3（3px）0.5108@150k；
  - M432_adaln4_sym（进行中，对称增强+REPA+adaLN4）。
- **归档**：`assets/results/_archive/`（29 runs，保留每 run 最后 2 个 ckpt）。

## 3. Insights（按证据强度排序）

1. **数据增强 + 长训练是 strict 的最大单一增益（+0.033）**：c41x_cos 0.5354@170k → cos_e 0.5680@360k（增强 113k + 训练 2×长）。⚠ 二者在实验里共变，未单独拆分。
2. **容量决定 seen 上限，对 strict 影响小**：S/2 30M seen 平台 ~0.52；M/2 49M 0.571@152k；Sp/2 59M 0.67–0.76。strict 各容量都在 0.51–0.57。**要视觉质量上 Sp/2，要 strict 容量不是瓶颈。**
3. **resume > 从头**：同配方 resume 线（cos_e 0.5680）> 从头线（scratch 0.51–0.52）。
4. **标准骨架 g 的价值 = 可部署的字身份 + 空间结构**：char-only 0.3469 vs std-skel 0.5680（+0.22）；实例骨架可到 0.73（不可部署）。VAE 对 1px 细线不保真（round-trip IoU 0.861 vs 3px 0.960），**3px 是更好的编码**（raw cos 0.90→0.84，字间残差两者都正交可分）。
5. **xattn（cross-attn 注入）strict ≈0、seen +0.01~0.02、成本 +33%**：sp(adaLN4) 0.5179@77.5k ≈ c41x(xattn12) 0.518@80k。当前线已改用 adaLN，adaLN4 比 adaLN12 省 ~17% 算力、-3M 参数。
6. **12ch 联合目标需要 per-group 权重**：等权时结构通道吃 ~51% final-layer 梯度（patch_embed aux 2.6× img）；按 ref 等权会压 seen。**保留 12ch（ref 有效），但必须调权**。
7. **REPA 是像素/DINO 的（非 latent）**；历史最优线都带 REPA0.03。latent-only 运行时它会自然失效——要作为显式配方变量。
8. **评测本身两个坑（均已修）**：flow 分支静默禁用 `w_latent_skel/canny`（结构 loss 从未生效）；`make_eval_cache` 对缺 shard 的样本给 g=0 → strict 出现"0.44 假崩"。历史 strict 值经覆盖率复核有效。
9. **三处"幻觉收益"被证伪**：`callig_spatial` 外挂 ±0.000；style token ≈0；xattn ≈0（strict）。
10. **数据增强 v3.3 有系统偏差**（朝 mid 收敛）→ v4 对称 ±（均值不变）已替换。

## 4. 哪些有用 / 哪些没用（结论）

**有用（保留）**
- 标准骨架 g（改用 **3px**）+ 12ch latent 联合目标（**per-group 加权**）+ REPA0.03 + adaLN 注入（**4 层**）+ 增强数据（**对称 v4**）+ resume/长训 + 容量 ≥ M/2。
- 工程：GPU in-mem eval + 评测期间暂停训练；g 覆盖率告警；`vae_io` 并行编码（encode bs128=19.8G/160 img/s）。

**没用/负作用（弃用）**
- xattn（strict 无增益、+33% 计算）、`callig_spatial`、style tokens、`skel_head`、IDS/DINO 字表（库外 4%）。
- 像素/probe 结构损失路线（latent 目标下用 12ch 监督替代）；1px 骨架；等权 12ch；把宽度朝中值收敛的增强 v3.3。

## 5. 悬而未决 / 下一步 A/B

1. **adaLN4 vs adaLN12**（单变量，验证简化无损）；**aux 权重扫描**（0/0.3·0.8/等权）。
2. **字身份第二通路**：g(3px) vs DINO 字表 vs 两者（同口径 strict），目标突破 0.57。
3. **增强 vs 长训解耦**：固定预算下 增强×长度 2×2。
4. 12ch 的 canny 也升 3px？skel/canny 权重的 dose-response。
5. 架构清理：删除上述死开关（旧 ckpt 用 `legacy-pre-simplify-20260913` 推理）。
