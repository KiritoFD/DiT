# 下一代条件注入架构演进方案：从全局扁平相加到空间-语义解耦

> **文档代号**：`DOC-105-03`  
> **制定日期**：2026年10月  
> **状态**：架构升级蓝图（Next-Gen Blueprint）  
> **核心目标**：突破当前 AdaLN 全局广播瓶颈，实现书法风格与汉字拓扑在空间域与时间域的双重物理级解耦。

---

## 1. 现状审视：当前条件注入的表达力天花板

在当前的基准主线（`v60` / `v61`）中，条件融合与注入采用了经典极简范式：
```text
y = [e_callig(32d) ; e_script(16d) ; e_char(256d)]   (共 304 维)
c = t_emb + Linear(SplitLN(y))                        (映射至 1024 维隐层)
h_block = DiTBlock(x, c)                              (AdaLN-Zero 全局调制)
```

### 1.1 核心缺陷物理诊断：
1. **空间敏感信息被强制扁平化（Spatial Blindness）**：
   - 书法生成包含两类截然不同的信息源：
     - **全局统计量（Global Style）**：书家的墨色枯润、运笔速度、笔画粗细偏好，属于全图统一分布的风格信号；
     - **空间拓扑量（Spatial Layout）**：汉字的笔画走向、偏旁布局、点画相对坐标，属于高度空间敏感（Spatially Dependent）的几何信号。
   - 当前做法把 256 维汉字 ID 压缩成一个一维全局标量，强制广播给整张图像的所有 256 个空间 Token（$16 \times 16$）。每个 Patch 拿到的调制量完全相同，网络必须完全凭借自注意力层（Self-Attention）在深层“盲目反解空间坐标”，导致复杂字、生僻字的起笔位置容易产生迷茫与局部粘连。
2. **时间步与条件信号的粗暴相加（$t_{\text{emb}} + y_{\text{emb}}$）**：
   - 扩散去噪是一个跨尺度演化过程：
     - **扩散初期（$t \to 1.0$）**：画布全为高斯白噪声，此阶段最核心的任务是**确立汉字的骨架大结构（拓扑搭建）**；
     - **扩散末期（$t \to 0.0$）**：骨架早已成型，去噪目标是**刻画墨色飞白、露锋折笔等微观质感（风格渲染）**。
   - 现有的简单相加使汉字内容与书家风格在所有时间步以固定比例混杂输入，无法根据去噪阶段动态倾斜侧重。
3. **推断时缺乏双轴独立的引导控制能力**：
   - 现有的无分类器引导（CFG）只能“全部条件一起放大”或“全部丢弃（Unconditional）”。无法在保证汉字 100% 正确不写错别字的前提下，单向拉大书法家的特异风格夸张度。

---

## 2. 下一代四大架构升级路径

```text
                  ┌────────────────────────────────────────────────────────┐
                  │                 下一代条件解耦注入架构                  │
                  └────────────────────────────────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
 ┌──────────────────────┐                             ┌──────────────────────┐
 │   全局风格通路 (Style)│                             │   空间内容通路 (Char)│
 │   书家(32d) + 书体(16d)│                             │   汉字拓扑(256d)     │
 └──────────┬───────────┘                             └──────────┬───────────┘
            │                                                    │
            ▼                                                    ▼
    AdaLN 全局调制                                      空间自适应交叉调制
    (控制墨润/粗细/质感)                                (控制间架/部首/空间落笔)
            │                                                    │
            └─────────────────────────┬──────────────────────────┘
                                      ▼
                      时间步动态门控混合 (Time-Gate)
                      t 接近 1: 拓扑强引导 (防错字)
                      t 接近 0: 风格强引导 (显笔锋)
```

---

### 2.1 改造一：风格与内容通道的物理级双流解耦 (Dual-Stream Disentanglement)

- **核心设计**：
  借鉴现代多模态扩散模型（Flux / SD3 的 MMDiT 理念），将“书家风格”与“汉字内容”彻底分流：
  1. **风格流（Style Stream）**：保留高效的 AdaLN-Zero。书家向量 $\mathbf{e}_{\text{callig}}$ 和书体向量 $\mathbf{e}_{\text{script}}$ 经过小型 MLP 投影为全局 scale/shift，只负责调制每个 Transformer Block 的特征振幅（墨色浓淡、整体骨力）；
  2. **内容流（Content Stream）**：将汉字 Embedding 升级为**结构空间先验（Spatial Token Prior）**。汉字先验表（256d）经浅层解耦映射为 $K$ 个关键部首 Token（或 $4 \times 4$ 空间拓扑网格），仅在 DiT 的前 4 层通过解耦的交叉注意力（Cross-Attention）注入图像 Patch。
- **数学表达式**：
  $$\mathbf{c}_{\text{style}} = \text{MLP}_{\text{style}}([\mathbf{e}_{\text{callig}} \,;\, \mathbf{e}_{\text{script}}])$$
  $$\mathbf{h}_{\text{norm}} = \text{AdaLN}(\mathbf{x}, \mathbf{t}_{\text{emb}}, \mathbf{c}_{\text{style}})$$
  $$\mathbf{h}_{\text{attn}} = \text{SelfAttn}(\mathbf{h}_{\text{norm}}) + \alpha \cdot \text{CrossAttn}\big(\mathbf{h}_{\text{norm}}, \, \text{Proj}(\mathbf{e}_{\text{char}})\big) \quad (\text{仅前 4 层})$$

---

### 2.2 改造二：时间步敏感的动态门控机制 (Time-Dependent Adaptive Gating)

- **核心设计**：
  让内容信号与风格信号在速度场中的权重显式依赖当前时间步 $t \in [0, 1]$：
  $$\mathbf{y}_{\text{fused}}(t) = w_{\text{char}}(t) \cdot \mathbf{y}_{\text{char}} + w_{\text{style}}(t) \cdot \mathbf{y}_{\text{style}}$$
- **动力学门控函数设计**：
  - **内容门控权重**：$w_{\text{char}}(t) = \text{Sigmoid}\Big(k_1 \cdot (t - t_0)\Big)$（高时间步强，低时间步衰减至近零）；
  - **风格门控权重**：$w_{\text{style}}(t) = \text{Sigmoid}\Big(k_2 \cdot (t_0 - t)\Big)$（高时间步弱，低时间步急剧增强）；
- **工程收益**：
  - 在去噪初始步（高噪区），模型 $100\%$ 精力用于搭建字形拓扑骨架，彻底消除生僻字“部首残缺”；
  - 在去噪后半程（低噪区），字形已被锁定，模型将全部自由度交给书法家笔墨质感，彻底消除“描边呆滞感”，释放飞白与枯润。

---

### 2.3 改造三：多尺度分层语义路由 (Hierarchical Semantic Routing)

- **核心设计**：
  Transformer 具有天然的“浅层学几何局部、深层学抽象全局”的表征特性，不应在所有 12 层无脑喂入相同的三元组。
- **分层路由方案**：
  - **Layers 1 ~ 4（结构成形层）**：主要注入汉字拓扑 $\mathbf{e}_{\text{char}}$ 与书体 $\mathbf{e}_{\text{script}}$，确立字形的绝对中宫和外拓走势；
  - **Layers 5 ~ 8（风格融合层）**：逐渐过渡，注入书家主效应 $\mathbf{e}_{\text{callig}}$，协调笔画交接处的连带关系；
  - **Layers 9 ~ 12（质感渲染层）**：完全阻断字符硬结构注入，纯粹由书法家微观风格 $\mathbf{e}_{\text{callig}}$ 驱动，雕刻笔锋锐度、枯湿浓淡与宣纸渗透纹理。

---

### 2.4 改造四：双轴独立分类器无关引导 (2-Axis Decoupled CFG)

- **核心设计**：
  在推理采样阶段，彻底摆脱单一 $s_{\text{cfg}}$ 的捆绑限制，支持独立外推**“书法家风格轴”**与**“汉字辨识轴”**。
- **四分支采样公式**：
  在单次 Batch 中并行评估 4 种条件组合（耗时仅增加少量矩阵乘法）：
  1. $\epsilon_{\text{full}} = \epsilon(\mathbf{x}_t, t, c, s, ch)$ —— 全条件
  2. $\epsilon_{\text{style}} = \epsilon(\mathbf{x}_t, t, c, s, \varnothing)$ —— 仅有书家/书体（无字）
  3. $\epsilon_{\text{content}} = \epsilon(\mathbf{x}_t, t, \varnothing, \varnothing, ch)$ —— 仅有汉字（无书家）
  4. $\epsilon_{\text{uncond}} = \epsilon(\mathbf{x}_t, t, \varnothing, \varnothing, \varnothing)$ —— 纯无条件底座
- **正交引导合成公式**：
  $$\hat{\epsilon} = \epsilon_{\text{uncond}} + \gamma_{\text{char}} (\epsilon_{\text{content}} - \epsilon_{\text{uncond}}) + \gamma_{\text{style}} (\epsilon_{\text{style}} - \epsilon_{\text{uncond}}) + \gamma_{\text{inter}} (\epsilon_{\text{full}} - \epsilon_{\text{content}} - \epsilon_{\text{style}} + \epsilon_{\text{uncond}})$$
- **控制效果**：
  - 遇到易错别字的生僻字：提高 $\gamma_{\text{char}} = 4.0$，降低 $\gamma_{\text{style}} = 1.5$（确保写对）；
  - 遇到常见简单汉字（如“大”、“一”）：降低 $\gamma_{\text{char}} = 1.5$，拉满 $\gamma_{\text{style}} = 5.0$（极致展现狂草跌宕或颜体厚重）。

---

## 3. 落地实施路线图

| 阶段 | 改动代号 | 架构变动范围 | 预期指标增益 |
|---|---|---|---|
| **Phase 1** | **`v62-2AxisCFG`** | 仅改造推理采样器，支持风格/内容双轴独立外推 | Strict SSIM $+0.005$ |
| **Phase 2** | **`v63-TimeGate`** | 在现有条件投影层加入可学习时间步自适应门控 $w(t)$ | 结构错误率下降 $50\%$ |
| **Phase 3** | **`v64-DualStream`** | 解耦浅层局部内容注意与深层全局风格 AdaLN | Strict SSIM 冲破 **`0.615+`** |
