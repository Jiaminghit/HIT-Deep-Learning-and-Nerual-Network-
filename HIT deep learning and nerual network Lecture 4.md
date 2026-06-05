# HIT deep learning and nerual network Lecture 4

## Transformer 模型

### 从序列到序列的范式转变

在 Transformer 出现之前，序列建模（如机器翻译、文本生成等 NLP 任务）的主流方法是基于循环神经网络（RNN）及其变体（LSTM、GRU）。这些模型存在以下根本性问题：

1. **顺序计算限制**：RNN 必须按时间步依次处理序列，无法并行化，训练效率低
2. **长程依赖问题**：梯度在长序列中容易消失或爆炸，即使 LSTM/GRU 也只能缓解而无法根治
3. **信息瓶颈**：Seq2Seq 模型将所有源序列信息压缩到一个固定长度的上下文向量中

2017 年，Google 团队的 Vaswani 等人在论文《Attention Is All You Need》中提出了 **Transformer** 模型，完全摒弃了循环结构，**仅依赖注意力机制**来建模序列中任意两个位置之间的关系，实现了并行计算与长程依赖的统一。

### Transformer 的整体架构

Transformer 沿用了 Encoder-Decoder 框架：

- **Encoder（编码器）**：由 $N$ 个相同的编码器层堆叠而成，每个层包含多头自注意力（Multi-Head Self-Attention）和前馈网络（Feed-Forward Network）两个子层
- **Decoder（解码器）**：同样由 $N$ 个相同的解码器层堆叠，除了上述两个子层外，还插入了一个交叉注意力（Cross-Attention）子层，用于关注编码器的输出
- 每个子层后都接有 **残差连接（Residual Connection）** 和 **层归一化（Layer Normalization）**

```
Encoder: Input → Input Embedding + Positional Encoding
              → [Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm] × N
              → Encoder Output

Decoder: Output (shifted right) → Output Embedding + Positional Encoding
              → [Masked Multi-Head Self-Attention → Add & Norm
                 → Cross-Attention (with Encoder Output) → Add & Norm
                 → FFN → Add & Norm] × N
              → Linear → Softmax → Output Probabilities
```

### 输入模块（Input Block）

#### 词嵌入（Token Embedding）

输入文本首先经过分词（Tokenization），将文本切分为 token 序列（常见的如 BPE、WordPiece、SentencePiece 等分词方法），然后每个 token 被映射为一个稠密向量：

$$
\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}
$$

其中 $n$ 为序列长度，$d_{\text{model}}$ 为嵌入维度（原始 Transformer 中 $d_{\text{model}} = 512$）。

#### 位置编码（Positional Encoding）

由于 Transformer 没有循环或卷积结构，模型本身不具备感知序列顺序的能力。因此需要显式地为每个位置注入位置信息。原始 Transformer 使用**正弦位置编码（Sinusoidal Positional Encoding）**：

对于位置 $pos$ 和维度索引 $i$（$i = 0, 1, \dots, d_{\text{model}}/2 - 1$）：

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
\end{aligned}
$$

> 使用正弦/余弦函数的好处：
> 1. 每个位置的编码是唯一且确定的
> 2. 不同位置之间的编码可以通过线性变换相互表示（$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性函数），利于模型学习相对位置关系
> 3. 可以外推到训练时未见过的序列长度

最终输入到 Encoder 的向量为：$\mathbf{X}_{\text{input}} = \mathbf{X}_{\text{embedding}} + \mathbf{PE}$

### 自注意力机制（Self-Attention）

自注意力是 Transformer 的核心。其核心思想是：对于序列中的每个 token，让它去"关注"序列中所有其他 token（包括自己），动态计算它们之间的关联权重，然后根据这些权重对信息进行加权聚合。

#### 缩放点积注意力（Scaled Dot-Product Attention）

给定输入序列 $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$，通过三个可学习的线性变换得到 **Query $\mathbf{Q}$**、**Key $\mathbf{K}$**、**Value $\mathbf{V}$**：

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}^V
$$

其中 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$（通常 $d_k = d_{\text{model}} / h$，$h$ 为注意力头数）。

注意力计算流程：

1. **计算注意力分数**：$\mathbf{Q}$ 与 $\mathbf{K}$ 做点积，得到每对 token 之间的"相似度"
2. **缩放**：除以 $\sqrt{d_k}$，防止点积值过大导致 Softmax 梯度消失
3. **Softmax 归一化**：得到注意力权重（概率分布）
4. **加权聚合**：用注意力权重对 $\mathbf{V}$ 加权求和

完整的数学表达：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

> **为什么要除以 $\sqrt{d_k}$？**
> 当 $d_k$ 较大时，点积 $\mathbf{Q}\mathbf{K}^\top$ 的方差会增大（因为点积是 $d_k$ 个独立随机变量之和）。较大的点积值会使 Softmax 输出趋近于 one-hot 分布，导致梯度接近零。除以 $\sqrt{d_k}$ 将方差控制为 1，保证梯度稳定。

#### 多头注意力（Multi-Head Attention）

单一注意力头可能只关注到一种类型的关系。多头注意力通过并行运行多个注意力头，使模型能够同时关注来自不同表示子空间的信息：

$$
\begin{aligned}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O \\
\text{head}_i &= \text{Attention}(\mathbf{X} \mathbf{W}_i^Q, \mathbf{X} \mathbf{W}_i^K, \mathbf{X} \mathbf{W}_i^V)
\end{aligned}
$$

其中 $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$\mathbf{W}^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$。

每个头在不同的低维子空间中计算注意力（$d_k = d_v = d_{\text{model}} / h$），原始 Transformer 使用 $h = 8$ 个头，每个头的维度为 $d_k = 64$。多头的总计算量与单头全维度注意力相当，但表达能力大大增强。

### 编码器（Encoder）

#### 前馈网络（Feed-Forward Network, FFN）

注意力子层之后是一个逐位置（position-wise）的前馈全连接网络，即对每个位置的表示独立应用相同的两层全连接：

$$
\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$，$\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$。原始 Transformer 中 $d_{ff} = 2048$（是 $d_{\text{model}} = 512$ 的 4 倍）。

> FFN 的作用类似于用 $1 \times 1$ 卷积进行逐点特征变换，为模型提供非线性变换能力。2 层结构在中间提供了更高维度的表示空间（$d_{ff} > d_{\text{model}}$），增强了表达能力。

#### 残差连接与层归一化（Add & Norm）

每个子层（注意力和 FFN）都采用残差连接后接层归一化：

$$
\text{Output} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))
$$

- **残差连接**：解决深层网络的梯度消失问题，使信息可以绕过子层直接传递
- **层归一化**：对每个样本的特征维度做归一化，加速训练收敛，提高训练稳定性

> 需要注意的是，原始论文采用的是 **Post-LN**（LayerNorm 在残差连接之后），而后来的研究（如 GPT 系列）发现 **Pre-LN**（LayerNorm 放在子层之前）可以带来更稳定的训练：
> - Post-LN: $\mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$ → Output = $\text{LayerNorm}(\text{上一步结果})$
> - Pre-LN: $\mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$

### 解码器（Decoder）

Decoder 与 Encoder 有两处关键差异：

#### 1. 掩码自注意力（Masked Self-Attention）

在 Decoder 的自注意力中，为了防止模型在预测第 $t$ 个 token 时"偷看"到未来的 token（第 $t+1, t+2, \dots$），需要将注意力矩阵的上三角部分屏蔽（mask）掉（设为 $-\infty$，使得 Softmax 后权重为零）：

$$
\text{MaskedAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}
$$

其中 $\mathbf{M}$ 为掩码矩阵，上三角部分为 $-\infty$，下三角（含对角线）为 $0$。

> 这确保了自回归生成（Auto-Regressive）的因果关系：当前位置只能依赖于之前的位置，保证了训练与推理的一致性。

#### 2. 交叉注意力（Cross-Attention / Encoder-Decoder Attention）

Decoder 的第二个注意力子层不再是自注意力，而是将 Decoder 当前的表示作为 $\mathbf{Q}$，将 Encoder 的输出作为 $\mathbf{K}$ 和 $\mathbf{V}$：

$$
\text{CrossAttn} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{dec}} \mathbf{K}_{\text{enc}}^\top}{\sqrt{d_k}}\right) \mathbf{V}_{\text{enc}}
$$

这使得 Decoder 可以在生成每个 token 时从源序列（Encoder 的输出）中动态提取相关信息。

### 输出层

Decoder 的最终输出通过一个线性层映射到词汇表大小维度，再经过 Softmax 得到每个位置上每个可能 token 的概率分布：

$$
P(y_t \mid y_{<t}, \mathbf{X}_{\text{source}}) = \text{softmax}(\mathbf{h}_t \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}})
$$

其中 $\mathbf{W}_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$，$|V|$ 为词汇表大小。

训练时使用**交叉熵损失**，对序列中每个位置的预测与真实标签计算损失并取平均。解码时使用**自回归生成**，从头开始逐个生成 token，每次将已生成的 token 序列送入 Decoder 预测下一个 token。

### 网络训练

#### 优化器与学习率调度

原始 Transformer 使用 Adam 优化器，配合自定义的学习率调度策略（warmup + decay）：

$$
lr = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step\_num}^{-0.5}, \;\text{step\_num} \cdot \text{warmup\_steps}^{-1.5}\right)
$$

- 前 $\text{warmup\_steps}$ 步，学习率线性增长（warmup 阶段）
- 之后学习率与步数的平方根成反比衰减

#### 正则化策略

1. **Dropout**：在每个子层输出之后（进入残差连接和归一化之前）、嵌入层和位置编码之和上应用 Dropout（$p = 0.1$）
2. **标签平滑（Label Smoothing）**：将 one-hot 标签与均匀分布混合，防止模型过度自信，提升泛化能力（$\epsilon_{ls} = 0.1$）

### Transformer 的优势与影响

| 优势 | 说明 |
|------|------|
| **并行计算** | 训练时所有位置的注意力可同时计算，无需像 RNN 一样顺序展开 |
| **长程依赖** | 任意两个位置之间都是 $O(1)$ 的路径长度（注意力直接连接），梯度传播不受距离影响 |
| **可解释性** | 注意力权重可以直接可视化，展示模型在关注输入的哪些部分 |
| **通用性** | 不仅适用于 NLP，还被推广到视觉（ViT）、语音、多模态等领域 |
| **可扩展性** | 通过堆叠更多层、增加维度，模型性能有可预测的提升（Scaling Law） |

> Transformer 的提出标志着深度学习进入了"大模型时代"，GPT、BERT、T5、LLaMA 等几乎所有现代大语言模型都基于 Transformer 架构或其变体。

---

## 视觉大模型基础

### 大模型概述

#### 什么是大模型

大模型（Large-Scale Model / Foundation Model）是近年来人工智能领域的核心范式。其核心思想是：**在海量数据上预训练一个大规模的通用模型，然后通过微调（Fine-tuning）或提示（Prompting）等方式适配到各种下游任务**。

大模型的主要特点：
- **参数量巨大**：从数十亿到数千亿甚至万亿参数级别
- **数据规模庞大**：训练数据通常涵盖互联网级别的语料（TB 级别）
- **涌现能力（Emergent Abilities）**：当模型规模超过某个阈值后，会出现小模型所不具备的能力（如上下文学习、思维链推理等）
- **多任务统一**：同一个模型可以处理翻译、摘要、问答、编程等多种任务

#### 大语言模型的训练范式

以 ChatGPT 为代表的现代大语言模型通常遵循以下训练流程：

1. **预训练（Pre-training）**：在海量文本语料上进行自监督学习（如下一个 token 预测），使模型获得广泛的语言知识和能力
2. **监督微调（Supervised Fine-Tuning, SFT）**：在高质量的人工标注指令数据上进行微调，使模型学会遵循人类指令
3. **奖励建模（Reward Modeling, RM）**：训练一个奖励模型（Reward Model）来评估模型输出与人类偏好的对齐程度——对同一 prompt 的多个回复进行人工排序，训练 RM 预测这些偏好排序
4. **强化学习（RLHF / PPO）**：使用强化学习（近端策略优化，PPO）在奖励模型的指导下进一步优化模型，使输出更符合人类偏好

> RLHF（Reinforcement Learning from Human Feedback）技术由 OpenAI 与 Anthropic 等机构共同推进发展，核心思想是将人类偏好信号注入模型训练过程中。InstructGPT 和 ChatGPT 的成功证明了该方法在提升模型有用性和安全性方面的有效性。

---

## Vision Transformer（ViT）

### 从 Transformer 到 ViT

2020 年，Google 团队的 Dosovitskiy 等人在论文《An Image is Worth 16x16 Words》中提出了 **Vision Transformer（ViT）**，首次将纯 Transformer 架构直接应用于图像分类任务，证明了在大规模数据预训练条件下，Transformer 可以完全替代卷积神经网络。

核心思想：**将图像视为 patch（图像块）序列，就像 NLP 中的 token 序列一样，直接送入 Transformer 编码器进行处理**。

### ViT 的 Embedding 层

#### Patch Embedding

将输入图像 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ 切分为不重叠的固定大小的 patch：

- 设图像尺寸为 $224 \times 224 \times 3$（标准 ImageNet 尺寸）
- Patch 大小为 $16 \times 16$
- 则 patch 数量为 $N = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196$
- 每个 patch 被展平为一个向量：$16 \times 16 \times 3 = 768$ 维

每个 patch 通过一个可训练的线性投影（Linear Projection）映射到 $D$ 维嵌入空间：

$$
\mathbf{z}_{p} = \mathbf{x}_{p} \mathbf{E}, \quad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}
$$

其中 $\mathbf{x}_{p} \in \mathbb{R}^{P^2 \cdot C}$ 为一个 patch 展平后的向量，$P$ 为 patch 大小，$D$ 为模型隐藏维度。

> 这实际上等价于使用一个卷积核大小和步长均为 $P$ 的 2D 卷积，输出通道数为 $D$。原始 ViT-Base 使用 $P = 16$，$D = 768$。

#### Class Token

ViT 借鉴了 BERT 的设计，在 patch 序列的最前面拼接一个可学习的 **[class] token**（表示为 $\mathbf{z}_0^0 = \mathbf{x}_{\text{class}}$），其最终对应的编码器输出作为整张图像的全局表示，用于接分类头进行分类：

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{class}};\; \mathbf{z}_p^1 \mathbf{E};\; \mathbf{z}_p^2 \mathbf{E};\; \dots;\; \mathbf{z}_p^N \mathbf{E}], \quad \mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D}
$$

> class token 的设计使得 ViT 不需要像 CNN 那样使用全局平均池化来聚合空间信息。class token 通过自注意力机制自然地与所有 patch 交互，聚合全局信息。

#### 位置编码

ViT 使用可学习的 1D 位置编码（而非原始 Transformer 的正弦编码），为序列中的每个位置（包括 class token）添加一个 $D$ 维的可学习向量：

$$
\mathbf{z}_0^{\text{final}} = \mathbf{z}_0 + \mathbf{E}_{\text{pos}}, \quad \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}
$$

> ViT 中的位置编码是**可学习的**（而非固定的正弦编码）。实验表明，可学习的位置编码和正弦位置编码在性能上差异不大，但可学习编码更为简洁。此外，$N+1 = 197$ 个位置编码分别对应 1 个 class token + 196 个 patch token。

### ViT 编码器

ViT 的编码器与原始 Transformer 的编码器几乎完全相同，由 $L$ 个相同的层堆叠而成，每层包含：

1. **多头自注意力（Multi-Head Self-Attention, MSA）**
2. **多层感知机（MLP）**

每个子层前后使用 Pre-LN 的残差连接结构：

$$
\begin{aligned}
\mathbf{z}'_{\ell} &= \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \quad \ell = 1, \dots, L \\
\mathbf{z}_{\ell} &= \text{MLP}(\text{LN}(\mathbf{z}'_{\ell})) + \mathbf{z}'_{\ell}, \quad \ell = 1, \dots, L
\end{aligned}
$$

#### 自注意力机制（与 Transformer 一致）

MSA 的操作与 NLP Transformer 完全相同——每个头独立计算缩放点积注意力，然后拼接并投影。在 ViT 中，序列长度 $N+1$ 为 197（ViT-B/16），自注意力使得每个 patch 都能与图像中的任意其他 patch 交互，实现了**全局感受野**。

#### MLP 结构

ViT 的 MLP 由两个全连接层和一个非线性激活函数组成。与原始 Transformer 不同，ViT 使用 **GELU（Gaussian Error Linear Unit）** 而非 ReLU：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

其中 $\Phi(x)$ 为标准正态分布的累积分布函数，$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$。

MLP 的表达式为：

$$
\text{MLP}(\mathbf{x}) = \text{GELU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{D \times D_{ff}}$，$\mathbf{W}_2 \in \mathbb{R}^{D_{ff} \times D}$，$D_{ff}$ 通常为 $D$ 的 4 倍（如 ViT-Base 中为 $768 \rightarrow 3072 \rightarrow 768$）。

> GELU 相比 ReLU 更加平滑，在原点附近不强制置零，保留了小的负值，其形状在期望意义上与 Dropout 和 ReLU 的组合相近，在现代 Transformer 模型中已成为标准选择。

### MLP Head（分类头）

经过 $L$ 层编码器处理后，取出 class token 对应的输出向量 $\mathbf{z}_L^0 \in \mathbb{R}^D$，通过层归一化后送入最终的分类头：

$$
\mathbf{y} = \text{LN}(\mathbf{z}_L^0) \mathbf{W}_{\text{head}}
$$

其中 $\mathbf{W}_{\text{head}} \in \mathbb{R}^{D \times K}$，$K$ 为类别数。分类头本质上是一个简单的线性层（无隐藏层），在预训练时用于 ImageNet 分类，在下游任务微调时替换为相应的任务头。

### ViT 的模型变体

| 模型 | 层数 $L$ | 隐藏维度 $D$ | MLP 维度 $D_{ff}$ | 注意力头数 $h$ | 参数量 |
|------|----------|-------------|-------------------|---------------|--------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

### ViT 的关键发现

1. **数据需求大**：与 CNN 相比，ViT 缺少归纳偏置（inductive bias）——即卷积的局部性和平移等变性。因此在小数据集上训练时，ViT 性能不如 CNN。但在大规模数据集（如 ImageNet-21k，约 1400 万张图，或 JFT-300M，约 3 亿张图）上预训练后，ViT 可以超越同等规模的 CNN。

2. **全局感受野**：ViT 从第一层开始就拥有全局感受野（所有 patch 之间都可以直接交互），而 CNN 的感受野是逐层增大的。

3. **patch 化处理**：将图像切分为 patch 并视为 token 序列的方式，使得图像处理与文本处理在模型架构层面实现了统一。

### 微调（Fine-Tuning）

在大规模数据集上预训练后，ViT 可以迁移到各种下游任务。微调时：
- 移除预训练的 MLP Head，替换为针对目标任务的分类头（零初始化）
- 使用更高分辨率的输入图像（如从 $224^2$ 提升到 $384^2$ 或 $512^2$），此时 patch 大小保持不变，序列长度会相应增加
- 由于位置编码是可学习的，需要对预训练的位置编码进行 2D 插值以适应新的序列长度
- 微调时可以使用较小的学习率和较少的训练轮数

---

## CLIP（Contrastive Language-Image Pre-training）

### 动机与背景

传统的计算机视觉模型存在两个主要局限：

1. **封闭的标签空间**：每训练一个模型只能识别固定数量的类别，扩展类别需要重新收集数据、重新训练
2. **视觉与语言的割裂**：模型只学习从图像到类别标签的映射，无法理解图像的语义描述，更无法进行灵活的跨模态推理

2021 年，OpenAI 提出了 **CLIP（Contrastive Language-Image Pre-training）**，通过大规模的图文对数据进行对比学习，将视觉和语言统一到同一个表示空间中，实现了强大的**零样本（Zero-shot）**分类能力。

### 核心思想

CLIP 的核心思想简洁而优雅：**利用自然语言作为视觉概念的监督信号**。

- 传统做法：图像 → CNN/ResNet → 固定的 1000 类分类器（如"猫"、"狗"等）
- CLIP 的做法：图像 → 图像编码器 → 图像特征向量；文本描述（如 "a photo of a cat"）→ 文本编码器 → 文本特征向量；通过对比学习使匹配的图文对在特征空间中尽量接近，不匹配的对尽量远离

### 模型架构

CLIP 由两个独立的编码器组成：

#### 图像编码器（Image Encoder）

可以使用两种架构：
- **ViT 架构**（如 ViT-B/32、ViT-L/14）：将图像切分为 patch，通过 Transformer 编码器得到图像特征
- **ResNet 架构**（如 ResNet-50、ResNet-101）：使用传统的 CNN 提取图像特征，通常做一定修改（如加入注意力池化层）

图像编码器的最终输出经过一个线性投影映射到联合嵌入空间中的一个向量：$\mathbf{I} \in \mathbb{R}^{d}$。

#### 文本编码器（Text Encoder）

使用 **Transformer 解码器**（GPT-2 风格，仅有 masked self-attention）架构：
- 输入的文本首先被分词并转换为 token 嵌入
- 加入可学习的位置编码
- 经过多层 Transformer 处理
- 取最后一个 token（[EOS]）位置的输出，通过线性投影映射到联合嵌入空间：$\mathbf{T} \in \mathbb{R}^{d}$

> 在原始 CLIP 中，文本编码器是一个 12 层、8 个注意力头、512 维的 Transformer（与 GPT-2 的 117M 参数量级相当），词表大小为 49,152（使用 BPE 分词）。图像编码器 ViT-L/14 约 307M 参数。

### 对比学习（Contrastive Learning）

CLIP 的训练使用**批量对比损失（Batch-wise Contrastive Loss）**，也称为 InfoNCE 损失。

给定一个包含 $N$ 个图文对的 batch $\{(\mathbf{I}_i, \mathbf{T}_i)\}_{i=1}^N$：

1. 分别计算所有图像特征 $\{\mathbf{I}_i\}$ 和所有文本特征 $\{\mathbf{T}_i\}$
2. 对特征向量做 L2 归一化
3. 计算图像-文本余弦相似度矩阵 $\mathbf{S} \in \mathbb{R}^{N \times N}$，其中 $\mathbf{S}_{ij} = \frac{\mathbf{I}_i^\top \mathbf{T}_j}{\|\mathbf{I}_i\| \|\mathbf{T}_j\|}$（实际上归一化后即 $\mathbf{I}_i^\top \mathbf{T}_j$）
4. 引入可学习的温度参数 $\tau$ 缩放相似度：$\mathbf{S}_{ij}' = \mathbf{S}_{ij} \cdot e^\tau$
5. 分别从图像侧和文本侧计算交叉熵损失，取平均：

$$
\begin{aligned}
\mathcal{L}_{\text{image}} &= -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathbf{I}_i^\top \mathbf{T}_i \cdot e^\tau)}{\sum_{j=1}^N \exp(\mathbf{I}_i^\top \mathbf{T}_j \cdot e^\tau)} \\
\mathcal{L}_{\text{text}} &= -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathbf{T}_i^\top \mathbf{I}_i \cdot e^\tau)}{\sum_{j=1}^N \exp(\mathbf{T}_i^\top \mathbf{I}_j \cdot e^\tau)} \\
\mathcal{L}_{\text{CLIP}} &= \frac{1}{2}(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})
\end{aligned}
$$

> 直观理解：$N \times N$ 的相似度矩阵中，对角线元素（$\mathbf{I}_i$ 与 $\mathbf{T}_i$，真实的图文匹配对）应该具有最高的相似度。损失函数鼓励模型拉近匹配对（正样本），推远所有非匹配对（负样本，共 $N-1$ 对）。双向（图像→文本和文本→图像）计算损失确保了表示空间在两个模态方向上的对称性。

### 训练数据

CLIP 使用了从互联网收集的 **4 亿个图文对（400M image-text pairs）**——一个名为 WIT（WebImageText）的私有数据集。这与 ImageNet（约 120 万张标注图像）相比，数据量大了约 300 倍，且不需要人工标注类别。训练使用了大规模分布式训练（数百块 GPU/TPU），batch size 达到 32,768。

> CLIP 训练的巨大数据规模和 batch size 是对比学习成功的关键——大 batch 意味着每个样本都能获得更多的负样本，从而使对比信号更强、表示质量更好。

### 零样本推理（Zero-Shot Inference）

这是 CLIP 最具吸引力的能力。在推理时：

1. **构建文本提示**：对于目标数据集的每个类别（如 ImageNet 的 1000 个类），将类别名称嵌入到自然语言提示模板中，例如：
   - `"a photo of a {class_name}"`
   - `"a picture of a {class_name}"`
   - 或更精细的模板如 `"a photo of a {class_name}, a type of pet"`（对于宠物类别）

2. **文本特征提取**：将所有类别的文本提示送入文本编码器，得到一批文本特征向量

3. **图像特征提取**：将待分类的图像送入图像编码器，得到图像特征向量

4. **相似度匹配**：计算图像特征与所有文本特征之间的余弦相似度，取 Softmax 得到分类概率：
   $$
   P(y \mid \mathbf{I}) = \frac{\exp(\mathbf{I}^\top \mathbf{T}_y \cdot e^\tau)}{\sum_{c=1}^K \exp(\mathbf{I}^\top \mathbf{T}_c \cdot e^\tau)}
   $$

5. **Prompt Engineering**：提示模板的设计对零样本性能有显著影响。CLIP 使用**提示集成（Prompt Ensemble）**——对多个不同模板的文本特征取平均，作为每个类别的最终文本嵌入：
   $$
   \bar{\mathbf{T}}_c = \frac{1}{M} \sum_{m=1}^M \mathbf{T}_{c, m}
   $$
   这显著提升了零样本分类的准确率（在 ImageNet 上提升约 5 个百分点）。

### CLIP 的性能与影响

#### 零样本能力

- 在 ImageNet 零样本分类上达到 76.2% top-1 准确率（ViT-L/14），与有监督的 ResNet-50 相当，而无需任何 ImageNet 训练样本
- 在 30 个不同的计算机视觉数据集上进行零样本评估，CLIP 在大多数数据集上超过了专门的监督模型

#### 鲁棒性与泛化

- CLIP 对自然分布偏移（distribution shift，如图像风格变化、抽象画等）表现出极强的鲁棒性，远超标准监督模型
- 在 Sketch、Rendered 等非自然图像上也有出色的表现

#### 多模态理解

- 开启了视觉-语言预训练的新范式（VLP，Vision-Language Pre-training）
- 成为后续许多多模态模型（如 ALIGN、BLIP、LLaVA）的基础

#### 局限性

- 在细粒度分类（如区分不同品种的花、不同型号的汽车等）上性能仍然不足
- 对抽象概念和系统性推理（如计数、空间关系）的理解有限
- 训练效率较低，需要海量数据和大规模计算资源
- 对训练数据中的偏见和刻板印象存在继承问题

---

## 知识蒸馏与 DINO

### 知识蒸馏（Knowledge Distillation）基础

知识蒸馏由 Hinton 等人于 2015 年提出，核心思想是：**用一个大型、训练好的"教师网络（Teacher Network）"的输出来指导一个较小的"学生网络（Student Network）"的学习**。

教师网络输出的"软标签（Soft Labels）"——即每个类别的概率分布（不是 one-hot 硬标签）——包含了比硬标签更丰富的信息（如"猫"与"老虎"之间的相似性），这种信息被称为"暗知识（Dark Knowledge）"。

蒸馏损失通常为两部分的加权和：

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{KD}}(p_s^\tau, p_t^\tau) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(p_s, y_{\text{true}})
$$

其中：
- $p_s^\tau$ 和 $p_t^\tau$ 分别为学生和教师网络在温度 $\tau$ 下的软化概率输出
- $\text{Softmax}_\tau(z_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$，温度 $\tau > 1$ 使得概率分布更平滑
- $\mathcal{L}_{\text{KD}}$ 通常使用 KL 散度
- $\mathcal{L}_{\text{CE}}$ 为传统的交叉熵损失（学生与原标签）

### DINO：自蒸馏与自监督学习的结合

**DINO（Self-Distillation with No Labels）** 由 Facebook AI Research（FAIR）在 2021 年提出，全称意为"无需标签的自蒸馏"。DINO 将知识蒸馏与自监督学习巧妙结合，在不需要任何标注数据的情况下，使 ViT 学到了极具语义的特征表示。

> DINO 的名字源自"DIstillation with NO labels"的缩写，强调了其核心创新——在完全无监督的条件下完成知识蒸馏。

### DINO 的核心架构

DINO 的框架包含两个共享架构但参数不同的网络：

- **教师网络（Teacher Network）**：$g_{\theta_t}$，参数为 $\theta_t$
- **学生网络（Student Network）**：$g_{\theta_s}$，参数为 $\theta_s$

两个网络接收同一图像的不同增强视图：

1. **全局视图（Global Views）**：从原图中裁剪出较大区域（如覆盖原图 50% 以上），通常取 2 个
2. **局部视图（Local Views）**：从原图中裁剪出较小区域（如覆盖原图 25% 以下），通常取多个（如 8 个）

$$
\begin{aligned}
\mathbf{x}_{\text{global}}^{(1)}, \mathbf{x}_{\text{global}}^{(2)} &\sim \text{Aug}_{\text{global}}(\mathbf{x}) \\
\mathbf{x}_{\text{local}}^{(1)}, \dots, \mathbf{x}_{\text{local}}^{(V)} &\sim \text{Aug}_{\text{local}}(\mathbf{x})
\end{aligned}
$$

**核心策略**：所有视图都送入学生网络，而只有全局视图送入教师网络。这鼓励学生从局部信息中推断出与教师从全局信息中看到的一致的语义内容——这种设计被称为**"局部到全局"（local-to-global）的对应关系学习**。

### 训练过程

#### 前向传播

1. 对输入的每个图像，生成 $2$ 个全局视图和多个局部视图
2. 教师网络处理全局视图，学生网络处理所有视图
3. 两个网络的输出经过 Softmax 得到类别概率分布（对特征在最后一个维度做 Softmax）

#### 损失函数 —— 交叉熵 + 停止梯度

DINO 的核心损失函数形式简单而优雅：

$$
\mathcal{L}_{\text{DINO}} = \sum_{\mathbf{x} \in \{\mathbf{x}_{\text{global}}, \mathbf{x}_{\text{local}}\}} \sum_{\mathbf{x}' \in \mathbf{X}_{\text{global}}, \mathbf{x}' \neq \mathbf{x}} H\left(P_t(\mathbf{x}'), P_s(\mathbf{x})\right)
$$

其中 $H(a, b) = -a \log b$ 为交叉熵损失，$P_t$ 和 $P_s$ 分别为教师和学生网络的 Softmax 输出。

> **直观理解**：给定图像的两个不同视图（如全局视图 A 和局部视图 B），教师网络从全局视图 A 中提取特征并输出一个概率分布（软标签），学生网络从局部视图 B 中提取特征，其输出概率分布应该尽量匹配教师的输出。这迫使学生网络学会：即使只看到局部区域，也要推断出与全局视图一致的语义内容。

#### 教师网络参数更新 —— 指数移动平均（EMA）

DINO 最关键的创新在于教师网络的参数更新策略。与学生网络通过反向传播更新不同，教师网络的参数 $\theta_t$ 由学生网络参数的**指数移动平均（Exponential Moving Average, EMA）**得到：

$$
\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s
$$

其中 $\lambda$ 为动量系数（Momentum Coefficient），通常从 0.996 到 1.0 按余弦调度逐渐增加。

> **为什么使用 EMA 而不是反向传播？**
>
> 1. **避免模型坍塌（Collapse）**：在自监督学习中，如果两个网络都通过梯度更新，模型容易找到捷径——将所有输入映射到同一个输出（退化解）。EMA 更新使教师网络成为学生网络的一个"时间平滑版本"，其输出更加稳定，为学生网络提供一致的回归目标，从而防止坍塌。
>
> 2. **提升教师质量**：EMA 相当于对过去多个训练步的学生参数进行加权平均，这天然具有模型集成（Model Ensemble）的效果，教师网络的输出质量通常优于学生的瞬时输出。实际上这是一种"时域自集成"（Temporal Self-Ensembling）。
>
> 3. **课程学习效应**：通过逐渐增加 $\lambda$（从 0.996 到 1.0），使教师网络在训练初期更快地跟随学生更新（快速学习基础特征），在训练后期趋于稳定（提供高质量的一致目标）。

#### 防止坍塌的额外技巧

除了 EMA，DINO 还使用了以下技巧防止模型坍塌：

1. **Sharpening（锐化）**：在教师网络的 Softmax 中使用较低的温度参数 $\tau_t$（如 $\tau_t = 0.04$），使学生输出更加尖锐（更接近 one-hot），避免均匀分布
2. **Centering（中心化）**：对教师输出减去一个运行均值（running mean of the teacher outputs），防止某一个维度主导分布：
   $$
   g_t(\mathbf{x}) \leftarrow g_t(\mathbf{x}) - \mathbf{c}, \quad \mathbf{c} \leftarrow m \mathbf{c} + (1 - m) \frac{1}{B} \sum_{i=1}^B g_t(\mathbf{x}_i)
   $$
   中心化可视为一种"反坍塌"正则化——防止模型将所有样本分配到少数类别
3. **Student Temperature**：学生在计算 Softmax 时使用高温度 $\tau_s$（如 $\tau_s = 0.1$），使输出更平滑，梯度信息更丰富

### DINO 的特征特性

DINO 训练出的 ViT 展现出许多令人惊讶的特性：

#### 1. 自发的语义分割能力

DINO 的自注意力图（Self-Attention Map）自动聚焦于图像中的显著物体，甚至可以实现**无监督的语义分割**：

- 最后一个 Transformer 层的 class token 的自注意力图可以清楚地"定位"到图像中的前景物体
- 不需要任何像素级标注，注意力图就自然地将物体与背景分离
- 这种"免费"的分割能力在之前的自监督方法中是前所未有的

#### 2. 高质量的语义特征

- DINO 特征在 $k$-NN 分类、线性探测（linear probing）等评估中显著优于之前的所有自监督方法
- 在 ImageNet 线性评估中达到 80%+ 的 top-1 准确率（ViT-S/16），接近监督训练的 ViT
- 特征空间具有良好的语义结构，相似的图像在特征空间中自然聚集

#### 3. 浅层与深层的分工

与 CNN 类似，DINO 训练的 ViT 也展现出层级化的特征加工模式：

- **浅层**：关注局部纹理、边缘等低级特征
- **深层**：关注语义区域、完整物体等高级语义特征

### DINOv2：下一代自监督视觉模型

Meta AI 在 2023 年进一步推出了 DINOv2，在 DINO 的基础上进行了若干关键改进：

1. **更大的训练数据**：使用自动构建的大规模高质量图像数据集（约 1.42 亿张图像，通过相似度去重和筛选得到），代替了 DINO 使用的 ImageNet
2. **更高效的训练策略**：
   - 使用 Flash Attention 加速自注意力计算
   - 使用嵌套的 patch 打包（Nested Patch Packing）提升训练效率
   - 改进的 iBOT 损失（结合了 masked image modeling 和自蒸馏）
3. **更大的模型规模**：训练了 ViT-g（11 亿参数）级别的大模型
4. **更通用的表示**：DINOv2 的特征在各种视觉任务（深度估计、语义分割、实例检索等）中都表现出色，且作为冻结特征提取器就可达到优秀的性能

### 总结对比

| 特性 | 监督学习 (CNN/ViT) | CLIP | DINO |
|------|-------------------|------|------|
| 是否需要标注 | 是（类别标签） | 否（图文对，天然存在） | 否（完全无监督） |
| 监督信号来源 | 人类标注 | 网页中的图文共现 | 数据增强的不变性 |
| 特征类型 | 判别性特征 | 视觉-语言对齐特征 | 语义结构化特征 |
| 零样本能力 | 无 | 强 | 有限 |
| 语义分割能力 | 需专门训练 | 需额外的文本引导 | 自发出现在注意力中 |
| 训练数据规模 | 中等（IM-1K） | 大（400M 图文对） | 大（LVD-142M 或 IM-1K） |
| 典型范式 | 分类头训练 | 对比学习（图文） | 自蒸馏（图像自身） |

---

## 从 Transformer 到视觉大模型：技术演进路线总结

1. **Transformer (2017)**：提出自注意力机制替代 RNN，奠定大模型基础架构
2. **BERT / GPT (2018-2020)**：预训练 + 微调范式在 NLP 领域全面成功，展示了大模型的 Scaling Law
3. **ViT (2020)**：将 Transformer 应用于图像分类，当数据规模足够时性能超越 CNN
4. **CLIP (2021)**：将视觉和语言在对比学习框架下对齐，实现强大的零样本视觉识别能力
5. **DINO (2021)**：通过自蒸馏实现自监督学习，ViT 的注意力自发具备语义分割能力
6. **DINOv2 (2023)**：结合自蒸馏和掩码图像建模，提取通用视觉特征，支撑多种下游任务

> 这一演进路线揭示了从**任务专用模型**到**通用基础模型**的范式转变：模型不再需要为每个具体任务从头训练，而是在海量（无标签或弱标签）数据上预训练后，通过简单适配即可迁移到各种下游任务——这正是"视觉大模型"乃至"基础模型"（Foundation Model）时代的核心主张。
