# HIT deep learning and nerual network Lecture 3

## 深度学习视觉应用概述

深度学习在计算机视觉领域的应用主要包括三大核心任务：

1. **图像分类（Image Classification）**：判断图像中包含哪类物体（整图级）
2. **目标检测（Object Detection）**：定位并分类图像中的多个物体（边界框级）
3. **语义分割（Semantic Segmentation）**：为图像中的每个像素分配类别标签（像素级）

> 此外还有实例分割（Instance Segmentation）、全景分割（Panoptic Segmentation）、姿态估计（Pose Estimation）等衍生任务。

## 目标检测基础

### 问题定义

目标检测需要同时回答两个问题：
- **定位（Where）**：物体在图像中的位置，通常用边界框（Bounding Box）表示：$(x, y, w, h)$ 或 $(x_1, y_1, x_2, y_2)$
- **分类（What）**：该物体的类别

### 评价指标

#### IoU（Intersection over Union）

IoU 衡量预测边界框与真实边界框的重叠程度：

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{A \cap B}{A \cup B}
$$

IoU 取值 $[0, 1]$，越接近 $1$ 表示预测框与真实框越吻合。

#### mAP（mean Average Precision）

- **Precision（精确率）**：$\frac{TP}{TP + FP}$，预测为正的样本中有多少是正确的
- **Recall（召回率）**：$\frac{TP}{TP + FN}$，真实正样本中有多少被正确召回
- 对每个类别，在不同置信度阈值下计算 Precision-Recall 曲线下的面积即为 AP
- **mAP** 为所有类别 AP 的平均值
- mAP@0.5：IoU 阈值取 0.5；mAP@0.5:0.95：IoU 阈值从 0.5 到 0.95 步长 0.05 的平均 mAP

### 目标检测方法的两大范式

#### 两阶段检测器（Two-Stage Detectors）

先生成候选区域（Region Proposals），再对候选区域进行分类和位置精修：

- **R-CNN 系列**：R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN
- 优点：精度高，对小目标友好
- 缺点：速度慢，难以实时

#### 单阶段检测器（One-Stage Detectors）

直接回归边界框和类别，无需候选区域生成步骤：

- **YOLO 系列**：YOLOv1 → YOLOv2/YOLO9000 → YOLOv3 → YOLOv4 → YOLOv5 → YOLOv6 → YOLOv7 → YOLOv8 → YOLOv9 → YOLOv10
- **SSD 系列**：SSD、RetinaNet
- 优点：速度快，适合实时应用
- 缺点：早期版本对小目标/密集目标精度较低（后续版本已大幅改善）

## YOLO 系列目标检测

### YOLO 的核心思想

YOLO（You Only Look Once）将目标检测视为一个端到端的回归问题，单次前向传播即可同时预测边界框和类别概率。

**核心流程**：

1. 将输入图像划分为 $S \times S$ 的网格（grid）
2. 每个网格预测 $B$ 个边界框及对应的置信度
3. 每个网格同时预测 $C$ 个类别的条件概率
4. 边界框置信度与类别概率相乘，得到每个框的最终得分
5. 通过非极大值抑制（NMS）去除冗余的检测框

### YOLOv1 — 统一检测的开端

#### 网络结构

YOLOv1 的网络结构受 GoogLeNet 启发：

- 24 个卷积层（用于特征提取）+ 2 个全连接层（用于边界框和类别预测）
- 在每个 $1 \times 1$ 降维卷积层后接 $3 \times 3$ 卷积层
- 输入图像尺寸：$448 \times 448 \times 3$
- 输出：$S \times S \times (B \times 5 + C) = 7 \times 7 \times 30$（其中 $B=2$ 个框，$C=20$ 个 VOC 类别）

#### 损失函数

YOLOv1 的损失函数由三部分组成：

$$
\mathcal{L} = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] + \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]
$$

$$
+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2
$$

$$
+ \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
$$

> 对宽高取平方根，使小框的位置偏差获得更大的惩罚权重，缓解大小框位置误差权重不平衡的问题。$\lambda_{\text{coord}} = 5$ 增大位置误差权重，$\lambda_{\text{noobj}} = 0.5$ 减小无物体时的置信度误差权重。

#### YOLOv1 的局限性

- 每个网格只预测一个类别，对小目标和密集目标检测效果差
- 全连接层限制了输入图像必须固定尺寸
- 使用均方误差作为位置损失，对边界框的评分不够准确

### YOLOv2 / YOLO9000 — 更好、更快、更强

YOLOv2 在 YOLOv1 基础上引入了多项改进：

| 改进 | 说明 |
|------|------|
| **Batch Normalization** | 每层卷积后添加 BN，提升收敛速度，同时起到正则化作用 |
| **高分辨率分类器** | 先在 $448 \times 448$ 图像上微调分类网络，再迁移到检测任务 |
| **锚框（Anchor Boxes）** | 借鉴 Faster R-CNN，使用预定义的锚框替代直接回归坐标 |
| **维度聚类（Dimension Clusters）** | 在训练集上使用 K-means 聚类自动确定锚框尺寸，而非手工设计 |
| **直接位置预测** | 直接预测相对于网格中心的偏移，避免锚框机制的不稳定性 |
| **细粒度特征** | 添加 passthrough 层，将浅层高分辨率特征连接到深层，改善小目标检测 |
| **多尺度训练** | 每 10 个 batch 随机改变输入尺寸（$320 \sim 608$），使网络适应不同分辨率 |
| **Darknet-19** | 新的骨干网络：19 层卷积 + 5 个最大池化，分类任务 top-1 76.8% |

#### 锚框回归方式

YOLOv2 对每个锚框预测 5 个值 $(t_x, t_y, t_w, t_h, t_o)$，最终边界框为：

$$
\begin{aligned}
b_x &= \sigma(t_x) + c_x \\
b_y &= \sigma(t_y) + c_y \\
b_w &= p_w \cdot e^{t_w} \\
b_h &= p_h \cdot e^{t_h}
\end{aligned}
$$

其中 $(c_x, c_y)$ 为网格左上角坐标，$(p_w, p_h)$ 为锚框宽高，$\sigma$ 为 sigmoid 函数将偏移限制在 $[0,1]$ 内。

### YOLOv3 — 多尺度与多标签

YOLOv3 是 YOLO 系列的经典版本，核心改进：

#### Darknet-53 骨干网络

- 53 层卷积，大量使用残差连接（Residual Connections）
- 结合了 ResNet 的跳跃连接思想，网络更深但训练稳定
- 分类精度与 ResNet-152 相当，但速度快 2 倍

#### 多尺度特征融合（FPN）

YOLOv3 在三个不同尺度上进行检测，类似特征金字塔网络（Feature Pyramid Network）：

| 检测尺度 | 特征图尺寸 | 感受野 | 适用目标 |
|----------|-----------|--------|---------|
| 大尺度（scale 1） | $13 \times 13$ | 大 | 大目标 |
| 中尺度（scale 2） | $26 \times 26$ | 中 | 中等目标 |
| 小尺度（scale 3） | $52 \times 52$ | 小 | 小目标 |

> 低层特征分辨率高、细节丰富适合检测小目标；高层特征语义强、感受野大适合检测大目标。通过上采样 + 通道拼接实现跨尺度特征融合。

#### 多标签分类

- 将 Softmax 分类替换为独立的逻辑回归（Sigmoid）分类器
- 每个类别独立预测，支持一个物体同时属于多个类别（多标签分类）
- 使用二值交叉熵损失

### YOLOv4 — 工程化的集大成者

YOLOv4（2020 年 4 月）由 Alexey Bochkovskiy 提出，将大量有效的训练技巧整合在一起：

#### 骨干网络：CSPDarknet53

- 在 Darknet53 中引入 CSP（Cross Stage Partial）连接
- CSP 将特征图分为两部分：一部分经过卷积块处理，另一部分直接与输出拼接
- 优点：减少计算量，增强梯度传播，提升推理速度

#### Neck：SPP + PANet

- **SPP（Spatial Pyramid Pooling）**：在 backbone 输出处使用多尺度池化（$1 \times 1$、$5 \times 5$、$9 \times 9$、$13 \times 13$），增大感受野
- **PANet（Path Aggregation Network）**：在 FPN 之后添加自底向上的路径增强，进一步促进特征流通

#### 训练技巧（Bag of Freebies / Bag of Specials）

| 技巧 | 类别 | 说明 |
|------|------|------|
| Mosaic 数据增强 | Freebies | 将 4 张图片拼接成一张进行训练，丰富上下文 |
| CIoU Loss | Freebies | 考虑边界框的中心距离和宽高比的 IoU 损失 |
| CmBN | Freebies | 跨 mini-batch 的批归一化 |
| DropBlock | Freebies | 在空间区域上做 dropout，而非随机丢弃单个神经元 |
| Mish 激活函数 | Specials | 平滑的非单调激活函数 |
| DIoU-NMS | Specials | 考虑中心距离的非极大值抑制 |

#### CIoU Loss

CIoU（Complete IoU）Loss 同时考虑重叠面积、中心点距离和宽高比：

$$
\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v
$$

其中 $\rho$ 为中心距离，$c$ 为外接框对角线长度，$v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$ 衡量宽高比一致性，$\alpha$ 为平衡系数。

### YOLOv5 — 工程落地标杆

YOLOv5 由 Ultralytics 团队发布，重点优化了部署和易用性：

- 使用 PyTorch 实现，提供多种模型尺寸：**n（nano）、s（small）、m（medium）、l（large）、x（xlarge）**
- 引入 **AutoAnchor**：自动学习最优锚框
- 使用 **自适应图片缩放**：根据数据集长宽比分布，自动调整训练时图像缩放策略
- 提供完善的推理和部署工具链（ONNX、TensorRT、CoreML 等）

### YOLOv6 — 工业级高效检测器

YOLOv6（2022 年 9 月）由美团视觉智能部提出，专为工业应用设计，在精度和速度之间取得了优异平衡。

#### 网络结构概述

YOLOv6 的网络结构由三个主要部分组成：

```
输入 → EfficientRep Backbone → RepBiFPN Neck → Efficient Decoupled Head → 输出
```

#### 骨干网络：EfficientRep

EfficientRep 的核心思想是在训练和推理阶段使用不同的网络结构（Reparameterization，结构重参数化）：

- **训练阶段**：使用多分支结构（类似 RepVGG），包含 $3 \times 3$ 卷积 + $1 \times 1$ 卷积分支 + 恒等映射分支
- **推理阶段**：将多分支等价融合为一个 $3 \times 3$ 卷积，推理速度大幅提升

$$
\text{RepConv}_{\text{train}} = \text{Conv}_{3\times3} + \text{Conv}_{1\times1} + \text{Identity}
$$

$$
\text{RepConv}_{\text{infer}} \approx \text{Conv}_{3\times3}^{\text{fused}}
$$

> 结构重参数化的核心原理：BN 层和卷积层在推理时都是线性运算，可以融合为一个卷积；多个并行的卷积也可以等价合并为一个卷积。

#### Neck：RepBiFPN

- 在 PANet（BiFPN）基础上引入重参数化思想
- 使用 RepBlock 作为基本构建块
- 高效的双向特征融合：高语义信息自上而下传递，低层定位信息自下而上传递

#### Head：Efficient Decoupled Head

YOLOv6 使用解耦头（Decoupled Head）将分类和定位任务分离：

- **分类分支**：经过 $3 \times 3$ 卷积 + 逐点卷积预测类别
- **回归分支**：经过 $3 \times 3$ 卷积 + 逐点卷积预测边界框偏移
- 相比 YOLOv5 的耦合头，解耦头让分类和回归任务各司其职，加速收敛

#### 其他关键改进

| 改进 | 说明 |
|------|------|
| **Anchor-free** | 放弃锚框机制，直接预测边界框中心到四边的距离，简化设计 |
| **SimOTA 标签分配** | 使用动态标签分配策略，根据损失值自动确定正样本 |
| **SIoU Loss** | 在 CIoU 基础上进一步考虑角度信息 |
| **VariFocal Loss** | 改进的分类损失函数，对困难样本赋予更高权重 |
| **量化训练** | 支持 INT8 量化推理，便于边缘设备部署 |

#### 模型尺寸

| 模型 | 参数量 | mAP@0.5:0.95 (COCO) |
|------|--------|---------------------|
| YOLOv6-N | 4.3M | 35.9% |
| YOLOv6-S | 17.2M | 43.1% |
| YOLOv6-M | 34.3M | 49.5% |
| YOLOv6-L | 58.5M | 52.5% |

### YOLOv7 — 可扩展的训练优化

YOLOv7（2022 年 7 月）提出了一系列可训练的优化策略：

- **E-ELAN（Extended ELAN）**：扩展的高效层聚合网络，使用分组卷积控制梯度路径
- **模型缩放（Model Scaling）**：同时考虑深度和宽度的复合缩放策略
- **计划的重参数化（Planned Reparameterization）**：针对不同网络层采用不同的重参数化策略
- **辅助训练头（Auxiliary Head）**：在中间层添加辅助检测头，训练时辅助梯度传播，推理时去除

### YOLOv8 ∼ YOLOv10 — 持续演进

| 版本 | 关键创新 |
|------|---------|
| **YOLOv8** | Anchor-free + Decoupled Head + C2f 模块（更丰富的梯度流），由 Ultralytics 维护 |
| **YOLOv9** | PGI（Programmable Gradient Information）+ GELAN（Generalized ELAN），解决深层网络信息丢失问题 |
| **YOLOv10** | 一致的 NMS-free 训练 + 双分配策略（一对多 + 一对一），实现端到端、无后处理的检测 |

### YOLO 系列的技术演进趋势

1. **锚框（Anchor）**：Anchor-based（v2~v5）→ Anchor-free（v6、v8+）—— 减少超参数设计
2. **检测头（Head）**：Coupled head（v1~v5）→ Decoupled head（v6+）—— 分类与回归分离
3. **标签分配**：固定 IoU 阈值 → 动态标签分配（SimOTA、TaskAlignedAssigner）—— 更精准的正负样本划分
4. **骨干网络**：Darknet → CSPDarknet → EfficientRep → GELAN —— 注重推理效率
5. **损失函数**：MSE → IoU Loss → GIoU → DIoU → CIoU → SIoU —— 越来越精确的定位损失
6. **训练技巧**：从简单的数据增强到 Mosaic、MixUp、Copy-Paste 等复杂增强

## 语义分割

### 问题定义

语义分割（Semantic Segmentation）是为图像中的**每一个像素**分配一个语义类别标签。与目标检测不同，语义分割不区分同类物体的不同实例。

- 输入：任意尺寸的 RGB 图像
- 输出：与输入同尺寸的类别分割图（每个像素一个类别 ID）
- 常见数据集：PASCAL VOC 2012（21 类）、Cityscapes（19 类）、ADE20K（150 类）、COCO-Stuff（171 类）

### 评价指标

#### 像素准确率（Pixel Accuracy）

$$
\text{PA} = \frac{\sum_{c=0}^{C-1} TP_c}{\sum_{c=0}^{C-1} (TP_c + FP_c)} = \frac{\text{分类正确的像素数}}{\text{总像素数}}
$$

#### 平均交并比（mIoU, mean Intersection over Union）

mIoU 是语义分割最核心的指标：

$$
\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}
$$

$$
\text{mIoU} = \frac{1}{C} \sum_{c=0}^{C-1} \text{IoU}_c
$$

其中 $C$ 为类别数，$TP_c$ 表示类别 $c$ 中预测正确的像素数。

> mIoU 比像素准确率更公平——对于类别不平衡场景（如大面积背景 vs 小面积物体），像素准确率会被背景类主导，而 mIoU 对所有类别平等对待。

### 全卷积网络（FCN）— 语义分割的开创性工作

FCN（Fully Convolutional Networks）由 UC Berkeley 于 2014 年提出（CVPR 2015），是将 CNN 用于像素级预测的开创性工作。

#### 核心思想

将分类网络（如 VGG-16）的全连接层替换为 $1 \times 1$ 卷积层，使网络能够接受任意尺寸输入并输出空间对应的分割图。

#### 编码器-解码器结构

1. **编码器（Encoder / Downsampling Path）**：使用 VGG-16 的卷积层提取语义特征，通过池化逐步降低分辨率、增大感受野
2. **解码器（Decoder / Upsampling Path）**：通过转置卷积（Transposed Convolution）或双线性插值逐步恢复空间分辨率

#### 跳跃连接（Skip Connections）

将编码器中的浅层高分辨率特征与解码器中的深层低分辨率特征结合：

- **FCN-32s**：直接从 conv7 上采样 32 倍 → 分割粗糙
- **FCN-16s**：conv7 上采样 2 倍 + pool4，再上采样 16 倍 → 分割变精细
- **FCN-8s**：conv7 上采样 2 倍 + pool4，再上采样 2 倍 + pool3，最后上采样 8 倍 → 分割最精细

> 跳跃连接使解码器能同时利用深层的语义信息和浅层的空间细节，大幅提升分割边界的精细程度。

#### 转置卷积（Transposed Convolution）

转置卷积又称反卷积（Deconvolution）或分数步长卷积（Fractionally Strided Convolution），将低分辨率特征图上采样到高分辨率：

- 对于 $s=2$ 的转置卷积，相当于在输入相邻像素之间插入零值，再进行标准卷积
- 转置卷积的卷积核是**可学习的**，可以学习最优的上采样方式

### U-Net — 生物医学分割的经典

U-Net（2015 年）由 Olaf Ronneberger 等人提出，专为生物医学图像分割设计，是最经典的编码器-解码器结构之一。

#### 对称 U 形结构

```
输入 → [编码器: 卷积×2 → 池化] ×4 → [瓶颈层] → [解码器: 上采样 → 拼接 → 卷积×2] ×4 → 输出
```

#### 关键设计

- **对称结构**：编码器 4 次下采样，解码器 4 次上采样，形如字母 "U"
- **跳跃连接**：每个解码器层的输入 = 上采样后的深层特征 $\oplus$ 对应层级的编码器特征（通道维度拼接）
- **加倍通道数**：每次下采样后将通道数翻倍（$64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 1024$）
- **无全连接层**：纯卷积网络，支持任意输入尺寸

#### U-Net 的优势

- 跳跃连接使分割边界非常精细
- 在小数据集上也能有效训练（数据增强 + 弹性变形）
- 广泛应用于医疗影像分割、卫星图像分割等领域

### DeepLab 系列 — 带有空洞卷积的语义分割

DeepLab 系列由 Google 提出，在多尺度上下文获取上做了重要贡献。

#### DeepLab v1（2015）

- 使用**空洞卷积（Dilated / Atrous Convolution）**：在标准卷积核参数之间插入孔洞（零值），在不增加参数量的情况下增大感受野
- 膨胀率 $r$：卷积核元素之间的间隔为 $r-1$ 个零。

  > 例如 $3 \times 3$、$r=2$ 的空洞卷积，其有效感受野等价于 $5 \times 5$ 标准卷积，但参数量不变。

- 使用全连接 CRF（Conditional Random Field）作为后处理，优化分割边界

#### 空洞卷积的数学定义

对于卷积核 $\mathbf{w}$ 和膨胀率 $r$：

$$
y_{i,j} = \sum_{m} \sum_{n} x_{i + r \cdot m,\; j + r \cdot n} \cdot w_{m,n}
$$

#### DeepLab v2（2017）

- **ASPP（Atrous Spatial Pyramid Pooling）**：使用多个不同膨胀率（如 $r=6, 12, 18, 24$）的并行空洞卷积，以不同感受野捕捉多尺度上下文信息
- 骨干网络升级为 ResNet-101

#### DeepLab v3（2017）和 DeepLab v3+（2018）

- **DeepLab v3**：改进 ASPP 模块，添加全局平均池化分支和 BN 层
- **DeepLab v3+**：引入编码器-解码器结构，以 Xception 或 ResNet 为骨干，将 ASPP 作为编码器和解码器之间的桥梁，进一步提升分割边界的精细度

### PSPNet — 金字塔场景解析

PSPNet（Pyramid Scene Parsing Network，CVPR 2017）的核心是**金字塔池化模块（Pyramid Pooling Module）**：

- 使用 4 个不同尺度（$1\times1$、$2\times2$、$3\times3$、$6\times6$）的平均池化
- 每个池化结果经 $1\times1$ 卷积降维后上采样至原分辨率
- 将原始特征图与多尺度池化结果拼接
- 有效利用了全局、子区域、局部等多层次上下文信息

### SegNet

SegNet 的核心创新在于将最大池化索引（Max-Pooling Indices）传递到解码器：

- 编码器每次做最大池化时，记录最大值的位置索引
- 解码器上采样时，直接将特征值放入记录的索引位置，其余位置填 0
- 优点：无需学习上采样参数，内存开销小

### Segment Anything Model（SAM）— 分割大模型时代

SAM（Segment Anything Model）是 Meta AI 于 2023 年 4 月发布的分割基础模型，标志着图像分割进入"大模型"时代。

#### SAM 的核心能力

- **零样本分割（Zero-shot Segmentation）**：无需针对特定任务训练，即可对任意图像中的任意物体进行分割
- **交互式提示**：通过点（Point）、框（Box）、掩码（Mask）或纯文本提示来指定要分割的目标
- **通用性**：在 SA-1B 数据集（1100 万张图片、超过 10 亿个掩码）上训练，泛化能力极强

#### SAM 的网络结构

SAM 由三个组件构成：

1. **图像编码器（Image Encoder）**
   - 基于 **ViT（Vision Transformer）** 架构，使用 MAE（Masked Autoencoder）预训练的 ViT-H/16
   - 输入：$1024 \times 1024$ 的图像
   - 输出：$64 \times 64 \times 256$ 的图像嵌入（Image Embedding）
   - 每张图像只需计算一次，可被多个提示共享

2. **提示编码器（Prompt Encoder）**
   - **稀疏提示（Sparse Prompts）**：
     - 点：位置编码 + 前景/背景指示编码
     - 框：左上角和右下角的位置编码对
     - 自由格式文本（可选扩展）
   - **稠密提示（Dense Prompts）**：
     - 掩码：通过卷积编码后与图像嵌入逐元素相加
   - 输出一组提示嵌入向量

3. **掩码解码器（Mask Decoder）**
   - 轻量级的 Transformer 解码器（两个 Transformer Block + 动态掩码预测头）
   - 将图像嵌入和提示嵌入进行交叉注意力（Cross-Attention）
   - 输出三个层级的掩码预测（整体、部分、子部分），配合 IoU 分数供用户选择

```
输入图像 → Image Encoder (ViT) ───→ 图像嵌入 ────────────────┐
                                                              ↓
提示(点/框/掩码) → Prompt Encoder → 提示嵌入 → Mask Decoder → 掩码 + IoU分数
```

#### SAM 的训练策略

| 阶段 | 数据 | 方式 |
|------|------|------|
| 阶段 1 | 公开分割数据集 | 标准的监督训练（在已有标注上以交互方式模拟提示） |
| 阶段 2 | 模型生成的掩码（SA-1B） | 数据引擎迭代：人工标注 → 模型训练 → 模型辅助标注 → 人工修正 → 再训练 |

> **数据引擎（Data Engine）** 的三个阶段：
> 1. 辅助人工标注：SAM 辅助标注员进行掩码标注
> 2. 半自动标注：SAM 自动生成部分掩码，仅不确定区域由人工确认
> 3. 全自动标注：SAM 在网格点上自动生成所有可能掩码，经后处理过滤

#### SAM 的优势与局限

| 优势 | 局限 |
|------|------|
| 强大的零样本泛化能力 | 推理速度相对较慢（ViT-H 编码器较大） |
| 灵活的交互式提示方式 | 在细粒度/模糊边界上可能产生不准确的分割 |
| 可作为其他视觉任务的基础组件 | 不提供类别语义信息（仅生成掩码，不分类） |
| 开源模型和数据集 | 对纹理复杂或严重遮挡的物体可能失败 |

#### SAM 2

SAM 2（2024 年 7 月发布）进一步扩展了 SAM 的能力：

- 支持**视频分割**：可在视频中跟踪和分割任意物体
- 引入**记忆机制**：通过记忆注意力（Memory Attention）在时间维度上传播信息
- 流式处理：对视频帧逐帧处理，只需在首帧或关键帧提供提示

### 语义分割模型的对比

| 模型 | 年代 | 核心创新 | 骨干网络 | mIoU (VOC 2012) |
|------|------|---------|---------|-----------------|
| FCN-8s | 2014 | 全卷积 + 跳跃连接 | VGG-16 | 62.2% |
| U-Net | 2015 | 对称编码器-解码器 + 跳跃连接 | 自定义 | 生物医学专用 |
| SegNet | 2015 | 池化索引传递 | VGG-16 | 59.9% |
| DeepLab v2 | 2017 | 空洞卷积 + ASPP | ResNet-101 | 79.7% |
| PSPNet | 2017 | 金字塔池化模块 | ResNet-101 | 82.6% |
| DeepLab v3+ | 2018 | 编码器-解码器 + ASPP | Xception | 89.0% |
| SAM | 2023 | ViT 基础模型 + 交互式提示 | ViT-H | 零样本，无固定 mIoU |

### 语义分割中的关键技术总结

1. **编码器-解码器结构**：编码器提取语义特征，解码器恢复空间分辨率
2. **跳跃连接**：将浅层的空间细节传递给深层，改善边界质量
3. **空洞卷积**：不增加参数量的前提下扩大感受野，保留空间分辨率
4. **多尺度上下文聚合**：ASPP、PSP 模块等方式获取多尺度信息
5. **Transformer 架构**：ViT、Swin Transformer 等替代 CNN 作为骨干网络，提供全局感受野
6. **大模型范式**：以 SAM 为代表的基础模型，通过大规模预训练实现零样本泛化

## 目标检测与语义分割的联系与区别

| 维度 | 目标检测 | 语义分割 |
|------|---------|---------|
| 预测粒度 | 边界框（粗粒度） | 像素（细粒度） |
| 输出 | 边界框坐标 + 类别 | 逐像素类别标签 |
| 实例区分 | 区分不同实例 | 不区分同类实例 |
| 典型模型 | YOLO、Faster R-CNN | FCN、U-Net、DeepLab、SAM |
| 应用场景 | 自动驾驶感知、安防监控、人脸检测 | 医学影像分析、卫星遥感、自动驾驶可行驶区域 |
| 主干网络 | CSPDarknet、EfficientRep | VGG、ResNet、Vision Transformer |

> 实例分割（如 Mask R-CNN、YOLACT）在目标检测的基础上为每个检测到的实例生成像素级掩码，相当于检测 + 分割的结合体。

## 深度学习视觉应用的发展趋势

1. **从 CNN 到 Transformer**：Vision Transformer（ViT）、Swin Transformer 等架构开始在检测和分割中替代纯 CNN 骨干网络
2. **从专用模型到基础模型**：SAM、DINOv2、CLIP 等大规模预训练模型提供通用视觉特征，支持多种下游任务
3. **Anchor-free 和 NMS-free**：检测头越来越简化，趋向于端到端训练（DETR、YOLOv10 等）
4. **多模态融合**：视觉 + 语言（如 Grounding DINO、CLIPSeg）、视觉 + 深度/点云等多模态联合
5. **轻量化和边缘部署**：更高效的网络结构和模型压缩技术，使深度学习视觉应用在移动端和嵌入式设备上广泛落地
6. **开放词汇检测与分割**：不局限于预定义类别，能检测/分割任意文本描述的物体（如 GLIP、Grounding SAM）
