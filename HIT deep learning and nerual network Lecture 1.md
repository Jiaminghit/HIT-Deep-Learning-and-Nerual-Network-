# HIT deep learning and nerual network Lecture 1
## 线性回归问题：
### 问题描述：
参数定义:$$\boldsymbol{\theta} = [\theta_1, \theta_2, \cdots \theta_n]^{\mathrm{T}}, \mathbf{x} = [x_1, x_2, \cdots x_n]^{\mathrm{T}}$$
假设函数:$$y = h_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta}^{\mathrm{T}}\mathbf{x}$$
给定样本:$$(\mathbf{x}^{(i)}, y^{(i)})$$
损失函数:$$J(\boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^{N} \left( y^{(i)} - h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) \right)^2$$
目标:$$\min_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$
### 求解：
求解条件:$$\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = 0$$
解析解:$$\boldsymbol{\theta} = (\mathbf{X}^{\mathrm{T}}\mathbf{X})^{-1}\mathbf{X}^{\mathrm{T}}\mathbf{y}$$
    其中
    $$
    \mathbf{X} = \begin{bmatrix} (\mathbf{x}^{(1)})^{\mathrm{T}} \\ (\mathbf{x}^{(2)})^{\mathrm{T}} \\ \dots \\ (\mathbf{x}^{(N)})^{\mathrm{T}} \end{bmatrix}, \mathbf{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \dots \\ y^{(N)} \end{bmatrix}
    $$
## 线性二分类问题：
### 结果变换 （S函数）
由于是二分类问题，结果只有0，1之分；但由于我们最终需要概率(结果落在0-1之间)，所以需要在回归问题的基础上对值做一个变换
* sigmoid函数：$$y = \frac{1}{1 + e^{-z}}$$
    > 其中 $z = \theta_1x_1 + \theta_2x_2 + \theta_0$
    > sigmoid函数性质：$y' = y(1 - y)$
### softmax 回归：
* 给定样本：$$(\mathbf{x}^{(i)}, y^{(i)})$$
* 损失函数：
    $$
    J(\boldsymbol{\theta}) = \frac{1}{2}\sum_{i=1}^{N}\left(y^{(i)} - h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})\right)^2 \\
    h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) = \frac{1}{1+e^{-\boldsymbol{\theta}^{\mathrm{T}}\mathbf{x}^{(i)}}} = \frac{1}{1+e^{-z(\mathbf{x}^{(i)})}}
    $$
* 目标：找到超平面参数$\theta$，使$J(\boldsymbol{\theta})$最小，$\min_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$
### 梯度下降法：
求解如上问题需要令$\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = 0$ 但是此时的损失函数为非线性函数无法求出精确解$\boldsymbol{\theta}$,所以采用迭代的方法，让$J(\boldsymbol{\theta}) \rightarrow 0$
* 方法步骤：
  1. 将问题转化为：构建迭代序列，为使其满足如下形式：
  $$\boldsymbol{\theta}_1, \boldsymbol{\theta}_2, \cdots \boldsymbol{\theta}_k \rightarrow \boldsymbol{\theta}^*$$
  最简单的构建方式为：
  $$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \Delta\boldsymbol{\theta}_k$$
  > 确定 $\Delta\boldsymbol{\theta}_k$ 即可确定如何构建序列
  2. 对代价函数进行一阶taylor展开近似
  $$J(\boldsymbol{\theta}_{k+1}) = J(\boldsymbol{\theta}_k) + \left[\frac{dJ}{d\boldsymbol{\theta}}\right]^{\mathrm{T}} \Delta\boldsymbol{\theta}_k$$
  3. 为了保证每次代价函数每次迭代都会下降$(J(\boldsymbol{\theta}_{k+1}) \leq J(\boldsymbol{\theta}_k))$,需要定义梯度下降的步长为：
  $$\Delta\boldsymbol{\theta}_k = -\alpha \frac{dJ}{d\boldsymbol{\theta}} = -\alpha \nabla_{\boldsymbol{\theta}} J$$
## 对数回归与多分类回归
### 对数回归
* 使用条件概率描述二分类问题：
$$P(y^{(i)} = 1|\mathbf{x}^{(i)}) = h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^{\text{T}}\mathbf{x}^{(i)}}} \\
P(y^{(i)} = 0|\mathbf{x}^{(i)}) = 1 - P(y^{(i)} = 1|\mathbf{x}^{(i)}) = 1 - h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})$$
* 根据 Bayes 公式，二分类问题可以使用条件概率统一描述为：
  $$P(y|\mathbf{x}, \boldsymbol{\theta}) = (h_{\boldsymbol{\theta}}(\mathbf{x}))^y(1 - h_{\boldsymbol{\theta}}(\mathbf{x}))^{1-y}$$
* 假设各样本相互独立，服从Bernoulli分布。则则$\theta$的合理估计值应当是让所有样本事件产生的几率最大， 即应当是极大似然的， 因此取似然函数：
  $$L(\boldsymbol{\theta}) = \prod_{i=1}^{N} P(y^{(i)}|x^{(i)}, \boldsymbol{\theta})$$
  对两边同时取自然对数得到：
  $$l(\boldsymbol{\theta}) = \log L(\boldsymbol{\theta}) = \sum_{i} \left( y^{(i)}\log\left(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})\right) + (1 - y^{(i)})\log\left(1 - h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})\right) \right) \\
  \min_{\boldsymbol{\theta}} \{-l(\boldsymbol{\theta})\} = \max_{\boldsymbol{\theta}} \{L(\boldsymbol{\theta})\}
  $$
* 重新修改指标函数与求导：
  1. 重新修改指标函数：$$J(\boldsymbol{\theta}) = -\sum_{i} \left( y^{(i)}\log\left(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})\right) + (1 - y^{(i)})\log\left(1 - h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})\right) \right)$$
  2. 对其最小化，有：$$\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \sum_{i} \mathbf{x}^{(i)} \left( h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)} \right)$$
  > 其中 $\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}$ 也可记为梯度算子 $\nabla_{\boldsymbol{\theta}}$ ，$$\nabla_{\boldsymbol{\theta}} = \frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \left[ \frac{\partial}{\partial \theta_1} \quad \cdots \quad \frac{\partial}{\partial \theta_n} \right]^{\text{T}}$$
### 多分类回归
* 对于有 $k$ 个标记的分类问题，分类函数如下：
  $$h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) = \begin{bmatrix} p(y^{(i)} = 1 | \mathbf{x}^{(i)}, \boldsymbol{\theta}) \\ p(y^{(i)} = 2 | \mathbf{x}^{(i)}, \boldsymbol{\theta}) \\ \vdots \\ p(y^{(i)} = k | \mathbf{x}^{(i)}, \boldsymbol{\theta}) \end{bmatrix} = \frac{1}{\sum_{c=1}^k e^{\boldsymbol{\theta}_c^{\text{T}}\mathbf{x}^{(i)}}} \begin{bmatrix} e^{\boldsymbol{\theta}_1^{\text{T}}\mathbf{x}^{(i)}} \\ e^{\boldsymbol{\theta}_2^{\text{T}}\mathbf{x}^{(i)}} \\ \vdots \\ e^{\boldsymbol{\theta}_k^{\text{T}}\mathbf{x}^{(i)}} \end{bmatrix}$$
  因为是多分类，所以需要多个分割超平面，因此有：$\boldsymbol{\theta} = \begin{bmatrix} \boldsymbol{\theta}_1^{\text{T}} \\ \boldsymbol{\theta}_2^{\text{T}} \\ \vdots \\ \boldsymbol{\theta}_k^{\text{T}} \end{bmatrix}$
* 取代价函数：
  $$J(\boldsymbol{\theta}) = -\left[ \sum_{i=1}^N \sum_{k=1}^K 1\{y^{(i)} = k\} \log \frac{\exp(\boldsymbol{\theta}^{(k)\text{T}}\mathbf{x}^{(i)})}{\sum_{j=1}^K \exp(\boldsymbol{\theta}^{(j)\text{T}}\mathbf{x}^{(i)})} \right]$$
  对应梯度：$$\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}^{(k)}} = - \sum_{i=1}^N \left[ \mathbf{x}^{(i)} \left( 1\{y^{(i)} = k\} - P(y^{(i)} = k | \mathbf{x}^{(i)}; \boldsymbol{\theta}) \right) \right]$$
* 以上代价函数恶意简写为**交叉熵损失**形式：
  $$l(y, \hat{y}) = - \sum_{j=1}^K y_j \log \hat{y}_j$$
  > 信息熵定义：$H(X) = - \sum_{i=1}^K p(x_i) \log(p(x_i))$

## 神经元模型
### 形式
$$y = f\left(\sum_{j=1}^{n} w_j x_j - \theta\right)$$
向量化表示：
$$z = \sum_{j=1}^{n} w_j x_j - \theta = \mathbf{w}^{T}\mathbf{x}\\
y = f(z)$$
### 激活函数
略

## 感知机模型
### 感知机原理
* 感知机从输入到输出的模型如下：$$y = f(x) = \text{sign}(\mathbf{w}^{\text{T}}\mathbf{x})$$
  > 其中 $\text{sign}$ 为符号函数：$\text{sign}(x) = \begin{cases} -1 & x < 0 \\ 1 & x \ge 0 \end{cases}$
* 感知机的损失函数：
  对于样本 $(\mathbf{x}^{(i)}, y^{(i)})$，注意到，如果样本正确分类，则有：
  $$\begin{cases} 
  \frac{y^{(i)}(\mathbf{w}^{\text{T}}\mathbf{x}^{(i)})}{\|\mathbf{w}\|} > 0 & \text{正确分类样本} \\ 
  \frac{y^{(i)}(\mathbf{w}^{\text{T}}\mathbf{x}^{(i)})}{\|\mathbf{w}\|} < 0 & \text{错误分类样本} 
  \end{cases}$$
  > 其中$d = \frac{\mathbf{w}^{\text{T}}\mathbf{x}}{\|\mathbf{w}\|}$为点到超平面的距离公式

  因此可定义损失函数如下：$$L(\mathbf{w}) = - \frac{1}{\|\mathbf{w}\|} \sum y^{(i)}(\mathbf{w}^{\text{T}}\mathbf{x}^{(i)})$$
  我们需要找到超平面参数 $\mathbf{w}^*$，满足：$$L(\mathbf{w}^*) = \min_{\mathbf{w}} \left( - \sum y^{(i)}(\mathbf{w}^{\text{T}}\mathbf{x}^{(i)}) \right)$$
* 感知机的学习算法：
  输入：训练数据集 $\{\mathbf{x}^{(i)}, y^{(i)}\}$ (监督学习)输出：$\mathbf{w}$
  1. 赋初值 $\mathbf{w}_0$。数据序号 $i = 1$，迭代次数 $k = 0$。
  2. 选择数据点 $(\mathbf{x}^{(i)}, y^{(i)})$
  3. 判断该数据点是否为当前模型的误分类点，即判断若$$y^{(i)}(\mathbf{w}^{\text{T}}\mathbf{x}^{(i)}) \le 0$$，则更新权值：$$\mathbf{w}_{k+1} = \mathbf{w}_k + \eta y^{(i)}\mathbf{x}^{(i)}$$
  > 注：此更新方法与 Hebb 规则相同
  4. 转到2，直到训练集中没有误分类点
## 多层感知机
$$y_1^{[1]} = f\left(w_{11}^{[1]}x_1 + w_{12}^{[1]}x_2 - \theta_1^{[1]}\right)$$
$$y_2^{[1]} = f\left(w_{21}^{[1]}x_1 + w_{22}^{[1]}x_2 - \theta_2^{[1]}\right)$$
$$y = f\left(w_1^{[2]}y_1^{[1]} + w_2^{[2]}y_2^{[1]} - \theta\right)$$
$$f(\cdot) = \begin{cases} 1, & \cdot \ge 0 \\ 0, & \cdot < 0 \end{cases}$$
> 多层感知器网络，有如下定理：
> **定理1** 若隐层节点（单元）可任意设置，用三层阈值节点的网络，可以实现任意的二值逻辑函数。
> **定理2** 若隐层节点（单元）可任意设置，用三层S型非线性特性节点的网络，可以一致逼近紧集上的连续函数或按 范数逼近紧集上的平方可积函数。
## 多层前馈网络及BP算法
### 多层前馈网络
### BP学习算法
BP学习算法由正向传播和反向传播组成：
1. 正向传播是输入信号从输入层经隐层，传向输出层，若输出层得到了期望的输出，则学习算法结束；否则，转至反向传播。
2. 反向传播是将误差(样本输出与网络输出之差)按原联接通路反向计算，由梯度下降法调整各层节点的权值和阈值，使误差减小。
## BP反传算法的推导
### 先来看一下前向传播 —— 其中涉及到了一些反传中用到的中间值
网络中第$l$ 层，其线性输出为：
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} $$
经过激活函数(Sigmoid)后其输出为：
$$\mathbf{a}^{[l]} = f(\mathbf{z}^{[l]})$$
对应的第$l$层第$i$个节点先行输出为：
$$ z_i^{[l]} = \sum_{j}w_{ij}^{[l]}a_j^{[l-1]}$$
> 其中$w_{ij}^{[l]}$表示连接第$l$层第$i$个节点和第$l - 1$层第$j$个接待你的权重值
### 输入输出
1. 输入：$\mathbf{a^{[0]}} = \mathbf{x}$
2. 输出：$\hat{\mathbf{y}} = \mathbf{a^{[L]}} = \mathbf{a}$
3. 输入输出样本：$\{\mathbf{x}^{(1)}, \mathbf{y}^{(1)}\}, \{\mathbf{x}^{(2)}, \mathbf{y}^{(2)}\}, \dots \{\mathbf{x}^{(N)}, \mathbf{y}^{(N)}\}$
### 训练目的
选择均方误差我作为代价函数：
$$J(\mathbf{x}^{(i)}; \mathbf{w}) = \frac{1}{2}(\mathbf{y}^{(i)} - \hat{\mathbf{y}}^{(i)}(\mathbf{x}; \mathbf{w}))^2 = \frac{1}{2}(\mathbf{e}^{(i)}(\mathbf{x}; \mathbf{w}))^2$$
> $\mathbf{e}^{(i)} = \mathbf{y}^{(i)} - \hat{\mathbf{y}}^{(i)}$
训练目的是：
$$\min_{\mathbf{w}}J(\mathbf{x}^{(i)}; \mathbf{w})$$
### 梯度下降迭代算法
设初始权值为 $\mathbf{w}_0$，$k$时刻权值为 $\mathbf{w}_k$，则使用泰勒级数展开，有：$$J(\mathbf{w}_{k+1}) = J(\mathbf{w}_k) + \left[ \frac{dJ}{d\mathbf{w}} \right]^\mathrm{T} \Bigg|_{\mathbf{w}=\mathbf{w}_k} \Delta\mathbf{w}_k + \cdots$$
选择$$\Delta\mathbf{w}_k = -\alpha \frac{dJ}{d\mathbf{w}}, \quad 0 < \alpha \le 1$$这样每一步都能保证 $J(\mathbf{w}_{k+1}) \le J(\mathbf{w}_k)$，从而使 $J$ 最终可收敛到最小
### 前向传播
### 误差反传 —— 输出层
以一个包含隐含层和输出层的二层网络为例，我们先求代价函数对输出层权值 $w_{ij}^{[2]}$ 的偏导。
这里必须使用微积分中的链式求导法则。因为权值 $w_{ij}^{[2]}$ 影响了线性组合 $z_i$，$z_i$ 影响了激活输出 $a_i$，$a_i$ 影响了误差 $e_i$，最后 $e_i$ 决定了代价函数 $J$ 。我们将这四步的偏导数相乘：
$$
\begin{aligned}
  \frac{\partial J}{\partial w_{ij}^{[2]}} &= \frac{\partial J}{\partial e_i} \cdot \frac{\partial e_i}{\partial a_i} \cdot \frac{\partial a_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_{ij}^{[2]}} \\ &= -e_i a_i(1 - a_i) a_j^{[1]}
\end{aligned}
$$
定义了一个局部梯度（误差项） $\delta_i^{[2]} = a_i(1 - a_i)e_i$ 
最终输出层的权值更新公式变为：
  $$\Delta w_{ij}^{[2]} = \alpha \delta_i^{[2]} \cdot a_j^{[1]}$$
### 误差反传 —— 隐含层
隐含层的输出 $a_i^{[1]}$ 不仅仅影响单一的输出节点，它通过多个连接影响了所有的输出节点 $y_m$ 。因此，在对代价函数求偏导时，需要对所有受影响的输出节点误差求和
* 第一步：构建链式求导的骨架
要算代价函数 $J$ 对隐含层权重 $w_{ij}^{[1]}$ 的偏导，且权重 $w_{ij}^{[1]}$ 仅仅直接影响了它所在的那个隐含层神经元的输出 $a_i^{[1]}$，故根据链式求导法：$$\frac{\partial J}{\partial w_{ij}^{[1]}} = \left[ \frac{\partial J}{\partial e} \right]^T \frac{\partial e}{\partial a_i^{[1]}} \frac{\partial a_i^{[1]}}{\partial w_{ij}^{[1]}}$$
> 这个公式把一个复杂的问题拆成了三部分。其中最棘手的是中间那一项 $\frac{\partial e}{\partial a_i^{[1]}}$（误差对隐含层输出的偏导）。因为 $e$ 是一个向量（包含了所有输出节点的误差），所以这一步展开后是一个雅可比矩阵（Jacobian）乘法的雏形。
* 第二步：算 $\frac{\partial e}{\partial a_i^{[1]}}$
隐含层的第 $i$ 个神经元输出 $a_i^{[1]}$，并不是只影响一个最终结果。它通过不同的权重 $w_{1i}^{[2]}, w_{2i}^{[2]}, \dots, w_{mi}^{[2]}$，分别连接到了所有的输出节点 $y_1, y_2, \dots, y_m$ 上;
以第 $m$ 个输出节点 $y_m$ 为例。根据前向传播的公式，$y_m$ 的线性输入是 $\sum w_{mj}^{[2]}a_j^{[1]}$。当我们对特定的隐含层节点 $a_i^{[1]}$ 求导时，除了第 $i$ 项，其他项常数化为 0，提取出系数 $w_{mi}^{[2]}$。结合 Sigmoid 的导数 $a_m(1-a_m)$，得到：$$\frac{\partial y_m}{\partial a_i^{[1]}} = a_m(1-a_m)w_{mi}^{[2]}$$
既然 $a_i^{[1]}$ 影响了所有的输出节点，那么在计算总梯度时，就必须把所有输出节点反馈回来的误差加起来。这就是下面这个求和公式的物理意义：$$\left[ \frac{\partial J}{\partial e} \right]^T \frac{\partial e}{\partial a_i^{[1]}} = -\sum_{j=1}^m a_j(1-a_j)w_{ji}^{[2]}e_j$$
* 第三步：引入了 $\delta$（局部梯度/误差项）表示法
我们还差最后一部分 $\frac{\partial a_i^{[1]}}{\partial w_{ij}^{[1]}}$。因为 $a_i^{[1]} = \sigma(\sum w_{ij}^{[1]}x_j)$，对其求导非常简单，就是 Sigmoid 导数乘以对应的输入 $x_j$：$$\frac{\partial a_i^{[1]}}{\partial w_{ij}^{[1]}} = a_i^{[1]}(1-a_i^{[1]})x_j$$
为了在代码实现中更高效地利用矩阵运算，我们定义隐含层的局部梯度 $\delta_i^{[1]}$ 为：$$\delta_i^{[1]} = \left[ \sum_{j=1}^m w_{ji}^{[2]}\delta_j^{[2]} \right] (a_i^{[1]})'$$这个定义极为精妙：它说明前一层的局部梯度，等于后一层的局部梯度乘上它们之间的权重矩阵，再点乘本层激活函数的导数。
有了 $\delta$，隐含层的权重更新规则与输出层达到了高度的形式统一：$$\Delta w_{ij}^{[1]} = \alpha \delta_i^{[1]} \cdot x_j$$即：权重变化量 = 学习率 × 目标节点的误差项（$\delta$） × 源节点的输入。
## 全连接网络问题
