---
author : MakiNaruto
title : LLM - GPU显存占用
toc : true
date : 2025-01-15
tags : 
  - DeepLearning
  - TrainingMethods
  - GPUMemory

header_img : content_img/NLP/WestWorld.jpg

---

## 显存占用分类
首先先看看模型计算过程中, 哪些过程需要被存储下来.<br>
### 推理阶段
推理阶段的显存占用主要来自于模型参数和激活值。模型参数是固定的，而激活值则取决于输入数据的大小和模型的结构。对于大型语言模型来说，输入序列较长时，激活值的占用会显著增加，因此需要优化内存使用以适应更长的输入序列。
$$M_{KV} = 2 × L × N_{𝑘𝑣}× D × S × B (bytes)$$
$$M_{per\_token} = 2 × L × N_{𝑘𝑣} × D (bytes/token)$$
- N_{𝑘𝑣}  为 K V矩阵的数量, 一个head 2 个(K + V各一个)
- D 为每个head的维度
- S 为输入序列长度
- B 为batch size
- L 为层数

### 训练阶段
存储主要分为两大块：静态显存 + 动态显存<br>
M_total = 模型参数 + 梯度 + 优化器状态 + Activation<br>

Model States 指和模型本身息息相关的，必须存储的内容，具体包括：
- parameters(固定)：模型参数
- gradients(固定)：梯度
- optimizer states(固定)：优化器状态，例如 Adam 优化器的 momentum 和 variance 等。
- activation(固定)：激活值。虽然它不是必须存储的，但在训练过程中会额外产生的内容.

下面是一个表格，列出了每个模型状态的精度和占用的字节数：
| Model States | 精度 | Bytes | 
| --- | --- | --- |
| Model Weights | FP16 | 2 |
| Gradients | FP16 | 2 |
| Master Weights| FP32 | 4 |
| Adam Optimizer (m/v) | FP32 | 8 |
| Total / 总计 |  - | 16 Bytes |


以一个7B的模型为例, 若不做任何优化, 其显存占用可以估算如下：<br>

Total = 模型 + 梯度 + 优化器状态 + activation<br>
  = 7B * (2bytes + 2bytes + 12bytes) + activation<br>
  = 7B * 16 bytes + activation<br>
  = 112 B bytes / (1024 ** 3) + activation<br>
  ≈ 104GB + activation<br>


#### Activation

$$ M_{act} ≈ C × L × B × S × H × bytes $$

$$ M_{act}≈L⋅B⋅S⋅H⋅(3(QKV)+1(attn_{out})+4(MLP_{expand})+4(MLP_{intermediate} + residual))⋅bytes $$

一般来说, C 的值在 10~15 之间, 取决于模型的具体实现和优化策略. 其中, QKV 的计算需要存储3倍的激活值, attention 输出需要存储1倍的激活值, MLP 扩展和中间层需要存储4倍的激活值, 最后残差连接也需要存储4倍的激活值. 因此, 总体上可以近似估算为:
$$ 
M_{act} ≈ 12⋅L⋅B⋅S⋅H⋅bytes 
$$ 


📊 显存优化效果对比<br>
无 Checkpoint : C≈10\~15 (显存占用极高)<br>
开启 Checkpoint : C≈2\~4 (显存降低3~5倍)<br>
FlashAttention : C≈1.5 (显存占用最低)<br>
Checkpoint的核心策略, **用计算换取显存**, 反向传播时重算部分激活值，大幅降低峰值.<br>

