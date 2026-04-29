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
## 存储分类
首先先看看模型计算过程中, 哪些过程需要被存储下来.<br>
存储主要分为两大块：静态显存 + 动态显存<br>
M_total = 模型参数 + 梯度 + 优化器状态 + Activation<br>

Model States 指和模型本身息息相关的，必须存储的内容，具体包括：
- parameters(固定)：模型参数
- gradients(固定)：梯度
- optimizer states(固定)：优化器状态，例如 Adam 优化器的 momentum 和 variance 等。
- activation(固定)：激活值。虽然它不是必须存储的，但在训练过程中会额外产生的内容.


| Model States | 精度 | Bytes | 
| --- | --- | --- |
| Model Weights | FP16 | 2 |
| Gradients | FP16 | 2 |
| Master Weights| FP32 | 4 |
| Adam Optimizer (m/v) | FP32 | 8 |
| Total / 总计 |  - | 16 Bytes |

### Activation
M_act ≈ C × L × B × S × H × bytes

📊 显存优化效果对比<br>
无 Checkpoint : C≈10\~15 (显存占用极高)<br>
开启 Checkpoint : C≈2\~4 (显存降低3~5倍)<br>
FlashAttention : C≈1.5 (显存占用最低)<br>
Checkpoint的核心策略, **用计算换取显存**, 反向传播时重算部分激活值，大幅降低峰值.<br>

