---
author : MakiNaruto
title : 大模型学习之小知识点
description : 大模型学习笔记
toc : true
date : 2025-01-21
tags : 
  - LLM
  - Tips
  
header_img : content_img/NLP/WestWorld.jpg

---
## Loss图像
常见激活函数的图像:

<div style="display: flex; justify-content: space-between;">
<img src="https://pytorch.org/docs/stable/_images/Sigmoid.png" width="30%" title="Sigmoid">
<img src="https://pytorch.org/docs/stable/_images/Tanh.png" width="30%" title="Tanh">
<img src="https://pytorch.org/docs/stable/_images/ReLU.png" width="30%" title="ReLU">
</div>

<div style="display: flex; justify-content: space-between;">
<img src="https://pytorch.org/docs/stable/_images/Sigmoid.png" width="30%" title="Sigmoid">
<img src="https://pytorch.org/docs/stable/_images/Tanh.png" width="30%" title="Tanh">
<img src="https://pytorch.org/docs/stable/_images/ReLU.png" width="30%" title="ReLU">
</div>

<b>其他激活函数: </b>[non-linear-activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)


## LoRA
### LoRA加速原理

比如有非常大维度的权重矩阵 W, 其维度为1024x2048.  

1. 首先，我们将这个权重矩阵冻结，即不再对其进行更新。
2. 引入低秩矩阵: 我们用两个更小的矩阵A, B 代替 W. A: 1024x32，B: 32x2048。
3. 计算新的输出: 在进行前向传播时，我们将输入向量先与A相乘，得到一个维度为32的中间向量。然后，将这个中间向量与B相乘，得到最终的输出。这个过程可以表示为：<br>
output = input @ A @ B
4. 训练低秩矩阵: 在训练过程中，我们只更新A和B这两个小矩阵的参数，而原始的权重矩阵保持不变。由于A和B的维度远小于原始的权重矩阵，因此训练速度会大大加快。
5. 训练结束后合并矩阵: 将更新加到原始权重矩阵: 将计算得到的 ΔW 加到原始的权重矩阵 W 上，得到新的权重矩阵 W'：<br>
ΔW = A @ B<br>
W' = W + ΔW<br>
也就是当我们训练完毕推理时, 使用的为合并后的矩阵.

LoRA只操作模型的线性层, 并且一般不会对lm_head进行参数的更新.

训练期间和训练后的 LoRA 示意图:
![](/content_img/NLP/LLM-PT/lora.png)


[将 LoRA 权重合并到基础模型中](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) 

## embedding
大模型本身是有embedding层的, 不需要去再加载一些单独训练的embedding了, 其embedding的维度和字典的维度不相同. 但在传统的Word2vec、bert等模型中, 其embedding是与字典的长度相同的. 为什么会有这个差异?

```python
model.base_model.embed_tokens
>>> Embedding(151936, 896)
tokenizer.vocab_size
>>> 151643
```

1.  特殊token: 除了常规的词汇，模型通常还会引入一些特殊的token，比如<PAD>、<UNK>、<CLS>等。这些token虽然不在原始的词表中，但也会被分配一个embedding向量，从而导致embedding层的维度略大于词表长度。
2. 预留空间: 在模型训练过程中，为了应对新的词汇或者一些特殊的需求，模型可能会预留一些额外的embedding向量。这些预留的向量可以用来表示未出现在训练数据中的词，或者用于一些特定的任务。
3. 模型架构设计: 某些模型架构可能会在embedding层上进行一些额外的操作，比如引入位置编码等。这些操作可能会导致embedding层的维度发生变化。

## attention_mask 的作用

```python
# 生成随机输入和掩码
X = torch.randn(5, 3)
mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1]])
```

当我们将一些信息选为不关注时, 对信息的关注也是不同的, 如下图所示:
![](/content_img/NLP/LLM-PT/attention_mask.png)
