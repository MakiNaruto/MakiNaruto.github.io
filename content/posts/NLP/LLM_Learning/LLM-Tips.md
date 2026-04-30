---
author : MakiNaruto
title : LLM - Learning Tips
toc : true
date : 2024-12-21
tags : 
  - LLM
  - Tips

header_img : content_img/NLP/WestWorld.jpg

---

## 模型参数量的计算
### 数据精度
要计算参数量, 我们首先要知道, 在计算机表示中, 对于数据的表示精度范围不同, 意味着该数据表示形式可能占用的内存空间也不同.<br>
浮点数在计算机中的存储分为三个部分：<br>
　1. 符号位（sign）：float和double符号位均为1位，0代表正数，1代表负数<br>
　2. 指数位（exponent）：存储科学计数法中的指数部分，采用移位存储<br>
　3. 尾数位（fraction）：存储科学计数法中的尾数部分<br>
![](/content_img/NLP/LLM_Learning/LLM-Tips/data_precision_struct.jpg)

计算方式:<br>
$$(-1)^{符号位} \times 2^{指数位 - 指数偏移} \times (1 + \frac{尾数位}{1024}) $$

例如fp16, 举例说明: <br>
　Exponent(指数位)：5 <br>
　　表示范围 00001(1)到11110(30) <br>
　　为了能表示负数，减去偏置（15），指数部分（-14 - 15 ）<br>
  Fraction 位数位：10 <br>
　　共10位 <br>
　　表示范围 0000000000 - 1111111111（ 0~1023） <br>
　　除以1024（0 - 1023）/1024 <br>

可以表示的最大数据(符号位=1，指数位=30，尾数位=1）：  <br>
$(-1)^{0} \times 2^{30 - 15} \times (1 + \frac{1023}{1024}) $ <br>
可以表示的最小数据(符号位=-1，指数位=30，尾数位=1）： <br>
$(-1)^{1} \times 2^{30 - 15} \times (1 + \frac{1023}{1024}) $ <br>

常用数据精度

|  简称  |                     全称                      |  符号位  | 指数位 | 小数位 | 总位数 | 字节数 |                表示范围                 |
|:----:|:-------------------------------------------:|:-----:|:---:|:---:|:---:|:---:|:-----------------------------------:|
| fp32 | 单精度浮点数<br>（Single-precision floating-point） |   1   |  8  | 23  | 32  |  4  |  $[-3.4 × 10^{38}, 3.4 × 10^{38}]$  |
| fp16 |    半精度浮点数<br>（Half-precision floating-point）    |   1   |  5  | 10  | 16  |  2  |          $[-65504, 65504]$          |
| bf16 |       Brain 浮点数<br>（Brain floating-point）       |   1   |  8  |  7  | 16  |  2  | $[-3.39 × 10^{38}, 3.39 × 10^{38}]$ |


## Loss图像
常见激活函数的图像:

<div style="display: flex; justify-content: space-between;">
<img src="https://pytorch.org/docs/stable/_images/Sigmoid.png" width="30%" title="Sigmoid">
<img src="https://pytorch.org/docs/stable/_images/Tanh.png" width="30%" title="Tanh">
<img src="https://pytorch.org/docs/stable/_images/ReLU.png" width="30%" title="ReLU">
</div>

<div style="display: flex; justify-content: space-between;">
<img src="https://pytorch.org/docs/stable/_images/Hardswish.png" width="30%" title="Hardswish">
</div>

<b>其他激活函数: </b>[non-linear-activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

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
![](/content_img/NLP/LLM_Learning/LLM-Tips/attention_mask.png)
