---
author : MakiNaruto
title : LLM - Multi Attention
toc : true
date : 2025-02-06
tags : 
  - Attention
  - MLA
  - LLM

header_img : content_img/NLP/WestWorld.jpg

---
## KV-cache
由于基于Transformer的LLM参数量都很大, 占用显存很大, 所以优化显存自然成了主要目的.
复习一下[MHA公式]({{% relref "/posts/NLP/LLM_Learning/Transformer.md/#2--multi-head-attention" %}}), 在推理时, 发现每个计算出的KQV前序列是一样的, 为了避免重复计算, 提出把前序计算好的缓存起来，使用的时候, 直接取就好. 这也就是目前主流的KV-cache的机制.

### 减小KV cache的方法
业界针对KV Cache的优化，衍生出很多方法，主要有四类：

1. 共享KV：多个Head共享使用1组KV，将原来每个Head一个KV，变成1组Head一个KV，来压缩KV的存储。代表方法：GQA，MQA等

2. 窗口KV：针对长序列控制一个计算KV的窗口，KV cache只保存窗口内的结果（窗口长度远小于序列长度），超出窗口的KV会被丢弃，通过这种方法能减少KV的存储，当然也会损失一定的长文推理效果。代表方法：Longformer等

3. 量化压缩：基于量化的方法，通过更低的Bit位来保存KV，将单KV结果进一步压缩，代表方法：INT8等

4. 计算优化：通过优化计算过程，减少访存换入换出的次数，让更多计算在片上存储SRAM进行，以提升推理性能，代表方法：flashAttention等

### 共享KV
![MHA, MQA, GQA, MLA](/content_img/NLP/LLM_Learning/Attention/DeepSeekV2.png)
1. <b>Multi-Head Attention</b>, 图1, 每一层的所有Head都独立拥有自己的KQV权重矩阵, 计算时各自使用自己的权重计算.
2. <b>Multi-Query Attention</b>, 图2, 每一层的所有Head，按照数量分组, 一组的成员, 共享同一个KQV权重矩阵来计算Attention。因此, 分最多组就是MHA(图左), 最少就是MQA(图右).
3. <b>Group-Query Attention</b>, 图3, 每一层的所有Head，都共享同一个KQV权重矩阵来计算Attention.
4. <b>Multi-Head Latent Attention</b>, 图4, 每个Transformer层，只缓存了权重$c_{t}^{KV}$和$k_{t}^{R}$, 个人认为可以理解为缓存了两个分解的低秩矩阵.
![MLA](/content_img/NLP/LLM_Learning/Attention/MLA-DeepSeek-V3.png)


### 参考地址
[[1] deepseek-v2](deepseek-v2:https://arxiv.org/pdf/2405.04434)<br>
[[2] deepseek-v3](deepseek-v3:https://arxiv.org/pdf/2412.19437)<br>
[[3] deepseek技术解读(1)-彻底理解MLA（Multi-Head Latent Attention）](https://blog.csdn.net/qq_27590277/article/details/145171014)<br>