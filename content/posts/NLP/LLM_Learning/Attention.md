---
author : MakiNaruto
title : LLM - Improving Attention
toc : true
date : 2025-02-06
tags : 
  - Attention
  - MLA
  - LLM

header_img : content_img/NLP/WestWorld.jpg

---

背景: 当输入序列（sequence length）较长时，Transformer的计算过程缓慢且耗费内存，这是因为self-attention的<b>计算时间</b>和<b>内存存取复杂度</b>会随着<b>输入序列</b>的增加成二次增长。因此业界提出了几种加速方案.

## FlashAttention

Attention标准实现没有考虑到对内存频繁的IO操作, 它基本上将HBM加载/存储操作视为0成本。因此FlashAttention的优化方案是通过“split attention”的方式, 将多个操作融合在一起, 只从HBM加载一次，然后将结果写回来。减少了内存带宽的通信开销，并且采用了高效的GPU实现, 极大地提高了效率.。 
![FlashAttention对内存读写的改进](/content_img/NLP/LLM_Learning/Attention/MemoryOperator.jpg)


[知乎: FlashAttention算法详解](https://zhuanlan.zhihu.com/p/651280772)

## 共享KV 
多个Head共享使用1组KV，将原来每个Head一个KV，变成1组Head一个KV，来压缩KV的存储。代表方法：GQA，MQA等
![MHA, MQA, GQA, MLA](/content_img/NLP/LLM_Learning/Attention/DeepSeekV2.png)
1. <b>Multi-Head Attention</b>, 图1, 每一层的所有Head都独立拥有自己的KQV权重矩阵, 计算时各自使用自己的权重计算.
2. <b>Multi-Query Attention</b>, 图2, 每一层的所有Head，按照数量分组, 一组的成员, 共享同一个KQV权重矩阵来计算Attention。因此, 分最多组就是MHA(图左), 最少就是MQA(图右).
3. <b>Group-Query Attention</b>, 图3, 每一层的所有Head，都共享同一个KQV权重矩阵来计算Attention.
4. <b>Multi-Head Latent Attention</b>, 图4, 每个Transformer层，只缓存了权重$c_{t}^{KV}$和$k_{t}^{R}$, 个人认为可以理解为缓存了两个分解的低秩矩阵.
![MLA](/content_img/NLP/LLM_Learning/Attention/MLA-DeepSeek-V3.png)

## 窗口KV
针对长序列控制一个计算KV的窗口，KV cache只保存窗口内的结果（窗口长度远小于序列长度），超出窗口的KV会被丢弃，通过这种方法能减少KV的存储，当然也会损失一定的长文推理效果。代表方法：Longformer等

## 量化压缩
基于量化的方法，通过更低的Bit位来保存KV，将单KV结果进一步压缩，代表方法：INT8等



## 参考地址
[[1] deepseek-v2](deepseek-v2:https://arxiv.org/pdf/2405.04434)<br>
[[2] deepseek-v3](deepseek-v3:https://arxiv.org/pdf/2412.19437)<br>
[[3] deepseek技术解读(1)-彻底理解MLA（Multi-Head Latent Attention）](https://blog.csdn.net/qq_27590277/article/details/145171014)<br>
[[4] 知乎: FlashAttention算法详解](https://zhuanlan.zhihu.com/p/651280772)<br>
