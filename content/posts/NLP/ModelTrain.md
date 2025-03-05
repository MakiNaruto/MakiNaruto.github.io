---
author : MakiNaruto
title : LLM - 模型训练的几种方式
toc : true
date : 2025-03-04
tags : 
  - DeepLearning
  - TrainingMethods

header_img : content_img/NLP/WestWorld.jpg

---

## 存储分类
首先先看看模型计算过程中, 哪些过程需要被存储下来.
存储主要分为两大块：Model States 和 Residual States

Model States指和模型本身息息相关的，必须存储的内容，具体包括：
- optimizer states：Adam优化算法中的 momentum 和 variance
- gradients：模型梯度
- parameters：模型参数W

Residual States指并非模型必须的，但在训练过程中会额外产生的内容，具体包括：
- activation：激活值。在backward过程中使用链式法则计算梯度时会用到。有了它算梯度会更快，但它不是必须存储的，因为可以通过重新做Forward来算它。 activation 加速例子： 
  - 假设有一个激活函数 f(x) = sigmoid(x)，其导数为 f'(x)。
  - 在正向传播时，计算 a = f(x) 并存储 a。
  - 在反向传播时，计算 f'(x)。如果存储了 a，则可以直接使用 a 来计算 f'(x)，例如，如果 f(x) 是  函数，则 f'(x) = f(x) * (1 - f(x))，可以直接利用之前存储的f(x)来计算。
  - 如果没有存储 a，则需要重新计算 f(x)，然后才能计算 f'(x)。
- temporary buffers: 临时存储。例如把梯度发送到某块GPU上做加总聚合时产生的存储。
- unusable fragment memory：碎片化的存储空间。虽然总存储空间是够的，但是如果取不到连续的存储空间，相关的请求也会被fail掉。对这类空间浪费可以通过内存整理来解决。

接下来, 看看在不同的工程实现中, 采用不同的存储策略, 对内存、带宽、GPU等设备传输和计算有哪些影响。

## 模型并行（Model Parallelism）
当你有一个单卡装不下的大模型时，把模型分割成不同的层，每一层都放指定的GPU上. 此时，模型做一轮forward和backward的过程如下：
![](/content_img/NLP/LLM_Learning/ModelTrain/model_parallelism_bubble.jpg)
这张图的含义是：我在GPU0上做完一次forward，然后将GPU0上最后一层的输入传给GPU1，继续做forward，直到四块GPU都做完forward后，我再依次做backward。等把四块GPU上的backward全部做完后，最后一个时刻我统一更新每一层的梯度。

这样做确实能训更大的模型了，但也带来了两个问题：<br>
1.GPU利用度不够。<br>
2.中间结果占据大量内存。在做backward计算梯度的过程中，我们需要用到每一层的中间结果z, 每一层的中间结果的保留随, 着模型的增大占据的显存也越大。

## 流水线并行（Pipeline Parallelism）
为了解决模型并行带来的问题, 而Gpipe提出了流水线并行.
流水线并行的核心思想是：在模型并行的基础上，进一步引入数据并行的办法，即把原先的数据再划分成若干个batch，送入GPU进行训练。未划分前的数据，叫mini-batch。在mini-batch上再划分的数据，叫micro-batch。

### 切分micro-batch
![](/content_img/NLP/LLM_Learning/ModelTrain/pipeline_parallelism_bubble.jpg)
其中，第一个下标表示GPU编号，第二个下标表示micro-batch编号。Gpipe通过实验证明，假设我们将mini-batch划分为M个，当batch>4时，bubble产生的空转时间占比对最终训练时长影响是微小的，可以忽略不计。
将batch切好，并逐一送入GPU的过程，就像一个流水生产线一样（类似于CPU里的流水线），因此也被称为Pipeline Parallelism。

### re-materalization(active checkpoint)
![](/content_img/NLP/LLM_Learning/ModelTrain/re-materialization.jpg)
Gpipe采用了一种非常简单粗暴但有效的办法：用时间换空间，在论文里，这种方法被命名为re-materalization，后人也称其为active checkpoint。
这种方法的主要思想是几乎不存中间结果，等到backward的时候，再重新算一遍forward.

## 数据并行（Data Parallelism）
数据并行的核心思想是：在各个GPU上都拷贝一份完整模型，各自吃一份数据，算一份梯度，最后对梯度进行累加来更新整体模型。如下图所示
![](/content_img/NLP/LLM_Learning/ModelTrain/dp_model.jpg)
但在大模型场景，巨大的存储和GPU间的通讯量，需要考虑的工程实现就是重点了.<br>
一个经典数据并行的过程如下：<br>
1.在每块计算GPU上都拷贝一份完整的模型参数。额外指定一块GPU做梯度收集<br>
2.把一份数据X（例如一个batch）均匀分给不同的计算GPU。<br>
3.每块计算GPU做一轮Forward和Back Forward后，算得一份梯度 G。<br>
4.每块计算GPU将自己的梯度push给梯度收集GPU，做聚合操作。这里的聚合操作一般指梯度累加。当然也支持用户自定义。<br>
5.梯度收集GPU聚合完毕后，计算GPU从它那pull下完整的梯度结果，用于更新模型参数W。更新完毕后，计算GPU上的模型参数依然保持一致。<br>
6.聚合再下发梯度的操作，称为AllReduce。

### 通讯瓶颈与梯度异步更新
实际操作中带来两个问题
存储开销大。每块GPU上都存了一份完整的模型，造成冗余。<br>
通讯开销大。Server需要和每一个Worker进行梯度传输。当Server和Worker不在一台机器上时，Server的带宽将会成为整个系统的计算效率瓶颈。
受通讯负载不均的影响，DP一般用于单机多卡场景。

## 分布式数据并行(Distributed Data Parallel)
DDP首先要解决的就是通讯问题：将Server上的通讯压力均衡转到各个Worker上。 实现这一点后，可以进一步去Server，留Worker。
目前最通用的AllReduce方法：Ring-AllReduce, 它由百度最先提出，非常有效地解决了数据并行中通讯负载不均的问题，使得DDP得以实现。<br>
Ring-AllReduce通过定义网络环拓扑的方式，将通讯压力均衡地分到每个GPU上，使得跨机器的数据并行（DDP）得以高效实现。<br>
DDP把通讯量均衡负载到了每一时刻的每个Worker上，而DP仅让Server做勤劳的搬运工。当越来越多的GPU分布在距离较远的机器上时，DP的通讯时间是会增加的。

## DeepSpeed ZeRO，零冗余优化
先看一下大模型训练的过程中，GPU都需要存什么内容。
![](/content_img/NLP/LLM_Learning/ModelTrain/model_calculation_process.jpg)


### 精度混合训练
在模型计算, forward和backward的过程中，fp32的计算开销也是庞大的。
那么能否在计算的过程中，引入fp16或bf16（半精度浮点数，存储占2byte），来减轻计算压力呢？
于是，混合精度训练就产生了，它的步骤如下图：
![](/content_img/NLP/LLM_Learning/ModelTrain/mixed_precision.jpg)

主要流程如下:
1. 存储一份fp32的parameter，momentum和variance（统称model states）
2. 在forward开始之前，额外开辟一块存储空间，将fp32 parameter减半到fp16 parameter。
3. 正常做forward和backward，在此之间产生的activation和gradients，都用fp16进行存储。
4. 用fp16 gradients去更新fp32下的model states。
5. 当模型收敛后，fp32的parameter就是最终的参数输出。<br>
通过这种方式，混合精度训练在计算开销和模型精度上做了权衡。



## 参考
[猛猿 - 图解大模型训练系列](https://zhuanlan.zhihu.com/p/613196255)