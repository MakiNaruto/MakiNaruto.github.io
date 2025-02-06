---
author : MakiNaruto
title : Torch 一些使用的小Tips
description : 使用过程中遇到的一些问题的记录.
date : 2022-02-13
tags:
  - Python
  - Pytorch
toc : true
header_img : img/Think-twice-code-once.jpg
---


## 为什么使用 contiguous()?
确保切片后的张量在内存中是连续存储的。这是因为切片操作可能会导致张量不再连续存储，而后续的计算（比如计算loss）需要连续存储的张量。

示例: 
``` python
import torch

# 创建一个连续的张量
x = torch.randn(3, 4)
print(x.is_contiguous())  # 输出 True

# 转置操作会使张量不连续
y = x.T
print(y.is_contiguous())  # 输出 False

# 将不连续张量转换为连续张量
z = y.contiguous()
print(z.is_contiguous())  # 输出 True
```

## 模型加载
### 限制哪些卡被使用.
```python
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
```

### 1. 使用的卡, 自动分配
```python
AutoModel.from_pretrained(glm, trust_remote_code=True, device_map='auto')

```

### 2. 分别指定哪些层加载到哪个卡上
```python
1. 加载完模型后查看都有哪些层, 例如
print(model.hf_device_map)

2. 自定义加载显卡
device_map = {
    'transformer.embedding': 0, 
    'transformer.rotary_pos_emb': 0, 
    'transformer.encoder.layers.0': 0, 
    'transformer.encoder.layers.1': 0, 
    'transformer.encoder.layers.2': 0, 
    ...
    'transformer.encoder.layers.27': 1, 
    'transformer.encoder.final_layernorm': 1, 
    'transformer.output_layer': 1
    }

3. 加载模型
AutoModel.from_pretrained(model, trust_remote_code=True, device_map=device_map)
```
