---
author : MakiNaruto
title : MarkDown 常用的语法
description : 在编写hugo博客中, 常用的markdown语法记录。
toc : true
date : 2025-01-29
tags : 
  - markdown
  - syntax

header_img : content_img/Other/HearthStone/drunkery.jpeg
---


## 图片
### 链接示例
```markdown
# markdown 原生不可修改图片比例
![](https://pytorch.org/docs/stable/_images/Sigmoid.png)
# 使用html img 使图片变化大小
<img src="https://pytorch.org/docs/stable/_images/Sigmoid.png" width="30%" title="Sigmoid">

# 一行, 多列图片展示.
<div style="display: flex; justify-content: space-between;">
<img src="https://pytorch.org/docs/stable/_images/Sigmoid.png" width="30%" title="Sigmoid">
<img src="https://pytorch.org/docs/stable/_images/Tanh.png" width="30%" title="Tanh">
<img src="https://pytorch.org/docs/stable/_images/ReLU.png" width="30%" title="ReLU">
</div>
```



### 实现效果
<img src="https://pytorch.org/docs/stable/_images/Sigmoid.png" width="30%" title="Sigmoid">


<div style="display: flex; justify-content: space-between;">
<img src="https://pytorch.org/docs/stable/_images/Sigmoid.png" width="30%" title="Sigmoid">
<img src="https://pytorch.org/docs/stable/_images/Tanh.png" width="30%" title="Tanh">
<img src="https://pytorch.org/docs/stable/_images/ReLU.png" width="30%" title="ReLU">
</div>


## 超链接
### 示例

```markdown
使用markdown原生的方式
[链接博客](posts/your_post_relative_path/)

使用hugo的relref  # 删除下面的 \`, 因为hugo 会检查 relref 本地的路径.
[链接博客]({{`%` relref "posts/your_post_relative_path/" `%`}})

打开页面并定位到标题位置
[链接博客+标题处](posts/your_post_relative_path/#title)
[链接博客+标题处]({{`%` relref "posts/your_post_relative_path/#title" `%`}})

互联网网址
[baidu](http://www.baidu.com)
```

上述两种方式, 在链接的填写中并无差异. 在链接到hugo内部的相对文档时, 若要定位到具体标题位置处时, 在文档后加上 <b>/#<标题名称></b>.
定位时按照标题唯一名称来定位, 不区分几级标题. <br>

若标题重复时:
1. 标题会按照顺序自动添加数字区分, 手动加上标题名称即可.
    ```markdown
    标题           # 第一个标题
    标题-1         # 第二个标题
    标题-2         # 第三个标题
    ```
2. 或者在右边的导航栏 <b>CATALOG</b> 找到自己想要定位的位置, 右键复制链接地址后, 查看被转换后的具体名称.

### 实现效果
1. [链接博客](posts/nlp/cs224n-01/)
2. [链接博客+标题处]({{% relref "posts/nlp/cs224n-01/#动画演示" %}})
3. [百度](http://www.baidu.com)