---
author : MakiNaruto
title : SGD梯度下降拟合线性回归
description : 利用梯度下降拟合一元回归学习笔记
toc : true
date : 2019-09-28
tags : 
  - SGD
  - Math
  - Gradient Descent

header_img : content_img/MechineLearning/neural_network.jpeg
---


# 画图

让我们以直线 y=2x+1 附近作散点图，误差内上下浮动。

![散点图](/content_img/MechineLearning/1.webp "散点图")


因为开始时，截距$\theta_{0}$和斜率$ \theta_{1}$初始都为0，绿点为初始点的分布。所以将其描点如图所示。

![初始状态](/content_img/MechineLearning/2.webp  "初始状态")

简单用三幅图表示学习过程

![学习中](/content_img/MechineLearning/3.webp  "学习中")

![最终](/content_img/MechineLearning/4.webp  "最终")

![连线图](/content_img/MechineLearning/5.webp  "连线图")

这里用到了两个公式：
①一元线性表达式：$$h_{\theta}(x)=\theta_{0}+\theta_{1} x \tag{1}$$
②代价函数：
$$
J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
\tag{2}$$
很明显，公式①一元线性表达式就是我们常用的 $y = b + ax \tag{3}$

我们的目的是使代价函数尽可能的小：$
\underset{\theta_{0}, \theta_{1}}{\operatorname{minimize}} J\left(\theta_{0}, \theta_{1}\right)\tag{4}$

接下来我们开始工作，进行梯度下降用到了求偏导。当对其进特征，也就是对$\theta_{0}，\theta_{1}$求偏导。每次求偏导后更新公式为

$\theta_{0} :=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)\tag{5}$

$\theta_{1} :=\theta_{1}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x^{(i)}\tag{6}$

那么将公式(1)  一元元线性表达带入式子(5)   得到式子(7)：

$$\theta_{0} :=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left((\theta_{0}+\theta_{1} x)-y^{(i)}\right)\tag{7}$$

在式(7)中，我们将$(\theta_{0}+\theta_{1} x)-y^{(i)}$ 抽取出来，暂时定义$y^{(d)} = \theta_{0}+\theta_{1} x$，这个就是我们正在变化的绿点，其横坐标$x$不会变化，这样就很明显，每一个蓝点都有对应的绿点在其垂直于$x$轴的连线上，且绿点通过学习不断的变动垂直距离上的位置，我们把公式(7)换一种表达方式，引入$y^{(d)}$

$\theta_{0} :=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(y^{(d)}-y^{(i)}\right)\tag{8}$

这样一看，我们很清楚的就能得到平均垂直距离，也就是$y^{(d)}$随之每次学习，$\theta_{0}$一点一点的变大，蓝点与绿点的垂直距离越来越小，$\theta_{0}$的变化也逐渐变小直至稳定，$\theta_{1}$同理。

代码如下
```python
import numpy as np
import matplotlib.pyplot as plt
import random

X = np.random.random(100).reshape(100,1)   #随机生成100行1列的随机值
Y = np.zeros(100).reshape(100,1)    
for i , x in enumerate(X):
    val = x * 2 + 1 + random.uniform(-0.2,0.2)
    Y[i] = val                     # 令y的值随机偏动在范围内

plt.figure()
plt.scatter(X,Y,color='g',s=10)         #X1 为横坐标轴值，y为纵坐标值
# plt.plot(X,2 * X + 1,color='r',linewidth=2.5,linestyle='-')
plt.show()
```


执行梯度下降以进行线性回归的代码段
```python
#最小二乘法，用来计算损失
def compute_error(theta0,theta1,x_data,y_data):
    totalError=0
    for i in range(0,len(x_data)):
        totalError += ( (theta0 + theta1 * x_data[i]) - y_data[i] ) ** 2
    return totalError/float(len(x_data))

#梯度下降法
def gradient_descent_runner(x_data,y_data,Theta0,Theta1,lr,epochs):
    #计算数据总量长度
    m = float(len(x_data))
    J_Cost = {}
    #循环epochs
    for i in range(epochs):
        J_Theta0 = 0
        J_Theta1 = 0
        J_CostValue = 0
        #每迭代5次，输出一次图像
        if i % 50 == 0:
            print("epochs",i)
            plt.plot(x_data,y_data,'b.')
            plt.plot(x_data, Theta1 * x_data + Theta0 ,'g.')
            plt.show()
            pass
        pass
        '''
        repeat until convergence{
            start
        '''
        #计算梯度
        for j in range(0,len(x_data)):
            #分别对西塔0和西塔1求偏导后的函数
            J_Theta0 += -(1/m) * (y_data[j] - ((Theta1 * x_data[j]) + Theta0))
            J_Theta1 += -(1/m) * (y_data[j] - ((Theta1 * x_data[j]) + Theta0)) * x_data[j]
            
        J_Cost[i] = J_CostValue
        #更新截距和梯度
        Theta0 = Theta0 - (lr * J_Theta0)
        Theta1 = Theta1 - (lr *J_Theta1)
        '''
            end
        }
        '''

    return Theta0,Theta1,J_Cost

lr = 0.1  #设置学习率
Theta0 = 0  #截距相当于Theta0
Theta1 = 0  #斜率,相当于Theta1
epochs = 500  #最大迭代次数

print("Starting Theta0={0},Theta1={1},error={2}".format(Theta0,Theta1,compute_error(Theta0,Theta1,X,Y)))

print("Running")
Theta0,Theta1,J_Cost = gradient_descent_runner(X,Y,Theta0,Theta1,lr,epochs)

print("After {0} iterations Theta0={1},Theta1={2},error={3}".format(epochs,Theta0,Theta1,compute_error(Theta0,Theta1,X,Y)))

# 画图
plt.plot(X, Y,'b.')
#也就是y的值k*x_data+b
plt.plot(X, X * Theta1 + Theta0, 'r')
plt.show()
```



同样的，利用神经网络也可以进行线性回归。

```python
b = np.ones(100).reshape(100,1) 
old_X = np.column_stack((old_X,b))

# 利用torch计算梯度并进行权重的修改
X = torch.tensor(old_X, dtype=torch.float, device=device)
Y = torch.tensor(old_Y, dtype=torch.float, device=device)

# pytorch 
w1 = torch.randn(2, 1, device=device, dtype=dtype, requires_grad=True)
loss = 0

learning_rate = 1e-4
for t in range(500):
    y_pred = X.mm(w1).clamp(min=0)
    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - Y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 利用torch进行反向传播自动更新梯度
    loss.backward()

    with torch.no_grad():
        # 更新权重向量，也就是ax+b 的a，b值
        w1 -= learning_rate * w1.grad

        # 将更新梯度清零
        w1.grad.zero_()

coefficient = torch.sum(w1,1)
Theta1 = coefficient[0].item()
Theta0 = coefficient[1].item()
plt.figure()
plt.scatter(X[:,0],Y,color='g',s=10)         #X1 为横坐标轴值，y为纵坐标值
plt.plot(X,Theta1 * X + Theta0,color='r',linewidth=2.5,linestyle='-')
plt.show()
```