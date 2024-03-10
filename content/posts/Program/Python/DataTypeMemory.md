---
author : MakiNaruto
title : Python - 数据类型所需内存
description : Python DataType Use Memory
date : 2022-02-13
tags:
  - Python
  - DataType

header_img : img/Think-twice-code-once.jpg
---

## Python内置的数据类型
Python提供一些内置数据类型，如：  
dict、list、set、frozenset、tuple、str、bytes、bytearray。    
str 这个类是用来存储Unicode字符串的。  
而 bytes 和 bytearray 这两个类是用来存储二进制数据的。  

## C语言数据类型所占空间
在C中，我们常见的数据类型所占空间为  
```text
char ：1个字节
char*(即指针变量): 8个字节
short int : 2个字节
int：  4个字节
unsigned int : 4个字节
float:  4个字节
double:   8个字节
long:   8个字节
long long:  8个字节
unsigned long:  8个字节
```

## 问题描述
数据类型占用空间一般各语言相差不大，但奇怪的是，在python中数据所占空间却与其他语言不一样，如下图所示

![](/content_img/Python/Memory/1.png)  
使用命令[sys.getsizeof()](https://docs.python.org/zh-cn/3/library/sys.html#module-sys)查询, 返回的数值以字节为单位.

## 究其原因
python中万物皆对象，数据类型也以对象的方式呈现, 我们可以通过对象id、对象值对其进行判断，示例如下
```python
a = 1.0
b = 1
a is b >> False
a == b >> True
```

python代码在运行的时候会由python解析器执行，具体会解析为C语言的某种结构。也就是说，python中的一个int（或其他）映射到c语言中会是一种复杂结构体。
以python的int为例说明，下面是python的int在C中的具体形式：

```c
typedef struct{
  Pyobject_HEAD
  long ob_ival;
}PyIntobject;

struct _longobject{
  long ob_.refcnt;          //用计数
  PyType0bject *ob_type;    //变量类型
  size_t ob_size;           //实际占用内容大小
  long ob_digit[1];         //存储的实际ython值
}
```
可以看出，python int 的实际的值只是相应C结构中的一个属性, 这也就是为什么python中的int 不是4个字节而是28

## Python常用数据结构占用空间及增长方式

```python
Empty
Bytes  type        scaling notes
28     int         +4 bytes about every 30 powers of 2
37     bytes       +1 byte per additional byte
49     str         +1-4 per additional character (depending on max width)
48     tuple       +8 per additional item
64     list        +8 for each additional
224    set         5th increases to 736; 21nd, 2272; 85th, 8416; 341, 32992
240    dict        6th increases to 368; 22nd, 1184; 43rd, 2280; 86th, 4704; 171st, 9320
136    func def    does not include default args and other attrs
1056   class def   no slots 
56     class inst  has a __dict__ attr, same scaling as dict above
888    class def   with slots
16     __slots__   seems to store in mutable tuple-like structure
                   first slot grows to 48, and so on.
```
例如int类型，每当数值超过2^30增加4字节的空间占用
![](/content_img/Python/Memory/2.png)  

## 参考资料
[A More Complete Answer](https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python)<br>
[Python sys 模块](https://docs.python.org/zh-cn/3/library/sys.html#module-sys)