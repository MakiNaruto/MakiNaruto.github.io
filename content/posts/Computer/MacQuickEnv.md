---
author : MakiNaruto
title : 一键安装你的Mac环境
description : 重置系统, 仅保留程序环境的一键化脚本.
date : 2024-03-18
tags:
  - Shell
  - MacOS
toc : true
header_img : img/Think-twice-code-once.jpg
---

## HomeBrew
应用软件, 开发程序环境, 一键安装. 非常方便. 
由于服务在境外, 访问速度慢, 推荐使用清华源 https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/

```shell
# 程序环境
brew install npm miniconda mas git blacktop/tap/lporg
# 可视化操作软件
brew install --cask google-chrome iterm2 visual-studio-code postman 
```

## 需要密码和权限的软件
比如ToDesk, 在安装时需要权限跳出密码输入, 可使用以下脚本完成, 省去了输入密码确认下一步等点击操作.
```shell
PASSWORD="your passwd"
expect -c "
    spawn brew install --cask todesk
    expect \"Password:\"
    send \"$PASSWORD\n\"
    expect \"to continue or any other key to abort:\"
    send \"\n\"
    interact
"
```

## 苹果应用商店
使用前提: 
- 安装[mas](https://github.com/mas-cli/mas), 推荐使用 HomeBrew 安装. `brew install mas`<br>
- 查询APP ID. Mac App Store 中每一个应用都有自己的应用识别码（Product Identifier）, 这可以在每个应用的链接中看到。mas 就是根据 Product Identifier 安装与更新应用，也提供了查询应用 ID 的命令。
- 使用mas安装, `mas install [APP ID]`

注意:
- 应用必须在商店登陆账号的已购列表中，因为命令行无法完成「购买」这个操作；
- 对于新上架的应用，可能无法查询到其识别码。因为 mas 的查询列表在缓存文件中，目前尚不清楚其列表更新周期，但若由其他途径（如应用链接）得知新上架应用识别码，仍可正常安装。

使用示例, 更多查看[mas官方教程](https://github.com/mas-cli/mas): 
```shell
# 搜索并锁定APP的ID
mas search [APP Name] 
# 应用商店程序安装
mas install [APP ID, APP ID2, ...] 
# 更新
mas upgrade [APP ID, APP ID2, ...] 
```
