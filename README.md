# The 5th homework for CFD

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

计算流体力学第五次课程作业

## 主要内容

- 关于第五次作业的报告
- 关于第五次作业的数值验证源码

## 前置要求

- Python 3.7+
- matplotlib  第三方库
- numpy 第三方库
- dataclasses 第三方库
- texlive/simple Tex

## 快速开始

基本使用：

- 在 `/doc` 路径下查看报告的 latex 源码和已经编译完成的.pdf文件
- 在 `/src` 路径下查看 python 源码，可以直接运行 `main.py`  来查看数值验证结果，结果储存为同文件夹下的.jpg文件。

## 项目结构

```plaintext
project-root/
├── src/                   # 源代码目录
│   ├── main.py            # 包含使用的参数，主循环
│   ├── func.py            # 各种数值计算函数的集合，还包括可视化函数和划分网格和应用边界条件的函数
│   └── param.py           # 定义了一个参数类
├── docs/                  # 报告目录
│   ├── picture/           # 图片存放目录
│   └── hw5.tex            # 报告的 latex 源码
|   └── hw5.pdf            # 编译完成的报告
└── ReadMe.md              # 说明文档
```