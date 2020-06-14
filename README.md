# Introduction
road_crack_detect ：利用深度学习技术，检测高速公路路面病害（裂缝、坑洞等）。

# How to Run
## 环境准备
- 需要安装python 3.5
- 以下python包需要安装：
  - tensorflow
  - keras
  - numpy
  - opencv
  - heatmap

## 运行程序
1. 运行predict_highway.py
  ```bash
  python predict_highway.py
  ```
运行结束后，标注后的图片会保存在./images_dst目录。



## 其他说明
1. 模型训练脚本未包含在仓库中。
2. 数据集未包含在仓库中。
