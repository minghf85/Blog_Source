+++
author = "Ming"
title = "Dense Optical Flow"
date = "2025-05-05"
description = "稠密光流估计"
tags = [
    "Optical Flow",
    "CV",
    "Farneback",
    "Dense Optical Flow",
    "writing"
]
categories = [
    "CV"
]
weight = 1
heroStyle = "thumbAndBackground"
series = ["Optical Flow"]
series_order = 2
+++
{{< katex >}}

# 光流估计算法
- 当描述部分像素时，称为：稀疏光流
- 当描述全部像素时，称为：稠密光流
## 稀疏光流估计
详细可看：

{{< article link="/posts/optical_flow_introduction/" >}}

下面简单介绍一下稀疏光流估计：  
稀疏光流估计（Sparse Optical Flow Estimation）是一种用于**追踪视频序列中稀疏特征点运动**的技术。其核心思想是通过分析连续帧之间的像素变化，计算预先选定的关键点（如角点、边缘等）的运动向量。

---

### **核心概念**
1. **光流（Optical Flow）**：
   - 描述图像中像素点在连续帧之间的表观运动（2D位移向量）。
   - 假设亮度恒定（同一像素在不同帧中亮度不变）和运动微小（时间连续）。

2. **稀疏性**：
   - 仅计算图像中少数显著特征点的光流，而非所有像素（稠密光流会计算全图运动）。

---

### **经典算法：Lucas-Kanade (LK) 方法**
- **基本思想**：
  - 基于局部窗口的亮度恒定假设，通过最小二乘法求解运动向量。
  - 适用于运动较小、纹理丰富的区域。
- **数学形式**：
  $$
  \begin{bmatrix}
  \sum I_x^2 & \sum I_x I_y \\\\
  \sum I_x I_y & \sum I_y^2
  \end{bmatrix}
  \begin{bmatrix}
  u \\\\ v
  \end{bmatrix}
  = - 
  \begin{bmatrix}
  \sum I_x I_t \\\\ \sum I_y I_t
  \end{bmatrix}
  $$
  其中 \(I_x, I_y\) 是空间梯度，\(I_t\) 是时间梯度，\(u, v\) 为光流向量。

- **改进**：
  - 金字塔LK算法（处理大运动）：通过图像金字塔分层计算，从粗到细修正运动估计。

---

### **优缺点**
- **优点**：
  - 计算高效（仅处理少量点）。
  - 对纹理丰富的场景鲁棒。
- **缺点**：
  - 依赖特征点质量，在低纹理区域表现差。
  - 无法反映全图运动信息。

---

## 稠密光流估计
稠密光流估计（Dense Optical Flow Estimation）是指对图像中每个像素点进行光流计算，得到全图的运动场。与稀疏光流不同，稠密光流关注的是整个图像的运动信息，适用于需要精确运动分析的场景，如视频稳定、物体跟踪等。

### Farneback方法

