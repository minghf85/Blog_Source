+++
author = "Ming"
title = "Optical_Flow_introduction"
date = "2025-05-05"
description = "光流估计介绍"
tags = [
    "Optical Flow",
    "CV",
    "writing"
]
weight = 1
+++
{{< katex >}}
# 光流估计
## 参考
- [哥伦比亚大学CV课程](https://www.youtube.com/watch?v=lnXFcmLB7sM)
## 介绍
CV中有一种特征估计和学习的方法叫做Optical Flow-光流，通常是用于获取视频相邻帧的信息。  
现在有的应用场景有：
- 自动驾驶和智能交通：可以通过相机分析周围车辆、行人等的运动来辅助驾驶；还可用于建模周围的3d场景，用于自动驾驶的环境理解。
- 视频监控与安防：检测异常行为、分析运动轨迹。
- 机器人导航和SLAM：和自动驾驶差不多，场景地图建模、避障、环境学习。
- 医学影像分析：血液器官的微小运动。
- 视频压缩与处理：通过存储帧间差距来编码，减小存储；稳定视频、补帧等。
- 体育动作微表情分析：光流估计人体运动姿态、表情等，可以识别动作或者分析动作语义等。
这篇文章简要介绍一下背后的原理，作为引导。  
## 运动场
![alt text](/img/Optical_Flow/motionfield.png)
Image velocity of a point that is moving in the scene.  
图片像素点的在一个场中的矢量速度，类似于磁场中磁感线。  
关键点在于三维场景中的点运动(`角点`)到相机成像2d平面中的像素点映射关系。
其中z是单位向量，表示相机的朝向，通过相似三角形可得：
$$
\frac{r_i}{f} = \frac{r_0}{r_0 \cdot z} \tag{1}
$$
再结合速度的表达式可得（相邻帧运动幅度小，可极限等价）：
$$
v_i = \frac{d r_i}{d t} = f \frac{(r_0 \cdot z)v_0-(v_0 \cdot z)r_0}{(r_0 \cdot z)^2} 
$$
再改写一下：
$$
v_i = f \frac{(r_0 \times v_0) \times z}{(r_0 \cdot z)^2} \tag{2}
$$
<video controls src="/video/QQ202556-16224.mp4" title="of-example"></video>

Optical Flow:Motion of brightness patterns in the image.  
理想情况下，光流和运动场是相同的，例外如：一个球在固定光源下旋转；一个球固定，光源旋转。
两帧之间的像素点位移速度矢量场，其中方向和大小表示了像素点的运动方向和速度。  

## 光流限制方程
### 假设
- 光度一致性：
    运动物体在相邻帧之间的亮度不变，假设光照不变。
$$
I(x,y,t) = I(x+dx,y+dy,t+dt) \tag{3}
$$
- 小位移和小时间：
    物体在相邻帧之间的位移和时间间隔都很小，假设相邻帧之间的运动幅度小。

`泰勒展开式`：
$$
f(x+dx) = f(x) + \frac{\partial f}{\partial x}dx + \frac{\partial ^2 f}{\partial x^2} \frac{dx^2}{2!} + \frac{\partial ^3 f}{\partial x^3} \frac{dx^3}{3!} + \cdots \tag{4}
$$
换成光强度函数，并根据假设小位移和小时间，忽略高阶项：
$$
I(x,y,t) \approx I(x,y,t) + \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt \tag{5}
$$

结合(4)(6)可得：
$$
I_x dx + I_y dy + I_t dt = 0 (I_x = \frac{\partial I}{\partial x})
$$
除以dt并让\\(dt \rightarrow 0\\)
$$
I_x \frac{\partial x}{\partial t} + I_y \frac{\partial y}{\partial t} + I_t = 0 \tag{6}
$$

得到**光流限制方程（Optical Flow Constraint Equation）**：
$$
I_x u + I_y v + I_t = 0 \tag{7}
$$
其中u、v是光流矢量的x、y分量，表示像素点在x、y方向上的速度。

## 计算偏导\\(I_x\ I_y\ I_t\\)

![PDcompute](/img/Optical_Flow/PDcompute.png)

以四个像素的两帧图片为例，计算偏导数。  
可以看到这一个方程有两个未知数u、v，是一个欠定方程组，目前无法求解。

## 补充
### 法线流和平行流
![NPflow](/img/Optical_Flow/NPflow.png)
### Aperture Problem
![ApertureP](/img/Optical_Flow/ApertureP.png)

## Lucas-Kanade方法
`假设`：在一个小的窗口W内，光流是一致的，即u、v是常数。    
$$
I_x (k,l) u + I_y (k,l) v + I_t (k,l) = 0 \tag{8}
$$
使用矩阵形式表达（W大小为nxn）：

$$
\begin{bmatrix}
I_x(1,1) & I_y(1,1) \\\\
\vdots & \vdots \\\\
I_x(k,l) & I_y(k,l) \\\\
\vdots & \vdots \\\\
I_x(n,n) & I_y(n,n)
\end{bmatrix}
\begin{bmatrix}
u \\\\
v
\end{bmatrix}
= -
\begin{bmatrix}
I_t(1,1) \\\\
\vdots \\\\
I_t(k,l) \\\\
\vdots \\\\
I_t(n,n)
\end{bmatrix}
\tag{9}
$$

\\(n^2\\)个方程，2个未知数u、v，求解u、v的最小二乘解：
$$
Au=B
$$
$$
u = (A^TA)^{-1}A^T B \tag{10}
$$

When does Optical Flow estimation work?(可自行了解)
- \\(A^TA\\)可逆
- \\(A^TA\\)良态  

不适合光流估计的情况：
- smooth region
- edge region

适合做光流估计的情况：
![workfine](/img/Optical_Flow/workfine.png)
纹理丰富的区域。

## 由粗糙到精细

### 问题
如果两帧图片的像素有一个大的运动怎么办