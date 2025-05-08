+++
author = "Ming"
title = "Optical_Flow_introduction"
date = "2025-05-05"
description = "光流估计介绍"
tags = [
    "Optical Flow",
    "CV"
]
categories = [
    "CV"
]
weight = 2
heroStyle = "thumbAndBackground"
series = ["Optical Flow"]
series_order = 1
+++
{{< katex >}}
# 光流估计
## 参考
- [哥伦比亚大学CV课程](https://www.youtube.com/watch?v=lnXFcmLB7sM)
- [哥伦比亚大学CV课程东君中文讲解](https://www.bilibili.com/video/BV1Mc411i7WE)
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
I_x(1,1) & I_y(1,1) \\
\vdots & \vdots \\
I_x(k,l) & I_y(k,l) \\
\vdots & \vdots \\
I_x(n,n) & I_y(n,n)
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix}
= -
\begin{bmatrix}
I_t(1,1) \\
\vdots \\
I_t(k,l) \\
\vdots \\
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
如果两帧图片的像素有一个大的运动怎么办?

### 下采样
![lowres](/img/Optical_Flow/lowres.png)

两帧都下采样之后，像素的motion就会很小了，因为移动单位是看的像素点的位移格子个数。  
然后就可以用之前的光流限制方程计算。  
由图像金字塔的概念，逐层下采样和计算光流。  
![Coarse-to-Fine Estimation Algorithm](/img/Optical_Flow/c2fe.png)
其中\\((u,v)^{(n)}\\)即是真实分辨率的光流。
### 模板匹配
![Template Match](/img/Optical_Flow/TempMatch.png)
原理：在一帧的图片中的一个小窗口T内，计算该窗口在另一帧中一个大一些的窗口S中和T相同大小窗口的相似度，找到最相似的窗口。  
缺点：计算比较慢；也有可能会出现错误匹配。

## 代码
使用`Opencv`中的`calcOpticalFlowFarneback`函数实现`Lucas-Kanade`光流估计检测车辆运动。  
每两秒重新检测角点，新的车辆进入画面也能检测到。   
```python
import cv2
import numpy as np
import time

# 读取视频文件
cap = cv2.VideoCapture('test.mp4')

# 获取视频帧率（FPS）并计算2秒对应的帧数
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_3sec = int(fps * 2)  # 每2秒的帧数

# LK光流参数
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 随机颜色（用于绘制轨迹）
color = np.random.randint(0, 255, (100, 3))

# 初始化
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
mask = np.zeros_like(old_frame)

frame_count = 0  # 帧计数器
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # 每3秒强制重新检测角点
    if frame_count % frames_per_3sec == 0:
        p0 = cv2.goodFeaturesToTrack(frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        mask = np.zeros_like(frame)  # 清空轨迹（可选）

    # 计算光流
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 筛选有效点
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 绘制轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        # 更新角点
        p0 = good_new.reshape(-1, 1, 2)

    # 显示结果
    img = cv2.add(frame, mask)
    cv2.imshow('LK Tracking (3s Reset)', img)

    # 退出条件
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 更新前一帧
    old_gray = frame_gray.copy()

# 释放资源
cap.release()
cv2.destroyAllWindows()
```
结果如封面所示。