+++
author = "Ming"
title = "Failed to load module gail and atk-bridge while compiling a simple C++ program"
date = "2025-04-28"
description = "ubuntu vscode终端运行opencv imshow 报错：Gtk-Message: 17:07:43.639: Failed to load module \"gail\" Gtk-Message: 17:07:43.639: Failed to load module \"atk-bridge\""
tags = [
    "bug",
    "vscode",
    "opencv"
]
+++

## System information
- ubuntu 22.04
- opencv 4.5.5
- g++ 11.4.0
- cmake 3.22.1

## Detailed description
### 由于我已经修复，所以以下信息来自[`github-issue`](https://github.com/opencv/opencv/issues/25235)  
Gtk-Message: 17:07:43.639: Failed to load module "gail"
Gtk-Message: 17:07:43.639: Failed to load module "atk-bridge"

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.646: Unable to locate theme engine in module_path: "adwaita",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.646: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.647: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.647: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.648: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.648: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.648: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.648: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.648: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.648: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.648: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.649: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.649: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.649: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.649: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.649: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.649: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.649: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.649: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.650: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.650: Unable to locate theme engine in module_path: "pixmap",

(DisplayImage:106670): Gtk-WARNING **: 17:07:43.650: Unable to locate theme engine in module_path: "adwaita",
./DisplayImage: symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0: undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE  

## Steps to reproduce
我参照这篇文章配置第一个opencv项目-[ubuntu22.04 OpenCV4.6.0(c++)环境配置](https://blog.csdn.net/qq_51022848/article/details/128095476)
问题出现在编译运行的地方
```bash
cd build
cmake ..
make
./demo # 执行
```
demo.cpp
```cpp
#include<opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include<iostream>

using namespace std;
using namespace cv;

Mat src;

int main(int argc, char ** argv)
{
        src = imread("/home/ming/PROJECT/digital/test.jpg");//这里是你的图片
        if (src.empty())
        {
		cout << "没有读取到图像" << endl;
		return -1;
        }
        imshow("hello", src);
        waitKey(0);
        return 0;
}
```

CMakeLists.txt
```txt
cmake_minimum_required(VERSION 2.8)
project( digital )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( demo src/demo.cpp )
target_link_libraries( demo ${OpenCV_LIBS} )
```
## Reason and solutions
- vscode installed by snap
一般是因为使用了snap安装的vscode，特别是使用鱼香一键安装命令。  
当时是因为安装ros2，想顺便将vscode这些安装了，使用了一键安装。  
- 解决办法
    - 卸载当前的vscode
    卸载命令
    ```bash
    sudo snap remove code
    ```
    - 安装[官方版本](https://code.visualstudio.com/download)
    选择对应的安装，我这里是选择.deb  
    安装命令
    ```bash
    sudo apt install <file>.deb
    ```
最后重新运行./demo即可


    