# 示例展示

本章节展示的所有示例位于[examples](https://github.com/OAID/Tengine/tree/tengine-lite/examples) 。

## 准备工作
### 环境准备
要编译和运行示例程序，你需要准备:

1.一台可以编译C/C++ 的Linux环境的电脑（x86或Arm架构均可）。

2.下载Tengine源码，位于 Tengine 的分支 tengine-lite 上：
```bash
git clone -b tengine-lite https://github.com/OAID/Tengine.git  Tengine
```



### 编译

build.sh 编译脚本默认配置已实现自动编译 examples 中的 demo 程序。

**以x86架构为例，编译后demo 存放在 ./build/install/bin/ 目录下：** 

```bash
bug1989@DESKTOP-SGN0H2A:/mnt/d/ubuntu/gitlab/build-linux$ tree install
install
├── bin
│   ├── tm_alphapose
│   ├── tm_classification
│   ├── tm_classification_int8
│   ├── tm_classification_uint8
│   ├── tm_crnn
│   ├── tm_efficientdet
│   ├── tm_efficientdet_uint8
│   ├── tm_hrnet
│   ├── tm_landmark
│   ├── tm_landmark_uint8
│   ├── tm_mobilefacenet
│   ├── tm_mobilefacenet_uint8
│   ├── tm_mobilenet_ssd
│   ├── tm_mobilenet_ssd_uint8
│   ├── tm_nanodet_m
│   ├── tm_openpose
│   ├── tm_retinaface
│   ├── tm_ultraface
│   ├── tm_unet
│   ├── tm_yolact
│   ├── tm_yolact_uint8
│   ├── tm_yolofastest
│   ├── tm_yolov3
│   ├── tm_yolov3_tiny
│   ├── tm_yolov3_tiny_uint8
│   ├── tm_yolov3_uint8
│   ├── tm_yolov4
│   ├── tm_yolov4_tiny
│   ├── tm_yolov4_tiny_uint8
│   ├── tm_yolov4_uint8
│   ├── tm_yolov5
│   └── tm_yolov5s
├── include
│   └── tengine
│       └── c_api.h                     C预测库头文件
└── lib
    ├── libtengine-lite-static.a        静态预测库
    └── libtengine-lite.so              动态预测库
```

### 模型仓库

模型仓库包含了运行examples所需模型、图片和文档。
- [百度网盘](https://pan.baidu.com/s/1JsitkY6FVV87Kao6h5yAmg) （提取码：7ke5）

- [Google Drive](https://drive.google.com/drive/folders/1hunePCa0x_R-Txv7kWqgx02uTCH3QWdS?usp=sharing)


## 分类任务 - [tm_classification.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification.c)

Tengine Lite 兼容 Tengine 原有的 C API 供用户使用，这里我们使用 C API 展示如何运行 tm_classification 例程运行 MobileNet v1 分类网络模型，实现指定图片分类的功能。让你快速上手 Tengine Lite C API。这里，我们使用在这个撸猫时代行业从业者大爱的 tiger cat 作为测试图片。

![](https://z3.ax1x.com/2021/06/30/RBIQIO.jpg)

将测试图片和模型文件放在 Tengine-Lite 根目录下，运行：

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_classification -m models/mobilenet.tmfile -i images/cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679
```

结果如下：

```bash
tengine-lite library version: 1.4-dev

model file : models/mobilenet.tmfile
image file : images/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 33.74 ms, max_time 33.74 ms, min_time 33.74 ms
--------------------------------------
8.574144, 282
7.880117, 277
7.812573, 278
7.286458, 263
6.357486, 281
--------------------------------------
```

## 人脸关键点检测任务 - [tm_landmark.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_landmark.cpp)

使用图片：

![](https://z3.ax1x.com/2021/06/30/RB5dC4.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_landmark -m models/landmark.tmfile -i images/mobileface02.jpg -r 1 -t 1
```

结果如下：

```bash
tengine-lite library version: 1.4-dev
Repeat [1] min 8.784 ms, max 8.784 ms, avg 8.784 ms
```

![](https://z3.ax1x.com/2021/07/01/RrPSuq.jpg)

## retinaface 人脸检测任务 - [tm_retinaface.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_retinaface.cpp)

使用图片：

![](https://z3.ax1x.com/2021/06/30/RBC311.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_retinaface -m models/retinaface.tmfile -i images/mtcnn_face4.jpg -r 1 -t 1
```

结果如下：

```bash
tengine-lite library version: 1.4-dev
img_h, img_w : 316, 474
Repeat 1 times, thread 1, avg time 28.78 ms, max_time 28.78 ms, min_time 28.78 ms
--------------------------------------
detected face num: 4
BOX 1.00:( 38.4053 , 86.142 ),( 46.3009 , 64.0174 )
BOX 0.99:( 384.076 , 56.9844 ),( 76.968 , 83.9609 )
BOX 0.99:( 169.196 , 87.1324 ),( 38.4133 , 46.8504 )
BOX 0.98:( 290.004 , 104.453 ),( 37.6346 , 46.7777 )
```

![](https://z3.ax1x.com/2021/07/01/Rrs6LF.jpg)

## yolact 实例分割任务 - [tm_yolact.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolact.cpp)

使用图片：

![](https://z3.ax1x.com/2021/06/30/RBFpTO.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolact -m models/yolact.tmfile -i images/ssd_car.jpg -r 1 -t 1
```

结果如下：

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 2064.44 ms, max_time 2064.44 ms, min_time 2064.44 ms
--------------------------------------
6 = 0.99966 at 130.82 57.77 340.78 x 237.36
3 = 0.99675 at 323.39 194.97 175.57 x 132.96
1 = 0.33431 at 191.24 195.78 103.06 x 179.22
```

![](https://z3.ax1x.com/2021/07/01/RrEbEq.jpg)

## unet 图像分割任务 - [tm_unet.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_unet.cpp)

使用图片：

![](https://z3.ax1x.com/2021/07/01/Rse0SK.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_unet -m models/unet_sim3.tmfile -i images/carvana01.jpg -r 1 -t 1
```

结果如下：

```bash
Image height not specified, use default 512
Image width not specified, use default  512
Scale value not specified, use default  0.00392, 0.00392, 0.00392
tengine-lite library version: 1.4-dev

model file : models/unet_sim3.tmfile
image file : images/carvana01.jpg
img_h, img_w, scale[3], mean[3] : 512 512 , 0.004 0.004 0.004, 0.0 0.0 0.0
Repeat 1 times, thread 1, avg time 4861.93 ms, max_time 4861.93 ms, min_time 4861.93 ms
--------------------------------------
segmentatation result is save as unet_out.png
```

![](https://z3.ax1x.com/2021/07/01/Rs8YjI.png)

## yolov5s目标检测任务 - [tm_yolov5s.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolov5s.cpp)

使用图片：

![](https://z3.ax1x.com/2021/06/30/RBVdq1.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolov5s -m models/yolov5s.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```
结果如下：

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 462.94 ms, max_time 462.94 ms, min_time 462.94 ms
--------------------------------------
detection num: 3
16:  89%, [ 135,  218,  313,  558], dog
 7:  86%, [ 472,   78,  689,  169], truck
 1:  75%, [ 123,  107,  578,  449], bicycle
```

![](https://z3.ax1x.com/2021/06/30/RBl7Wt.jpg)

## hrnet人体姿态识别任务 - [tm_hrnet.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_hrnet.cpp)

使用图片：

![](https://s1.ax1x.com/2020/09/01/dvJm8A.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_hrnet -m models/hrnet.tmfile -i images/pose.jpg -r 1 -t 1
```

结果如下：

```bash
tengine-lite library version: 1.4-dev
Repeat [1] min 416.223 ms, max 416.223 ms, avg 416.223 ms
x: 27, y: 58, score: 0.91551
x: 27, y: 45, score: 0.865156
x: 28, y: 30, score: 0.831916
x: 34, y: 29, score: 0.839507
x: 38, y: 44, score: 0.88559
x: 35, y: 55, score: 0.891349
x: 31, y: 30, score: 0.873104
x: 31, y: 14, score: 0.928233
x: 30, y: 10, score: 0.948434
x: 29, y: 1, score: 0.915752
x: 23, y: 31, score: 0.811694
x: 24, y: 24, score: 0.935574
x: 24, y: 14, score: 0.899991
x: 37, y: 13, score: 0.908696
x: 41, y: 22, score: 0.902927
x: 41, y: 29, score: 0.847032
```

![](https://z3.ax1x.com/2021/07/01/RrSvg1.jpg)


## 汉字识别任务 - [tm_crnn.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_crnn.cpp)

模型文件：`crnn_lite_dense.tmfile`
测试图片：`o2_resize.jpg`
字库文件：`keys.txt`
测试图片：

![](https://s1.ax1x.com/2020/10/20/BSlFPS.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_crnn -m models/crnn_lite_dense.tmfile -i images/o2_resize.jpg -l files/keys.txt
```

结果如下：

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 23.30 ms, max_time 23.30 ms, min_time 23.30 ms
--------------------------------------
如何突破自己的颜值上限
--------------------------------------
```

其中ocr的识别结果会直接打印到终端中, 同时如果需要保存为txt文件可以修改源码使其重定向到文件。

我们将持续更新各种有趣的 demo ，敬请期待......
