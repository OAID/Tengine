# examples

[**English Version**](README_EN.md)

Tengine Lite 的 examples 将提供简单的、好玩的 demo。
- [分类任务](#分类任务---tm_classificationc)
- [人脸关键点检测任务](#人脸关键点检测任务---tm_landmarkcpp)
- [ssd 目标检测任务](#ssd-目标检测任务---tm_mobilenet_ssdcpp)
- [retinaface 人脸检测任务](#retinaface-人脸检测任务---tm_refinafacecpp)
- [yolact 实例分割任务](#yolact-实例分割任务---tm_yolactcpp)
- [yolov3 目标检测任务](#yolov3-目标检测任务---tm_yolov3cpp)
- [yolov4-tiny目标检测任务](#yolov4-tiny目标检测任务---tm_yolov4_tinycpp)
- [openpose人体姿态识别任务](#人体姿态识别任务---tm_openposecpp)
- [crnn汉字识别任务](#汉字识别任务---tm_crnncpp)
  
----------
## 分类任务 - [tm_classification.c](tm_classification.c)

Tengine Lite 兼容 Tengine 原有的 C API 供用户使用，这里我们使用 C API 展示如何运行 tm_classification 例程运行 MobileNet v1 分类网络模型，实现指定图片分类的功能。让你快速上手 Tengine Lite C API。这里，我们使用在这个撸猫时代行业从业者大爱的 tiger cat 作为测试图片。

![lu mao](https://github.com/OAID/Tengine/blob/master/tests/images/cat.jpg)

### 模型仓库

- [百度网盘](https://pan.baidu.com/s/1JsitkY6FVV87Kao6h5yAmg) （提取码：7ke5）

- [Google Drive](https://drive.google.com/drive/folders/1hunePCa0x_R-Txv7kWqgx02uTCH3QWdS?usp=sharing)

### 源码参考

[tm_classification.c](tm_classification.c)

### 编译

build.sh 编译脚本默认配置已实现自动编译 examples 中的 demo 程序，以 x86 平台为例，demo 存放在 ./build/install/bin/ 目录下。

```bash
bug1989@DESKTOP-SGN0H2A:/mnt/d/ubuntu/gitlab/build-linux$ tree install
install
├── bin
│   ├── cpp_tm_classification
│   ├── cpp_tm_mobilenet_ssd
│   ├── tm_benchmark
│   ├── tm_classification
│   ├── tm_classification_fp16
│   ├── tm_classification_uint8
│   ├── tm_classification_vulkan
│   ├── tm_crnn
│   ├── tm_landmark
│   ├── tm_landmark_uint8
│   ├── tm_mobilefacenet
│   ├── tm_mobilenet_ssd
│   ├── tm_mobilenet_ssd_acl
│   ├── tm_mobilenet_ssd_uint8
│   ├── tm_openpose
│   ├── tm_retinaface
│   ├── tm_yolact
│   ├── tm_yolov3_tiny
│   ├── tm_yolov3_uint8
│   ├── tm_yolov4
│   └── tm_yolov4_tiny
├── include
│   └── tengine_c_api.h
└── lib
    └── libtengine-lite.so
```

### 运行结果

将测试图片和模型文件放在 Tengine-Lite 根目录下，运行：

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_classification -m models/mobilenet.tmfile -i images/cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679
```

结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev

model file : ./temp/models/mobilenet.tmfile
image file : ./temp/images/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 656.76 ms, max_time 656.76 ms, min_time 656.76 ms
--------------------------------------
8.574148, 282
7.880116, 277
7.812579, 278
7.286453, 263
6.357488, 281
--------------------------------------
```

## 人脸关键点检测任务 - [tm_landmark.cpp](tm_landmark.cpp)

使用图片：

![](https://github.com/OAID/Tengine/blob/master/tests/images/mobileface02.jpg)

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_landmark -m models/landmark.tmfile -i images/mobileface02.jpg -r 1 -t 1
```

结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat [1] min 17.461 ms, max 17.461 ms, avg 17.461 ms
```

![](https://s1.ax1x.com/2020/08/28/doZQxO.jpg)

## ssd 目标检测任务 - [tm_mobilenet_ssd.cpp](tm_mobilenet_ssd.cpp)

使用图片：

![](https://github.com/OAID/Tengine/blob/master/tests/images/ssd_dog.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_mobilenet_ssd -m models/mobilenet_ssd.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```

结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat 1 times, thread 1, avg time 206.30 ms, max_time 206.30 ms, min_time 206.30 ms
--------------------------------------
detect result num: 3
dog	:99.8%
BOX:( 138 , 209 ),( 324 , 541 )
car	:99.7%
BOX:( 467 , 72 ),( 687 , 171 )
bicycle	:99.5%
BOX:( 107 , 141 ),( 574 , 415 )
======================================
[DETECTED IMAGE SAVED]:
======================================
```

![](https://s1.ax1x.com/2020/08/28/doeJ6U.jpg)

## retinaface 人脸检测任务 - [tm_refinaface.cpp](tm_refinaface.cpp)

使用图片：

![](https://github.com/OAID/Tengine/blob/master/tests/images/mtcnn_face4.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_retinaface -m models/retinaface.tmfile -i images/mtcnn_face4.jpg -r 1 -t 1
```

结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
img_h, img_w : 316, 474
Repeat 1 times, thread 1, avg time 75.72 ms, max_time 75.72 ms, min_time 75.72 ms
--------------------------------------
detected face num: 4
BOX 0.99:( 38.9179 , 86.3346 ),( 45.7028 , 63.2934 )
BOX 0.99:( 168.12 , 86.14 ),( 37.5249 , 47.7839 )
BOX 0.98:( 383.673 , 56.4136 ),( 77.176 , 83.8093 )
BOX 0.98:( 289.365 , 103.773 ),( 38.0025 , 47.6989 )
```

![](https://s1.ax1x.com/2020/08/28/doeBfx.jpg)

## yolact 实例分割任务 - [tm_yolact.cpp](tm_yolact.cpp)

使用图片：

![](https://github.com/OAID/Tengine/blob/master/tests/images/ssd_car.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolact -m models/yolact.tmfile -i images/ssd_car.jpg -r 1 -t 1
```

结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat 1 times, thread 1, avg time 15833.47 ms, max_time 15833.47 ms, min_time 15833.47 ms
--------------------------------------
6 = 0.99966 at 130.82 57.77 340.78 x 237.36
3 = 0.99675 at 323.39 194.97 175.57 x 132.96
1 = 0.33431 at 191.24 195.78 103.06 x 179.22
```

![](https://s1.ax1x.com/2020/08/28/doe4ht.png)

## yolov3 目标检测任务 - [tm_yolov3.cpp](tm_yolov3.cpp)

使用图片：

![](https://github.com/OAID/Tengine/blob/master/tests/images/ssd_dog.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolov3_tiny -m models/yolov3_tiny.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```

结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat 1 times, thread 1, avg time 262.52 ms, max_time 262.52 ms, min_time 262.52 ms
--------------------------------------
num_detections,4
16: 57%
left = 129,right = 369,top = 186,bot = 516
2: 65%
left = 465,right = 677,top = 74,bot = 171
1: 60%
left = 205,right = 576,top = 153,bot = 447
```

![](https://s1.ax1x.com/2020/08/28/domYCt.jpg)

## yolov4-tiny目标检测任务 - [tm_yolov4_tiny.cpp](tm_yolov4_tiny.cpp)

使用图片：

![](https://github.com/OAID/Tengine/blob/master/tests/images/ssd_dog.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolov4_tiny -m models/yolov4_tiny.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```
结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat 1 times, thread 1, avg time 177.72 ms, max_time 177.72 ms, min_time 177.72 ms
--------------------------------------
16: 75%, [ 126, 223, 326, 536], dog
 7: 84%, [ 455,  77, 703, 168], truck
 1: 28%, [  56,  85, 603, 496], bicycle
```

![](https://s1.ax1x.com/2020/10/19/0zpvfU.jpg)

## 人体姿态识别任务 - [tm_openpose.cpp](tm_openpose.cpp)

使用图片：

![](https://s1.ax1x.com/2020/09/01/dvJm8A.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_openpose -m models/openpose_coco.tmfile -i image/pose.jpg -r 1 -t 1
```

结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat 1 times, thread 1, avg time 15350.25 ms, max_time 15350.25 ms, min_time 15350.25 ms
--------------------------------------
KeyPoints Coordinate:
0:[292.174, 55.6522]
1:[306.087, 125.217]
2:[250.435, 139.13]
3:[236.522, 222.609]
4:[222.609, 306.087]
5:[361.739, 125.217]
6:[403.478, 208.696]
7:[417.391, 292.174]
8:[264.348, 306.087]
9:[264.348, 431.304]
10:[264.348, 570.435]
11:[347.826, 306.087]
12:[375.652, 431.304]
13:[333.913, 542.609]
14:[278.261, 41.7391]
15:[306.087, 41.7391]
16:[264.348, 55.6522]
17:[320, 55.6522]
```

![](https://s1.ax1x.com/2020/09/01/dvJ2x1.jpg)
![](https://s1.ax1x.com/2020/09/01/dvJxZ8.jpg)

人体姿态识别结果会保存为图片，名称为：`Output-Keypionts.jpg`和`Output-Skeleton.jpg`。

## 汉字识别任务 - [tm_crnn.cpp](tm_crnn.cpp)

模型文件：`crnn_lite_dense.tmfile`
测试图片：`o2_resize.jpg`
字库文件：`keys.txt`
测试图片：

![](https://s1.ax1x.com/2020/10/20/BSlFPS.jpg)

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_crnn -m model/crnn_lite_dense.tmfile -i model/o2_resize.jpg -l model/keys.txt
```

结果如下：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat 1 times, thread 1, avg time 43.32 ms, max_time 43.32 ms, min_time 43.32 ms
--------------------------------------
如何突破自己的颜值上限
--------------------------------------
```

其中ocr的识别结果会直接打印到终端中, 同时如果需要保存为txt文件可以修改源码使其重定向到文件。

我们将持续更新各种有趣的 demo ，敬请期待......
