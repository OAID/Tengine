## examples

Tengine Lite 的 examples 将提供简单的、好玩的 demo。

Tengine Lite 兼容 Tengine 原有的 C API 供用户使用，这里我们使用 C API 展示如何运行 tm_classification 例程运行 MobileNet v1 分类网络模型，实现指定图片分类的功能。让你快速上手Tengine Lite C API。这里，我们使用在这个撸猫时代行业从业者大爱的 tiger cat 作为测试图片。

![lu mao](https://github.com/OAID/Tengine/blob/master/tests/images/cat.jpg)

### 源码参考

[tm_classification.c](example/tm_classificaton.c)

### 编译

build.sh 编译脚本默认配置已实现自动编译 examples 中的 demo 程序，以 x86 平台为例，demo 存放在 ./build/install/bin/ 目录下。

```bash
bug1989@DESKTOP-SGN0H2A:/mnt/d/ubuntu/gitlab/build-linux$ tree install
install
├── bin
│   ├── tm_benchmark
│   ├── tm_classification
│   ├── tm_mobilenet_ssd
│   ├── tm_retinaface
│   └── tm_yolov3_tiny
├── include
│   └── tengine_c_api.h
└── lib
    └── libtengine-lite.so
```

### 运行结果

将测试图片和模型文件放在 Tengine-Lite 根目录下。

- 图像分类

运行：

```bash
$ ./build/examples/tm_classification -m models/mobilenet.tmfile -i images/cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679
```

结果如下：

```bash
model file : models/mobilenet.tmfile
image file : images/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 89.72 ms, max_time 89.72 ms, min_time 89.72 ms
--------------------------------------
8.574146, 282
7.880116, 277
7.812573, 278
7.286457, 263
6.357488, 281
--------------------------------------
```

- 目标检测

运行：

```bash
$ ./build/examples/tm_mobilenet_ssd -m models/mobilenet_ssd.tmfile -i images/cat.jpg
```

结果如下：

```bash
Repeat 1 times, thread 1, avg time 149.96 ms, max_time 149.96 ms, min_time 149.96 ms
--------------------------------------
detect result num: 1
cat	:100.0%
BOX:( 171 , 27 ),( 345 , 356 )
======================================
[DETECTED IMAGE SAVED]:
======================================
```
目标检测结果会保存为图片，名称为：`tengine_example_out.jpg`。

- 人脸检测

运行：

```bash
$ ./build/examples/tm_retinaface -m models/retinaface.tmfile -i images/mobileface01.jpg
```

结果如下：

```bash
img_h, img_w : 112, 112
Repeat 1 times, thread 1, avg time 30.81 ms, max_time 30.81 ms, min_time 30.81 ms
--------------------------------------
detected face num: 1
BOX 0.93:( 19.7433 , 15.3631 ),( 74.7011 , 95.6369 )
```

人脸检测结果会保存为图片，名称为：`tengine_example_out.jpg`。

- 人脸特征点检测

运行：

```bash
$ ./build/examples/tm_landmark -m models/landmark.tmfile -i images/mobileface01.jpg
```

结果如下：

```bash
Repeat [1] min 29.110 ms, max 29.110 ms, avg 29.110 ms
```

人脸特征点检测结果会保存为图片，名称为：`tengine_example_out.jpg`。

我们将持续更新各种有趣的 demo ，敬请期待......
