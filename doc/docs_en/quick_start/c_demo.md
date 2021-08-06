# Examples display

All examples shown in this section are located at [examples](https://github.com/OAID/Tengine/tree/tengine-lite/examples) 。

## Preparation
### Environmental preparation
To compile and run the c sample program, you need to prepare: 

1. a computer that can compile C/C++ Linux environment (x86 or Arm architecture can be used).
2. Download Tengine Lite source code，which is located on the branch of Tengine-lite：
```bash
git clone -b tengine-lite https://github.com/OAID/Tengine.git  Tengine
```


### Compilation
build.sh compiles example folders demo programs on default.

**Taking x86 architecture as an example, the compilation generated demos are stored in `./build/install/bin/` folder：**

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
│       └── c_api.h                     C Forecast Library Header File
└── lib
    ├── libtengine-lite-static.a        Static Forecast Library
    └── libtengine-lite.so              Dynamic Forecast Library
```

### Model Zoo

The model zoo contains the models, images, and files needed to run examples

- [Baidu Netdisk](https://pan.baidu.com/s/1JsitkY6FVV87Kao6h5yAmg) (password: 7ke5)

- [Google Drive](https://drive.google.com/drive/folders/1hunePCa0x_R-Txv7kWqgx02uTCH3QWdS?usp=sharing)

## Classification task - [tm_classification.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification.c)

Tengine Lite is compatible with original Tengine's C API. Here we demonstrate how to run MobileNet v1 via tm_classification example code, providing image classification functionality. This would help you get involve with Tengine Lite C API. We use the popular tiger cat image for test.

![](https://z3.ax1x.com/2021/06/30/RBIQIO.jpg)

Put testing images and models under root folder of Tengine-Lite project, and run it:

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_classification -m models/mobilenet.tmfile -i images/cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679
```

output:

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

## Facial Landmark Detection Task - [tm_landmark.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_landmark.cpp)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RB5dC4.jpg)

Run it with:
```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_landmark -m models/landmark.tmfile -i images/mobileface02.jpg -r 1 -t 1
```

output:

```bash
tengine-lite library version: 1.4-dev
Repeat [1] min 8.784 ms, max 8.784 ms, avg 8.784 ms
```

![](https://z3.ax1x.com/2021/07/01/RrPSuq.jpg)

## RetinaFace Face Detection Task - [tm_retinaface.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_retinaface.cpp)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBC311.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_retinaface -m models/retinaface.tmfile -i images/mtcnn_face4.jpg -r 1 -t 1
```

output：

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

## Yolact Instance Segmentation Task  - [tm_yolact.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolact.cpp)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBFpTO.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolact -m models/yolact.tmfile -i images/ssd_car.jpg -r 1 -t 1
```

output:

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 2064.44 ms, max_time 2064.44 ms, min_time 2064.44 ms
--------------------------------------
6 = 0.99966 at 130.82 57.77 340.78 x 237.36
3 = 0.99675 at 323.39 194.97 175.57 x 132.96
1 = 0.33431 at 191.24 195.78 103.06 x 179.22
```

![](https://z3.ax1x.com/2021/07/01/RrEbEq.jpg)

## U-Net Image Segmentation Task - [tm_unet.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_unet.cpp)

We use this image:

![](https://z3.ax1x.com/2021/07/01/Rse0SK.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_unet -m models/unet_sim3.tmfile -i images/carvana01.jpg -r 1 -t 1
```

output:

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

## YoloV5s Object Detection Task - [tm_yolov5s.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolov5s.cpp)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBVdq1.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolov5s -m models/yolov5s.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```
output：

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

## HRNet Human Pose Estimation Task - [tm_hrnet.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_hrnet.cpp)

We use this image:

![](https://s1.ax1x.com/2020/09/01/dvJm8A.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_hrnet -m models/hrnet.tmfile -i images/pose.jpg -r 1 -t 1
```

output:

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


## Chinese character recognition task - [tm_crnn.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_crnn.cpp)

model file:`crnn_lite_dense.tmfile`
image file:`o2_resize.jpg`
font file:`keys.txt`

![](https://s1.ax1x.com/2020/10/20/BSlFPS.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_crnn -m models/crnn_lite_dense.tmfile -i images/o2_resize.jpg -l files/keys.txt
```

result:

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 23.30 ms, max_time 23.30 ms, min_time 23.30 ms
--------------------------------------
如何突破自己的颜值上限
--------------------------------------
```

The result of ocr recognition is displayed in terminal, you may also modify the source code to save it to file.

We will continously updating more fancy demos, please stay tuned...

