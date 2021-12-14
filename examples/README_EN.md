# examples

[**中文版本**](README.md)

Tengine Lite's examples providing simple yet fancy demos.

  - [Classification](#classification-task---tm_classificationc)
  - [Facial Landmark Detection](#facial-landmark-detection-task---tm_landmarkcpp)
  - [SSD Object Detection](#ssd-object-detection-task---tm_mobilenet_ssdc)
  - [RetinaFace Face Detection](#retinaface-face-detection-task---tm_refinafacecpp)
  - [Scrfd Face Detection](#scrfd-face-detection-task---tm_scrfdcpp)
  - [Yolact Instance Segmentation](#yolact-instance-segmentation-task---tm_yolactcpp)
  - [U-Net Image Segmentation](#u-net-image-segmentation-task---tm_unetcpp)
  - [YoloV3 Object Detection Task](#yolov3-object-detection-task---tm_yolov3cpp)
  - [YoloV4-tiny Object Detection Task](#yolov4-tiny-object-detection-task---tm_yolov4_tinycpp)
  - [YoloV5s Object Detection Task](#yolov5s-object-detection-task---tm_yolov5scpp)
  - [NanoDet Object Detection Task](#nanodet-object-detection-task---tm_nanodet_mcpp)
  - [EfficientDet Object Detection Task](#efficientdet-object-detection-task---tm_efficientdetc)
  - [Yolox Object Detection Task](#yolox-object-detection-task---tm_yoloxcpp)
  - [OpenPose Human Pose Estimation Task](#openpose-human-pose-estimation-task---tm_openposecpp)
  - [HRNet Human Pose Estimation Task](#hrnet-human-pose-estimation-task---tm_hrnetcpp)
  - [CRNN Chinese character recognition](#chinese-character-recognition-task---tm_crnncpp)
  - [PaddleSeg Human Segmentation](#human-segmentation-task---tm_seghumancpp)

In addition to single-image single-model-inference example, Tengine Lite gives pipeline application based on video and image stream.

  - [Distance Estimation](#Distance-Estimation)
  - [Facial Feature Extraction](#Facial-Feat-Extraction)

----------
## Classification task - [tm_classification.c](tm_classification.c)

Tengine Lite is compatible with original Tengine's C API. Here we demonstrate how to run MobileNet v1 via tm_classification example code, providing image classification functionality. This would help you get involve with Tengine Lite C API. We use the popular tiger cat image for test.

![](https://z3.ax1x.com/2021/06/30/RBIQIO.jpg)

### Model Zoo
The model zoo contains the models, images, and files needed to run examples

- [Baidu Netdisk](https://pan.baidu.com/s/1JsitkY6FVV87Kao6h5yAmg) (password: 7ke5)

- [Google Drive](https://drive.google.com/drive/folders/1hunePCa0x_R-Txv7kWqgx02uTCH3QWdS?usp=sharing)


### Reference Code

[tm_classification.c](tm_classification.c)

### Compilation

build.sh compiles example folders demo programs on default. Take x86 platform as example, the compilation generated demos are stored in `./build/install/bin/` folder.

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
│   ├── tm_scrfd
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
│       └── c_api.h
└── lib
    ├── libtengine-lite-static.a
    └── libtengine-lite.so
```

### Running Result

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

## Facial Landmark Detection Task - [tm_landmark.cpp](tm_landmark.cpp)

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

## SSD Object Detection Task - [tm_mobilenet_ssd.c](tm_mobilenet_ssd.c)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBVdq1.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_mobilenet_ssd -m models/mobilenet_ssd.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```

output:

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 78.89 ms, max_time 78.89 ms, min_time 78.89 ms
--------------------------------------
detect result num: 3 
dog     :99.8%
BOX:( 138 , 209 ),( 324 , 541 )
car     :99.7%
BOX:( 467 , 72 ),( 687 , 171 )
bicycle :99.5%
BOX:( 107 , 141 ),( 574 , 415 )
======================================
[DETECTED IMAGE SAVED]:
======================================
```

![](https://z3.ax1x.com/2021/07/01/RrPnDx.jpg)

## RetinaFace Face Detection Task - [tm_refinaface.cpp](tm_refinaface.cpp)

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


## Scrfd Face Detection Task - [tm_scrfd.cpp](tm_scrfd.cpp)

We use this image:

![](https://z3.ax1x.com/2021/11/25/oAaGVS.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_scrfd -m models/scrfd_2.5g_kps.tmfile -i images/face5.jpg -r 1 -t 1
```

output：


```bash
tengine-lite library version: 1.5-dev
Repeat 1 times, thread 1, avg time 289.97 ms, max_time 289.97 ms, min_time 289.97 ms
--------------------------------------
detection num: 5
0.90917 at 199.37 54.92 28.52 x 38.12
0.89985 at 70.50 29.96 32.26 x 41.25
0.88838 at 111.36 48.00 33.53 x 46.77
0.88484 at 247.54 51.15 30.21 x 37.29
0.83953 at 149.23 49.48 27.89 x 38.50


```

![](https://z3.ax1x.com/2021/11/25/oAUxN4.jpg)

## Yolact Instance Segmentation Task - [tm_yolact.cpp](tm_yolact.cpp)

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

## U-Net Image Segmentation Task - [tm_unet.cpp](tm_unet.cpp)

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
segmentation result is save as unet_out.png
```

![](https://z3.ax1x.com/2021/07/01/Rs8YjI.png)

## YoloV3 Object Detection Task - [tm_yolov3.cpp](tm_yolov3.cpp)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBVdq1.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolov3 -m models/yolov3.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```

output：

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 1131.67 ms, max_time 1131.67 ms, min_time 1131.67 ms
--------------------------------------
detection num: 3
16: 100%, [ 123,  223,  320,  544], dog
 1:  99%, [ 160,  117,  568,  435], bicycle
 7:  94%, [ 473,   87,  693,  166], truck
```

![](https://z3.ax1x.com/2021/06/30/RBJSBT.jpg)

## YoloV4-tiny Object Detection Task - [tm_yolov4_tiny.cpp](tm_yolov4_tiny.cpp)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBVdq1.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolov4_tiny -m models/yolov4-tiny.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```
output：

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 152.50 ms, max_time 152.50 ms, min_time 152.50 ms
--------------------------------------
detection num: 3
16:  87%, [ 136,  206,  318,  542], dog
 7:  81%, [ 463,   79,  703,  170], truck
 1:  61%, [  72,  100,  577,  479], bicycle
```
![](https://z3.ax1x.com/2021/06/30/RBKqQU.jpg)

## YoloV5s Object Detection Task - [tm_yolov5s.cpp](tm_yolov5s.cpp)

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

## NanoDet Object Detection Task - [tm_nanodet_m.cpp](tm_nanodet_m.cpp)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBVdq1.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_nanodet_m -m models/nanodet.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```
output：

```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 35.96 ms, max_time 35.96 ms, min_time 35.96 ms
--------------------------------------
detection num: 3
 1: 59.313%, [141.945, 160.890, 563.568, 429.829], bicycle
16: 50.605%, [132.646, 205.861, 312.255, 511.470], dog
 2: 48.931%, [462.477,  72.462, 701.777, 170.343], car
```

![](https://z3.ax1x.com/2021/07/01/RsVkff.jpg)

## EfficientDet Object Detection Task - [tm_efficientdet.c](tm_efficientdet.c)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBVdq1.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_efficientdet -m ../models/efficientdet.tmfile -i ../images/ssd_dog.jpg -r 1 -t 1
```
output：

```bash
tengine-lite library version: 1.4-dev
model file : ../models/efficientdet.tmfile
image file : ../images/ssd_dog.jpg
img_h, img_w, scale[3], mean[3] : 512 512 , 0.017 0.018 0.017, 123.7 116.3 103.5
Repeat 1 times, thread 1, avg time 598.86 ms, max_time 598.86 ms, min_time 598.86 ms
--------------------------------------
17:  80%, [ 132,  222,  315,  535], dog
 7:  73%, [ 467,   74,  694,  169], truck
 1:  42%, [ 103,  119,  555,  380], bicycle
 2:  29%, [ 687,  113,  724,  156], car
 2:  25%, [  57,   77,  111,  124], car
```

![](https://z3.ax1x.com/2021/07/08/RqxsmR.jpg)

## Yolox Object Detection Task - [tm_yolox.cpp](tm_yolox.cpp)

We use this image:

![](https://z3.ax1x.com/2021/06/30/RBVdq1.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolox -m ../models/yolox_nano.tmfile -i ../images/ssd_dog.jpg -r 1 -t 1
```
output：

```bash
tengine-lite library version: 1.5-dev
Repeat 1 times, thread 1, avg time 97.64 ms, max_time 97.64 ms, min_time 97.64 ms
--------------------------------------
detection num: 3
16:  85%, [ 132,  216,  318,  545], dog
 1:  83%, [ 112,  140,  568,  427], bicycle
 2:  69%, [ 466,   77,  693,  168], car

```

![](https://z3.ax1x.com/2021/11/19/IHwcJ1.jpg)



## OpenPose Human Pose Estimation Task - [tm_openpose.cpp](tm_openpose.cpp)

We use this image:

![](https://s1.ax1x.com/2020/09/01/dvJm8A.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_openpose -m models/openpose_coco.tmfile -i images/pose.jpg -r 1 -t 1
```

output:


```bash
tengine-lite library version: 1.4-dev
Repeat 1 times, thread 1, avg time 7296.71 ms, max_time 7296.71 ms, min_time 7296.71 ms
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

![](https://z3.ax1x.com/2021/06/30/RBdWa6.jpg)
![](https://z3.ax1x.com/2021/06/30/RBdfIK.jpg)

The result of human pose estimation will be saved as images, whose names are: `Output-Keypionts.jpg` and `Output-Skeleton.jpg`.

## HRNet Human Pose Estimation Task - [tm_hrnet.cpp](tm_hrnet.cpp)

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


## Chinese character recognition task - [tm_crnn.cpp](tm_crnn.cpp)

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


## Human segmentation task - [tm_seghuman.cpp](tm_seghuman.cpp)

model file：`paddleSegSim.tmfile`

image file：`human_image.jpg`

image file：

![](https://s1.ax1x.com/2021/12/09/offIJK.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_seghuman -m models/paddleSegSim.tmfile -i images/human_image.jpeg
```

result：

```bash
tengine-lite library version: 1.5-dev
Repeat 1 times, avg time 123.766 ms, max_time 123.766 ms, min_time 123.766 ms
```

![](https://s1.ax1x.com/2021/12/09/of4XPP.jpg)

segmentation result image is saved as seg_human_result.jpg.


## Distance Estimation

model file:`mobilenet_ssd.tmfile`

run (GPU recommended)
```bash
$ cd build/examples
$ ln -s models/mobilenet_ssd.tmfile
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./tm_pipeline_estimate_ped_distance
detect result num: 1 
person	:100.0%
BOX:( 35 , 78 ),( 587 , 478 )
...
```

## Facial Feature Extraction

model list:
* `rfb-320.tmfile`  face detection
* `landmark.tmfile`  face landmark
* `mobilefacenet.tmfile`  face feature

```bash
$ cd build/examples
$ ln -s models/rfb-320.tmfile
$ ln -s models/landmark.tmfile
$ ln -s models/mobilefacenet.tmfile
```

run (GPU recommanded):
```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./tm_pipeline_enroll_face  ./images
```

face feature would serialized to `feature0.bin`


We will continously updating more fancy demos, please stay tuned...
