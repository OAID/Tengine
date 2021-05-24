# examples

[**中文版本**](README.md)

Tengine Lite's examples providing simple yet fancy demos.

  - [Classification](#classification-task---tm_classificationc)
  - [Facial Landmark Detection](#facial-landmark-detection-task---tm_landmarkcpp)
  - [SSD Object Detection](#ssd-object-detection-task---tm_mobilenet_ssdcpp)
  - [RetinaFace Face Detection](#retinaface-face-detection-task---tm_refinafacecpp)
  - [Yolact Instance Segmentation](#yolact-instance-segmentation-task---tm_yolactcpp)
  - [YoloV3 Object Detection Task](#yolov3-object-detection-task---tm_yolov3cpp)
  - [Yolov4-tiny Object Detection Task](#yolov4-tiny-object-detection-task---tm_yolov4_tinycpp)
  - [Human Pose Estimation Task](#human-pose-estimation-task---tm_openposecpp)
  - [Chinese character recognition](#chinese-character-recognition-task---tm_crnncpp)
  
----------
## Classification task - [tm_classification.c](tm_classification.c)

Tengine Lite is compatible with original Tengine's C API. Here we demonstrate how to run MobileNet v1 via tm_classification example code, providing image classification functionality. This would help you get involve with Tengine Lite C API. We use the popular tiger cat image for test.

![lu mao](https://github.com/OAID/Tengine/blob/master/tests/images/cat.jpg)

### Model Zoo

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

### Running Result

Put testing images and models under root folder of Tengine-Lite project, and run it:

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_classification -m models/mobilenet.tmfile -i images/cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679
```

output:

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

## Facial Landmark Detection Task - [tm_landmark.cpp](tm_landmark.cpp)

We use this image:

![](https://github.com/OAID/Tengine/blob/master/tests/images/mobileface02.jpg)

Run it with:
```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_landmark -m models/landmark.tmfile -i images/mobileface02.jpg -r 1 -t 1
```

output:

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat [1] min 17.461 ms, max 17.461 ms, avg 17.461 ms
```

![](https://s1.ax1x.com/2020/08/28/doZQxO.jpg)

## SSD Object Detection Task - [tm_mobilenet_ssd.cpp](tm_mobilenet_ssd.cpp)

We use this image:

![](https://github.com/OAID/Tengine/blob/master/tests/images/ssd_dog.jpg)

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_mobilenet_ssd -m models/mobilenet_ssd.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```

output:

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

## RetinaFace Face Detection Task - [tm_refinaface.cpp](tm_refinaface.cpp)

We use this image:

![](https://github.com/OAID/Tengine/blob/master/tests/images/mtcnn_face4.jpg)

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_retinaface -m models/retinaface.tmfile -i images/mtcnn_face4.jpg -r 1 -t 1
```

output：

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

## Yolact Instance Segmentation Task - [tm_yolact.cpp](tm_yolact.cpp)

We use this image:

![](https://github.com/OAID/Tengine/blob/master/tests/images/ssd_car.jpg)

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_yolact -m models/yolact.tmfile -i images/ssd_car.jpg -r 1 -t 1
```

output:

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

## YoloV3 Object Detection Task - [tm_yolov3.cpp](tm_yolov3.cpp)

We use this image:

![](https://github.com/OAID/Tengine/blob/master/tests/images/ssd_dog.jpg)

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_yolact -m models/yolov3_tiny.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```

output：

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

## Yolov4-tiny Object Detection Task - [tm_yolov4_tiny.cpp](tm_yolov4_tiny.cpp)

We use this image:

![](https://github.com/OAID/Tengine/blob/master/tests/images/ssd_dog.jpg)

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
$ ./build/install/bin/tm_yolov4_tiny -m models/yolov4_tiny.tmfile -i images/ssd_dog.jpg -r 1 -t 1
```
output：

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat 1 times, thread 1, avg time 177.72 ms, max_time 177.72 ms, min_time 177.72 ms
--------------------------------------
num_detections,10
16: 74%
left = 125,right = 327,top = 221,bot = 537
2: 40%
7: 84%
left = 455,right = 703,top = 77,bot = 168
1: 28%
left = 56,right = 603,top = 85,bot = 496
```

![](https://s1.ax1x.com/2020/10/19/0zpvfU.jpg)

## Human Pose Estimation Task - [tm_openpose.cpp](tm_openpose.cpp)

We use this image:

![](https://s1.ax1x.com/2020/09/01/dvJm8A.jpg)

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_openpose -m models/openpose_coco.tmfile -i image/pose.jpg -r 1 -t 1
```

output:

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

The result of human pose estimation will be saved as images, whose names are: `Output-Keypionts.jpg` and `Output-Skeleton.jpg`.

## Chinese character recognition task - [tm_crnn.cpp](tm_crnn.cpp)

model file:`crnn_lite_dense.tmfile`
image file:`o2_resize.jpg`
font file:`keys.txt`

![](https://s1.ax1x.com/2020/10/20/BSlFPS.jpg)

```bash
export LD_LIBRARY_PATH=./build/install/lib
./build/install/bin/tm_crnn -m model/crnn_lite_dense.tmfile -i model/o2_resize.jpg -l model/keys.txt
```

result:

```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev
Repeat 1 times, thread 1, avg time 43.32 ms, max_time 43.32 ms, min_time 43.32 ms
--------------------------------------
如何突破自己的颜值上限
--------------------------------------
```

The result of ocr recognition is displayed in terminal, you may also modify the source code to save it to file.

We will continously updating more fancy demos, please stay tuned...
