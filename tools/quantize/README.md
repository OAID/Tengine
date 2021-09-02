# Tengine Post Training Quantization Tools

To support int8 model deployment on AIoT devices, we provide some universal post training quantization tools which can convert the **Float32** tmfile model to **Int8**/**UInt8** tmfile model.

## 1 Compile

### 1.1 Install dependent libraries

```
sudo apt install libopencv-dev
```

### 1.2 Compile from source file

```
git clone https://github.com/OAID/Tengine.git  tengine-lite
cd tengine-lite
mkdir build 
cd build
cmake -DTENGINE_BUILD_QUANT_TOOL=ON ..
make && make install
```

Those quantization tools should be in `./install/bin/` directory

```
$ tree install/bin/
install/bin/
├── quant_tool_int8
├── quant_tool_uint8
├── ......
```

## 2 Symmetric per-channel quantization tool

| Type                  | Note                                                         |
| --------------------- | ------------------------------------------------------------ |
| Adaptive              | TENGINE_MODE_INT8                                            |
| Activation data       | Int8                                                         |
| Weight date           | Int8                                                         |
| Bias date             | Int32                                                        |
| Example               | [**tm_classification_int8.c**](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_int8.c) |
| Execution environment | Ubuntu 18.04                                                 |

### 2.1 Description params

```
$ ./quant_tool_int8 -h
---- Tengine Post Training Quantization Tool ----

Version     : v1.2, 15:20:21 Jul 25 2021
Status      : int8, per-channel, symmetric
[Quant Tools Info]: The input file of Float32 tmfile file not specified!
[Quant Tools Info]: optional arguments:
        -h    help            show this help message and exit
        -m    input model     path to input float32 tmfile
        -i    image dir       path to calibration images folder
        -f    scale file      path to calibration scale file
        -o    output model    path to output int8 tmfile
        -a    algorithm       the type of quant algorithm(0:min-max, 1:kl, 2:aciq, default is 0)
        -g    size            the size of input image(using the resize the original image,default is 3,224,224)
        -w    mean            value of mean (mean value, default is 104.0,117.0,123.0)
        -s    scale           value of normalize (scale value, default is 1.0,1.0,1.0)
        -b    swapRB          flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)
        -c    center crop     flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)
        -y    letter box      the size of letter box process image is necessary([rows, cols], default is [0, 0])
        -k    focus           flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)
        -t    num thread      count of processing threads(default is 1)

[Quant Tools Info]: example arguments:
        ./quant_tool_int8 -m ./mobilenet_fp32.tmfile -i ./dataset -o ./mobilenet_int8.tmfile -g 3,224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017
```

### 2.2 Demo

Before use the quant tool, **you need Float32 tmfile and Calibration Dataset**, the image num of calibration dataset we suggest to use 500-1000.

```
$ .quant_tool_int8  -m ./mobilenet_fp32.tmfile -i ./dataset -o ./mobilenet_int8.tmfile -g 3,224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017 -z 1

---- Tengine Post Training Quantization Tool ----

Version     : v1.1, 15:46:24 Mar 14 2021
Status      : int8, per-channel, symmetric
Input model : ./mobilenet_fp32.tmfile
Output model: ./mobilenet_int8.tmfile
Calib images: ./dataset
Algorithm   : KL
Dims        : 3 224 224
Mean        : 104.007 116.669 122.679
Scale       : 0.017 0.017 0.017
BGR2RGB     : ON
Center crop : OFF
Letter box  : OFF
Thread num  : 1

[Quant Tools Info]: Step 0, load FP32 tmfile.
[Quant Tools Info]: Step 0, load FP32 tmfile done.
[Quant Tools Info]: Step 0, load calibration image files.
[Quant Tools Info]: Step 0, load calibration image files done, image num is 55.
[Quant Tools Info]: Step 1, find original calibration table.
[Quant Tools Info]: Step 1, find original calibration table done, output ./table_minmax.scale
[Quant Tools Info]: Step 2, find calibration table.
[Quant Tools Info]: Step 2, find calibration table done, output ./table_kl.scale
[Quant Tools Info]: Thread 1, image nums 55, total time 1964.24 ms, avg time 35.71 ms
[Quant Tools Info]: Calibration file is using table_kl.scale
[Quant Tools Info]: Step 3, load FP32 tmfile once again
[Quant Tools Info]: Step 3, load FP32 tmfile once again done.
[Quant Tools Info]: Step 3, load calibration table file table_kl.scale.
[Quant Tools Info]: Step 4, optimize the calibration table.
[Quant Tools Info]: Step 4, quantize activation tensor done.
[Quant Tools Info]: Step 5, quantize weight tensor done.
[Quant Tools Info]: Step 6, save Int8 tmfile done, ./mobilenet_int8.tmfile
[Quant Tools Info]: Step Evaluate, evaluate quantitative losses
cosin   0    32  avg  0.995317  ### 0.000000 0.953895 0.998249 0.969256 ...
cosin   1    32  avg  0.982403  ### 0.000000 0.902383 0.964436 0.873998 ...
cosin   2    64  avg  0.976753  ### 0.952854 0.932301 0.982766 0.958503 ...
cosin   3    64  avg  0.981889  ### 0.976637 0.981754 0.987276 0.970671 ...
cosin   4   128  avg  0.979728  ### 0.993999 0.991858 0.990438 0.992766 ...
cosin   5   128  avg  0.970351  ### 0.772556 0.989541 0.986996 0.989563 ...
cosin   6   128  avg  0.954545  ### 0.950125 0.922964 0.946804 0.972852 ...
cosin   7   128  avg  0.977192  ### 0.994728 0.972071 0.995353 0.992700 ...
cosin   8   256  avg  0.977426  ### 0.968429 0.991248 0.991274 0.994450 ...
cosin   9   256  avg  0.962224  ### 0.985255 0.969171 0.958762 0.967461 ...
cosin  10   256  avg  0.954253  ### 0.984353 0.935643 0.656188 0.929778 ...
cosin  11   256  avg  0.971987  ### 0.997596 0.967681 0.476525 0.999115 ...
cosin  12   512  avg  0.972861  ### 0.968920 0.905907 0.993918 0.622953 ...
cosin  13   512  avg  0.959161  ### 0.935686 0.000000 0.642560 0.994388 ...
cosin  14   512  avg  0.963903  ### 0.979613 0.957169 0.976440 0.902512 ...
cosin  15   512  avg  0.963226  ### 0.977065 0.965819 0.998149 0.905297 ...
cosin  16   512  avg  0.960935  ### 0.861674 0.972926 0.950579 0.987609 ...
cosin  17   512  avg  0.961057  ### 0.738472 0.987884 0.999124 0.995397 ...
cosin  18   512  avg  0.960127  ### 0.935455 0.968909 0.970831 0.981240 ...
cosin  19   512  avg  0.963755  ### 0.972628 0.992305 0.999518 0.799737 ...
cosin  20   512  avg  0.949364  ### 0.922776 0.896038 0.945079 0.971338 ...
cosin  21   512  avg  0.961256  ### 0.902256 0.896438 0.923361 0.973974 ...
cosin  22   512  avg  0.946552  ### 0.963806 0.982075 0.878965 0.929992 ...
cosin  23   512  avg  0.953677  ### 0.953880 0.996364 0.936540 0.930796 ...
cosin  24  1024  avg  0.941197  ### 0.000000 0.992507 1.000000 0.994460 ...
cosin  25  1024  avg  0.973546  ### 1.000000 0.889181 0.000000 0.998084 ...
cosin  26  1024  avg  0.869351  ### 0.522966 0.000000 0.987009 0.000000 ...
cosin  27     1  avg  0.974982  ### 0.974982 
cosin  28     1  avg  0.974982  ### 0.974982 
cosin  29     1  avg  0.974982  ### 0.974982 
cosin  30     1  avg  0.978486  ### 0.978486 

---- Tengine Int8 tmfile create success, best wish for your INT8 inference has a low accuracy loss...\(^0^)/ ----
```

## 3 Asymmetric per-layer quantization tool

| Type                  | Note                                                         |
| --------------------- | ------------------------------------------------------------ |
| Adaptive              | TENGINE_MODE_UINT8                                           |
| Activation data       | UInt8                                                        |
| Weight date           | UInt8                                                        |
| Bias date             | Int32                                                        |
| Example               | [**tm_classification_uint8.c**](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_uint8.c) |
| Execution environment | Ubuntu 18.04                                                 |

### 3.1 Description params

```
$ ./quant_tool_uint8 -h
---- Tengine Post Training Quantization Tool ----

Version     : v1.2, 15:20:08 Jul 25 2021
Status      : uint8, per-layer, asymmetric
[Quant Tools Info]: The input file of Float32 tmfile file not specified!
[Quant Tools Info]: optional arguments:
        -h    help            show this help message and exit
        -m    input model     path to input float32 tmfile
        -i    image dir       path to calibration images folder
        -f    scale file      path to calibration scale file
        -o    output model    path to output uint8 tmfile
        -a    algorithm       the type of quant algorithm(0:min-max, 1:kl, 2:aciq, default is 0)
        -g    size            the size of input image(using the resize the original image,default is 3,224,224)
        -w    mean            value of mean (mean value, default is 104.0,117.0,123.0)
        -s    scale           value of normalize (scale value, default is 1.0,1.0,1.0)
        -b    swapRB          flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)
        -c    center crop     flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)
        -y    letter box      the size of letter box process image is necessary([rows, cols], default is [0, 0])
        -k    focus           flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)
        -t    num thread      count of processing threads(default is 1)

[Quant Tools Info]: example arguments:
        ./quant_tool_uint8 -m ./mobilenet_fp32.tmfile -i ./dataset -o ./mobilenet_uint8.tmfile -g 3,224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017
```

### 3.2 Demo

Before use the quant tool, **you need Float32 tmfile and Calibration Dataset**, the image num of calibration dataset we suggest to use 500-1000.

```
$ .quant_tool_uint8  -m ./mobilenet_fp32.tmfile -i ./dataset -o ./mobilenet_uint8.tmfile -g 3,224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017

---- Tengine Post Training Quantization Tool ----

Version     : v1.2, 18:32:53 May 30 2021
Status      : uint8, per-layer, asymmetric
Input model : ./mobilenet_fp32.tmfile
Output model: ./mobilenet_uint8.tmfile
Calib images: ./dataset
Scale file  : NULL
Algorithm   : MIN MAX
Dims        : 3 224 224
Mean        : 104.000 117.000 123.000
Scale       : 0.017 0.017 0.017
BGR2RGB     : ON
Center crop : OFF
Letter box  : 0 0
YOLOv5 focus: OFF
Thread num  : 4

[Quant Tools Info]: Step 0, load FP32 tmfile.
[Quant Tools Info]: Step 0, load FP32 tmfile done.
[Quant Tools Info]: Step 0, load calibration image files.
[Quant Tools Info]: Step 0, load calibration image files done, image num is 5.
[Quant Tools Info]: Step 1, find original calibration table.
[Quant Tools Info]: Step 1, images 00005 / 00005
[Quant Tools Info]: Step 1, find original calibration table done, output ./table_minmax.scale
[Quant Tools Info]: Thread 4, image nums 5, total time 37.23 ms, avg time 87.45 ms
[Quant Tools Info]: Calibration file is using table_minmax.scale
[Quant Tools Info]: Step 3, load FP32 tmfile once again
[Quant Tools Info]: Step 3, load FP32 tmfile once again done.
[Quant Tools Info]: Step 3, load calibration table file table_minmax.scale.
[Quant Tools Info]: Step 4, optimize the calibration table.
[Quant Tools Info]: Step 4, quantize activation tensor done.
[Quant Tools Info]: Step 5, quantize weight tensor done.
[Quant Tools Info]: Step 6, save Int8 tmfile done, mobilenet_uint8.tmfile

---- Tengine Int8 tmfile create success, best wish for your INT8 inference has a low accuracy loss...\(^0^)/ ----
```
