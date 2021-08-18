# Model Quantization-Asymmetric Quantization
To support the deployment of Uint8 model on AIoT equipment, we provide some general post training quantization tools, which can transform Float32 tmfile model into Uint8 tmfile model.

## Asymmetric Layered Quantization

| Type                  | Note                                                         |
| --------------------- | ------------------------------------------------------------ |
| Adaptive              | TENGINE_MODE_UINT8                                           |
| Activation data       | UInt8                                                        |
| Weight date           | UInt8                                                        |
| Bias date             | Int32                                                        |
| Example               | [**tm_classification_uint8.c**](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_uint8.c) |
| Execution environment | Ubuntu 18.04                                                 |

## Adapting Hardware

- CPU Uint8 mode
- TIM-VX NPU（such as A311D、i.MX8M Plus、RK1126？）

## Download

At present, we provide precompiled executable files, which can be obtained from here [quant_tool_uint8](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_uint8).

## Install Dependent Libraries

```
sudo apt install libopencv-dev
```

## Operational Parameter

```bash
$ ./quant_tool_uint8 -h
[Quant Tools Info]: optional arguments:
-h    help            show this help message and exit
-m    input model     path to input float32 tmfile
-i    image dir       path to calibration images folder
-o    output model    path to output int8 tmfile
-a    algorithm       the type of quant algorithm(0:min-max, 1:kl, default is 1)
-g    size            the size of input image(using the resize the original image,default is 3,224,224
-w    mean            value of mean (mean value, default is 104.0,117.0,123.0
-s    scale           value of normalize (scale value, default is 1.0,1.0,1.0)
-b    swapRB          flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)
-c    center crop     flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)
-y    letter box      flag which indicates that letter box process image is necessary(maybe using for YOLO, 0:OFF, 1:ON, default is 0)
-t    num thread      count of processing threads(default is 4)
```

## Example

Before using the quantization tool, **you need Float32 tmfile and Calibration Dataset**。

- Calibrate the data content and cover all the application scenarios of the model as much as possible. Generally, our experience is randomly selected from the training set；
- The number of calibration data sheets is 500-1000 according to experience.

```bash
$ .quant_tool_uint8  -m ./mobilenet_fp32.tmfile -i ./dataset -o ./mobilenet_uint8.tmfile -g 3,224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017

---- Tengine Post Training Quantization Tool ----

Version     : v1.0, 18:06:10 Mar  4 2021
Status      : uint8, per-layer, asymmetric
Input model : ./mobilenet_fp32.tmfile
Output model: ./mobilenet_uint8.tmfile
Calib images: ./dataset
Algorithm   : MIN MAX
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
[Quant Tools Info]: Step 1, images 00055 / 00055
[Quant Tools Info]: Step 1, find original calibration table done, output ./table_minmax.scale
[Quant Tools Info]: Step 2, find calibration table.
[Quant Tools Info]: Step 2, images 00001 / 00055
[Quant Tools Info]: Step 2, find calibration table done, output ./table_kl.scale
[Quant Tools Info]: Thread 1, image nums 55, total time 1195.07 ms, avg time 21.73 ms
[Quant Tools Info]: Calibration file is using table_minmax.scale
[Quant Tools Info]: Step 3, load FP32 tmfile once again
[Quant Tools Info]: Step 3, load FP32 tmfile once again done.
[Quant Tools Info]: Step 3, load calibration table file table_minmax.scale.
[Quant Tools Info]: Step 4, optimize the calibration table.
[Quant Tools Info]: Step 4, quantize activation tensor done.
[Quant Tools Info]: Step 5, quantize weight tensor done.
[Quant Tools Info]: Step 6, save Int8 tmfile done, ./mobilenet_uint8.tmfile

---- Tengine Int8 tmfile create success, best wish for your INT8 inference has a low accuracy loss...\(^0^)/ ----
```