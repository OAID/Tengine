# Tengine YOLOv5s Optimize Tool

To support **YOLOv5s** detection model deployment on AIoT devices, we provide some universal network optimize tools which can optimize the **onnx type** of [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.onnx) to easy convert to tmfile model to deployment on CPU/GPU/**NPU** with **Tengine Framework**.

## 1 The object of this optimization

- Remove the focus nodes of prepare process;
- Remove the YOLO detection nodes of postprocess;
- Fusion the activation HardSwish node replace the Sigmoid and Mul.

## 2 How to use

## 2.1 Install dependent libraries

```
sudo pip install onnx onnx-simplifier
```

## 2.2 Description params

```
$ python3 yolov5s-opt.py -h
usage: yolov5s-opt.py [-h] [--input INPUT] [--output OUTPUT]
                      [--in_tensor IN_TENSOR] [--out_tensor OUT_TENSOR]
                      [--verbose]

YOLOv5 Optimize Tool Parameters

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         input model path
  --output OUTPUT       output model path
  --in_tensor IN_TENSOR
                        input tensor name
  --out_tensor OUT_TENSOR
                        output tensor names
  --verbose             show verbose info
```

## 2.3 Demo

Download the original YOLOv5s model.

```
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt -O yolov5s.v5.pt # yolov5
```

Export the original pytorch model to onnx type by official script.

```
python3 models/export.py --weights yolov5s.v5.pt --simplify
```

Using this tool.

```
$ python3 yolov5s-opt.py --input yolov5s.v5.onnx --output yolov5s.v5.opt.onnx --in_tensor 167 --out_tensor 397,458,519
---- Tengine YOLOv5 Optimize Tool ----

Input model      : yolov5s.v5.onnx
Output model     : yolov5s.v5.opt.onnx
Input tensor     : 167
Output tensor    : 397,458,519
[Quant Tools Info]: Step 0, load original onnx model from yolov5s.v5.onnx.
256
[Quant Tools Info]: Step 1, Remove the focus and postprocess nodes.
[Quant Tools Info]: Step 2, Using hardswish replace the sigmoid and mul.
[Quant Tools Info]: Step 3, Rebuild onnx graph nodes.
[Quant Tools Info]: Step 4, Update input and output tensor.
[Quant Tools Info]: Step 5, save the new onnx model to yolov5s.v5.opt.onnx.

---- Tengine YOLOv5s Optimize onnx create success, best wish for your inference has a high accuracy ...\(^0^)/ ----
```

## 2.4 The result

[Download]() the YOLOv5s-opt.onnx.

An overall comparison between
![a complicated model](https://github.com/BUG1989/tengine_test_data/blob/main/yolov5s.png)
and its simplified version:

![Comparison between old model and new model](https://github.com/BUG1989/tengine_test_data/blob/main/yolov5s-opt.png)
