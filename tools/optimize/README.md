# Tengine YOLOv5s Optimize Tool

To support **YOLOv5s** detection model deployment on AIoT devices, we provide some universal network optimize tools which can optimize the **onnx type** of [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.onnx) to easy convert to tmfile model to deployment on CPU/GPU/**NPU** with **Tengine Framework**.

## 1 The object of this optimization

- Remove the focus nodes of prepare process;
- Remove the YOLO detection nodes of postprocess;
- Fusion the activation HardSwish node replace the Sigmoid and Mul.

## 2 How to use

## 2.1 Install dependent libraries

```
sudo pip install onnx
```

## 2.2 Description params

```
$ ./python3 yolov5s-opt.py -h
usage: yolov5s-opt.py [-h] [--input INPUT] [--output OUTPUT]
                      [--in_cut_name IN_CUT_NAME] [--output_num OUTPUT_NUM]
                      [--out_cut_names OUT_CUT_NAMES]

YOLOv5 Optimize Tool Parameters

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         input model path
  --output OUTPUT       output model path
  --in_cut_name IN_CUT_NAME
                        input cut node name
  --output_num OUTPUT_NUM
                        output num
  --out_cut_names OUT_CUT_NAMES
                        output cut node names
```

## 2.3 Demo

Download the original onnx type YOLOv5s model.

```
wget https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.onnx
```
Using this tool.

```
$ python3 yolov5s-opt.py --input yolov5s.onnx --output yolov5s-opt.onnx --in_cut_name 167 --out_cut_names 381,420,459 --output_num 3
---- Tengine YOLOv5 Optimize Tool ----

Input model      : yolov5s.onnx
Output model     : yolov5s-opt.onnx
Input node       : 167
Output nodes     : 381,420,459
Output node num  : 3

[Quant Tools Info]: Step 0, load original onnx model from yolov5s.onnx.
[Quant Tools Info]: Step 1, Remove the focus and postprocess nodes.
[Quant Tools Info]: Step 2, Using hardswish replace the sigmoid and mul.
[Quant Tools Info]: Step 3, Rebuild new onnx model.
[Quant Tools Info]: Step 4, save the new onnx model to yolov5s-opt.onnx

---- Tengine YOLOv5s Optimize onnx create success, best wish for your inference has a high accuracy ...\(^0^)/ ----
```

## 2.4 The result

[Download]() the YOLOv5s-opt.onnx.

An overall comparison between
![a complicated model](https://github.com/BUG1989/tengine_test_data/blob/main/yolov5s.png)
and its simplified version:

![Comparison between old model and new model](https://github.com/BUG1989/tengine_test_data/blob/main/yolov5s-opt.png)
