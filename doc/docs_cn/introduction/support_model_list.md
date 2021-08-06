# 模型支持

Tengine 已完成对主流的计算机视觉模型的进行支持，包括分类、检测、识别、分割、关键点、OCR等功能。

- 支持的CPU架构：ARM、X86、MIPS、RISC-V ；

- 支持的GPU架构：NVIDIA valta TensorCore、Adreno、Mali  ；

- 支持的NPU架构：NPU ；

| 类别   | 模型            | 支持平台                 	|
| ------ | --------------  | ------------------------  |
| 分类   | MobileNet V1    | CPU、GPU、NPU            	|
| 检测   | MobileNet-SSD   | CPU、GPU、NPU            	|
| 识别   | MobileFaceNets  | CPU、GPU                 	|
| 检测   | YOLOv3          | CPU、GPU、NPU            	|
| 检测   | YOLOv3-Tiny     | CPU、GPU、NPU            	|
| 检测   | YOLOv4          | CPU、GPU、NPU            	|
| 检测   | YOLOv4-Tiny     | CPU、GPU、NPU            	|
| 检测   | YOLOv5          | CPU、GPU、NPU            	|
| 检测   | YOLOv5s         | CPU、GPU、NPU            	|
| 检测   | YOLOvfastest    | CPU、GPU、NPU            	|
| 人脸   | retinaface	   | CPU、GPU、NPU            	|
| 人脸   | ultraface	   | CPU、GPU		     	    |
| 分割   | YOLCAT          | CPU、GPU                 	|
| 关键点 | Landmark        | CPU、GPU                 	|
| 关键点 | Alphapose       | CPU、GPU 	              	|
| 关键点 | Openpose        | CPU、GPU 	              	|
| OCR    | crnn_lite_dense | CPU、GPU 	               	|

**提示**：

- 模型链接来自 Tengine Model Zoo，我们将持续更新;
- 支持平台列表中的 NPU 中，部分模型采用异构计算实现，即 CPU+NPU。

**模型仓库**

- [百度网盘](https://pan.baidu.com/s/1JsitkY6FVV87Kao6h5yAmg) （提取码：7ke5）

- [Google Drive](https://drive.google.com/drive/folders/1hunePCa0x_R-Txv7kWqgx02uTCH3QWdS?usp=sharing)