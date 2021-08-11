# Model Support

Tengine has finished supporting mainstream computer vision models, including classification, detection, recognition, segmentation, key points and other functions.

- The supported CPU architecture：ARM、X86、MIPS、RISC-V ；
- The supported GPU architecture：NVIDIA valta TensorCore、Adreno、Mali  ；
- The supported NPU architecture：NPU ；

| Catalog 	    | Model          | Support Platform         |
| ----------------- | -------------- | ------------------------ |
| classification    | MobileNet V1   | CPU、GPU、NPU 		|
| detection    	    | MobileNet-SSD  | CPU、GPU、NPU 		|
| recognition       | MobileFaceNets | CPU、GPU            	|
| detection         | YOLOv3         | CPU、GPU、NPU            	|
| detection         | YOLOv3-Tiny    | CPU、GPU、NPU            	|
| detection         | YOLOv4         | CPU、GPU、NPU            	|
| detection         | YOLOv4-Tiny    | CPU、GPU、NPU            	|
| detection         | YOLOv5         | CPU、GPU、NPU            	|
| detection         | YOLOv5s        | CPU、GPU、NPU            	|
| detection         | YOLOvfastest   | CPU、GPU、NPU            	|
| faceDetection     | retinaface     | CPU、GPU、NPU            	|
| faceDetection	    | ultraface	     | CPU、GPU			|
| segmentation      | YOLCAT         | CPU、GPU                	|
| keypoint   	    | Landmark       | CPU、GPU            	|
| keypoint          | Alphapose      | CPU、GPU                	|
| keypoint          | Openpose       | CPU、GPU                	|
| OCR    	    | crnn_lite_dense | CPU、GPU 	        |

**Tips**：

- The model link comes from Tengine Model Zoo, and we will continue to update it;
- Among the NPUs in the list of supported platforms, some models are implemented by heterogeneous computing, namely CPU+NPU.

**Model Zoo**

- [Baidu Netdisk](https://pan.baidu.com/s/1JsitkY6FVV87Kao6h5yAmg) (password: 7ke5)

- [Google Drive](https://drive.google.com/drive/folders/1hunePCa0x_R-Txv7kWqgx02uTCH3QWdS?usp=sharing)
