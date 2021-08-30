# Tengine Video Capture User Manual
## constraints
The current version only supports NPU network model reasoning demonstration based on the Khadas VIM3 SBC, we will gradually improve to support more hardware platform demonstration.
By default, the firmware in Khadas VIM3 is the latest version.
Hardware description
| | | item description
| ----------- | ------------------------------------------------------------ |
| Khadas VIM3 | built-in A311D SoC single board computer, built-in 5 tops NPU accelerator |
| | USB camera | input real-time video streams
LCD | | console operation, real-time output sample | run results
| HDMI cable | because Khadas VIM3 TYPE C interface with HDMI interface is too tight, need to find a little bit small interface HMDI cable |
### Software description
The following is a description of software on a Khadas VIM3 single-board computer.
- Ubuntu 20.04
- OpenCV 4.2
- GCC 9.3.0
- cmake 3.16.3
### Operation instructions
The command line operations in the following steps are based on the operations on the Khadas VIM3 board computer, where:
- **Download**, **compile** steps can be performed via SSH login or directly in the Khadas VIM3 Ubuntu desktop boot console;
- **The run** step is only performed in the Ubuntu desktop startup console of Khadas VIM3.
## compiler
### download tim-vx
```
$ git clone https://github.com/VeriSilicon/TIM-VX.git
```
Download Tengine ###
```
$ git clone https://github.com/OAID/Tengine.git tengine-lite
$ cd tengine-lite
```
### Prepare the code
```
$ cd &lt;tengine-lite-root-dir&gt;
$ cp -rf ... / TIM - ag/include/source/device/TIM - ag /
$ cp -rf ... / TIM - ag/SRC/source/device/TIM - ag /
```
### Execute compile
```
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON -DTENGINE_ENABLE_MODEL_CACHE=ON -DTENGINE_BUILD_DEMO=ON ..
$ make demo_yolo_camera -j`nproc`
```
After compilation, 'libtenine-lite. so' and 'demo_yolo_camera' are stored in the following directory:
- `>tengine-lite-root-dir&gt;/build/source/libtengine-lite.so`
- `>tengine-lite-root-dir&gt;/build/demos/demo_yolo_camera`
## to run

Model file 'yolov3_uint8.tmfile' can be downloaded from Model ZOO and stored in the following order:
```
......
├── demo_yolo_camera
├── libtengine-lite.so
├── models
│   └── yolov3_uint8.tmfile
......
```
Execute 'demo_yolo_camera' in the current path:
```
./demo_yolo_camera
```
*P.S. : The first run will compile the kernel file that the NPU depends on online, and the wait time is about 30 seconds. The next run is the cache file file in the directory where the NPU is directly loaded (less than 1 second).*
## About containers
- We provide a container version based on the Khadas VIM3 platform, see [deploy_superEdge](deploy_SuperEdge.md) for more information;
- We provide a SuperEdge version of Tencent Cloud, please refer to it (to be added).
## FAQ
Other questions (including Khadas VIM3 purchase channels) are referred to [compile_timvx](compile_timvx.md).
