# Tengine Video Capture User Manual



## 约束

当前版本仅支持基于 Khadas VIM3 SBC 上的 NPU 网络模型推理演示，我们后续会逐步完善，支持基于更多硬件平台的功能演示。

默认大家手上的 Khadas VIM3 中的固件为最新版本。

### 硬件说明

| 物品        | 描述                                                         |
| ----------- | ------------------------------------------------------------ |
| Khadas VIM3 | 内置 A311D SoC 的单板计算机，内置 5Tops NPU 加速器           |
| USB 摄像头  | 输入实时视频流                                               |
| 液晶显示器  | 控制台操作，实时输出示例运行结果                             |
| HDMI连接线  | 由于Khadas VIM3 的 TYPE C 接口与 HDMI 接口过于紧凑，需要寻找小一点接口的 HMDI 连接线 |

### 软件说明

以下均为 Khadas VIM3 单板计算机上的软件描述。

- Ubuntu 20.04
- OpenCV 4.2
- gcc 9.3.0
- cmake 3.16.3

### 操作说明

后续步骤中的命令行操作均为基于 Khadas VIM3 单板计算机上的操作，其中：

- **下载**、**编译**步骤 可以过 SSH 登陆或者直接在 Khadas VIM3 的 Ubuntu 桌面启动控制台中执行；
- **运行**步骤仅在 Khadas VIM3 的 Ubuntu 桌面启动控制台中执行。

## 编译

### 下载 NPU 依赖库 TIM-VX

```
$ git clone https://github.com/VeriSilicon/TIM-VX.git
```

### 下载 Tengine

```
$ git clone https://github.com/OAID/Tengine.git tengine-lite
$ cd tengine-lite
```

### 准备代码

```
$ cd <tengine-lite-root-dir>
$ cp -rf ../TIM-VX/include  ./source/device/tim-vx/
$ cp -rf ../TIM-VX/src      ./source/device/tim-vx/
```

### 执行编译

```
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON -DTENGINE_ENABLE_MODEL_CACHE=ON -DTENGINE_BUILD_DEMO=ON ..
$ make demo_yolo_camera -j`nproc`
```

编译完成后，`libtengine-lite.so` 和 `demo_yolo_camera` 存放在以下路径：

- `<tengine-lite-root-dir>/build/source/libtengine-lite.so`
- `<tengine-lite-root-dir>/build/demos/demo_yolo_camera`

## 运行

模型文件 `yolov3_uint8.tmfile` 可从 Model ZOO 中下载，按照以下顺序方式存放文件：

```
......
├── demo_yolo_camera
├── libtengine-lite.so
├── models
│   └── yolov3_uint8.tmfile
......
```

执行当前路径下的 `demo_yolo_camera` ：

```
./demo_yolo_camera
```

*P.S. ：第一次运行因为会在线编译生成 NPU 运行依赖的 kernel file，会有一定的等待时间（大约30秒），后续运行直接加载所在目录下的 cache file 文件（小于1秒）。*

## 关于容器

- 我们提供了基于 Khadas VIM3 平台的容器版本，具体操作可以参考 [deploy_superedge](deploy_SuperEdge.md)；
- 我们提供了腾讯云的 SuperEdge 版本，请参考（待补充）。



## FAQ

Khadas VIM3 编译 Tengine + TIMVX 其余问题（包括 Khadas VIM3 购买渠道）可以参考 [compile_timvx](compile_timvx.md)。



