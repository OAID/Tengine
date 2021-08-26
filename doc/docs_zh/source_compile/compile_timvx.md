# 源码编译（TIM-VX）

## 简介

[TIM-VX](https://github.com/VeriSilicon/TIM-VX.git) 的全称是 Tensor Interface Module for OpenVX，是 VeriSilicon 提供的用于在支持 OpenVX 的其自研 ML 加速器 IP 上实现深度学习神经网络模型部署。它可以做为 Android NN、TensorFlow-Lite、MLIR、TVM、Tengine 等 Runtime Inference Framework 的 Backend 模块。

Tengine 基于 [Khadas VIM3](https://www.khadas.cn/product-page/vim3) (Amlogic A311D)单板计算机，完成了 TIM-VX 的集成，充分发挥出其内置 NPU **高性能**和 Tengine 异构计算自动切图的**易用性**。
目前支持的芯片平台有：

- [Amlogic](https://www.amlogic.com) 的 [A311D](https://www.amlogic.com/#Products/393/index.html) / [S905D3](https://www.amlogic.com/#Products/392/index.html) ，[C305X](https://www.amlogic.com/#Products/412/index.html) / [C308X](https://www.amlogic.com/#Products/409/index.html) ；
- [Rockchip](https://www.rock-chips.com) 的 [RV1109](https://www.rock-chips.com/a/cn/product/RV11xilie/2020/0427/1073.html) / [RV1126](https://www.rock-chips.com/a/cn/product/RV11xilie/2020/0427/1075.html) ；
- [NXP](https://www.nxp.com/) 的 [i.MX 8M Plus](https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/i-mx-applications-processors/i-mx-8-processors/i-mx-8m-plus-arm-cortex-a53-machine-learning-vision-multimedia-and-industrial-iot:IMX8MPLUS) ；
- [瓴盛科技(JLQ)](https://www.jlq.com) 的 [JA308](https://www.jlq.com/images/products/ja310/ja308.pdf) / [JA310](https://www.jlq.com/images/products/ja310/ja310.pdf) / [JA312](https://www.jlq.com/images/products/ja310/ja312.pdf) ;

## 准备工作

由于 TIM-VX 官方只提供了 A311D 的预编译库和 x86_64 NPU 模拟计算的预编译库，因此本文档只基于上述两种平台进行源码编译、安装、运行说明。

### 下载 TIM-VX 源码 

```bash
$ git clone https://github.com/VeriSilicon/TIM-VX.git
```

### Download Tengine Lite

```bash
$ git clone https://github.com/OAID/Tengine.git tengine-lite
$ cd tengine-lite
```

### 2.1 Prepare for x86_64 simulator platform

**non-cross-compilation**

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/lib/x86_64
$ mkdir -p ./3rdparty/tim-vx/include
$ cp -rf ../TIM-VX/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/src    ./source/device/tim-vx/
$ cp -rf ../TIM-VX/prebuilt-sdk/x86_64_linux/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/prebuilt-sdk/x86_64_linux/lib/*    ./3rdparty/tim-vx/lib/x86_64/
$ rm ./source/device/tim-vx/src/tim/vx/*_test.cc
```

Build Tengine

```bash
$ export LD_LIBRARY_PATH=<tengine-lite-root-dir>/3rdparty/tim-vx/lib/x86_64

$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j4
```

### 2.2 Prepare for Khadas VIM3 platform

Prepare for VIM3 prebuild sdk:

```bash
$ wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz
$ tar zxvf aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz
$ mv aarch64_A311D_D312513_A294074_R311680_T312233_O312045 prebuild-sdk-a311d

$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/lib/aarch64
$ mkdir -p ./3rdparty/tim-vx/include
$ cp -rf ../TIM-VX/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/src    ./source/device/tim-vx/
$ cp -rf ../prebuild-sdk-a311d/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../prebuild-sdk-a311d/lib/*    ./3rdparty/tim-vx/lib/aarch64/
$ rm ./source/device/tim-vx/src/tim/vx/*_test.cc
```

#### 2.2.1 cross-compilation

TOOLCHAIN_FILE in the <tengine-lite-root-dir>/toolchains
```bash
$ export LD_LIBRARY_PATH=<tengine-lite-root-dir>/3rdparty/tim-vx/lib/aarch64

$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j4
```

#### 2.2.2 non-cross-compilation

Check for galcore:

```bash
$ sudo dmesg | grep Galcore
```

if  ( Galcore version < 6.4.3.p0.286725 )

```bash
$ rmmod galcore
$ insmod galcore.ko
```

Check for libOpenVX.so*:

```bash
$ sudo find / -name "libOpenVX.so*"
```

if  ( libOpenVX.so version <   libOpenVX.so.1.3.0  in  /usr/lib )

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p Backup
$ mv /usr/lib/libOpenVX.so* ./Backup
$ cp -rf ../prebuild-sdk-a311d/lib/libOpenVX.so* /usr/lib
```

Build Tengine

```bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j4
```


### 3.  编译 Rockchip RV1109/RV1126 buildroot 平台
RV1109/RV1126 只有 buildroot，没有完整系统的概念，所以不能进行本地编译，只能交叉编译。
解压缩 Rockchip 提供(或板卡厂商代为提供)的 RV1109/RV1126 SDK 后，找到 <rv1109-rv1126>external/rknpu/drivers/linux-armhf-puma/usr/lib，**注意**此目录下的文件[列表](#list)我们后面会用到，为了方便起见，此路径我们称之为 <rk_sdk_npu_lib>。

#### 3.1 Prepare for Khadas VIM3 platform

```bash
$ cd <tengine-lite-root-dir>
$ cp -rf ../TIM-VX/include  ./source/device/tim-vx/
$ cp -rf ../TIM-VX/src      ./source/device/tim-vx/
```

#### 3.2 准备3rdparty 依赖

```bash
$ wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_S905D3_D312513_A294074_R311680_T312233_O312045.tgz #不要介意是64位，我们只是需要头文件
$ tar zxvf aarch64_S905D3_D312513_A294074_R311680_T312233_O312045.tgz
$ mv aarch64_S905D3_D312513_A294074_R311680_T312233_O312045 prebuild-sdk-s905d3
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/include
$ mkdir -p ./3rdparty/tim-vx/lib/aarch32
$ cp -rf ../prebuild-sdk-s905d3/include/*  	./3rdparty/tim-vx/include/      #拷贝头文件
$ cp -rf <rk_sdk_npu_lib>/*      			./3rdparty/tim-vx/lib/aarch32/  #拷贝库
```



#### 3.3 编译

**准备好交叉编译链**

> * 交叉编译链：`gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf`
> * 且设置路径 `export PATH=<cross_tool_chain>/bin:$PATH` 

因为板上没有 OpenMP 的支持库，因此在 CMake 配置时关闭OpenMP选项，

```bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ export PATH=<cross_tool_chain>/bin:$PATH                      #把交叉编译链加入PATH
$ export LD_LIBRARY_PATH=../3rdparty/tim-vx/lib/aarch32         #设置依赖库的路径
$ ln -s ../3rdparty/tim-vx/lib/aarch32/libOpenVX.so.1.2 ../3rdparty/tim-vx/lib/aarch32/libOpenVX.so
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake  -DTENGINE_ENABLE_TIM_VX=ON -DTENGINE_OPENMP=OFF ..
$ make -j`nproc` && make install
```

### 4. 编译 Android 32bit 平台

目前只有 VIM3/VIM3L 和 i.MX 8M Plus 的 EVK 正式支持 Android 系统，编译时需要使用 NDK 进行编译。编译之前需要准备 3rdparty 的全部文件。
3rdparty 的结构同前面 Linux 的情况一致，但此时提取到的 so 放置的目录是 `3rdparty/tim-vx/lib/android`。

#### 4.1  准备代码

代码准备和前面典型的 Linux 准备过程相同，参考代码如下：

``` bash
$ cd <tengine-lite-root-dir>
$ cp -rf ../TIM-VX/include  ./source/device/tim-vx/
$ cp -rf ../TIM-VX/src      ./source/device/tim-vx/
```

#### 4.2 准备 3rdparty

假定采用下载的预编译 Android 库，参考准备的命令如下：

``` bash
$ wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/arm_android9_A311D_6.4.3.tgz
$ tar zxvf arm_android9_A311D_6.4.3.tgz
$ mv arm_android9_A311D_6.4.3 prebuild-sdk-android
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/include
$ mkdir -p ./3rdparty/tim-vx/lib/android
$ cp -rf ../prebuild-sdk-android/include/*  ./3rdparty/tim-vx/include/
$ cp -rf ../prebuild-sdk-android/lib/*      ./3rdparty/tim-vx/lib/android/
```

使用的 Android 系统内置的 NPU 驱动版本和相关的 so 不一定和下载到的 `6.4.3` 版本匹配，只需要保证不低于这个版本即可。如果确有问题，可以根据下载到的压缩包解压缩出来的 lib 目录里面的文件做列表，从板卡中用 adb pull 命令从 `/vendor/lib/` 目录中提取一套出来，放入 3rdparty/tim-vx/lib/android 目录里。

#### 4.3  编译

```bash
$ export ANDROID_NDK=<your-ndk-root-dir>
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-25 ..
$ make -j`nproc` && make install
```

完成编译后，建议使用 ADB Shell 跑测一下 example，确保板卡环境正确。**APK 能够运行**还需要**放行 NPU 驱动**的 so，具体**参见 FAQ 章节**。

### FAQ
Q：如何查看 NPU 驱动已经加载？
A：用 lsmod 命令查看相关的驱动模块加载情况；以 VIM3 为例，检查 Galcore 内核驱动是否正确加载：
``` bash
khadas@Khadas:~$ sudo lsmod
Module                  Size  Used by
iv009_isp_sensor      270336  0
iv009_isp_lens         69632  0
iv009_isp_iq          544768  0
galcore               663552  0
mali_kbase            475136  0
iv009_isp             540672  2
vpu                    49152  0
encoder                53248  0
# 中间打印略过
dhd                  1404928  0
sunrpc                446464  1
btrfs                1269760  0
xor                    20480  1 btrfs
raid6_pq              106496  1 btrfs
khadas@Khadas:~$
```
可以看到，`galcore 663552  0` 的打印说明了 galcore.ko 已经成功加载。

Q：如何查看 Galcore 的版本？
A：使用 dmesg 命令打印驱动加载信息，由于信息较多，可以通过 grep 命令进行过滤。
Linux 系统典型命令和打印如下：
``` bash
khadas@Khadas:~$ sudo dmesg | grep Galcore
[sudo] password for khadas: 
[   17.817600] Galcore version 6.4.3.p0.286725
khadas@Khadas:~$
```
Android 典型命令打印如下：
``` bash
kvim3:/ $ dmesg | grep Galcore
[   25.253842] <6>[   25.253842@0] Galcore version 6.4.3.279124+1
kvim3:/ $
```
可以看出，这个 linux 的 A311D 板卡加载的 galcore.ko 版本是 6.4.3.p0.286725，满足 linux 的版本最低要求。

Q：如何替换 galcore.ko？
A：在 SDK 和内核版本升级过程中，有可能有需要升级对应的 NPU 部分的驱动，尽管推荐这一部分由板卡厂商完成，但实际上也有可能有测试或其他需求，需要直接使用最新的 NPU 版本进行测试。这时需要注意的是首先卸载 galcore.ko，然后再加载新的版本。具体命令为(假设新版本的 galcore.ko 就在当前目录)：
``` bash
khadas@Khadas:~$ ls
galcore.ko
khadas@Khadas:~$ sudo rmmod galcore
khadas@Khadas:~$ sudo insmod galcore.ko
khadas@Khadas:~$ sudo dmesg | grep Galcore
[   17.817600] Galcore version 6.4.3.p0.286725
khadas@Khadas:~$
```
这样完成的是临时替换，临时替换在下次系统启动后就会加载回系统集成的版本；想要直接替换集成的版本可以通过 `sudo find /usr/lib -name galcore.ko` 查找一下默认位置，一个典型的路径是 `/usr/lib/modules/4.9.241/kernel/drivers/amlogic/npu/galcore.ko`，将 galcore.ko 替换到这个路径即可。
替换完成后，还需要替换用户态的相关驱动文件，<span id="list">**列表**一般有</span>：

``` bash
libGAL.so
libNNGPUBinary.so
libOpenCL.so
libOpenVXU.so
libVSC.so
libCLC.so
libNNArchPerf.so
libNNVXCBinary.so
libOpenVX.so
libOvx12VXCBinary.so
libarchmodelSw.so
```
其中部分文件大小写、文件名、版本扩展名等可能不尽相同，需要保证替换前后旧版本的库及其软连接清理干净，新版本的库和软连接正确建立不疏失(有几个 so 可能在不同的版本间是多出来或少掉的，是正常情况)。
这些文件一般在 `/usr/lib/` 文件夹里面(一些板卡可能没有预置用户态的驱动和内核驱动，这时自行添加后增加启动脚本加载内核驱动即可)。

Q：替换 galcore.ko 后，怎么检查细节状态？
A：有时 insmod galcore.ko 后，lsmod 时还是有 galcore 模块的，但确实没加载成功。此时可以用 dmesg 命令确认下返回值等信息，核查是否有其他错误发生。
Linux 典型打印如下：
``` bash
khadas@Khadas:~$ sudo dmesg | grep galcore
[    0.000000] OF: reserved mem: initialized node linux,galcore, compatible id shared-dma-pool
[   17.793965] galcore: no symbol version for module_layout
[   17.793997] galcore: loading out-of-tree module taints kernel.
[   17.817595] galcore irq number is 37.
khadas@Khadas:~$
```
Android 典型打印如下：
``` bash
kvim3:/ $ dmesg | grep galcore
[    0.000000] <0>[    0.000000@0]      c6c00000 - c7c00000,    16384 KB, linux,galcore
[   25.253838] <4>[   25.253838@0] galcore irq number is 53.
kvim3:/ $
```

Q：打印提示依赖库是未识别的 ELF 格式？
A：

Q：为什么我的 Android 跑不起来对应的 APK，但 ADB Shell 跑测试程序却可以？
A：Android 系统不同于 Linux 系统，可以很方便的通过 GDB Server 进行远程调试，所以建议 APP 里面的集成算法部分，先在 ADB Shell 里验证一下正确性后再进行 APK 的集成。
如果已经在 ADB Shell 里验证了典型的用例是正确的，APK 里面的 JNI 部分也没有其他问题，那么 APP 运行不了可以检查一下对应的 NPU 用户态驱动是否已经放行。许可文件路径是 `/vendor/etc/public.libraries.txt` 。许可没有放行一般提示包含有 `java.lang.UnsatisfiedLinkError` 错误。已经放行的 Android 许可文件大致如下图所示，libCLC.so 等已经包含进来：
``` bash
kvim3:/vendor/etc $ cat public.libraries.txt
libsystemcontrol_jni.so
libtv_jni.so
libscreencontrol_jni.so
libCLC.so
libGAL.so
libOpenVX.so
libOpenVXU.so
libVSC.so
libarchmodelSw.so
libNNArchPerf.so
kvim3:/vendor/etc $
```
如果没有放行，需要在 ADB Shell 里面转到 root 权限，并重新挂载文件系统；重新进入 ADB Shell 后，修改完成后重启一次系统。大致操作如下：
``` bash
adb root                              # 获取 root 权限
adb remount                           # 重新挂载文件系统
adb shell                             # 进入 ADB Shell
vi /vendor/etc/public.libraries.txt   # 编辑许可文件
```
如果对 vi 和相关命令不熟悉，可以考虑 `adb pull /vendor/etc/public.libraries.txt` 拉到 PC 上进行修改，然后再 `adb push public.libraries.txt /vendor/etc/` 推送回板卡。

### 附：部分支持的板卡链接

* 限于许可，Tengine-Lite 不能二次分发已经准备好的 3rdparty，请谅解。
* 如果本文档描述的过程和 FAQ 没有覆盖您的问题，也欢迎加入 QQ 群 829565581 进一步咨询。
* 不同版本的 TIM-VX 和 Tengine 对 OP 支持的情况有一定区别，请尽可能拉取最新代码进行测试评估。
* 如果已有 OP 没有满足您的应用需求，可以分别在 TIM-VX 和 Tengine 的 issue 里创建一个新的 issue 要求支持；紧急或商业需求可以加入 QQ 群联系管理员申请商业支持。
* Tengine 和 OPEN AI LAB 对文档涉及的板卡和芯片不做单独的保证，诸如芯片或板卡工作温度、系统定制、配置细节、价格等请与各自芯片或板卡供应商协商。
* 如果贵司有板卡想要合作，可以加入 OPEN AI LAB 的 QQ 群联系管理员进一步沟通。
