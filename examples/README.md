## examples

Tengine Lite 的 examples 将提供简单的、好玩的 demo。

Tengine Lite 兼容 Tengine 原有的 C API 供用户使用，这里我们使用 C API 展示如何运行 tm_classification 例程运行 MobileNet v1 分类网络模型，实现指定图片分类的功能。让你快速上手Tengine C++ API。这里，我们使用在这个撸猫时代行业从业者大爱的 tiger cat 作为测试图片。

![lu mao](https://github.com/OAID/Tengine/blob/master/tests/images/cat.jpg)

### 源码参考

[tm_classification.c](example/tm_classificaton.c)

### 编译

build.sh 编译脚本默认配置已实现自动编译 examples 中的 demo 程序，以 x86 平台为例，demo 存放在 ./build/install/bin/ 目录下。

```bash
bug1989@DESKTOP-SGN0H2A:/mnt/d/ubuntu/gitlab/build-linux$ tree install
install
├── bin
│   ├── tm_benchmark
│   ├── tm_classification
│   └── tm_mobilenet_ssd
├── include
│   └── tengine_c_api.h
└── lib
    └── libtengine-lite.so
```

### 运行结果

将测试图片和模型文件放在 Tengine-Lite 根目录下，运行：

```bash
$ export LD_LIBRARY_PATH=./build/install/lib
<<<<<<< examples/README.md
$ ./build/install/example/tm_classification -m models/mobilenet.tmfile -i images/cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679
=======
$ ./build/install/example/tm_classification -l models/synset_words.txt -m models/mobilenet.tmfile -i images/cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679
>>>>>>> examples/README.md
```

结果如下：

```bash
model file : ./temp/models/mobilenet_v1.tmfile
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

我们将持续更新各种有趣的 demo ，敬请期待......
