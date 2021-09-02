# C++ 完整示例

本章节展示的所有C++示例位于[examples](https://github.C++om/OAID/Tengine/tree/tengine-lite/examples) 。

## 1.环境准备
要编译和运行C++示例程序，你需要准备:
1.一台可以编译C/C++ 的Linux环境的电脑（x86或Arm架构均可）。


## 2.下载预编译的预测库
下载Tengine预编译库，
请根据你的架构选择正确的版本

| Arch  | with_extra | arm_stl    | with_cv | 下载                                      |
| ----- | ---------- | ---------- | ------- | ----------------------------------------- |
| ARMv8 | OFF        | c++_static |         | (<font size=4 color=orange>待补充</font>) |
| ARMv7 | OFF        |            |         | (<font size=4 color=orange>待补充</font>) |
| x86   | OFF        |            |         | (<font size=4 color=orange>待补充</font>) |

**以ARMv8架构为例，解压后内容结构如下：**(<font size=4 color=orange>待补充</font>)

```shell
tengine_lite_lib.cxx.armv8           Tengine 预测库
├── cxx                              C++预测库
│   ├── include                      C++预测库头文件
│   └── lib                          C++预测库文件
│       ├── libtengine-lite.a        静态预测库
│       └── libtengine-lite.so       动态预测库
└── demo                             示例 Demo
    └── cxx                          C++示例 Demo
```

## 3. 准备预测部署模型

(1) 模型下载：下载mobilenet_v1(<font size=4 color=orange>待补充</font>)模型后解压，可得到.tmfile后缀的模型文件。可通过模型可视化工具 [Netron](https://lutzroeder.github.io/netron/) 打开文件夹下的model文件，查看模型结构。

## 4. 编译预测示例程序

export LD_LIBRARY_PATH=(<font size=4 color=orange>待补充</font>)

g++ -I(<font size=4 color=orange>待补充</font>) -L(<font size=4 color=orange>待补充</font>) -l(<font size=4 color=orange>待补充</font>) cpp_tm_classification.cpp -o  cpp_tm_classification
## 5.执行预测

```bash
export LD_LIBRARY_PATH=./build-linux-x86/install/lib
cd ./build-linux-x86/install/examples/
$ ./cpp_tm_classification -m /path/to/mobilenet.tmfile -l /path/to/labels.txt -i /path/to/img.jpg -g 224,224 -s 0.017 -w 104.007,116.669,122.679

--------------------------------------
0.3465 - "tiger cat"
0.1609 - "tabby"
0.1564 - "weasel"
0.0844 - "Egyptian cat"
0.0258 - "bucket"
--------------------------------------
ALL TEST DONE
```

可以看出，我们已经成功的分类出测试图片是虎猫了。
到这里最基本的上手指北已经完成，剩下的小伙伴们可以自行探索，我们也会不定期更新各种教程范例给大家~
