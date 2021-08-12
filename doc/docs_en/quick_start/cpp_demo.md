# C++ Complete Example

All C++ examples shown in this section are located at [examples](https://github.C++om/OAID/Tengine/tree/tengine-lite/examples) .

## 1. Environmental preparation
To compile and run the C++ sample program, you need to prepare: 1. a computer that can compile C/C++ Linux environment (x86 or Arm architecture can be used).


## 2. Download the precompiled forecast library
Download the Tengine-Lite precompiled library. 

Please choose the correct version according to your architecture

| Arch  | with_extra | arm_stl    | with_cv | Download                                              |
| ----- | ---------- | ---------- | ------- | ----------------------------------------------------- |
| ARMv8 | OFF        | c++_static |         | (<font size=4 color=orange>To be supplemented</font>) |
| ARMv7 | OFF        |            |         | (<font size=4 color=orange>To be supplemented</font>) |
| x86   | OFF        |            |         | (<font size=4 color=orange>To be supplemented</font>) |

**Taking ARMv8 architecture as an example, the decompressed content structure is as follows：**(<font size=4 color=orange>To be supplemented</font>)

```shell
tengine_lite_lib.cxx.armv8           Tengine-Lite Forecast Library
├── cxx                              C++ Forecast Library
│   ├── include                      C++ Forecast Library Header File
│   └── lib                          C++ Forecast Library Library File
│       ├── libtengine-lite.a        Static Forecast Library
│       └── libtengine-lite.so       Dynamic Forecast Library
└── demo                             Demo
    └── cxx                          C++ Demo
```

## 3. Prepare A Predictive Deployment Model

(1) Model download: Download mobilenet_v1(<font size=4 color=orange>To be supplemented</font>). The model file with. tmfile suffix can be obtained by decompressing the model. You can open the model file under the folder through the model visualization tool [Netron](https://lutzroeder.github.io/netron/) to view the model structure.

## 4. Compile the Prediction Sample Program

export LD_LIBRARY_PATH=(<font size=4 color=orange>To be supplemented</font>)

g++ -I(<font size=4 color=orange>To be supplemented</font>) -L(<font size=4 color=orange>To be supplemented</font>) -l(<font size=4 color=orange>To be supplemented</font>) cpp_tm_classification.cpp -o  cpp_tm_classification
## 5. Execute Forecast

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

It can be seen that we have successfully classified the test pictures as tiger and cat. Here, the most basic finger north has been completed, and the rest of the friends can explore on their own. We will also update various tutorial examples to everyone from time to time ~