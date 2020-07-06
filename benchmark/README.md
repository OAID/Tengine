## Benchmark

Benchmark 是评估目标硬件平台网络模型运行速度的简单途径，只依赖于网络结构（xxx_benchmark.tmfile）即可。

### 生成 benchmark 专用 tmfile

虽然可以直接使用完整的 tmfile 运行 benchmark 示例，但是我们建议采用 benchmark 专用 tmfile 模型，节省文件传输时间。

1. 在使用模型转换工具 [convert_model_to_tm]() 之前，设置环境变量 ：

   ```shell
   $ export TM_FOR_BENCHMARK=1
   ```

2. 将原始框架模型转换为 tmfile benchmark 专用模型，以 Caffe 框架的 mobilenet_v1 举例：

   ```shell
   $ ./comvert_tm_tool -f caffe -p mobilenet_v1.prototxt -m mobilenet_v1.caffemodel -o mobilenet_v1_benchmark.tmfile
   ```

   我们已经提前转换了一小部分评估模型在 [benchmark/models](benchmark/models) 中。

---

### 编译

默认完成 Tengine Lite 编译，目标平台的 benchmark 可执行程序存放在  build-dir/install/bin/tm_benchmark 

```shell
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

### 使用方法

```shell
$ ./tm_benchmark -h
[Usage]:  [-h] [-r repeat_count] [-t thread_count] [-p cpu affinity, 0:auto, 1:big, 2:middle, 3:little] [-s net]
```

#### 例子，如何在 android 平台上运行 tm_benchmark

```shell
# for running on android device, upload to /data/local/tmp/ folder
$ adb push tm_benchmark /data/local/tmp/
$ adb push <tengine-lite-root-dir>/benchmark/models /data/local/tmp/
$ adb shell

# executed in android adb shell
$ cd /data/local/tmp/
$ ./tm_benchmark
```

---

Typical output (executed in linux)

EAIDK610 (Cortex-A72 1.8GHz x 2 + Cortex-A53 1.4GHz x 4)

```shell
[openailab@localhost benchmark]$ ../cmake-build-debug/benchmark/tm_benchmark -r 8
loop_counts  = 8
num_threads  = 1
power        = 0
     squeezenet_v1.1  min =   60.95 ms   max =   64.99 ms   avg =   61.91 ms
         mobilenetv1  min =  107.07 ms   max =  110.94 ms   avg =  108.07 ms
         mobilenetv2  min =  103.30 ms   max =  106.83 ms   avg =  104.08 ms
         mobilenetv3  min =   68.91 ms   max =   70.60 ms   avg =   69.44 ms
        shufflenetv2  min =   31.73 ms   max =   33.16 ms   avg =   32.14 ms
            resnet18  min =  209.66 ms   max =  211.33 ms   avg =  210.19 ms
            resnet50  min =  572.76 ms   max =  577.32 ms   avg =  575.06 ms
           googlenet  min =  253.46 ms   max =  256.21 ms   avg =  254.89 ms
         inceptionv3  min = 1014.39 ms   max = 1021.56 ms   avg = 1018.37 ms
               vgg16  min = 1165.28 ms   max = 1182.80 ms   avg = 1171.24 ms
                mssd  min =  219.30 ms   max =  225.62 ms   avg =  221.70 ms
          retinaface  min =   33.99 ms   max =   35.46 ms   avg =   34.41 ms
         yolov3_tiny  min =  309.41 ms   max =  317.77 ms   avg =  312.79 ms
      mobilefacenets  min =   46.79 ms   max =   49.18 ms   avg =   47.22 ms
ALL TEST DONE
```
