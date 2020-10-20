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

Khadas VIM3 (Cortex-A73 2.2GHz x 4 + Cortex-A53 1.8GHz x 2)

```bash
khadas@Khadas:~/tengine-lite/benchmark$ ../build/benchmark/tm_benchmark -r 5 -t 1 -p 1
start to run register cpu allocator
loop_counts = 5
num_threads = 1
power       = 1
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   55.66 ms   max =   56.19 ms   avg =   56.04 ms
         mobilenetv1  min =  103.18 ms   max =  105.37 ms   avg =  104.26 ms
         mobilenetv2  min =   91.46 ms   max =   93.07 ms   avg =   91.92 ms
         mobilenetv3  min =   56.30 ms   max =   57.17 ms   avg =   56.64 ms
        shufflenetv2  min =   29.92 ms   max =   30.62 ms   avg =   30.29 ms
            resnet18  min =  162.31 ms   max =  162.74 ms   avg =  162.48 ms
            resnet50  min =  495.61 ms   max =  498.00 ms   avg =  496.99 ms
           googlenet  min =  199.16 ms   max =  200.32 ms   avg =  199.72 ms
         inceptionv3  min =  801.93 ms   max =  813.71 ms   avg =  807.08 ms
               vgg16  min =  866.41 ms   max =  877.53 ms   avg =  871.45 ms
                mssd  min =  204.10 ms   max =  208.92 ms   avg =  206.05 ms
          retinaface  min =   28.57 ms   max =   29.06 ms   avg =   28.86 ms
         yolov3_tiny  min =  233.68 ms   max =  235.12 ms   avg =  234.19 ms
      mobilefacenets  min =   44.32 ms   max =   44.82 ms   avg =   44.60 ms
ALL TEST DONE
khadas@Khadas:~/tengine-lite/benchmark$ ../build/benchmark/tm_benchmark -r 5 -t 4 -p 1
start to run register cpu allocator
loop_counts = 5
num_threads = 4
power       = 1
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   22.10 ms   max =   22.33 ms   avg =   22.24 ms
         mobilenetv1  min =   32.07 ms   max =   32.68 ms   avg =   32.49 ms
         mobilenetv2  min =   40.16 ms   max =   40.59 ms   avg =   40.32 ms
         mobilenetv3  min =   32.37 ms   max =   32.60 ms   avg =   32.49 ms
        shufflenetv2  min =   12.67 ms   max =   12.91 ms   avg =   12.76 ms
            resnet18  min =   69.67 ms   max =   70.34 ms   avg =   69.91 ms
            resnet50  min =  174.66 ms   max =  175.34 ms   avg =  174.94 ms
           googlenet  min =   84.43 ms   max =   85.01 ms   avg =   84.82 ms
         inceptionv3  min =  274.61 ms   max =  276.78 ms   avg =  275.74 ms
               vgg16  min =  379.63 ms   max =  385.95 ms   avg =  382.01 ms
                mssd  min =   66.67 ms   max =   67.28 ms   avg =   67.01 ms
          retinaface  min =   15.15 ms   max =   15.34 ms   avg =   15.24 ms
         yolov3_tiny  min =  110.07 ms   max =  110.81 ms   avg =  110.50 ms
      mobilefacenets  min =   16.97 ms   max =   17.16 ms   avg =   17.06 ms
ALL TEST DONE
khadas@Khadas:~/tengine-lite/benchmark$ ../build/benchmark/tm_benchmark -r 5 -t 1 -p 3
start to run register cpu allocator
loop_counts = 5
num_threads = 1
power       = 3
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =  116.30 ms   max =  116.43 ms   avg =  116.34 ms
         mobilenetv1  min =  236.10 ms   max =  236.35 ms   avg =  236.21 ms
         mobilenetv2  min =  198.35 ms   max =  198.58 ms   avg =  198.42 ms
         mobilenetv3  min =  128.56 ms   max =  128.99 ms   avg =  128.76 ms
        shufflenetv2  min =   66.71 ms   max =   66.85 ms   avg =   66.75 ms
            resnet18  min =  358.30 ms   max =  358.49 ms   avg =  358.44 ms
            resnet50  min = 1094.14 ms   max = 1094.90 ms   avg = 1094.45 ms
           googlenet  min =  434.48 ms   max =  434.83 ms   avg =  434.61 ms
         inceptionv3  min = 1778.71 ms   max = 1779.36 ms   avg = 1779.03 ms
               vgg16  min = 1903.84 ms   max = 1932.26 ms   avg = 1909.85 ms
                mssd  min =  462.74 ms   max =  463.72 ms   avg =  463.13 ms
          retinaface  min =   59.83 ms   max =   59.94 ms   avg =   59.89 ms
         yolov3_tiny  min =  501.01 ms   max =  501.60 ms   avg =  501.32 ms
      mobilefacenets  min =   99.05 ms   max =   99.22 ms   avg =   99.13 ms
ALL TEST DONE
khadas@Khadas:~/tengine-lite/benchmark$ ../build/benchmark/tm_benchmark -r 5 -t 2 -p 3
start to run register cpu allocator
loop_counts = 5
num_threads = 2
power       = 3
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   63.93 ms   max =   64.02 ms   avg =   63.97 ms
         mobilenetv1  min =  115.33 ms   max =  115.47 ms   avg =  115.40 ms
         mobilenetv2  min =  105.52 ms   max =  105.74 ms   avg =  105.58 ms
         mobilenetv3  min =   83.13 ms   max =   84.02 ms   avg =   83.63 ms
        shufflenetv2  min =   40.04 ms   max =   40.13 ms   avg =   40.09 ms
            resnet18  min =  208.76 ms   max =  209.16 ms   avg =  208.88 ms
            resnet50  min =  600.78 ms   max =  607.13 ms   avg =  603.52 ms
           googlenet  min =  252.26 ms   max =  252.46 ms   avg =  252.34 ms
         inceptionv3  min =  949.61 ms   max =  960.68 ms   avg =  953.56 ms
               vgg16  min = 1105.32 ms   max = 1120.49 ms   avg = 1108.90 ms
                mssd  min =  237.19 ms   max =  237.38 ms   avg =  237.30 ms
          retinaface  min =   36.85 ms   max =   36.96 ms   avg =   36.89 ms
         yolov3_tiny  min =  297.31 ms   max =  298.04 ms   avg =  297.62 ms
      mobilefacenets  min =   53.09 ms   max =   53.18 ms   avg =   53.14 ms
ALL TEST DONE

```

EAIDK610 (Cortex-A72 1.8GHz x 2 + Cortex-A53 1.4GHz x 4)

```bash
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

Raspberry Pi 3B  (Cortex-A53 1.2GHZ x 4)

```
pi@raspberrypi:~/Tengine-Lite/build $ ./benchmark/tm_benchmark -r 8
start to run register cpu allocator
loop_counts = 8
num_threads = 1
power       = 0
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =  190.74 ms   max =  191.98 ms   avg =  191.15 ms
         mobilenetv1  min =  364.62 ms   max =  365.88 ms   avg =  364.92 ms
         mobilenetv2  min =  323.45 ms   max =  325.61 ms   avg =  323.85 ms
         mobilenetv3  min =  249.12 ms   max =  250.35 ms   avg =  249.39 ms
        shufflenetv2  min =  108.03 ms   max =  108.22 ms   avg =  108.12 ms
            resnet18  min =  598.48 ms   max =  605.05 ms   avg =  600.50 ms
            resnet50  min = 1754.92 ms   max = 1760.45 ms   avg = 1757.52 ms
           googlenet  min =  704.96 ms   max =  710.59 ms   avg =  705.90 ms
         inceptionv3  min = 2937.00 ms   max = 2940.33 ms   avg = 2939.03 ms
               vgg16  min = 3365.99 ms   max = 3546.59 ms   avg = 3391.13 ms
                mssd  min =  733.63 ms   max =  737.31 ms   avg =  735.61 ms
          retinaface  min =  112.00 ms   max =  114.12 ms   avg =  112.59 ms
         yolov3_tiny  min =  886.04 ms   max =  908.04 ms   avg =  889.82 ms
      mobilefacenets  min =  161.90 ms   max =  163.71 ms   avg =  162.18 ms
ALL TEST DONE
pi@raspberrypi:~/Tengine-Lite/build $ ./benchmark/tm_benchmark -r 8 -t 4
start to run register cpu allocator
loop_counts = 8
num_threads = 4
power       = 0
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   85.47 ms   max =   86.43 ms   avg =   86.05 ms
         mobilenetv1  min =  122.97 ms   max =  123.52 ms   avg =  123.29 ms
         mobilenetv2  min =  139.47 ms   max =  139.92 ms   avg =  139.76 ms
         mobilenetv3  min =  154.04 ms   max =  154.79 ms   avg =  154.41 ms
        shufflenetv2  min =   42.62 ms   max =   43.07 ms   avg =   42.82 ms
            resnet18  min =  362.03 ms   max =  364.59 ms   avg =  363.25 ms
            resnet50  min =  834.65 ms   max =  844.14 ms   avg =  838.60 ms
           googlenet  min =  364.03 ms   max =  367.16 ms   avg =  365.25 ms
         inceptionv3  min = 1074.93 ms   max = 1091.14 ms   avg = 1082.19 ms
               vgg16  min = 2622.68 ms   max = 2902.42 ms   avg = 2687.51 ms
                mssd  min =  258.68 ms   max =  260.33 ms   avg =  259.32 ms
          retinaface  min =   61.80 ms   max =   62.40 ms   avg =   61.98 ms
         yolov3_tiny  min =  673.53 ms   max =  695.12 ms   avg =  678.93 ms
      mobilefacenets  min =   72.38 ms   max =   72.78 ms   avg =   72.54 ms
ALL TEST DONE
```

Raspberry Pi 4B  (Cortex-A72 1.5GHZ x 4)
```bash
pi@raspberrypi:~/Tengine/benchmark $ ../build/benchmark/tm_benchmark -r 8
start to run register cpu allocator
loop_counts = 8
num_threads = 1
power       = 0
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   70.35 ms   max =   72.14 ms   avg =   70.99 ms
         mobilenetv1  min =  125.71 ms   max =  126.87 ms   avg =  126.30 ms
         mobilenetv2  min =  124.22 ms   max =  125.28 ms   avg =  124.67 ms
         mobilenetv3  min =   78.73 ms   max =   79.78 ms   avg =   79.10 ms
        shufflenetv2  min =   38.69 ms   max =   39.25 ms   avg =   38.96 ms
            resnet18  min =  219.02 ms   max =  220.24 ms   avg =  219.49 ms
            resnet50  min =  632.10 ms   max =  633.48 ms   avg =  632.85 ms
           googlenet  min =  264.98 ms   max =  385.50 ms   avg =  287.44 ms
         inceptionv3  min = 1035.28 ms   max = 1060.50 ms   avg = 1039.35 ms
               vgg16  min = 1163.56 ms   max = 1409.93 ms   avg = 1222.29 ms
                mssd  min =  254.38 ms   max =  255.45 ms   avg =  254.99 ms
          retinaface  min =   40.51 ms   max =   45.60 ms   avg =   41.24 ms
         yolov3_tiny  min =  301.45 ms   max =  304.59 ms   avg =  303.26 ms
      mobilefacenets  min =   59.49 ms   max =   60.40 ms   avg =   59.76 ms
ALL TEST DONE
pi@raspberrypi:~/Tengine/benchmark $ ../build/benchmark/tm_benchmark -r 8 -t 4
start to run register cpu allocator
loop_counts = 8
num_threads = 4
power       = 0
tengine-lite library version: 1.0-dev
     squeezenet_v1.1  min =   40.91 ms   max =   42.42 ms   avg =   41.44 ms
         mobilenetv1  min =   54.45 ms   max =   55.20 ms   avg =   54.84 ms
         mobilenetv2  min =   66.10 ms   max =   66.99 ms   avg =   66.39 ms
         mobilenetv3  min =   56.95 ms   max =   57.37 ms   avg =   57.14 ms
        shufflenetv2  min =   19.91 ms   max =   20.39 ms   avg =   20.10 ms
            resnet18  min =  157.12 ms   max =  160.92 ms   avg =  158.37 ms
            resnet50  min =  330.70 ms   max =  335.26 ms   avg =  332.28 ms
           googlenet  min =  169.96 ms   max =  172.73 ms   avg =  171.61 ms
         inceptionv3  min =  502.01 ms   max =  526.95 ms   avg =  511.20 ms
               vgg16  min =  818.95 ms   max =  854.09 ms   avg =  839.29 ms
                mssd  min =  110.79 ms   max =  113.91 ms   avg =  111.96 ms
          retinaface  min =   25.39 ms   max =   42.38 ms   avg =   27.72 ms
         yolov3_tiny  min =  188.28 ms   max =  190.19 ms   avg =  188.88 ms
      mobilefacenets  min =   28.96 ms   max =   31.83 ms   avg =   29.59 ms
ALL TEST DONE

```
