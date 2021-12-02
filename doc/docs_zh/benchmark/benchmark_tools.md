# 测试方法

Benchmark 是评估目标硬件平台网络模型运行速度的简单途径，只依赖于网络结构（xxx_benchmark.tmfile）即可。

## 测试模型获取

虽然可以直接使用完整的 tmfile 运行 benchmark 示例，但是我们建议采用 benchmark 专用 tmfile 模型，节省文件传输时间。

- 使用模型转换工具转换前，设置以下环境变量，将生成不带参数的 tmfile 文件，专门用于 benchmark 测试。

```
$ export TM_FOR_BENCHMARK=1
```

- 将原始框架模型转换为 tmfile benchmark 专用模型，以 Caffe 框架的 mobilenet_v1 举例：

```
$ ./convert_tm_tool -f caffe -p mobilenet_v1.prototxt -m mobilenet_v1.caffemodel -o mobilenet_v1_benchmark.tmfile
```

我们已经提前转换了一小部分评估模型在 benchmark/models 中。

## 获取 

默认完成 Tengine 编译，目标平台的 benchmark 可执行程序存放在 `build/install/bin/tm_benchmark`

## 使用方法

```
$ ./tm_benchmark -h
[Usage]:  [-h] [-r repeat_count] [-t thread_count] [-p cpu affinity, 0:auto, 1:big, 2:middle, 3:little] [-s net]
```

```
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
```
