benchmark can be used to test neural network inference performance

Only the network definition files (tengine model file) are required.

Set the ENV with "export TM_FOR_BENCHMARK=1" when you convert tmfile(tengine model file) used tengine convert tools, the models params will not save to tmfile(tengine model file).

The large  models params are not loaded but generated randomly for speed test.

More model networks may be added later.

---
Build
```
# assume you have already build tengine library successfully
# uncomment the following line in <tengine-root-dir>/CMakeLists.txt with your favorite editor

# add_subdirectory(benchmark)

$ cd <tengine-root-dir>/<your-build-dir>
$ make -j4

# you can find benchmark binary in <tengine-root-dir>/<your-build-dir>/benchmark/src
```

Usage
```
# The models file in <tengine-root-dir>/models is convert without model params.
$ cd <tengine-root-dir>
$ ./<your-build-dir>/benchmark/src/benchmark -r [loop count] -s [model index]
```
run benchmark on android device
```
# for running on android device, upload to /data/local/tmp/ folder
$ adb push benchncnn /data/local/tmp/
$ adb push <tengine-root-dir>/models /data/local/tmp/
$ adb shell

# executed in android adb shell
$ cd /data/local/tmp/
$ ./benchmark -r [loop count] -s [model index]
```

Parameter

| param       | options             | default |
| ----------- | ------------------- | ------- |
| loop count  | 1~N                 | 10      |
| model index | 0~N                 | -1      |

---

Typical output (executed in linux)

HiSilicon Hi3559V100 (Cortex-A73 1.6GHz x 2 + Cortex-A53 1.2GHz x 2)
```
root@hi3559V100:~/Tengine$ ./build-linux-native/benchmark/src/benchmark -s 10 -r 100
loop_count = 100
model index = 10
         mobilenetv1  min =   61.06 ms   max =   84.96 ms   avg =   62.55 ms
--------------------------------------
     squeezenet_v1.1  min =   40.33 ms   max =   43.88 ms   avg =   42.72 ms
--------------------------------------
               vgg16  min =  710.65 ms   max =  733.53 ms   avg =  716.25 ms
--------------------------------------
                mssd  min =  124.67 ms   max =  128.28 ms   avg =  125.98 ms
--------------------------------------
            resnet50  min =  322.88 ms   max =  330.16 ms   avg =  324.46 ms
--------------------------------------
          retinaface  min =  349.14 ms   max =  366.08 ms   avg =  358.09 ms
--------------------------------------
              yolov3  min = 1891.51 ms   max = 1902.97 ms   avg = 1896.89 ms
--------------------------------------
         mobilenetv2  min =   66.86 ms   max =   74.51 ms   avg =   68.94 ms
--------------------------------------
         mobilenetv3  min =   55.73 ms   max =   59.72 ms   avg =   58.45 ms
--------------------------------------

```


