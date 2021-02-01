# Tengine Lite TensorRT GPU User Manual

## Brief

Todo

## How to build

### Build for Linux

On Ubuntu

### build
```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-linux-trt
$ cmake -DTENGINE_ENABLE_TENSORRT=ON \
    -DTENSORRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
    -DTENSORRT_LIBRARY_DIR=/usr/lib/aarch64-linux-gnu ..

$ make -j4
$ make install
```

## Demo

```
nvidia@xaiver:~/tengine-lite-tq/build-linux-trt$ ./tm_classification_trt -m mobilenet_v1.tmfile -i cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679 -r 10
Tengine plugin allocator TRT is registered.
tengine-lite library version: 1.2-dev
Tengine: Try using inference precision TF32 failed, rollback.

model file : /home/nvidia/tengine-test/models/mobilenet_v1.tmfile
image file : /home/nvidia/tengine-test/images/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 2.10 ms, max_time 3.10 ms, min_time 2.03 ms
--------------------------------------
8.574147, 282
7.880117, 277
7.812574, 278
7.286457, 263
6.357487, 281
--------------------------------------
```
