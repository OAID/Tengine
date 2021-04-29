# Tengine Lite CUDA GPU User Manual

## Brief

Todo

## How to build

### Build for Linux

On Ubuntu

### build
```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-linux-cuda
$ cmake -DTENGINE_ENABLE_CUDA=ON ..

$ make -j4
```

## Demo

```
nvidia@xaiver:~/tengine-lite-tq/build-linux-cuda$ ./tm_classification_cuda -m mobilenet_v1.tmfile -i cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679 -r 10
Tengine plugin allocator CUDA is registered.
tengine-lite library version: 1.2-dev

model file : /home/nvidia/tengine-test/models/mobilenet_v1.tmfile
image file : /home/nvidia/tengine-test/images/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 10 times, thread 1, avg time 4.58 ms, max_time 5.72 ms, min_time 4.24 ms
--------------------------------------
8.574145, 282
7.880118, 277
7.812578, 278
7.286452, 263
6.357486, 281
--------------------------------------
```
