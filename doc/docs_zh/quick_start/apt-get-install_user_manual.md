
# apt-get 安装和使用Tengine 

### 功能说明 
   基于标准的ubuntu系统，Tengine提供了命令行的安装方式，通过apt-get命令，可下载Tengine库，并且同步提供相应的预置Demo，欢迎大家试用。
	

 **以下所有命令均在设备的命令行运行**
 
| 硬件     | 系统 | 
| ------- | ------- |
|   x86   |  ubuntu18.04     |
|   A311D |  ubuntu18.04     |


## 1. 客户端配置APT源 

### X86配置源命令:

```
echo deb [trusted=yes] http://58.33.201.109:8099/x86/ focal main | sudo tee -a /etc/apt/sources.list
```

### A311D板子配置源命令：

```
echo deb [trusted=yes] http://58.33.201.109:8099/a311d/ focal main | sudo tee -a /etc/apt/sources.list
```

## 2. 更新源

```
sudo apt-get update

```


## 3. 客户端下载tengine

```
sudo apt-get install tengine
```

## 4. 下载的tengine目录结构

###  X86:
```
/usr/
├── bin
│   ├── convert_tool
│   ├── quant_tool_uint8
│   ├── tm_classification
│   └── tm_yolov5s
├── include
│   └── tengine
│       ├── c_api.h
│       └── defines.h
├── lib
│   └── libtengine-lite.so
└── share
    ├── cat.jpg
    ├── mobilenet.tmfile
    ├── ssd_dog.jpg
    └── yolov5s.tmfile
```

###  A311D:

```
/usr/
├── bin
│   ├── tm_classification
│   ├── tm_yolov5s
│   └── tm_yolov5s_timvx
├── include
│   └── tengine
│       ├── c_api.h
│       └── defines.h
├── lib
│   └── libtengine-lite.so
└── share
    ├── cat.jpg
    ├── mobilenet.tmfile
    ├── ssd_dog.jpg
    ├── yolov5s.tmfile
    └── yolov5s_uint8.tmfile
```

## 5.Demo运行示例

###  X86平台

**分类网络示例**
```
peter@test-server:/usr/bin$ ./tm_classification -m /usr/share/mobilenet.tmfile -i /usr/share/cat.jpg 
Image height not specified, use default 224
Image width not specified, use default  224
Scale value not specified, use default  0.0, 0.0, 0.0
Mean value not specified, use default   104.0, 116.7, 122.7
tengine-lite library version: 1.5-dev

model file : /usr/share/mobilenet.tmfile
image file : /usr/share/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 23.36 ms, max_time 23.36 ms, min_time 23.36 ms
--------------------------------------
8.574143, 282
7.880117, 277
7.812573, 278
7.286458, 263
6.357487, 281
--------------------------------------
```

**检测网络示例**
```
peter@test-server:/usr/bin$ ./tm_yolov5s -m /usr/share/yolov5s.tmfile -i /usr/share/ssd_dog.jpg 
tengine-lite library version: 1.5-dev
Repeat 1 times, thread 1, avg time 306.23 ms, max_time 306.23 ms, min_time 306.23 ms
--------------------------------------
detection num: 3
16:  89%, [ 135,  218,  313,  558], dog
 7:  86%, [ 472,   78,  689,  169], truck
 1:  75%, [ 124,  107,  578,  449], bicycle
--------------------------------------
```



### A311D

**分类网络示例-cpu**

```
cd /usr/bin/
khadas@Khadas:/usr/bin$ ./tm_classification -m /usr/share/mobilenet.tmfile -i /usr/share/cat.jpg 
Image height not specified, use default 224
Image width not specified, use default  224
Scale value not specified, use default  0.0, 0.0, 0.0
Mean value not specified, use default   104.0, 116.7, 122.7
tengine-lite library version: 1.5-dev

model file : /usr/share/mobilenet.tmfile
image file : /usr/share/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 104.37 ms, max_time 104.37 ms, min_time 104.37 ms
--------------------------------------
8.574153, 282
7.880111, 277
7.812575, 278
7.286450, 263
6.357493, 281
--------------------------------------
```

**检测网络示例-CPU**
```
khadas@Khadas:/usr/bin$ ./tm_yolov5s -m /usr/share/yolov5s.tmfile -i /usr/share/ssd_dog.jpg 
tengine-lite library version: 1.5-dev
Repeat 1 times, thread 1, avg time 1369.74 ms, max_time 1369.74 ms, min_time 1369.74 ms
--------------------------------------
detection num: 3
16:  89%, [ 135,  218,  313,  558], dog
 7:  86%, [ 472,   78,  689,  169], truck
 1:  75%, [ 123,  107,  578,  449], bicycle
--------------------------------------
```

**检测网络示例-NPU**
```
khadas@Khadas:/usr/bin$ ./tm_yolov5s_timvx -m /usr/share/yolov5s_uint8.tmfile -i /usr/share/ssd_dog.jpg 
Please make sure Galcore Sdk version > 6.4.4,Please refer to https://github.com/OAID/Tengine/blob/tengine-lite/doc/npu_tim-vx_user_manual_zh.md for more information
tengine-lite library version: 1.5-dev
Repeat 1 times, thread 1, avg time 68.33 ms, max_time 68.33 ms, min_time 68.33 ms
--------------------------------------
detection num: 3
16:  89%, [ 136,  224,  313,  550], dog
 7:  87%, [ 473,   72,  692,  171], truck
 1:  75%, [ 129,  108,  578,  443], bicycle
--------------------------------------
```
