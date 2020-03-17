我们提供2类模型示例，一类是分类网络，一类是检测网络。

#### 如何编译

这里我们假设你已经完成Tengine的编译。

如果不知道如何编译Tengine，请参考[Tengine快速上手指南](https://github.com/OAID/Tengine/wiki)。

我们提供2套编译脚本，一个用于Linux平台测试程序生成，一个用于Android平台测试程序生成。

- ###### Linux编译

1. 编辑linux_build.sh脚本，设置正确的Tengine代码路径

```
vim ./linux_build.sh

TENGINE_PATH=/path/to/tengine
```

2. 执行linux_build.sh脚本

```
chmod +x ./linux_build.sh
./linux_build.sh
```

- ###### Android编译

Android测试程序的编译是在Linux Host机上使用[NDK](https://developer.android.google.cn/ndk/guides)编译的。

1. 编辑android_build.sh脚本，设置正确的Tengine代码路径、ANDROID_NDK路径、ABI/API_LEVEL属性

```
vim ./android_build.sh


ANDROID_NDK="/path/to/android-ndk-r16"
ABI="" # 比如armeabi-v7a
API_LEVEL= # 比如22

TENGINE_PATH=/path/to/tengine
```

2. 执行android_build.sh脚本

```
chmod +x ./android_build.sh
./linux_build.sh
```

#### 模型文件获取


我们在百度网盘中提供了所有examples的原始框架模型和Tengine模型。


[获取模型文件](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2FTengine_models&parentPath=%2F) 提取码：hhgc

#### 执行样例

编译完成后，将下载的模型放在样例代码的tengine目录下，与source和data同级。


直接执行以下命令：

```
chmod +x ./tf_example.sh
./tf_example.sh
```

tf_example.sh是执行tensorflow模型转换成的Tengine模型的执行脚本。

相应的，如果想执行caffe模型转换成的Tengine模型，请执行./caffe_example.sh。
