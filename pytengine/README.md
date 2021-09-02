### 编译Pytengine ubuntu 18.04
Linux jx 5.4.0-73-generic #82~18.04.1-Ubuntu SMP Fri Apr 16 15:10:02 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux

#### 1.安装依赖

~~~
pip3 install numpy
sudo apt install python3-opencv
~~~

#### 2. 安装pytengine

~~~
cd <tengine-lite>/pytengine/
sudo python3 setup.py install
~~~
If the directory does not exist, create it and try again. Note: python3.X must be your own python3 version.
~~~
cd <tengine-lite>/pytengine/
sudo mkdir -p /usr/local/lib/python3.X/{dist-packages,site-packages}
sudo python3 setup.py install
~~~

If the 'libtengine-lite.so' file does not found. Note:After [compilation](https://tengine-docs.readthedocs.io/en/latest/source_compile/compile_linux.html), the build/install/lib directory will generate `libtengine-lite.so` as shown below:

```bash
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