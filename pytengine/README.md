### 编译Pytengine ubuntu 18.04
Linux jx 5.4.0-73-generic #82~18.04.1-Ubuntu SMP Fri Apr 16 15:10:02 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux

#### 1.安装依赖

~~~
pip3 install numpy
sudo apt install python3-opencv
~~~

#### 2. 安装pytengine

~~~
sudo python3 setup.py install
~~~
If the directory dose not exist, create it and try again. Note: python3.X must be your won python3 version.
~~~
sudo mkdir -p /usr/local/lib/python3.X/{dist-packages,site-packages}
sudo python3 setup.py install
~~~