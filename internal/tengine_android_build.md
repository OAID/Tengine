
1. 从ftp下载ndk
   ftp://ftp.openailab.net/tengine/android-ndk-r16-linux-x86_64.
   
   解压为你的路径
   `/home/linaro/android-ndk-r16`

2. tengine
    
   ```
   git clone tengine
   git checkout dev/haitao
   ```
3. 修改配置为armv7
  - 修改`tengine/android_build.sh`
    - line6:    `-DANDROID_ABI="armeabi-v7a"`
    - line3: `export ANDROID_NDK=/home/linaro/android/android-ndk-r16`
  - 修改`CMakeLists.txt`,line24-25
    ```
    option(CONFIG_ARCH_ARM64 "build arm64 version" OFF)
    option(CONFIG_ARCH_ARM32 "build arm32 version" ON)
    ```


4. cmake版本>3/6
  我的是apt-get 安装的cmake 3.5, 要求至少3.6版本
  - 卸载旧版本 `sudo apt-get autoremove cmake`
  - 安装新版本
    ```
    wget  http://www.cmake.org/files/v3.7/cmake-3.7.2.tar.gz
    tar xf cmake-7.7.2.tar.gz
    cd cmake-3.7.2
    ./bootstrap
    make
    sudo make install
    ```
  - 检查版本 `cmake --version`

5. protobuf的问题
  - 查看当前版本 `protoc --version`,是2.6.0

  - 卸载
  `sudo apt-get autoremove libprotobuf-dev protobuf-compiler`
  
  - 源码安装
  https://github.com/google/protobuf/tree/master/src
  按照readme安装 `sudo apt-get install autoconf automake libtool`

  解压protobuf-3.0.0.zip
  ```
  cd protobuf-3.0.0
  ./autogen.sh
  ./configure
  make
  make install
  ```
  - 添加路径
  在`~/.profil`中添加`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`,保存
  - 执行 `source ~/.profile`
  - 检查protoc版本 `protoc --version`，是3.0.0
  

6. 安卓protobuf
  可能遇到的问题·安装安卓protobuf就好
      ```
    fatal error: 'google/protobuf/io/coded_stream.h' file not found
    #include <google/protobuf/io/coded_stream.h>
    ```
  - 解压`proto.tgz` 为`protobuf_lib`
  - 移动到tengine目录下 `mv protobuf_lib ~/tengine/`



7. build
   ```
   mkdir build
   cd build
   ../android_build.sh
   make
  
8. window安装adb,adb_driver
9. agb push
   需要push的有
   - etc/config
   - tengine.so
   - protobuf.so
   - model
   - test_bin
   把这些文件统一放到一个文件夹 test，一起push到android板子上
   ```
   gdb push tengine/test /data/local/tmp/
   ```
10. 执行adb shell
   ```
   cd /data/local/tmp
   export LD_LIBRARY_PATH=.
   ./test_bin 
   ```