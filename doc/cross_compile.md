# Note for cross compiling

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](../LICENSE)

<br \>
## 1. install the arm64 rootfs

```cd sysroot
make```

## 2. install the aarch64 cross toolchains

```sudo apt install g++-aarch64-linux-gnu``` 

## 3. install protobuf

```sudo apt install libprotobuf-dev```
    
## 4. Compile protobuf from source for cross_compiler 

Download protobuf from [github.com/google/Protobuf](https://github.com/google/protobuf/releases)

NOTE：The vesion must be same as host.

```sudo apt-get install autoconf automake libtool 
cd protobuf
./autogen.sh
./configure --with-protoc=protoc --host=aarch64-linux-gnu --prefix=/opt/protobuf_3.3.0
make 
sudo make install ```

>copy the libraries and headers into arm64 rootfs

``` cp /opt/protobuf_3.3.0/lib/libprotobuf.so Tengine_DIR/sysroot/ubuntu_rootfs/usr/lib/
 cp -r /opt/protobuf_3.3.0/include/google Tengine_DIR/sysroot/ubuntu_rootfs/usr/include```

## 5. un-comment the CROSS_COMPILE  in Makefile

```CROSS_COMPILE=aarch64-linux-gnu- ```


## 6. build as normal



