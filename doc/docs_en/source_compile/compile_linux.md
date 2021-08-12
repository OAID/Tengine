# Source Code Compilation (Linux)

## Local Compilation

### Download Tengine Lite Source Code

Download Tengine Lite source code，which is located on the branch of Tengine-lite：

```bash
git clone -b tengine-lite https://github.com/OAID/Tengine.git  Tengine-Lite
```

### Compile Tengine Lite

```bash
cd Tengine-Lite
mkdir build 
cd build
cmake ..
make
make install
```

After compilation, the build/install/lib directory will generate `libtengine-lite.so` as shown below:

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

## Cross-compiling Arm32/64 Linux

### Download Source Code

```bash
git clone -b tengine-lite https://github.com/OAID/Tengine.git  Tengine-Lite
```

### Install the Cross-compilation Tool Chain

For Arm64 Linux：

```bash
sudo apt install g++-aarch64-linux-gnu
```

For Arm32 Linux：

```bash
sudo apt install g++-arm-linux-gnueabihf
```

### Complie Tengine Lite

Arm64 Linux Cross Compilation

```bash
cd Tengine-Lite
mkdir build 
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make
make install
```

Arm32 Linux Cross Compilation

```bash
cd Tengine-Lite
mkdir build 
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake ..
make
make install
```

After compilation,  `libtengine-lite.so` file will be generated. Related header files、`libtengine-lite.so`  and related test programs will be copied to `build/install` directory.

