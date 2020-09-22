## 转换工具

### 1 源码编译

Tengine Lite 采用与 Tengine 相同的模型存储格式 tmfile，因此可以通过源码编译 Tengine 的源码生成 convert_model_to_tm。

#### 1.1 依赖工具安装

Tengine x86（ubuntu）平台版本编译依赖 `git,g++,cmake,make` 等一下基本编译依赖项，如果没有安装，可以使用 apt-get 进行安装，命令如下：

```bash
sudo apt-get install cmake make g++ git
```

#### 1.2 下载 Tengine 源码

下载 Tengine 源码：

```bash
git clone https://github.com/OAID/Tengine.git
```

#### 1.3 编译 Tengine

```
cd Tengine
mkdir build 
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/x86.gcc.toolchain.cmake ..
make -j4 && make install
```

编译完成后在 `Tengine/install` 中可找到 `convert_model_to_tm` 。



### 2 Web 转换工具

社区大佬巨作，基于 WebAssembly 技术内嵌 Tengine convert_model_to_tm 在本地浏览器，实现模型本地转换。

[convertmodel.com](https://convertmodel.com/)
