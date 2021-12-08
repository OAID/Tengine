# 简介

align tool是一个用来对齐onnx和tengine每一层输出结果的工具。



# 安装

- 安装依赖库

  ```bash
  cd Tengine/tools/align_tool
  pip install -r requirements.txt
  ```

- 安装pytengine

  首先参照[源码编译（Linux） — Tengine 文档 (tengine-docs.readthedocs.io)](https://tengine-docs.readthedocs.io/zh_CN/latest/source_compile/compile_linux.html) 编译，生成libtengine-lite.so.

  然后执行下列命令：

  ```bash
  sudo apt install python3-opencv
  cd Tengine/pytengine
  python3 setup.py install
  ```

  如果安装失败，请看[Tengine/pytengine](https://github.com/OAID/Tengine/tree/tengine-lite/pytengine)



# 使用示例

仓库里提供了mnist.onnx和mnist.tmfile，输入：

`python align_with_onnx.py --m mnist.onnx  --tm mnist.tmfile --a --s`

会得到结果如下：

```bash
---- align tools: tengine and onnx ----

input onnx model      : mnist_sim.onnx
is save result        : True
save text path        : ./output_onnx
is align by layer     : True
tengine output path   : None
tengine model         : mnist.tmfile
onnx inference over
-------------------------------------------------
export onnx output to text , please wait a moment
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 1695.29it/s]
onnx export ok
-------------------------------------------------
lib_path= ['/home/tool/align_onnx/Tengine/pytengine/tengine/libtengine-lite.so']
self.dtype.enum: 0
tengine inference over
-------------------------------------------------
TEXT                                              L1 DISTANCE         L2 DISTANCE         
Conv_0_in_blob_data.txt                           0.0019135           1.00000000
Conv_0_out_blob_data.txt                          1581.5              0.80794900
Conv_2_in_blob_data.txt                           0.00088148          1.00000000
Conv_2_out_blob_data.txt                          4152.6              0.79788300
Conv_4_in_blob_data.txt                           0.00087139          1.00000000
Conv_4_out_blob_data.txt                          1.4426e+04          0.31296200
MaxPool_6_in_blob_data.txt                        0.00013564          1.00000000
MaxPool_6_out_blob_data.txt                       5.277e-05           1.00000000
Reshape_8_in_blob_data.txt                        5.277e-05           1.00000000
Reshape_8_out_blob_data.txt                       5.277e-05           1.00000000
Gemm_9_in_blob_data.txt                           5.277e-05           1.00000000
Gemm_9_out_blob_data.txt                          0.00045572          1.00000000
Relu_10_in_blob_data.txt                          0.00045572          1.00000000
Relu_10_out_blob_data.txt                         0.00033022          1.00000000
Gemm_11_in_blob_data.txt                          0.00033022          1.00000000
Gemm_11_out_blob_data.txt                         4.0812e-05          1.00000000
```

在执行目录下会生成output和output_onnx文件夹，里面包含了每一层输入和输出的结果，在output_onnx文件夹里，compare.txt中保存了逐层对比的结果。



