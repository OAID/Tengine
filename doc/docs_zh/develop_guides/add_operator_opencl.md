# OpenCL 后端添加自定义 OP 指南

## 0. 简介
OpenCL(Open Computing Language) 是第一个面向异构系统通用目的并行编程的开放式、免费标准，也是一个统一的编程环境，便于软件开发人员为高性能计算服务器、桌面计算系统、手持设备编写高效轻便的代码，而且广泛适用于多核心处理器(CPU)、图形处理器(GPU)、Cell 类型架构以及数字信号处理器(DSP)等其他并行处理器，在游戏、娱乐、科研、医疗等各种领域都有广阔的发展前景。

[Tengine](https://github.com/OAID/Tengine) 已经完成 OpenCL 的支持和集成，在 ARM Mali GPU、NVIDIA GPU 上已经可以完成 Tengine 模型的 FP32 推理。关于 OpenCL 的算子支持及性能优化也在持续进行中。

Tengine 的 DAG 内存结构主要包含三个部分，graph(存储完整的模型架构)，node(存储每个 OP 节点的参数信息)，tensor(存储每个 OP 节点的输入输出张量)。
Tengine 的 OpenCL 后端主要通过 graph 对接 OpenCL 的执行队列，通过 tensor 的 `data` 对接 OpenCL 的内存分配，以及通过 node 的参数，完成 OpenCL 的 kernel 实现。大致映射关系如下图所示：

``` c
ir_graph                        >>>>>>>>>>>        OCLqueue(OpenCL的执行队列)
|
|---ir_tensor0                  >>>>>>>>>>>        cl_buffer0(输入buffer0)
|---    |       ir_tensor1      >>>>>>>>>>>            |___________cl_buffer1(输入buffer1)
|       |           |                                        |
|       |___________|                            |-----  Add OP Node 
|             |                                  |           |
|---       ir_node              >>>>>>>>>>>      |-----  Build OpenCL Kernel
|             |                                              |
|---      ir_tensor2            >>>>>>>>>>>              cl_buffer2(输出buffer)
```

为 OpenCL 后端添加新的算子主要包含以下 5 个步骤：

> 1. 添加 tensor 映射函数。
为新添加OP的输入输出分配内存。OpenCL 的内存分配方式有两种，目前 Tengine 中，通过 `OCLEngine::OCLTensorMap` 函数实现了 buffer 的分配方式。如使用 buffer 这种方式，除 winograd 等需要分配额外内存的实现，其他实现已统一分配好内存，不需要再进行额外操作。如需要使用 image 的内存分配方式，则需在 `OCLEngine::OCLTensorMap` 函数中另外添加 `create_image()` 相关实现。
> 2. 添加 node 映射函数。
为新添加 OP 的加入对应函数申明及实现，并完成 node 和 tensor 之间的关系衔接。
> 3. 完成 OpenCL 的 kernel 实现。
为 node 的映射添加完整 .cl 的 kernel 实现。
> 4. graph 映射。
完成 OpenCL 执行队列设置，及添加 OpenCL 所必须的参数设置，如 `global_work_size` 和 `local_work_size` 等。
> 5. 在 `limit.hpp` 文件中添加新增 OP 枚举。
切图机制需要获知后端 OP 支持情况，所以需要在 `limit.hpp` 文件中增加新的 OP 支持声明。


**下面将介绍新OP算子添加的具体操作细节，以下内容除单独强调外，都假设在 `<tengine-lite-root-dir>/source/device/opencl` 目录下进行**

## 1. 添加 tensor 映射

### 1.1 OpenCL 内存分配方式选择

选择对应的 OpenCL 内存分配方式，buffer 或 image。如选择 image 方式，则在 `OCLEngine::OCLTensorMap` 函数中添加对应实现，如选择 buffer 方式，这一步不需要进行额外操作。

### 1.2 OpenCL 内存分配大小设置

如果内存分配大小与 OP 输入输出 shape 一致，这一步不需要进行额外操作。如果 kernel 为类似 winograd 这样需要分配额外的内存做缓存的类型，则需要在 `OCLEngine::OCLTensorMap` 函数中添加对应内存分配实现。如果不能自动销毁，还需要注意添加析构或释放相关的实现。

## 2. 添加 node 映射

下面以 Dropout 算子为例，对流程进行说明。

### 2.1 添加 AddNode 函数申明

在 `ocl_executor.hpp` 中添加

``` c
bool AddDropoutNode(struct node* ir_node);
```

在 `ocl_executor.cc` 文件中的 `OCLEngine::BuildKernel(struct subgraph* subgraph)` 函数中添加

``` c
case OP_DROPOUT:
    this->AddDropoutNode(ir_node);
    break;
```
需要注意，`OP_DROPOUT` 是 Tengine 枚举的 OP 类型，其余枚举类型可以参考 `<tengine-lite-root-dir>/source/operator/op.h` 文件。

### 2.2 添加 OP 函数实现

在 `./op` 文件夹下添加 `ocl_droput.cc`  内容为 `AddDropoutNode(ir_node)` 的函数实现。
在 `./cl` 文件夹下添加 `drouput.cl`     内容为 OpenCL 的 kernel 实现。

其中，`AddDropoutNode(ir_node)` 函数需要包含以下内容:

#### 2.2.1 添加 .cl 文件路径

``` c
char* cl_env = getenv("ROOT_PATH");
char cl_kernel_path[500] = "";
strcat(cl_kernel_path, cl_env);
strcat(cl_kernel_path, "/source/device/opencl/cl/dropout.cl");
this->build_kernel(&cl_kernel_path[0], "dropout");
```

#### 2.2.2 添加 .cl 文件对应 kernel 的参数

``` c
int arg_idx = 0;
CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[output_tensor->index]));
CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(cl_mem), (void *)&this->ocl_tensor_map[input_tensor->index]));
CHECK_SET_KERNEL_STATUS(clSetKernelArg(kernel, ++arg_idx, sizeof(int), &output_tensor->elem_num));
```

#### 2.2.3 设置 OpenCL kernel 执行队列

``` c
struct OCLqueue Dropout;
Dropout.name = "Dropout";
Dropout.dims = 1;
Dropout.queue_kernel = this->kernel;
Dropout.queue_global_work_size = (size_t*)malloc(sizeof(size_t));
Dropout.queue_global_work_size[0] = output_tensor->elem_num;
Dropout.queue_local_work_size = (size_t*)malloc(sizeof(size_t));
Dropout.queue_local_work_size[0] =  1;
this->queue_list.push_back(Dropout);
```

## 3. 添加 OpenCL 的 kernel 实现

完成.cl后缀的kernel实现。

## 4. 添加OpenCL可支持op

在`./ocl_limit.hpp`文件中，找到对应的 OP 枚举，打开注释，标明已经实现了支持。

---

经过以上流程，一个 OP 的实现就已经完成了。接下来需要在实际 GPU 或其他 OpenCL 设备上运行评估性能，并进行适当调优，逐渐接近理论性能上限。