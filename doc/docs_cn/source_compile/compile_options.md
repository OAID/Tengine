# 编译选项说明

## 基础选项

| 编译选项                  | 说明                               | 默认值 |
| ------------------------- | ---------------------------------- | ------ |
| TENGINE_ENABLE_ALL_SYMBOL | 编译时是否打开所有符号             | ON     |
| TENGINE_OPENMP            | 编译时启用 OpenMP 并行单元         | ON     |
| TENGINE_BUILD_BENCHMARK   | 编译性能测试模块                   | ON     |
| TENGINE_BUILD_EXAMPLES    | 编译 Example 模块                  | ON     |
| TENGINE_BUILD_TESTS       | 编译单元测试模块                   | OFF    |
| TENGINE_COVERAGE          | 编译时启用代码覆盖率测试功能       | OFF    |
| TENGINE_BUILD_CPP_API     | 编译时启用 C++ API                 | OFF    |
| TENGINE_DEBUG_DATA        | 编译时启用调试选项，数据提取       | OFF    |
| TENGINE_DEBUG_TIME        | 编译时启用调试选项，单层耗时分析   | OFF    |
| TENGINE_DEBUG_MEM_STAT    | 编译时启用调试选项，内存状态分析   | OFF    |
| TENGINE_ARCH_ARM_82       | 编译时启用 ARM 架构的 armv8.2 指令 | OFF    |

## HCL 选项

| 编译选项                         | 说明                      | 默认值 |
| -------------------------------- | ------------------------- | ------ |
| TENGINE_STANDALONE_HCL           | 编译时单独生成 HCL 库     | OFF    |
| TENGINE_STANDALONE_HCL_AUTO_LOAD | 编译时指定 HCL 库自动加载 | ON     |

## 插件选项

| 编译选项                   | 说明               | 默认值 |
| -------------------------- | ------------------ | ------ |
| TENGINE_ENABLE_ACL         | 编译 ACL 插件      | OFF    |
| TENGINE_ENABLE_VULKAN      | 编译 Vulkan插件    | OFF    |
| TENGINE_ENABLE_TENSORRT    | 编译 TensorRT 插件 | OFF    |
| TENGINE_ENABLE_CUDABACKEND | 编译 CUDA 插件     | OFF    |
| TENGINE_ENABLE_OPENCL      | 编译 OpenCL 插件   | OFF    |
| TENGINE_ENABLE_TIM_VX      | 编译 TIM-VX 插件   | OFF    |
| TENGINE_ENABLE_NNIE        | 编译 NNIE 插件     | OFF    |
