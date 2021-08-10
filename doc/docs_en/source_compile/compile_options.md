# Compilation Option Description

## Basic Options

| Compilation Options       | Description                                                  | Default |
| ------------------------- | ------------------------------------------------------------ | ------- |
| TENGINE_ENABLE_ALL_SYMBOL | Open all symbols at compile time                             | ON      |
| TENGINE_OPENMP            | Enable OpenMP parallel units at compile time                 | ON      |
| TENGINE_BUILD_BENCHMARK   | Compile performance test module                              | ON      |
| TENGINE_BUILD_EXAMPLES    | Compile the Example module                                   | ON      |
| TENGINE_BUILD_TESTS       | Compile unit test module                                     | OFF     |
| TENGINE_COVERAGE          | Enable code coverage testing at compile time                 | OFF     |
| TENGINE_BUILD_CPP_API     | Enable C++ API at compile time                               | OFF     |
| TENGINE_DEBUG_DATA        | Enable debugging options at compile time, and extract data   | OFF     |
| TENGINE_DEBUG_TIME        | Enable debugging option at compile time, single-layer time-consuming analysis | OFF     |
| TENGINE_DEBUG_MEM_STAT    | Enable debugging options at compile time, and analyze memory status | OFF     |
| TENGINE_ARCH_ARM_82       | Enable ARMv8.2 instructions of arm architecture at compile time | OFF     |

## HCL Options

| Compilation Options              | Description                                                  | Default |
| -------------------------------- | ------------------------------------------------------------ | ------- |
| TENGINE_STANDALONE_HCL           | Generate HCL library separately at compile time              | OFF     |
| TENGINE_STANDALONE_HCL_AUTO_LOAD | Specifies that the HCL library is automatically loaded at compile time | ON      |

## Plugin Options

| Compilation Options        | Description              | Default |
| -------------------------- | ------------------------ | ------- |
| TENGINE_ENABLE_ACL         | Compile ACL plugin       | OFF     |
| TENGINE_ENABLE_VULKAN      | Compile Vulkan plugin    | OFF     |
| TENGINE_ENABLE_TENSORRT    | Compile TensorRT  plugin | OFF     |
| TENGINE_ENABLE_CUDABACKEND | Compile CUDA  plugin     | OFF     |
| TENGINE_ENABLE_OPENCL      | Compile OpenCL  plugin   | OFF     |
| TENGINE_ENABLE_TIM_VX      | Compile TIM-VX  plugin   | OFF     |
| TENGINE_ENABLE_NNIE        | Compile NNIE  plugin     | OFF     |
