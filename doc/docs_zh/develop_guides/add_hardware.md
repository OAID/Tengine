# 扩展硬件后端

------

## 背景知识
**Tengine** 在设计上将可扩展性作为第一优先级纳入考量，较早的版本注册机制依赖 `GCC GNU` 扩展，而 `GNU` 扩展并不是标准 `C` 的内容。当社区呼唤需要扩展支持到 `Microsoft Visual Studio` 上时，遇到了较多的困难。
在决定重新设计后，注册模块的易用性有了很大的提升。新的机制通过 `CMake` 额外的处理过程，取得类似遍历和注册的效果，完成模块的注册。具体的设计和改进可以参考**架构详解**中的**重要模块介绍**。

**Tengine** 在设计上将所有可以运行 `CNN` 的硬件单元均视为设备，`CPU` 就是一个典型的设备，在所有的编译选项里，`CPU` 设备都是默认包含的。如果描述一个新设备并注册，通常意义上这潜在上意味着要求编译的 **Tengine** 支持异构设备切图(相关内容可以阅读**混合设备**部分)；如果注册的设备也描述了**混合精度**的接口，那么设备还支持**混合精度**。
**Tengine** 通过一个嵌套的结构体完成一个设备的描述：
``` C
/*!
 * @struct nn_device_t
 * @brief  Abstract neural network runnable device description struct
 */
typedef struct device
{
    const char* name;
    struct interface* interface;      //!< device scheduler operation interface
    struct allocator* allocator;      //!< device allocation operation interface
    struct optimizer* optimizer;      //!< device optimizer operation interface
    struct scheduler* scheduler;      //!< device scheduler
    void*  privacy;                   //!< device privacy data
} ir_device_t;
```
从结构体 `ir_device_t` 上可以看出，设计上将一个设备(`device`)分成 6 部分，第一部分 `name` 描述了设备的名字，设备名字不允许重复；`interface` 描述了设备接口；`allocator`描述了设备相关子图的操作；`optimizer` 描述了切图和混合精度的接口；`scheduler` 描述了设备独特的调度接口。
以上接口通常不需要全部填充，**Tengine** 提供一组丰富的示例指导如何自定义并添加用户自己的设备。

-----------------------------------------------

## 添加自定义设备
### 创建目录，编写 CMakeLists 文件
首先在`source/device`创建一个以用户设备命名的文件夹，文件夹可以是用户的设备缩写或其他用户认为比较酷的名字(这里假设起名为`TPU`，那么目录就是`source/device/tpu`)，并从其他已经实现的 `device/xxx` 目录中复制一份 `CMakeLists.txt` 文件到当前文件夹；现在只需要对此 `CMakeLists.txt` 做些微的修改，而不需要从头创建。我们以从 `source/device/acl/CMakeLists.txt` 复制一份为例进行说明。该文件完整示例如下：
``` cmake
# 0. clear var
UNSET (_DEV_ACL_HEADER_PATH)
UNSET (_ACL_BASE_SOURCE)
UNSET (_ACL_OPS_SOURCE)
UNSET (_DEV_ACL_DEVICE_SOURCE)
UNSET (_DEV_ACL_COMPILER_DEFINES)
UNSET (_DEV_ACL_COMPILER_OPTIONS)
UNSET (_DEV_ACL_LINKER_OPTIONS)
UNSET (_DEV_ACL_LINK_LIBRARIES)


# 1.  set source root path
SET(_ACL_ROOT ${CMAKE_SOURCE_DIR}/source/device/acl)


# 2.  add header file path
LIST (APPEND _DEV_ACL_HEADER_PATH      ${_ACL_ROOT})
LIST (APPEND _DEV_ACL_HEADER_PATH      ${CMAKE_SOURCE_DIR}/3rdparty/acl/include)


# 3.  add linking lib searching path
LIST (APPEND _DEV_ACL_LINK_PATH        ${CMAKE_SOURCE_DIR}/3rdparty/acl/lib)


# 4.  add source files
AUX_SOURCE_DIRECTORY("${_ACL_ROOT}"    _ACL_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_ACL_ROOT}/op" _ACL_OPS_SOURCE)
LIST (APPEND _DEV_ACL_DEVICE_SOURCE    ${_ACL_BASE_SOURCE})
LIST (APPEND _DEV_ACL_DEVICE_SOURCE    ${_ACL_OPS_SOURCE})


# 5.  add build options for cpu device
# 5.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
    IF (TENGINE_COMPILER_GCC AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "6.1"))
        LIST (APPEND _DEV_ACL_COMPILER_OPTIONS -Wno-ignored-attributes)
    ENDIF()
ENDIF()


# 5.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 6.  add link options


# 7.  add link libs
LIST (APPEND _DEV_ACL_LINK_LIBRARIES   arm_compute)
LIST (APPEND _DEV_ACL_LINK_LIBRARIES   arm_compute_core)


# 8. set all to cmake cache
SET (TENGINE_ACL_HEADER_PATH       ${_DEV_ACL_HEADER_PATH}        CACHE INTERNAL  "Tengine Arm Compute Library device header files searching path"   FORCE)
SET (TENGINE_ACL_LINK_PATH         ${_DEV_ACL_LINK_PATH}          CACHE INTERNAL  "Tengine Arm Compute Library device link libraries searching path" FORCE)
SET (TENGINE_ACL_DEVICE_SOURCE     ${_DEV_ACL_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine Arm Compute Library device main source files"             FORCE)
SET (TENGINE_ACL_COMPILER_DEFINES  ${_DEV_ACL_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine Arm Compute Library about compiler defines"               FORCE)
SET (TENGINE_ACL_COMPILER_OPTIONS  ${_DEV_ACL_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine Arm Compute Library about compiler options"               FORCE)
SET (TENGINE_ACL_LINKER_OPTIONS    ${_DEV_ACL_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine Arm Compute Library about linker options"                 FORCE)
SET (TENGINE_ACL_LINK_LIBRARIES    ${_DEV_ACL_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine Arm Compute Library about link libraries"                 FORCE)


# 9. install device option
INSTALL (FILES ${_ACL_ROOT}/acl_define.h DESTINATION include/tengine RENAME acl_device.h)

```
首先需要将使用的 `CMake` 变量的前缀进行修改，以避免潜在的变量冲突；将所有的 `ACL` 替换为`TPU`；然后修改模块的搜索根路径 `_TPU_ROOT` 为  `source/device/tpu`。
``` cmake
# 1.  set source root path
SET(_TPU_ROOT ${CMAKE_SOURCE_DIR}/source/device/tpu)
```
自定义设备常常需要一些额外的`3rdparty`依赖，在 `_DEV_TPU_HEADER_PATH` 和 `_DEV_TPU_LINK_PATH` 中进行相应的修改；在 `ACL` 中，增加了 `ACL` 预编译库路径 `${CMAKE_SOURCE_DIR}/3rdparty/acl/lib`。
``` cmake
# 2.  add header file path
LIST (APPEND _DEV_TPU_HEADER_PATH      ${_TPU_ROOT})
LIST (APPEND _DEV_TPU_HEADER_PATH      ${CMAKE_SOURCE_DIR}/3rdparty/tpu/include)


# 3.  add linking lib searching path
LIST (APPEND _DEV_TPU_LINK_PATH        ${CMAKE_SOURCE_DIR}/3rdparty/tpu/lib)
```
源码搜集部分按实际情况修改即可。
``` cmake
# 4.  add source files
AUX_SOURCE_DIRECTORY("${_TPU_ROOT}"    _TPU_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_TPU_ROOT}/op" _TPU_OPS_SOURCE)
LIST (APPEND _DEV_TPU_DEVICE_SOURCE    ${_TPU_BASE_SOURCE})
LIST (APPEND _DEV_TPU_DEVICE_SOURCE    ${_TPU_OPS_SOURCE})
```
接下来的部分是编译相关的选项，根据实际情况修改即可。**Tengine** 默认打开了 C/C++ 支持，并尝试打开标准到 `C99/C++14`，如果工具链不支持会降级为 `C98/C++11`；如果用户的代码有其他特殊要求可以根据情况调整 `_DEV_TPU_COMPILER_DEFINES`，`_DEV_TPU_COMPILER_OPTIONS`,`_DEV_TPU_LINKER_OPTIONS` 这 3 个变量。
``` cmake
# 5.  add build options for cpu device
# 5.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
ENDIF()


# 5.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 6.  add link options
```
根据实际情况调整链接库情况，修改 `_DEV_TPU_LINK_LIBRARIES` 变量。
``` cmake
# 7.  add link libs
LIST (APPEND _DEV_TPU_LINK_LIBRARIES   tpu_runtime)
```
汇总一下临时变量到模块接口变量，接口变量设计为 `cache` 的，以便跨模块进行传递(另一方面这也是不同 `device` 不应重名的原因)。
``` cmake
SET (TENGINE_TPU_HEADER_PATH       ${_DEV_TPU_HEADER_PATH}        CACHE INTERNAL  "Tengine TPU device header files searching path"   FORCE)
SET (TENGINE_TPU_LINK_PATH         ${_DEV_TPU_LINK_PATH}          CACHE INTERNAL  "Tengine TPU device link libraries searching path" FORCE)
SET (TENGINE_TPU_DEVICE_SOURCE     ${_DEV_TPU_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine TPU device main source files"             FORCE)
SET (TENGINE_TPU_COMPILER_DEFINES  ${_DEV_TPU_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine TPU about compiler defines"               FORCE)
SET (TENGINE_TPU_COMPILER_OPTIONS  ${_DEV_TPU_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine TPU about compiler options"               FORCE)
SET (TENGINE_TPU_LINKER_OPTIONS    ${_DEV_TPU_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine TPU about linker options"                 FORCE)
SET (TENGINE_TPU_LINK_LIBRARIES    ${_DEV_TPU_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine TPU about link libraries"                 FORCE)
```
如果设备有特殊选项，可以考虑将其插入到 `install` 阶段。
``` cmake
# 9. install device option
INSTALL (FILES ${_TPU_ROOT}/tpu_define.h DESTINATION include/tengine RENAME tpu_device.h)
```
在根目录下的 `CMakeLists.txt` 中添加 `option` 以便编译时条件打开。
``` cmake
OPTION (TENGINE_ENABLE_TPU "With Awesome TPU support" OFF)
```
还需要修改 `source/device/CMakeLists.txt` 添加 `Option` 相关的处理。
``` cmake
# Awesome TPU
IF (TENGINE_ENABLE_TPU)
    ADD_SUBDIRECTORY (tpu)

    LIST (APPEND _TENGINE_DEVICE_HEADER_PATH        ${TENGINE_TPU_HEADER_PATH})
    LIST (APPEND _TENGINE_DEVICE_LINK_PATH          ${TENGINE_TPU_LINK_PATH})
    LIST (APPEND _TENGINE_DEVICE_COMPILER_DEFINES   ${TENGINE_TPU_COMPILER_DEFINES})
    LIST (APPEND _TENGINE_DEVICE_COMPILER_OPTIONS   ${TENGINE_TPU_COMPILER_OPTIONS})
    LIST (APPEND _TENGINE_DEVICE_LINKER_OPTIONS     ${TENGINE_TPU_LINKER_OPTIONS})
    LIST (APPEND _TENGINE_DEVICE_LINK_LIBRARIES     ${TENGINE_TPU_LINK_LIBRARIES})
    LIST (APPEND _TENGINE_DEVICE_SOURCE             ${TENGINE_TPU_DEVICE_SOURCE})
    LIST (APPEND _REGISTER_DEVICE_LIST              "${CMAKE_SOURCE_DIR}/source/device/tpu/tpu_device.cc")
ENDIF()
```
其中，`_REGISTER_DEVICE_LIST` 是设备注册的核心文件，需要根据实际情况进行填写。

### 填充结构体完成设备的注册
从某种意义上说，完成一个新设备的注册，只需要填充 `ir_device_t` 结构体，所有的其他代码工作都是围绕这个核心展开的。
``` c
/*!
 * @struct nn_device_t
 * @brief  Abstract neural network runnable device description struct
 */
typedef struct device
{
    const char* name;
    struct interface* interface;      //!< device scheduler operation interface
    struct allocator* allocator;      //!< device allocation operation interface
    struct optimizer* optimizer;      //!< device optimizer operation interface
    struct scheduler* scheduler;      //!< device scheduler
    void*  privacy;                   //!< device privacy data
} ir_device_t;
```

回顾 `ir_device_t` 结构体，`struct interface` 结构体描述了基本 `API` 接口：
``` c
/*!
 * @struct ir_interface_t
 * @brief  Abstract neural network runnable device interface struct
 */
typedef struct interface
{
    //!< interface of init this neural network device
    int (*init)(struct device* device);

    //!< interface of prepare runnable subgraph on device
    int (*pre_run)(struct device* device, struct subgraph* subgraph, void* options);

    //!< interface of run runnable subgraph on device
    int (*run)(struct device* device, struct subgraph* subgraph);

    //!< interface of post run runnable subgraph on device
    int (*post_run)(struct device* device, struct subgraph* subgraph);

    //!< interface of async run runnable subgraph on device
    int (*async_run)(struct device* device, struct subgraph* subgraph);

    //!< interface of async wait runnable subgraph on device
    int (*async_wait)(struct device* device, struct subgraph* subgraph, int try_wait);

    //!< interface of release runnable subgraph on device
    int (*release_graph)(struct device* device, void* device_graph);

    //!< interface of release this neural network device
    int (*release_device)(struct device* device);
} ir_interface_t;
```
参考 `ACL`，一个可能的 `TPU` 的实现填充如下：
``` c
static struct interface tpu_interface = {
        .init           = tpu_dev_init,
        .pre_run        = tpu_dev_prerun,
        .run            = tpu_dev_run,
        .post_run       = tpu_dev_postrun,
        .async_run      = nullptr,
        .async_wait     = nullptr,
        .release_graph  = nullptr,
        .release_device = tpu_dev_release,
};
```
`tpu_dev_init()` 是设备的全局初始化函数，注册设备时调用一次，反注册调用 `release_device()`。这个函数一般用来预申请设备内存作全局缓存，与设备驱动互操作初始化一些寄存器等。
`tpu_dev_prerun()` 是网络预处理部分，常见的处理包含申请 `tensor` 内存、转换数据 `layout`、创建设备运行图、编译设备 `kernel` 等。这部分申请的空间等需要在 `tpu_release_graph()` 中进行清理。
`tpu_post_run()` 与 `tpu_release_graph()` 可能会引发混淆，`tpu_post_run()` 常常用来只是清除运行一次的相关状态，与 `tpu_dev_prerun()` 相反，真正的释放工作可以放到 `tpu_release_graph()` 中进行。一个可能的场景是，运行一次分辨率的模型 `tpu_dev_prerun()` 后，换一个分辨率前运行 `tpu_post_run()`，然后再运行 `tpu_dev_prerun()`。当需要真正销毁时，运行 `tpu_release_graph()`。

`ir_device_t` 结构体中，`struct interface` 结构体描述了基本 `API` 接口， `struct allocator` 描述了设备能力上报接口和评估和调度的接口，`struct optimizer` 描述了切图和优化相关的接口，`struct scheduler` 描述了调度相关的接口。这几个接口的核心是 `struct scheduler`，设备并不总假设实现一个 `struct scheduler`，如果设备的这个接口描述是 `nullptr`，那么引擎会使用默认注册的 **`sync scheduler`** 运行网络，详情参考 `source/scheduler/scheduler.c` 中的 `static ir_scheduler_t sync_scheduler`。用户也可以实现一份自己的 `struct scheduler` 来完成特殊的任务；结合 `struct allocator` 和 `struct optimizer` 可以产生丰富的可能。下面的描述是假设用户不实现 `struct scheduler` 的情况下的逻辑。


``` cmake
static struct allocator tpu_allocator = {
        .describe       = tpu_describe,
        .evaluation     = tpu_evaluation,
        .allocate       = tpu_allocate,
        .release        = tpu_release,
};
```
在 `tpu_allocator` 中，`tpu_describe()` 上报模型的 `OP` 支持情况和精度支持情况，这里的 `OP` 和精度支持的描述并不会随网络变化而改变，潜在的含义是这种状态下总是假设用户特定设备是 `OP` 或精度 全场景支持的。以卷积为例，这意味着用户的设备支持所有模式的卷积，无论 `pad`、`stride` 、`h` 、`w` 、`c` 情况如何。如果设备实现确实需要在运行时评估，那么可以自定 `struct scheduler` 完成自定义过程。
`tpu_evaluation()` 用来运行前评估一下已经实现的设备子图是否可运行；这在需要编译 `kernel` 时特别有用。
`tpu_allocate()` 用来支持设备存储池的相关内容，在默认 `scheduler` 下无需填充这个入口。`tpu_release()` 是相反的释放过程。


``` cmake
static struct optimizer tpu_optimizer = {
        .split_graph    = tpu_split_graph,
        .optimize_graph = nullptr,
};
```
在 `tpu_optimizer` 结构体中，`tpu_split_graph()` 用来实现切图，`tpu_optimize_graph()` 用来实现混合精度，其中 `tpu_split_graph()` 可以调用默认实现的 `split_graph_node_to_sub_graph()` 进行普通切图；如果有特殊需求可以结合其他结构体的不同字段形成组合。

最后，需要编写注册函数和反注册函数 `int register_tpu_device()` 和 `int unregister_tpu_device()`，需要注意的是注册函数和反注册函数的后半段就是文件名，需要和实际文件名匹配，CMake 会自动的完成注册函数的调用过程的链接。

## 总结
通过上文的描述，可以知道添加一个自定义设备的核心工作就是填充 `ir_device_t` 结构体，描述完成后，设备注册的所有工作就完成了。模块化的 `device` 使得 **Tengine** 非常易于扩展，并有足够的灵活性。

## 彩蛋
`init_tengine(void)` 函数中，当 `operator prototype` 完成注册后，注册的就是 `serializer` 和 `devices`，但在静态代码状态下函数并不会跳转，用户可以安装一款集成开发环境，比如 `Microsoft Visual Studio` 或 `JetBrains CLion`，打开文件夹后生成 `CMake` 过程后即可进行跳转。
