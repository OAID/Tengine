# Extend the hardware backend

------

## background knowledge
**Tengine Lite** is designed to consider scalability as the first priority. The earlier version registration mechanism relies on the `GCC GNU` extension, and the `GNU` extension is not part of the standard `C`. When the community called for the need to extend support to `Microsoft Visual Studio`, they encountered more difficulties.
After deciding to redesign, the ease of use of the registration module has been greatly improved. The new mechanism achieves similar traversal and registration effects through the additional processing of `CMake`, and completes module registration. For specific design and improvement, please refer to **Important Module Introduction** in **Architecture Details**.

**Tengine Lite** treats all hardware units that can run `CNN` as devices in the design. `CPU` is a typical device. In all compilation options, `CPU` devices are included by default. If you describe a new device and register it, in the usual sense, this potentially means that the compiled **Tengine Lite** supports heterogeneous device cutting (for related content, please read the **mixed device** section); if the registered device is also Describes the **mixed precision** interface, then the device also supports **mixed precision**.
**Tengine Lite** completes the description of a device through a nested structure:
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
It can be seen from the structure `ir_device_t` that the design divides a device (`device`) into 6 parts. The first part `name` describes the name of the device, and the device name is not allowed to be repeated; `interface` describes the device interface; `allocator` describes the operation of device-related subgraphs; `optimizer` describes the interface of cutting graphs and mixed precision; `scheduler` describes the unique scheduling interface of the device.
The above interfaces usually do not need to be filled in. **Tengine Lite** provides a rich set of examples to guide how to customize and add users' own devices.

-----------------------------------------------

## Add a custom device step by step
### Create a directory and write a CMakeLists file
First, create a folder named after the user's device in `source/device`. The folder can be the user's device abbreviation or a name that other users think is cool (here assuming the name is `TPU`, then the directory is `source/devicetpu`), and copy a `CMakeLists.txt` file from other implemented `device/xxx` directories to the current folder; now you only need to make slight modifications to this ` CMakeLists.txt`, no need Created from scratch. Let's take a copy from `source/device/acl/CMakeLists.txt` as an example. The complete example of the file is as follows:
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
First, you need to modify the prefix of the used `CMake` variables to avoid potential variable conflicts; replace all `ACL` with `TPU`; then modify the search root path of the module `_TPU_ROOT` to `source/device/tpu` .
``` cmake
# 1.  set source root path
SET(_TPU_ROOT ${CMAKE_SOURCE_DIR}/source/device/tpu)
```
Custom devices often require some additional `3rdparty` dependencies, which should be modified accordingly in `_DEV_TPU_HEADER_PATH` and `_DEV_TPU_LINK_PATH`; in `ACL`, `ACL` precompiled library path `${CMAKE_SOURCE_DIR}/3rdparty/ acl/lib`.
``` cmake
# 2.  add header file path
LIST (APPEND _DEV_TPU_HEADER_PATH      ${_TPU_ROOT})
LIST (APPEND _DEV_TPU_HEADER_PATH      ${CMAKE_SOURCE_DIR}/3rdparty/tpu/include)


# 3.  add linking lib searching path
LIST (APPEND _DEV_TPU_LINK_PATH        ${CMAKE_SOURCE_DIR}/3rdparty/tpu/lib)
```
Source code collection part according to the actual situation can be modified.
``` cmake
# 4.  add source files
AUX_SOURCE_DIRECTORY("${_TPU_ROOT}"    _TPU_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_TPU_ROOT}/op" _TPU_OPS_SOURCE)
LIST (APPEND _DEV_TPU_DEVICE_SOURCE    ${_TPU_BASE_SOURCE})
LIST (APPEND _DEV_TPU_DEVICE_SOURCE    ${_TPU_OPS_SOURCE})
```
The next part is the compilation-related options, which can be modified according to the actual situation. **Tengine Lite** turns on C/C++ support by default, and try to open the standard to `C99/C++14`, if the tool chain does not support it, it will be downgraded to `C98/C++11`; if the user code has Other special requirements can be adjusted according to the situation `_DEV_TPU_COMPILER_DEFINES`, `_DEV_TPU_COMPILER_OPTIONS`, `_DEV_TPU_LINKER_OPTIONS` these three variables.
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
Adjust the link library according to the actual situation and modify the `_DEV_TPU_LINK_LIBRARIES` variable.
``` cmake
# 7.  add link libs
LIST (APPEND _DEV_TPU_LINK_LIBRARIES   tpu_runtime)
```
Summarize the temporary variables to the module interface variables. The interface variables are designed as `cache` so that they can be passed across modules (on the other hand, this is also the reason why different `device`s should not be named).
``` cmake
SET (TENGINE_TPU_HEADER_PATH       ${_DEV_TPU_HEADER_PATH}        CACHE INTERNAL  "Tengine TPU device header files searching path"   FORCE)
SET (TENGINE_TPU_LINK_PATH         ${_DEV_TPU_LINK_PATH}          CACHE INTERNAL  "Tengine TPU device link libraries searching path" FORCE)
SET (TENGINE_TPU_DEVICE_SOURCE     ${_DEV_TPU_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine TPU device main source files"             FORCE)
SET (TENGINE_TPU_COMPILER_DEFINES  ${_DEV_TPU_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine TPU about compiler defines"               FORCE)
SET (TENGINE_TPU_COMPILER_OPTIONS  ${_DEV_TPU_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine TPU about compiler options"               FORCE)
SET (TENGINE_TPU_LINKER_OPTIONS    ${_DEV_TPU_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine TPU about linker options"                 FORCE)
SET (TENGINE_TPU_LINK_LIBRARIES    ${_DEV_TPU_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine TPU about link libraries"                 FORCE)
```
If the device has special options, consider inserting it into the `install` stage.
``` cmake
# 9. install device option
INSTALL (FILES ${_TPU_ROOT}/tpu_define.h DESTINATION include/tengine RENAME tpu_device.h)
```
Add `option` to `CMakeLists.txt` in the root directory to enable conditions during compilation.
``` cmake
OPTION (TENGINE_ENABLE_TPU "With Awesome TPU support" OFF)
```
It also needs to modify `source / device / CMakeLists.txt` to add `Option` related processing.
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
Among them, `_REGISTER_DEVICE_LIST` is the core file of device registration, which needs to be filled in according to the actual situation.

### Fill the structure to complete the device registration
In a sense, to complete the registration of a new device, you only need to fill in the `ir_device_t` structure, and all other code work is carried out around this core.
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

Reviewing the `ir_device_t` structure, the `struct interface` structure describes the basic `API` interface:
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
Referring to `ACL`, a possible implementation of `TPU` is filled as follows:
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
`tpu_dev_init()` is the global initialization function of the device. It is called once when registering the device, and `release_device()` is called for anti-registration. This function is generally used to pre-apply for device memory as a global cache, interoperate with device drivers to initialize some registers, etc.
`tpu_dev_prerun()` is the network pre-processing part. Common processing includes applying for `tensor` memory, converting data `layout`, creating device running diagram, compiling device `kernel`, etc. This part of the application space needs to be cleaned up in `tpu_release_graph()`.
`tpu_post_run()` and `tpu_release_graph()` may cause confusion. `tpu_post_run()` is often used to clear the related state of a run. Contrary to `tpu_dev_prerun()`, the real release work can be placed in `tpu_release_graph() )` in progress. One possible scenario is to run `tpu_dev_prerun()` after running the model with one resolution, then run `tpu_post_run()` before changing the resolution, and then run `tpu_dev_prerun()`. When it needs to be destroyed, run `tpu_release_graph()`.

In the `ir_device_t` structure, the `struct interface` structure describes the basic `API` interface, the `struct allocator` describes the device capability reporting interface and the evaluation and scheduling interface, and the `struct optimizer` describes the picture cutting and optimization related Interface, `struct scheduler` describes the interface related to scheduling. The core of these interfaces is `struct scheduler`. The device does not always assume to implement a `struct scheduler`. If the interface description of the device is `nullptr`, then the engine will use the default registered **`sync scheduler`** Run the network, refer to `static ir_scheduler_t sync_scheduler` in `source/scheduler/scheduler.c` for details. Users can also implement their own `struct scheduler` to complete special tasks; combining `struct allocator` and `struct optimizer` can produce rich possibilities. The following description assumes that the user does not implement `struct scheduler`.


``` cmake
static struct allocator tpu_allocator = {
        .describe       = tpu_describe,
        .evaluation     = tpu_evaluation,
        .allocate       = tpu_allocate,
        .release        = tpu_release,
};
```
In `tpu_allocator`, `tpu_describe()` reports the model's `OP` support and accuracy support. The description of `OP` and accuracy support here will not change with network changes. The potential meaning is this state The following always assumes that the user-specific device is supported by `OP` or precision full-scene. Take convolution as an example. This means that the user's device supports all modes of convolution, regardless of the conditions of `pad`, `stride`, `h`, `w`, and `c`. If the device implementation really needs to be evaluated at runtime, you can customize the `struct scheduler` to complete the customization process.
`tpu_evaluation()` is used to evaluate whether the implemented device submap can be run before running; this is especially useful when you need to compile `kernel`.
`tpu_allocate()` is used to support the related content of the device storage pool, there is no need to fill this entry under the default `scheduler`. `tpu_release()` is the opposite release process.

``` cmake
static struct optimizer tpu_optimizer = {
        .split_graph    = tpu_split_graph,
        .optimize_graph = nullptr,
};
```
In the `tpu_optimizer` structure, `tpu_split_graph()` is used to implement graph cutting, and `tpu_optimize_graph()` is used to implement mixed precision. Among them, `tpu_split_graph()` can call the default implementation of `split_graph_node_to_sub_graph()` for ordinary graph cutting ; If you have special requirements, you can combine different fields of other structures to form a combination.

Finally, you need to write the registration function and the de-registration function `int register_tpu_device()` and `int unregister_tpu_device()`. It should be noted that the second half of the registration function and the de-registration function is the file name, which needs to match the actual file name. CMake The link of the calling process of the registered function will be automatically completed.

## Summary
From the above description, we can know that the core work of adding a custom device is to fill the `ir_device_t` structure. After the description is completed, all the work of device registration is completed. The modular `device` makes **Tengine Lite** very easy to expand and has enough flexibility.

## Surprise
In the `init_tengine(void)` function, when the `operator prototype` completes the registration, the registered ones are `serializer` and `devices`, but the function does not jump in the static code state, and the user can install an integrated development environment , Such as `Microsoft Visual Studio` or `Jetbrains Clion`, after opening the folder and generating the `CMake` process, you can jump.