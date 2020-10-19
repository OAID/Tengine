# add include path
list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/op)
list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm)
if(TENGINE_USE_TINY_SERIALIZER)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tiny)
endif()


# add lib files, for utility
if (TENGINE_BUILD_CPP_API)
    file(GLOB_RECURSE TENGINE_LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c ${CMAKE_CURRENT_SOURCE_DIR}/lib/tengine_cpp_api.cpp)
else()
    file(GLOB_RECURSE TENGINE_LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c)
endif()

# add front end files
file(GLOB_RECURSE TENGINE_FRONT_END_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op/*.c)

# add serializer files
file(GLOB_RECURSE TENGINE_SERIALIZER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm/*.c ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm/op/*.c)
if(TENGINE_USE_TINY_SERIALIZER)
    file(GLOB_RECURSE TENGINE_TINY_SERIALIZER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tiny/*.c)
endif()

# add core srcs
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_allocator.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_device.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_module.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_node_ops.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_probe.c")

# add reference operator files
file(GLOB_RECURSE TENGINE_BACKEND_REF_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*ref.c")

# add hcl operator files
# arm
if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
    if (TENGINE_ARCH_ARM_82)
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*hcl.c"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*fp16_arm82.c")
    else()
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*hcl.c"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c")
    endif()

    if (TENGINE_TARGET_PROCESSOR_32Bit)
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch32/*.S")
    else()
        if (TENGINE_ARCH_ARM_82)
            add_definitions(-march=armv8.2-a+fp16)
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch64/*.S"
                                                            "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/armv8_2/*.S")
        else()
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch64/*.S")
        endif()
    endif()
endif()

# X86
if (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    if (TENGINE_ARCH_X86_AVX)
        add_definitions(-mfma -mf16c)
    endif()
    file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*hcl.c"
                                                "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*x86.c")
endif()

# add cmsis operator files
file(GLOB_RECURSE TENGINE_BACKEND_CMSIS_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*cmsis.c")
# add ACL GPU support
if (TENGINE_ENABLE_ACL)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/dev/acl)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/acl/include)
    list(APPEND TENGINE_ACL_LIB_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/acl/lib)

    #link_directories(${TENGINE_ACL_LIB_DIRS})
    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'TENGINE_ACL_LIB_DIRS' is ${TENGINE_ACL_LIB_DIRS}.")
    endif()

    file(GLOB_RECURSE TENGINE_BACKEND_ACL_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/acl/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/dev/acl/*.c")
endif ()


# add vulkan support
if (TENGINE_ENABLE_VULKAN)
    add_subdirectory(dev/vulkan)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/dev/vulkan)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_BINARY_DIR}/dev/vulkan)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${Vulkan_INCLUDE_DIR})
    list(APPEND TENGINE_VULKAN_LIB_DIRS $ENV{VULKAN_SDK}/lib)
    
    link_directories(${TENGINE_VULKAN_LIB_DIRS})
    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'TENGINE_VULKAN_LIB_DIRS' is ${TENGINE_VULKAN_LIB_DIRS}.")
    endif()
endif()


# add VeriSilicon VIP8000 NPU support
if (TENGINE_ENABLE_VIPNPU)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/dev/vxnpu)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/vsi_sdk/include)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/acuity-ovxlib-dev/include)
    list(APPEND TENGINE_VIPNPU_LIB_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/acuity-ovxlib-dev/lib)

    link_directories(${TENGINE_VIPNPU_LIB_DIRS})
    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'TENGINE_VIPNPU_LIB_DIRS' is ${TENGINE_VIPNPU_LIB_DIRS}.")
    endif()

    file(GLOB_RECURSE TENGINE_BACKEND_VXNPU_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/vxnpu/*.c")
    file(GLOB_RECURSE TENGINE_BACKEND_VXNPU_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/vxnpu/op/*.c")
endif ()

# add libraries path
list(APPEND TENGINE_PRIVATE_LIB_DIRS ${TENGINE_ACL_LIB_DIRS})
list(APPEND TENGINE_PRIVATE_LIB_DIRS ${TENGINE_VULKAN_LIB_DIRS})
set(TENGINE_COMMON_LIB_DIRS ${TENGINE_PRIVATE_LIB_DIRS} CACHE INTERNAL "" FORCE)
link_directories(${TENGINE_PRIVATE_LIB_DIRS})

# add advanced hcl source files
if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS} ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS}
        ${TENGINE_BACKEND_HCL_OPS}
        ${TENGINE_BACKEND_HCL_ASM_OPS}
        ${TENGINE_BACKEND_ACL_BASE}
        ${TENGINE_BACKEND_VULKAN_BASE}
        ${TENGINE_BACKEND_VULKAN_OPS})
elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS} ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS}
        ${TENGINE_BACKEND_HCL_OPS}
        ${TENGINE_BACKEND_VULKAN_BASE}
        ${TENGINE_BACKEND_VULKAN_OPS})
else()
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS}
        ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS})
endif()


# show linking libraries
if(TENGINE_VERBOSE)
    message (STATUS "TENGINE: 'TENGINE_LINKING_LIBS' is ${TENGINE_LINKING_LIBS}.")
endif()

# add include path
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${TENGINE_PRIVATE_INC_DIRS})


if (ANDROID)
    target_link_libraries(${CMAKE_PROJECT_NAME} android log ${TENGINE_LINKING_LIBS})
else()
    target_link_libraries(${CMAKE_PROJECT_NAME} pthread dl m ${TENGINE_LINKING_LIBS})
endif()

if (TENGINE_ENABLE_ACL)
    target_link_libraries(${CMAKE_PROJECT_NAME} arm_compute arm_compute_core)
endif ()

if (TENGINE_ENABLE_VULKAN)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${Vulkan_LIBRARY})
    message("===== vulkan library === ${Vulkan_LIBRARY}")
    add_dependencies(${CMAKE_PROJECT_NAME} generate-spirv)
endif ()
install (TARGETS ${CMAKE_PROJECT_NAME} DESTINATION lib)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tengine_c_api.h DESTINATION include)
