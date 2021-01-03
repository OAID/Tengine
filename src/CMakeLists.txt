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
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch64/*.S")
        endif()
    endif()
endif()

# x86
if (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    if (TENGINE_ARCH_X86_AVX)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            add_definitions(/arch:AVX2)
        else()
            add_definitions(-mfma -mf16c -mavx2)
        endif()
    endif()
endif()

# add operator files
if (TENGINE_DYNAMIC_COMPILE AND EXISTS "${CMAKE_SOURCE_DIR}/cmake/operators.cmake")
    # macro for adding include op dir
    macro (ADD_OP name)
        # add reference operator files 
        set (REF_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*ref.c")
        list (APPEND REF_SOURCE ${REF_OP_PATH})
        
        # add hcl operator files
        # arm
        if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
            if (TENGINE_ARCH_ARM_82)
                set (ARM_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*arm.c")
                set (ARM82_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*fp16_arm82.c")
            else()
                set (ARM_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*arm.c")
            endif()
            list (APPEND HCL_SOURCE ${ARM_OP_PATH})
            list (APPEND HCL_SOURCE ${ARM82_OP_PATH})
        endif()

        # X86
        if (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
            set (X86_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*x86.c")
            list (APPEND HCL_SOURCE ${X86_OP_PATH})
        endif()

        # MIPS64
        if (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
            set (MIPS_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*mips.c")
            list (APPEND HCL_SOURCE ${MIPS_OP_PATH})
        endif()
    endmacro()
    include(${CMAKE_SOURCE_DIR}/cmake/operators.cmake)

    # add reference operator files
    file(GLOB_RECURSE TENGINE_BACKEND_REF_OPS ${REF_SOURCE})

    # add hcl operator files
    # arm or x86 or mips64
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" OR ${TENGINE_TARGET_PROCESSOR} MATCHES "X86" OR ${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   ${HCL_SOURCE})
    endif()
else()
    # add reference operator files
    file(GLOB_RECURSE TENGINE_BACKEND_REF_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*ref.c")

    # add hcl operator files
    # arm
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
        if (TENGINE_ARCH_ARM_82)
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c"
                                                        "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*fp16_arm82.c")
        else()
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c")
        endif()                                                    
    endif()

    # X86
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*x86.c")
    endif()

    # MIPS64
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*mips.c")
    endif()
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


# add libraries path
list(APPEND TENGINE_PRIVATE_LIB_DIRS ${TENGINE_ACL_LIB_DIRS})
list(APPEND TENGINE_PRIVATE_LIB_DIRS ${TENGINE_VULKAN_LIB_DIRS})
set(TENGINE_COMMON_LIB_DIRS ${TENGINE_PRIVATE_LIB_DIRS} CACHE INTERNAL "" FORCE)
link_directories(${TENGINE_PRIVATE_LIB_DIRS})


# add standalone hcl lib
set (TENGINE_STANDALONE_HCL_LIB_NAME hclcpu)
if (TENGINE_STANDALONE_HCL)
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
        add_library (
                ${TENGINE_STANDALONE_HCL_LIB_NAME} SHARED
                ${TENGINE_BACKEND_HCL_OPS}
                ${TENGINE_BACKEND_HCL_ASM_OPS}
        )
    elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
        add_library (
                ${TENGINE_STANDALONE_HCL_LIB_NAME} SHARED
                ${TENGINE_BACKEND_HCL_OPS}
        )
    elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
        add_definitions(-mips64r2)
        add_definitions(-mabi=64)
        add_definitions(-mmsa)
        add_library (
                ${TENGINE_STANDALONE_HCL_LIB_NAME} SHARED
                ${TENGINE_BACKEND_HCL_OPS}
        )
    endif()

    if (TENGINE_BACKEND_HCL_OPS)
        unset(TENGINE_BACKEND_HCL_OPS)
    endif()

    if (TENGINE_BACKEND_HCL_ASM_OPS)
        unset(TENGINE_BACKEND_HCL_ASM_OPS)
    endif()

    if (TENGINE_STANDALONE_HCL_AUTO_LOAD)
        add_definitions(-DTENGINE_AUTO_LOAD_HCL)
    endif()
endif()


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
elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS} ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS}
        ${TENGINE_BACKEND_HCL_OPS})
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

if (NOT TENGINE_ENABLE_ALL_SYMBOL)
    if (NOT (CMAKE_SYSTEM_NAME MATCHES "Windows"))
        set_property (TARGET ${CMAKE_PROJECT_NAME} PROPERTY COMPILE_FLAGS "-fvisibility=hidden")
        set_property (TARGET ${CMAKE_PROJECT_NAME} PROPERTY LINK_FLAGS    "-fvisibility=hidden")
    endif()
endif()

if (ANDROID)
    target_link_libraries(${CMAKE_PROJECT_NAME} android ${TENGINE_LINKING_LIBS})
elseif(MSVC)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${TENGINE_LINKING_LIBS})
elseif(OHOS)
    target_link_libraries(${CMAKE_PROJECT_NAME} dl m ${TENGINE_LINKING_LIBS})
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

if (TENGINE_STANDALONE_HCL)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${TENGINE_STANDALONE_HCL_LIB_NAME})
endif()

if(TENGINE_COVERAGE)
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")    
    target_link_libraries(${CMAKE_PROJECT_NAME} -coverage -lgcov)
endif()

# add install files
install (TARGETS ${CMAKE_PROJECT_NAME} DESTINATION lib)
if (TENGINE_STANDALONE_HCL)
    install (TARGETS ${TENGINE_STANDALONE_HCL_LIB_NAME} DESTINATION lib)
endif()
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tengine_c_api.h DESTINATION include)
if (TENGINE_BUILD_CPP_API)
    install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tengine_cpp_api.h DESTINATION include)
endif()
