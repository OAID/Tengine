# add include path
list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/op)
list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm)
if(TENGINE_USE_TINY_SERIALIZER)
    list(APPEND TENGINE_PRIVATE_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tiny)
endif()


# add lib files, for utility
file(GLOB_RECURSE TENGINE_LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c)

# add front end files
file(GLOB_RECURSE TENGINE_FRONT_END_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op/*.c)

# add serializer files
file(GLOB_RECURSE TENGINE_SERIALIZER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm/*.c ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm/op/*.c)
if(TENGINE_USE_TINY_SERIALIZER)
    file(GLOB_RECURSE TENGINE_TINY_SERIALIZER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tiny/*.c)
endif()

# add core srcs
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_device.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_module.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_node_ops.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_probe.c")

# add reference operator files
#file(GLOB_RECURSE TENGINE_BACKEND_REF_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*ref.c")
file(GLOB_RECURSE TENGINE_BACKEND_REF_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/conv/conv_ref.c")
list(APPEND TENGINE_BACKEND_REF_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/conv/ref/conv_kernel_ref.c")

# add hcl operator files
file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*hcl.c"
                                            "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c")

# add hcl asm operator files
if ("${TENGINE_TARGET_PROCESSOR}" MATCHES "ARM")
    if (TENGINE_TARGET_PROCESSOR_32Bit)
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch32/*.S")
    else()
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch64/*.S")
    endif()
endif()

# add cmsis operator files
file(GLOB_RECURSE TENGINE_BACKEND_CMSIS_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*cmsis.c")


# add advanced hcl source files
if ("${TENGINE_TARGET_PROCESSOR}" MATCHES "ARM")
    add_library(${PROJECT_NAME} STATIC
        ${TENGINE_LIB_SRCS} ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS}
        ${TENGINE_BACKEND_HCL_OPS}
        ${TENGINE_BACKEND_HCL_ASM_OPS})
else()
    add_library(${PROJECT_NAME} STATIC
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
target_include_directories(${PROJECT_NAME} PRIVATE ${TENGINE_PRIVATE_INC_DIRS})


if (ANDROID)
    target_link_libraries(${PROJECT_NAME} android ${TENGINE_LINKING_LIBS})
else()
    target_link_libraries(${PROJECT_NAME} pthread dl m ${TENGINE_LINKING_LIBS})
endif()

if(NOT BUILD_SHARED_LIBS)
    ocv_install_target(tengine EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
#install (TARGETS ${CMAKE_PROJECT_NAME} DESTINATION lib)
#install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tengine_c_api.h DESTINATION include)
