# 0. clear var
UNSET (_DEV_VULKAN_HEADER_PATH)
UNSET (_VULKAN_BASE_SOURCE)
UNSET (_VULKAN_OPS_SOURCE)
UNSET (_DEV_VULKAN_DEVICE_SOURCE)
UNSET (_DEV_VULKAN_COMPILER_DEFINES)
UNSET (_DEV_VULKAN_COMPILER_OPTIONS)
UNSET (_DEV_VULKAN_LINKER_OPTIONS)
UNSET (_DEV_VULKAN_LINK_LIBRARIES)



find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH REQUIRED)
message(STATUS "Tengine: found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")

# add shader spv header generate macro
include(${CMAKE_SOURCE_DIR}/cmake/generate_shader_spv_header.cmake)

macro(add_shader SHADER_SRC)
    message(STATUS "SHADER_SRC: ${SHADER_SRC}")
    generate_shader_spv_header(SHADER_SPV_HEADER SHADER_SPV_HEX_HEADERS ${SHADER_SRC})


    get_filename_component(SHADER_SPV_HEADER_NAME ${SHADER_SPV_HEADER} NAME)
    string(APPEND layer_shader_spv_data "#include \"${SHADER_SPV_HEADER_NAME}\"\n")

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_spv_data,sizeof(${SHADER_SRC_NAME_WE}_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_fp16p_spv_data,sizeof(${SHADER_SRC_NAME_WE}_fp16p_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_fp16pa_spv_data,sizeof(${SHADER_SRC_NAME_WE}_fp16pa_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_fp16s_spv_data,sizeof(${SHADER_SRC_NAME_WE}_fp16s_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_fp16sa_spv_data,sizeof(${SHADER_SRC_NAME_WE}_fp16sa_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_image_spv_data,sizeof(${SHADER_SRC_NAME_WE}_image_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_image_fp16p_spv_data,sizeof(${SHADER_SRC_NAME_WE}_image_fp16p_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_image_fp16pa_spv_data,sizeof(${SHADER_SRC_NAME_WE}_image_fp16pa_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_image_fp16s_spv_data,sizeof(${SHADER_SRC_NAME_WE}_image_fp16s_spv_data)},\n")
    string(APPEND layer_shader_registry "{${SHADER_SRC_NAME_WE}_image_fp16sa_spv_data,sizeof(${SHADER_SRC_NAME_WE}_image_fp16sa_spv_data)},\n")

    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEADER})
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_HEADERS})

    # generate layer_shader_type_enum file
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE} = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_fp16p = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_fp16pa = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_fp16s = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_fp16sa = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_image = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_image_fp16p = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_image_fp16pa = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_image_fp16s = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
    set(layer_shader_type_enum "${layer_shader_type_enum}${SHADER_SRC_NAME_WE}_image_fp16sa = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")

endmacro()

macro(add_layer class)
    string(TOLOWER ${class} name)

    file(GLOB_RECURSE SHADER_SRCS "shaders/${name}.comp")
    file(GLOB_RECURSE SHADER_SUBSRCS "shaders/${name}_*.comp")
    list(APPEND SHADER_SRCS ${SHADER_SUBSRCS})
    foreach(SHADER_SRC ${SHADER_SRCS})
        add_shader(${SHADER_SRC})
    endforeach()

    # generate layer_type_enum file
    set(layer_type_enum "${layer_type_enum}${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()

set(SHADER_SPV_HEX_FILES)

set(__LAYER_TYPE_ENUM_INDEX 0)
set(__LAYER_SHADER_TYPE_ENUM_INDEX 0)

add_layer(Convolution)
add_layer(ConvolutionDepthWise)
add_layer(Pooling)
add_layer(Padding)
add_layer(Packing)
add_layer(InnerProduct)
add_layer(Flatten)
add_layer(Relu)
add_layer(Eltwise)
add_layer(Softmax)
add_layer(Dropout)
add_layer(PriorBox)
add_layer(Permute)
add_layer(Reshape)
add_layer(Concat)
add_layer(Interp)
add_layer(Crop)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

# create new registry file
configure_file(layer_shader_registry.h.in   ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_registry.h)
configure_file(layer_shader_spv_data.h.in   ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_spv_data.h)
configure_file(layer_type_enum.h.in         ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)
configure_file(layer_shader_type_enum.h.in  ${CMAKE_CURRENT_BINARY_DIR}/layer_shader_type_enum.h)

# find_package(Vulkan QUIET)
set(VULKAN_LIBRARY "/usr/lib/x86_64-linux-gnu/" CACHE INTERNAL " " FORCE)
set(VULKAN_INCLUDE_DIRS "/usr/include/vulkan/" CACHE INTERNAL " " FORCE)

# 1.  set source root path
SET(_VULKAN_ROOT ${CMAKE_SOURCE_DIR}/source/device/vulkan)
SET(_VULKAN_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR})


# 2.  add header file path
LIST (APPEND _DEV_VULKAN_HEADER_PATH           ${_VULKAN_BUILD_ROOT})
LIST (APPEND _DEV_VULKAN_HEADER_PATH           ${_VULKAN_ROOT})
LIST (APPEND _DEV_VULKAN_HEADER_PATH           ${VULKAN_INCLUDE_DIRS})


# 3.  add linking lib searching path
LIST (APPEND _DEV_VULKAN_LINK_PATH             ${VULKAN_LIBRARY})


# 4.  add source files
AUX_SOURCE_DIRECTORY("${_VULKAN_ROOT}"         _VULKAN_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_VULKAN_ROOT}/layer"      _VULKAN_OPS_SOURCE)
LIST (APPEND _DEV_VULKAN_DEVICE_SOURCE         ${_VULKAN_BASE_SOURCE})
LIST (APPEND _DEV_VULKAN_DEVICE_SOURCE         ${_VULKAN_OPS_SOURCE})


# 5.  add build options for cpu device
# 5.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
    IF (TENGINE_COMPILER_GCC AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "6.1"))
        LIST (APPEND _DEV_VULKAN_COMPILER_OPTIONS -Wno-ignored-attributes)
    ENDIF()
ENDIF()


# 5.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 6.  add link options


# 7.  add link libs
LIST (APPEND _DEV_VULKAN_LINK_LIBRARIES   "libvulkan.so")


# 8. set all to cmake cache
SET (TENGINE_VULKAN_HEADER_PATH       ${_DEV_VULKAN_HEADER_PATH}        CACHE INTERNAL  "Tengine VULKAN device header files searching path"   FORCE)
SET (TENGINE_VULKAN_LINK_PATH         ${_DEV_VULKAN_LINK_PATH}          CACHE INTERNAL  "Tengine VULKAN device link libraries searching path" FORCE)
SET (TENGINE_VULKAN_DEVICE_SOURCE     ${_DEV_VULKAN_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine VULKAN device main source files"             FORCE)
SET (TENGINE_VULKAN_COMPILER_DEFINES  ${_DEV_VULKAN_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine VULKAN about compiler defines"               FORCE)
SET (TENGINE_VULKAN_COMPILER_OPTIONS  ${_DEV_VULKAN_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine VULKAN about compiler options"               FORCE)
SET (TENGINE_VULKAN_LINKER_OPTIONS    ${_DEV_VULKAN_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine VULKAN about linker options"                 FORCE)
SET (TENGINE_VULKAN_LINK_LIBRARIES    ${_DEV_VULKAN_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine VULKAN about link libraries"                 FORCE)
SET (TENGINE_VULKAN_DEPENDS_FORWARD   generate-spirv                    CACHE INTERNAL  "Tengine VULKAN about depends project"                FORCE)


# 9. install device option
INSTALL (FILES ${_VULKAN_ROOT}/VULKAN_define.h DESTINATION include/tengine RENAME VULKAN_device.h)
