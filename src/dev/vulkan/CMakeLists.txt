find_package(Vulkan REQUIRED)

# TODO: move to check.cmake
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

# message(STATUS "Tengine: add vulkan layer ${SHADER_SPV_HEX_FILES}")


set(CMAKE_SHARED_LINKER_FLAGS "-Bsymbolic -Bsymbolic-functions")

file(GLOB TENGINE_BACKEND_VULKAN_BASE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB TENGINE_BACKEND_VULKAN_OPS "${CMAKE_CURRENT_SOURCE_DIR}/layer/*.cpp")
set(TENGINE_BACKEND_VULKAN_BASE ${TENGINE_BACKEND_VULKAN_BASE} CACHE INTERNAL " " FORCE)
set(TENGINE_BACKEND_VULKAN_OPS ${TENGINE_BACKEND_VULKAN_OPS} CACHE INTERNAL " " FORCE)
