FILE(GLOB_RECURSE operator_src operator/plugin/*.cpp operator/operator/*.cpp)

include_directories(operator/include)

list(APPEND TENGINE_LIB_SRCS ${operator_src})

