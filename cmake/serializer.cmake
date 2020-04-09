
include_directories(serializer/include)

if(CONFIG_TENGINE_SERIALIZER)
    include_directories(serializer/include/tengine)
    include_directories(serializer/include/tengine/v1)
    include_directories(serializer/include/tengine/v2)
    FILE(GLOB_RECURSE tengine_serializer_cpp_src "serializer/tengine/*.cpp")
    FILE(GLOB_RECURSE tengine_serializer_c_src "serializer/tengine/*.c")
    list(APPEND TENGINE_LIB_SRCS ${tengine_serializer_cpp_src} ${tengine_serializer_c_src})
endif()

#FILE(GLOB_RECURSE source_serializer_cpp_src "serializer/source/*.cpp")
#list(APPEND TENGINE_LIB_SRCS ${source_serializer_cpp_src})

FILE(GLOB plugin_init "serializer/plugin/init.cpp")

list(APPEND TENGINE_LIB_SRCS ${plugin_init})
