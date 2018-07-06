
include_directories(serializer/include)

if(CONFIG_CAFFE_SERIALIZER)
    FILE(GLOB_RECURSE serializer_src "serializer/caffe/*.cpp")

    list(APPEND TENGINE_LIB_SRCS ${serializer_src})


    # the generated caffe.pb.cc 

    set(caffe_proto_cc ${CMAKE_CURRENT_BINARY_DIR}/serializer/caffe/te_caffe.pb.cc)

    set(caffe_dir ${CMAKE_CURRENT_SOURCE_DIR}/serializer/caffe)
    set(dest_dir  ${CMAKE_CURRENT_BINARY_DIR}/serializer/caffe)

    

    ADD_CUSTOM_COMMAND(OUTPUT ${caffe_proto_cc} 
                       COMMAND mkdir -p ${dest_dir}
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${caffe_dir} ${caffe_dir}/te_caffe.proto
                       COMMAND mv ${dest_dir}/te_caffe.pb.h ${caffe_dir}/../include/
                       DEPENDS ${caffe_dir}/te_caffe.proto)

   ADD_CUSTOM_TARGET(CAFFE_SERIALIZER_TARGET DEPENDS ${caffe_proto_cc})

   message("CAFFE SERIALIZER DEFINED: " ${caffe_proto_cc})
   list(APPEND TENGINE_LIB_SRCS ${caffe_proto_cc})

endif()

if(CONFIG_TENGINE_SERIALIZER)
    FILE(GLOB_RECURSE tengine_serializer_cpp_src "serializer/tengine/*.cpp")
    FILE(GLOB_RECURSE tengine_serializer_c_src "serializer/tengine/*.c")
    list(APPEND TENGINE_LIB_SRCS ${tengine_serializer_cpp_src} ${tengine_serializer_c_src})
endif()

FILE(GLOB plugin_init "serializer/plugin/init.cpp")

list(APPEND TENGINE_LIB_SRCS ${plugin_init})
