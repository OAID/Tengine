
include_directories(serializer/include)

if( CONFIG_CAFFE_SERIALIZER)
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

FILE(GLOB plugin_init "serializer/plugin/init.cpp")

list(APPEND TENGINE_LIB_SRCS ${plugin_init})
