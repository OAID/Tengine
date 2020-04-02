include_directories(${CONFIG_TENGINE_ROOT}/core/include)

if(PROTOBUF_FOUND)
	STRING(REGEX REPLACE "(.+)\\.so.*" "\\1" PROTOBUF_STATIC_LIB ${PROTOBUF_LIB})
	set( PROTOBUF_STATIC_LIB "${PROTOBUF_STATIC_LIB}.a" )
	message(STATUS "pb static : ${PROTOBUF_STATIC_LIB}")
else()
	set( PROTOBUF_STATIC_LIB ${PROTOBUF_LIB} )
endif()

set(STATIC_LIBS "-static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -lresolv -ldl -L${CONFIG_TENGINE_ROOT}/build ${PROTOBUF_STATIC_LIB}")

if(TENGINE_IN_BIND)
	set(TENIGNE_STATIC_LIBS tengine-static)
else()
	set(TENIGNE_STATIC_LIBS "-Wl,-rpath,${CONFIG_TENGINE_ROOT}/build -static -ltengine")
endif()

set(SERIAL_LIBS caffe-serializer-static mxnet-serializer-static onnx-serializer-static tensorflow-serializer-static tflite-serializer-static darknet-serializer-static)
function(create_target file_list)
    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" fname ${file})
    add_executable(${fname} ${file})
    target_link_libraries(${fname} ${TENGINE_LIBRARY} ${SERIAL_LIBS} ${TENIGNE_STATIC_LIBS} ${STATIC_LIBS})

    if(SYS_ROOT)
        set_property(TARGET ${fname} APPEND PROPERTY LINK_FLAGS "--sysroot=${SYS_ROOT} -Wl,-allow-shlib-undefined")
        set_property(SOURCE ${fname} APPEND PROPERTY COMPILE_FLAGS "--sysroot=${SYS_ROOT}")
    endif()

    install(TARGETS ${fname} DESTINATION convert_tools) 
endfunction()


FILE(GLOB file_list "*.cpp" "*.c")

foreach(file ${file_list})
    create_target(${file})
endforeach()

