include_directories(${CONFIG_TENGINE_ROOT}/core/include)
include_directories(${CONFIG_TENGINE_ROOT}/include)

set(TENGINE_LIBRARY ${CONFIG_TENGINE_ROOT}/build/libtengine.so)

function(create_target file_list)
    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" fname ${file})
    add_executable(${fname} ${file})
    if (ANDROID )
    	target_link_libraries(${fname} ${TENGINE_LIBRARY} ${OPENCV_LIB})
    else()
	    target_link_libraries(${fname} ${TENGINE_LIBRARY} ${OPENCV_LIB} pthread)    
    endif()

    if(SYS_ROOT)
        set_property(TARGET ${fname} APPEND PROPERTY LINK_FLAGS "--sysroot=${SYS_ROOT} -Wl,-allow-shlib-undefined")
        set_property(SOURCE ${fname} APPEND PROPERTY COMPILE_FLAGS "--sysroot=${SYS_ROOT}")
    endif()

    install(TARGETS ${fname} DESTINATION bin) 
endfunction()


FILE(GLOB file_list "*.cpp" "*.c")

foreach(file ${file_list})
    create_target(${file})
endforeach()

