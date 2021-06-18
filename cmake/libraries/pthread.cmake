# find & use pthread

FUNCTION(TENGINE_CHECK_LIB_PTHREAD _has_flag)
    IF(CMAKE_C_COMPILER_LOADED)
        INCLUDE(CheckIncludeFile)
        CHECK_INCLUDE_FILE("pthread.h" ${_has_flag})
    ELSE()
        INCLUDE(CheckIncludeFileCXX)
        CHECK_INCLUDE_FILE_CXX("pthread.h" ${_has_flag})
    ENDIF()
ENDFUNCTION()


FUNCTION(TENGINE_COMPILE_LIB_PTHREAD _target)
    # common compile option settings
    IF (TENGINE_ENABLE_CUDA OR TENGINE_ENABLE_TENSORRT)
        TARGET_COMPILE_OPTIONS(${_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler -pthread> $<$<NOT:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>>:-pthread>)
    ENDIF()

    # accoding to EMSCRIPTEN doc, compile & link option "-pthread" was needed.
    # if settings worked, macro "__EMSCRIPTEN_PTHREADS__" will be defined. 
    # see https://emscripten.org/docs/porting/pthreads.html
    IF(EMSCRIPTEN)
        # TARGET_LINK_OPTIONS was added after 3.13
        # see https://cmake.org/cmake/help/latest/command/target_link_options.html
        CMAKE_MINIMUM_REQUIRED(VERSION 3.13 FATAL_ERROR)

        # LINK_LANGUAGE keyword is added to generator expressions in 3.18
        IF((${CMAKE_MAJOR_VERSION} VERSION_EQUAL 3) AND (${CMAKE_MINOR_VERSION} VERSION_EQUAL 18 OR ${CMAKE_MINOR_VERSION} VERSION_GREATER 18))
            TARGET_LINK_OPTIONS(${_target} PRIVATE $<$<LINK_LANGUAGE:C>:-pthread> $<$<LINK_LANGUAGE:CXX>:-pthread>)
        ELSE()
            TARGET_LINK_OPTIONS(${_target} PRIVATE -pthread)
        ENDIF()
    ENDIF()
ENDFUNCTION()


FUNCTION(TENGINE_LINK_LIB_PTHREAD _target _is_static)
    IF(ANDROID)
        # is libandroid.so enough for using pthread?
        IF (_is_static)
            TARGET_LINK_LIBRARIES(${_target} PUBLIC  android)
        ELSE()
            TARGET_LINK_LIBRARIES(${_target} PRIVATE android)
        ENDIF()
    ELSEIF(UNIX AND NOT APPLE AND NOT MSVC)
        IF (_is_static)
            TARGET_LINK_LIBRARIES(${_target} PUBLIC  pthread)
        ELSE()
            TARGET_LINK_LIBRARIES(${_target} PRIVATE pthread)
        ENDIF()
    ELSEIF(EMSCRIPTEN)
        # nothing need to do when building for emscripten
    ELSE()
        IF(TENGINE_VERBOSE)
            MESSAGE (AUTHOR_WARNING "FULCRUM(VERBOSE): CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}, a condition check should be added(TENGINE_LINK_PTHREAD).")
        ENDIF()
        IF (_is_static)
            TARGET_LINK_LIBRARIES(${_target} PUBLIC  pthread)
        ELSE()
            TARGET_LINK_LIBRARIES(${_target} PRIVATE pthread)
        ENDIF()
    ENDIF()
ENDFUNCTION()


FUNCTION(TENGINE_USE_LIB_PTHREAD _target _is_static)
    TENGINE_COMPILE_LIB_PTHREAD(${_target})
    TENGINE_LINK_LIB_PTHREAD(${_target} ${_is_static})
ENDFUNCTION()
