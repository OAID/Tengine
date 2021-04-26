# check <syslog.h>

FUNCTION(TENGINE_CHECK_INC_SYSLOG _has_flag)
    IF(CMAKE_C_COMPILER_LOADED)
        INCLUDE(CheckIncludeFile)
        CHECK_INCLUDE_FILE("syslog.h" ${_has_flag})
    ELSE()
        INCLUDE(CheckIncludeFileCXX)
        CHECK_INCLUDE_FILE_CXX("syslog.h" ${_has_flag})
    ENDIF()
ENDFUNCTION()
