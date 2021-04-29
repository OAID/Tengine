# test flexible array
#   At the very begening, declaring zero-length arrays is allowed in GNU C as an extension. 
#   See: http://gcc.gnu.org/onlinedocs/gcc/Zero-Length.html


FUNCTION(TENGINE_CHECK_LANG_FLEXIBLE_ARRAY _support_flag)
    MESSAGE("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
    MESSAGE("PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
    MESSAGE("CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
    TRY_COMPILE(${_support_flag} ${CMAKE_BINARY_DIR} ${TENGINE_CMAKE_MODULE_LANG_DIR}/source/flexible_array.cc C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
ENDFUNCTION()
