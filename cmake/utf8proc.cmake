include (ExternalProject)

set(utf8proc_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/utf8proc/include)
set(utf8proc_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/utf8proc/lib/libutf8proc.a)

ExternalProject_Add(utf8proc
        PREFIX utf8proc
        GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc.git
        GIT_TAG v2.4.0
        GIT_SHALLOW 1
        DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/utf8proc/src/utf8proc
        CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/utf8proc
        )