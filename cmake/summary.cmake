# C/C++ Compilier report

MESSAGE (STATUS "")
MESSAGE (STATUS "")
MESSAGE (STATUS "Infomation Summary:")
MESSAGE (STATUS "")
# CMake infomation
MESSAGE (STATUS "CMake infomation:")
MESSAGE (STATUS "  - CMake version:              ${CMAKE_VERSION}")
MESSAGE (STATUS "  - CMake generator:            ${CMAKE_GENERATOR}")
MESSAGE (STATUS "  - CMake building tools:       ${CMAKE_BUILD_TOOL}")
MESSAGE (STATUS "  - Target System:              ${CMAKE_SYSTEM_NAME}")
MESSAGE (STATUS "  - Target CPU arch:            ${TENGINE_TARGET_PROCESSOR}")
MESSAGE (STATUS "  - Target building type:       ${CMAKE_BUILD_TYPE}")
IF (TENGINE_TARGET_PROCESSOR_32Bit)
    MESSAGE (STATUS "  - Target CPU bus width:       32 Bit")
ENDIF()
IF (TENGINE_TARGET_PROCESSOR_64Bit)
    MESSAGE (STATUS "  - Target CPU bus width:       64 Bit")
ENDIF()
MESSAGE (STATUS "")


# C/C++ Compilier infomation
MESSAGE (STATUS "${PROJECT_NAME} toolchain infomation:")
MESSAGE (STATUS "  Cross compiling: ${CMAKE_CROSSCOMPILING}")
MESSAGE (STATUS "  C/C++ compilier:")
MESSAGE (STATUS "    - C   standard version:     C${CMAKE_C_STANDARD}")
MESSAGE (STATUS "    - C   standard required:    ${CMAKE_C_STANDARD_REQUIRED}")
MESSAGE (STATUS "    - C   standard extensions:  ${CMAKE_C_EXTENSIONS}")
MESSAGE (STATUS "    - C   compilier version:    ${CMAKE_C_COMPILER_VERSION}")
MESSAGE (STATUS "    - C   compilier:            ${CMAKE_C_COMPILER}")
MESSAGE (STATUS "    - C++ standard version:     C++${CMAKE_CXX_STANDARD}")
MESSAGE (STATUS "    - C++ standard required:    ${CMAKE_CXX_STANDARD_REQUIRED}")
MESSAGE (STATUS "    - C++ standard extensions:  ${CMAKE_CXX_EXTENSIONS}")
MESSAGE (STATUS "    - C++ compilier version:    ${CMAKE_CXX_COMPILER_VERSION}")
MESSAGE (STATUS "    - C++ compilier:            ${CMAKE_CXX_COMPILER}")
MESSAGE (STATUS "  C/C++ compilier flags:")
MESSAGE (STATUS "    - C   compilier flags:      ${CMAKE_C_FLAGS}")
MESSAGE (STATUS "    - C++ compilier flags:      ${CMAKE_CXX_FLAGS}")
MESSAGE (STATUS "  OpenMP:")
IF (TENGINE_ENABLED_OPENMP)
MESSAGE (STATUS "    - OpenMP was found:         YES")
MESSAGE (STATUS "    - OpenMP version:           ${TENGINE_ENABLED_OPENMP}")
ELSE()
MESSAGE (STATUS "    - OpenMP was found:         NO")
ENDIF()
MESSAGE (STATUS "")


# CMake project infomation
MESSAGE (STATUS "${PROJECT_NAME} building infomation:")
MESSAGE (STATUS "  - Project source path is:     ${PROJECT_SOURCE_DIR}")
MESSAGE (STATUS "  - Project building path is:   ${CMAKE_BINARY_DIR}")
MESSAGE (STATUS "")


MESSAGE (STATUS "${PROJECT_NAME} other infomation:")
# show building install path
MESSAGE (STATUS "  Package install path:         ${CMAKE_INSTALL_PREFIX}")
MESSAGE (STATUS "")
