# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2021, OPEN AI LAB
# Author: lswang@openailab.com
#

# 0. clear var
UNSET (_DEV_TIM_VX_HEADER_PATH)
UNSET (_TIM_VX_BASE_SOURCE)
UNSET (_TIM_VX_OPS_SOURCE)
UNSET (_DEV_TIM_VX_DEVICE_SOURCE)
UNSET (_DEV_TIM_VX_COMPILER_DEFINES)
UNSET (_DEV_TIM_VX_COMPILER_OPTIONS)
UNSET (_DEV_TIM_VX_LINKER_OPTIONS)
UNSET (_DEV_TIM_VX_LINK_LIBRARIES)


# 1. integrate TIM-VX for default
IF (NOT (DEFINED TENGINE_ENABLE_TIM_VX_INTEGRATION))
    SET (TENGINE_ENABLE_TIM_VX_INTEGRATION ON)
ENDIF()

# 2. check arch
IF (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
    SET (_VX_ARCH "aarch64")
ELSEIF (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    SET (_VX_ARCH "x86_64")
ELSE()
    MESSAGE (FATAL_ERROR "Tengine: Unsupported TIM-VX arch:${TENGINE_TARGET_PROCESSOR}")
ENDIF()

# 3. check & set TIM-VX integration library
IF (TENGINE_ENABLE_TIM_VX_INTEGRATION)
    # set
    SET (_TIM_VX_NAME "tim_vx_internal")

    IF (TIM_VX_SOURCE_DIR)      # if TIM-VX source directory is set manually
        SET (_VX_ROOT ${TIM_VX_SOURCE_DIR}/src)
        IF ((EXISTS ${_VX_ROOT}/tim) AND (IS_DIRECTORY ${_VX_ROOT}/tim))
            MESSAGE (FATAL_ERROR "Tengine: TIM-VX source was manually set as ${TIM_VX_SOURCE_DIR}, but it's not exists ${TIM_VX_SOURCE_DIR}/src.")
        ENDIF()
    ELSE()                      # if TIM-VX source directory is not set, than check if copied manually
        SET (_VX_ROOT "${CMAKE_SOURCE_DIR}/source/device/tim-vx/src")
        IF ((EXISTS ${_VX_ROOT}/tim) AND (IS_DIRECTORY ${_VX_ROOT}/tim))
            MESSAGE (STATUS "Tengine: TIM-VX source was found in ${_VX_ROOT}/tim.")
        ELSE()                  # TODO: to download TIM-VX, for not exist
            MESSAGE (FATAL_ERROR "Tengine: TIM-VX source was not found. Please read doc/npu_tim-vx_user_manual.md for more info.")
        ENDIF()
    ENDIF()

    # for ovx api
    SET (_VX_OVX_ROOT ${_VX_ROOT}/tim/vx/internal/src)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}                       _VX_OVX_SRC)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/kernel                _VX_OVX_KERNEL)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/kernel/cl             _VX_OVX_KERNEL_CL)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/kernel/cpu            _VX_OVX_KERNEL_CPU)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/kernel/evis           _VX_OVX_KERNEL_EVIS)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/kernel/vx             _VX_OVX_KERNEL_VX)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/ops                   _VX_OVX_OPS)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/client                _VX_OVX_CLIENT)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/libnnext              _VX_OVX_LIBNNEXT)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/libnnext/ops/kernel   _VX_OVX_LIBNNEXT_OPS_KERNEL)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/quantization          _VX_OVX_QUANTIZATION)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/custom/ops            _VX_OVX_CUSTOM_OPS)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/custom/ops/kernel     _VX_OVX_CUSTOM_OPS_KERNEL)
    AUX_SOURCE_DIRECTORY (${_VX_OVX_ROOT}/utils                 _VX_OVX_UTILS)

    # for TIM-VX
    AUX_SOURCE_DIRECTORY (${_VX_ROOT}/tim/vx     _VX_BASE_SRC)
    AUX_SOURCE_DIRECTORY (${_VX_ROOT}/tim/vx/ops _VX_OPS_SRC)

    
    # all
    SET (_VX_SRC)
    LIST (APPEND      _VX_SRC ${_VX_BASE_SRC} ${_VX_OPS_SRC})
    LIST (REMOVE_ITEM _VX_SRC ${_VX_ROOT}/tim/vx/context_test.cc)
    LIST (REMOVE_ITEM _VX_SRC ${_VX_ROOT}/tim/vx/graph_test.cc)

    LIST (APPEND _VX_SRC ${_VX_OVX_SRC})
    LIST (APPEND _VX_SRC ${_VX_OVX_KERNEL})
    LIST (APPEND _VX_SRC ${_VX_OVX_KERNEL_CL})
    LIST (APPEND _VX_SRC ${_VX_OVX_KERNEL_CPU})
    LIST (APPEND _VX_SRC ${_VX_OVX_KERNEL_EVIS})
    LIST (APPEND _VX_SRC ${_VX_OVX_KERNEL_VX})
    LIST (APPEND _VX_SRC ${_VX_OVX_OPS})
    LIST (APPEND _VX_SRC ${_VX_OVX_CLIENT})
    LIST (APPEND _VX_SRC ${_VX_OVX_LIBNNEXT})
    LIST (APPEND _VX_SRC ${_VX_OVX_LIBNNEXT_OPS_KERNEL})
    LIST (APPEND _VX_SRC ${_VX_OVX_QUANTIZATION})
    LIST (APPEND _VX_SRC ${_VX_OVX_CUSTOM_OPS})
    LIST (APPEND _VX_SRC ${_VX_OVX_CUSTOM_OPS_KERNEL})
    LIST (APPEND _VX_SRC ${_VX_OVX_UTILS})
    LIST (APPEND _VX_SRC ${_VX_SRC})

    # build
    ADD_LIBRARY (${_TIM_VX_NAME} STATIC ${_VX_SRC})

    SET_TARGET_PROPERTIES (${_TIM_VX_NAME} PROPERTIES CXX_STANDARD_REQUIRED 14)
    SET_TARGET_PROPERTIES (${_TIM_VX_NAME} PROPERTIES CXX_STANDARD 14)

    SET (_VX_OVX_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
    TARGET_COMPILE_DEFINITIONS (${_TIM_VX_NAME} PRIVATE "-DOVXLIB_API=${_VX_OVX_API_ATTR}")

    TARGET_COMPILE_OPTIONS (${_TIM_VX_NAME} PRIVATE -fPIC)

    TARGET_INCLUDE_DIRECTORIES (${_TIM_VX_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/tim-vx/include)
    TARGET_INCLUDE_DIRECTORIES (${_TIM_VX_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source/device/tim-vx/include)
    TARGET_INCLUDE_DIRECTORIES (${_TIM_VX_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source/device/tim-vx/include/tim/vx)
    TARGET_INCLUDE_DIRECTORIES (${_TIM_VX_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source/device/tim-vx/src/tim/vx)
    TARGET_INCLUDE_DIRECTORIES (${_TIM_VX_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/source/device/tim-vx/src/tim/vx/internal/include)

    TARGET_LINK_DIRECTORIES (${_TIM_VX_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/tim-vx/lib/${_VX_ARCH})

    TARGET_LINK_LIBRARIES (${_TIM_VX_NAME} PRIVATE CLC GAL OpenVX OpenVXU VSC ArchModelSw NNArchPerf)
ENDIF()


# 4.  set source root path
SET(_TIM_VX_ROOT ${CMAKE_SOURCE_DIR}/source/device/tim-vx)

# 5.  add header file path
LIST (APPEND _DEV_TIM_VX_HEADER_PATH        ${_TIM_VX_ROOT})
LIST (APPEND _DEV_TIM_VX_HEADER_PATH        ${CMAKE_SOURCE_DIR}/3rdparty/tim-vx/include)
LIST (APPEND _DEV_TIM_VX_HEADER_PATH        ${CMAKE_SOURCE_DIR}/source/device/tim-vx/include)


# 6.  add linking lib searching path
LIST (APPEND _DEV_TIM_VX_LINK_PATH          ${CMAKE_SOURCE_DIR}/3rdparty/tim-vx/lib/${_VX_ARCH})


# 7.  add source files
AUX_SOURCE_DIRECTORY ("${_TIM_VX_ROOT}"     _TIM_VX_BASE_SOURCE)
AUX_SOURCE_DIRECTORY ("${_TIM_VX_ROOT}/op"  _TIM_VX_OPS_SOURCE)
LIST (APPEND _DEV_TIM_VX_DEVICE_SOURCE      ${_TIM_VX_BASE_SOURCE})
LIST (APPEND _DEV_TIM_VX_DEVICE_SOURCE      ${_TIM_VX_OPS_SOURCE})


# 8.  add build options for cpu device
# 8.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
ENDIF()


# 8.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 9.  add link options


# 10.  add link libs
IF (TENGINE_ENABLE_TIM_VX_INTEGRATION)
    LIST (APPEND _DEV_TIM_VX_LINK_LIBRARIES   -Wl,--whole-archive ${_TIM_VX_NAME} -Wl,--no-whole-archive)
ELSE()
    LIST (APPEND _DEV_TIM_VX_LINK_LIBRARIES   tim-vx)
ENDIF()


# 11. set all to cmake cache
SET (TENGINE_TIM_VX_HEADER_PATH       ${_DEV_TIM_VX_HEADER_PATH}        CACHE INTERNAL  "Tengine TIM_VX device header files searching path"   FORCE)
SET (TENGINE_TIM_VX_LINK_PATH         ${_DEV_TIM_VX_LINK_PATH}          CACHE INTERNAL  "Tengine TIM_VX device link libraries searching path" FORCE)
SET (TENGINE_TIM_VX_DEVICE_SOURCE     ${_DEV_TIM_VX_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine TIM_VX device main source files"             FORCE)
SET (TENGINE_TIM_VX_COMPILER_DEFINES  ${_DEV_TIM_VX_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine TIM_VX about compiler defines"               FORCE)
SET (TENGINE_TIM_VX_COMPILER_OPTIONS  ${_DEV_TIM_VX_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine TIM_VX about compiler options"               FORCE)
SET (TENGINE_TIM_VX_LINKER_OPTIONS    ${_DEV_TIM_VX_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine TIM_VX about linker options"                 FORCE)
SET (TENGINE_TIM_VX_LINK_LIBRARIES    ${_DEV_TIM_VX_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine TIM_VX about link libraries"                 FORCE)


# 12. install device option
INSTALL (FILES ${_TIM_VX_ROOT}/timvx_define.h DESTINATION include/tengine RENAME timvx_device.h)
