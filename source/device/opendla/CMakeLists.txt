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
# Copyright (c) 2021, Institute of Computing Technology
# Author: wanglei21c@mails.ucas.ac.cn
#

# 0. clear var
UNSET (_DEV_ODLA_HEADER_PATH)
UNSET (_ODLA_BASE_SOURCE)
UNSET (_ODLA_OPS_SOURCE)
UNSET (_DEV_ODLA_DEVICE_SOURCE)
UNSET (_DEV_ODLA_COMPILER_DEFINES)
UNSET (_DEV_ODLA_COMPILER_OPTIONS)
UNSET (_DEV_ODLA_LINKER_OPTIONS)
UNSET (_DEV_ODLA_LINK_LIBRARIES)


# 1.  set source root path
SET(_ODLA_ROOT ${CMAKE_SOURCE_DIR}/source/device/opendla)

# 2.  add header file path
LIST (APPEND _DEV_ODLA_HEADER_PATH        ${_ODLA_ROOT})
LIST (APPEND _DEV_ODLA_HEADER_PATH        ${CMAKE_CURRENT_BINARY_DIR})
LIST (APPEND _DEV_ODLA_HEADER_PATH        ${CMAKE_SOURCE_DIR}/source/device/opendla/include/)
LIST (APPEND _DEV_ODLA_HEADER_PATH        ${CMAKE_SOURCE_DIR}/source/device/opendla/include/common)
LIST (APPEND _DEV_ODLA_HEADER_PATH        ${CMAKE_SOURCE_DIR}/source/device/opendla/include/compiler)
LIST (APPEND _DEV_ODLA_HEADER_PATH        ${CMAKE_SOURCE_DIR}/source/device/opendla/include/runtime)
LIST (APPEND _DEV_ODLA_HEADER_PATH        ${CMAKE_SOURCE_DIR}/source/device/opendla/include/external)


# 3.
LIST (APPEND _DEV_ODLA_COMPILER_DEFINES    NVDLA_UTILS_ERROR_TAG="DLA")

# 3.  add source files
AUX_SOURCE_DIRECTORY ("${_ODLA_ROOT}"     _ODLA_BASE_SOURCE)
AUX_SOURCE_DIRECTORY ("${_ODLA_ROOT}/op"  _ODLA_OPS_SOURCE)

LIST (APPEND _DEV_ODLA_DEVICE_SOURCE      ${_ODLA_BASE_SOURCE})
LIST (APPEND _DEV_ODLA_DEVICE_SOURCE      ${_ODLA_OPS_SOURCE})

SET (_ODLA_LIB_NAME "odla")
ADD_LIBRARY (${_ODLA_LIB_NAME} STATIC)
TARGET_SOURCES (${_ODLA_LIB_NAME} PRIVATE )
TARGET_SOURCES (${_ODLA_LIB_NAME} PRIVATE ${_ODLA_BASE_SOURCE})
TARGET_SOURCES (${_ODLA_LIB_NAME} PRIVATE ${_ODLA_OPS_SOURCE})
# 4.  add build options for cpu device
# 4.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
ENDIF()


# 4.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 5.  add link options

# 6.  add link libs
LIST (APPEND _DEV_ODLA_LINK_PATH            ${_ODLA_ROOT}/lib)
LIST (APPEND _DEV_ODLA_LINK_LIBRARIES       "nvdla_compiler" )
LIST (APPEND _DEV_ODLA_LINK_LIBRARIES       "libprotobuf.a" )
LIST (APPEND _DEV_ODLA_LINK_LIBRARIES       "pthread")
LIST (APPEND _DEV_ODLA_LINK_LIBRARIES       "nvdla_runtime" )


# 7. set all to cmake cache
SET (TENGINE_ODLA_HEADER_PATH       ${_DEV_ODLA_HEADER_PATH}        CACHE INTERNAL  "Tengine ODLA device header files searching path"   FORCE)
SET (TENGINE_ODLA_LINK_PATH         ${_DEV_ODLA_LINK_PATH}          CACHE INTERNAL  "Tengine ODLA device link libraries searching path" FORCE)
SET (TENGINE_ODLA_DEVICE_SOURCE     ${_DEV_ODLA_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine ODLA device main source files"             FORCE)
SET (TENGINE_ODLA_COMPILER_DEFINES  ${_DEV_ODLA_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine ODLA about compiler defines"               FORCE)
SET (TENGINE_ODLA_COMPILER_OPTIONS  ${_DEV_ODLA_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine ODLA about compiler options"               FORCE)
SET (TENGINE_ODLA_LINKER_OPTIONS    ${_DEV_ODLA_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine ODLA about linker options"                 FORCE)
SET (TENGINE_ODLA_LINK_LIBRARIES    ${_DEV_ODLA_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine ODLA about link libraries"                 FORCE)


# 8. install device option
INSTALL (FILES ${CMAKE_CURRENT_BINARY_DIR}/odla_define.h DESTINATION include/tengine RENAME odla_device.h)
