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
UNSET (_DEV_TORCH_HEADER_PATH)
UNSET (_TORCH_BASE_SOURCE)
UNSET (_TORCH_OPS_SOURCE)
UNSET (_DEV_TORCH_DEVICE_SOURCE)
UNSET (_DEV_TORCH_COMPILER_DEFINES)
UNSET (_DEV_TORCH_COMPILER_OPTIONS)
UNSET (_DEV_TORCH_LINKER_OPTIONS)
UNSET (_DEV_TORCH_LINK_LIBRARIES)


# 1.  set source root path
SET(_TORCH_ROOT ${CMAKE_SOURCE_DIR}/source/device/torch)


# 2.  add header file path
LIST (APPEND _DEV_TORCH_HEADER_PATH      ${_TORCH_ROOT})
LIST (APPEND _DEV_TORCH_HEADER_PATH      ${TORCH_INCLUDE_DIRS})


# 3.  add linking lib searching path
LIST (APPEND _DEV_TORCH_LINK_PATH        ${TORCH_LIBS})


# 4.  add source files
AUX_SOURCE_DIRECTORY("${_TORCH_ROOT}"    _TORCH_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_TORCH_ROOT}/op" _TORCH_OPS_SOURCE)
LIST (APPEND _DEV_TORCH_DEVICE_SOURCE    ${_TORCH_BASE_SOURCE})
LIST (APPEND _DEV_TORCH_DEVICE_SOURCE    ${_TORCH_OPS_SOURCE})


# 5.  add build options for cpu device
# 5.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
ENDIF()


# 5.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 6.  add link options


# 7.  add link libs
LIST (APPEND _DEV_TORCH_LINK_LIBRARIES   ${TORCH_LIBRARIES})


# 8. set all to cmake cache
SET (TENGINE_TORCH_HEADER_PATH       ${_DEV_TORCH_HEADER_PATH}        CACHE INTERNAL  "Tengine TensorRT device header files searching path"   FORCE)
SET (TENGINE_TORCH_LINK_PATH         ${_DEV_TORCH_LINK_PATH}          CACHE INTERNAL  "Tengine TensorRT device link libraries searching path" FORCE)
SET (TENGINE_TORCH_DEVICE_SOURCE     ${_DEV_TORCH_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine TensorRT device main source files"             FORCE)
SET (TENGINE_TORCH_COMPILER_DEFINES  ${_DEV_TORCH_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine TensorRT about compiler defines"               FORCE)
SET (TENGINE_TORCH_COMPILER_OPTIONS  ${_DEV_TORCH_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine TensorRT about compiler options"               FORCE)
SET (TENGINE_TORCH_LINKER_OPTIONS    ${_DEV_TORCH_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine TensorRT about linker options"                 FORCE)
SET (TENGINE_TORCH_LINK_LIBRARIES    ${_DEV_TORCH_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine TensorRT about link libraries"                 FORCE)


# 9. install device option
INSTALL (FILES ${_TORCH_ROOT}/torch_define.h DESTINATION include/tengine RENAME torch_device.h)



