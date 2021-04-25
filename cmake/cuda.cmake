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

IF (TENGINE_ENABLE_CUDA OR TENGINE_ENABLE_TENSORRT)
    IF (${CMAKE_MINOR_VERSION} LESS 17)
        MESSAGE (FATAL_ERROR "Tengine: Backend needs CMake version >= 3.17.")
    ENDIF()

    FIND_PACKAGE(CUDAToolkit REQUIRED)

    SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules)
    FIND_PACKAGE(CUDNN REQUIRED)
ENDIF()

IF (TENGINE_ENABLE_CUDA)
    # enable language CUDA
    SET (CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

    INCLUDE (CheckLanguage)
    CHECK_LANGUAGE (CUDA)

    IF (CMAKE_CUDA_COMPILER)
        ENABLE_LANGUAGE(CUDA)
    ELSE()
        MESSAGE (WARNING "Tengine: To enable CUDA backend, \"CMAKE_CUDA_HOST_COMPILER\" must be set.")
        MESSAGE (FATAL_ERROR "Tengine: Cannot find nvcc from any CUDA version.")
    ENDIF()
ENDIF()
