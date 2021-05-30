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
# Author: qtang@openailab.com
#

FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)

FIND_PACKAGE(OpenCV QUIET)

IF (OpenCV_FOUND AND ${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    # macro for adding quant tool
    FUNCTION (TENGINE_QUANT_TOOL name file)
        ADD_EXECUTABLE(
            ${name}
            ./quant_save_graph.cpp 
            
            ../savegraph/save_graph.cpp 
            ../savegraph/tm2_op_save.cpp 
            ../savegraph/tm2_generate.c 
            ../common/common.cpp
            ../common/quant_utils.cpp
            "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})       
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_BINARY_DIR}")     
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source/operator/prototype")       
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tools/quantize/common")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tools/quantize/savegraph")
        TARGET_LINK_LIBRARIES(${name} PRIVATE -Wl,--whole-archive ${CMAKE_PROJECT_NAME}-static -Wl,--no-whole-archive ${OpenCV_LIBS})

        INSTALL (TARGETS ${name} DESTINATION bin)
    ENDFUNCTION()

    TENGINE_QUANT_TOOL(quant_tool_uint8 quant_tool_uint8.cpp)
ELSE()
    MESSAGE (WARNING "OpenCV not found, quantization tool won't be built")
ENDIF()
