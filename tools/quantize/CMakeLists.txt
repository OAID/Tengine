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

UNSET (TENGINE_TOOL_OPTIONS)
UNSET (TENGINE_TOOL_LINK_LIBRARIES)

FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)

FIND_PACKAGE(OpenCV QUIET)

IF (OpenCV_FOUND AND ${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    # macro for adding quant tool
    FUNCTION (TENGINE_QUANT_TOOL name file)
        ADD_EXECUTABLE(
            ${name}
            ./quant_save_graph.cpp
            ./quant_utils.cpp
            ./savegraph/save_graph.cpp 
            ./savegraph/tm2_op_save.cpp 
            ./savegraph/tm2_generate.c     
            "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})       
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_BINARY_DIR}")     
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source/operator/prototype")       
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tools/quantize/common")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tools/quantize/savegraph")

        IF (${TENGINE_COVERAGE})
            LIST (APPEND TENGINE_TOOL_OPTIONS "-fprofile-arcs")
            LIST (APPEND TENGINE_TOOL_OPTIONS "-ftest-coverage")
            LIST (APPEND TENGINE_TOOL_LINK_LIBRARIES -coverage)
            LIST (APPEND TENGINE_TOOL_LINK_LIBRARIES -lgcov)
        ENDIF()
        TARGET_COMPILE_OPTIONS (${name} PRIVATE ${TENGINE_TOOL_OPTIONS})
        TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME}-static ${OpenCV_LIBS} ${TENGINE_TOOL_LINK_LIBRARIES})

        INSTALL (TARGETS ${name} DESTINATION bin)

        # add to a virtual project group
        SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "tools/quant_tool")
    ENDFUNCTION()

    TENGINE_QUANT_TOOL(quant_tool_uint8 quant_tool_uint8.cpp)
ELSE()
    MESSAGE (WARNING "OpenCV not found, quantization tool won't be built")
ENDIF()
