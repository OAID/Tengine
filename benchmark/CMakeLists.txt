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

# generate tengine header file
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)


# examples function
FUNCTION (TENGINE_BENCHMARK name)
    ADD_EXECUTABLE (${name})

    # add common source file
    TARGET_SOURCES (${name} PRIVATE common/timer.cc)

    # add source files
    FOREACH (_VAR ${ARGN})
        TARGET_SOURCES (${name} PRIVATE ${_VAR})
    ENDFOREACH()

    # header file path
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    # add linking library
    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME})

    # add to a virtual project group
    SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "benchmark")
ENDFUNCTION()


# add benchmark
TENGINE_BENCHMARK (tm_benchmark      tm_benchmark.cc)
