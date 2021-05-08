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
# Author: hhchen@openailab.com
#

#[=======================================================================[.rst:
FindCUDNN
---------

This script locates the NVIDIA CUDA Deep Neural Network library (cuDNN) is
a GPU-accelerated library.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines :prop_tgt:`IMPORTED` target ``cuDNN::cuDNN``, if
cuDNN has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables::

  CUDNN_FOUND          - True if CUDNN was found
  CUDNN_INCLUDE_DIR    - include directories for CUDNN
  CUDNN_LIBRARY        - link against this library to use CUDNN
  CUDNN_LIBRARIES      - link against this full path library to use CUDNN
  CUDNN_LIBRARY_DIR    - link directories for CUDNN
  CUDNN_VERSION_STRING - Highest supported CUDNN version (eg. 1.2)
  CUDNN_VERSION_MAJOR  - The major version of the CUDNN implementation
  CUDNN_VERSION_MINOR  - The minor version of the CUDNN implementation


#]=======================================================================]

INCLUDE (FindPackageHandleStandardArgs)

SET (_INC_SEARCH_PATH)
LIST (APPEND _INC_SEARCH_PATH "/usr/include")
LIST (APPEND _INC_SEARCH_PATH "/usr/include/x86_64-linux-gnu")
LIST (APPEND _INC_SEARCH_PATH "/usr/include/aarch64-linux-gnu")
LIST (APPEND _INC_SEARCH_PATH ${CUDNN_ROOT})
LIST (APPEND _INC_SEARCH_PATH $ENV{CUDNN_ROOT})
LIST (APPEND _INC_SEARCH_PATH ${CUDNN_ROOT}/include)
LIST (APPEND _INC_SEARCH_PATH $ENV{CUDNN_ROOT}/include)
LIST (APPEND _INC_SEARCH_PATH ${CUDNN_INCLUDE_DIR})
LIST (APPEND _INC_SEARCH_PATH $ENV{CUDNN_INCLUDE_DIR})
LIST (APPEND _INC_SEARCH_PATH ${CUDA_INCLUDE_DIRS})
LIST (APPEND _INC_SEARCH_PATH $ENV{CUDNN_INCLUDE_DIRS})
LIST (APPEND _INC_SEARCH_PATH ${CUDAToolkit_INCLUDE_DIRS})

SET (_LIB_SEARCH_PATH)
LIST (APPEND _LIB_SEARCH_PATH "/usr/lib")
LIST (APPEND _LIB_SEARCH_PATH "/usr/lib/aarch64-linux-gnu")
LIST (APPEND _LIB_SEARCH_PATH ${CUDNN_ROOT})
LIST (APPEND _LIB_SEARCH_PATH $ENV{CUDNN_ROOT})
LIST (APPEND _LIB_SEARCH_PATH ${CUDNN_LIBRARY_DIR})
LIST (APPEND _LIB_SEARCH_PATH $ENV{CUDNN_LIBRARY_DIR})
LIST (APPEND _LIB_SEARCH_PATH ${CUDA_SDK_ROOT_DIR})
LIST (APPEND _LIB_SEARCH_PATH ${CUDA_TOOLKIT_ROOT_DIR})
LIST (APPEND _LIB_SEARCH_PATH ${CUDAToolkit_LIBRARY_ROOT})
LIST (APPEND _LIB_SEARCH_PATH ${CUDAToolkit_BIN_DIR})
LIST (APPEND _LIB_SEARCH_PATH ${CUDAToolkit_LIBRARY_DIR})


# find cudnn.h
FIND_PATH (CUDNN_INCLUDE_DIR cudnn.h PATHS ${_INC_SEARCH_PATH} DOC "location of cudnn.h" NO_DEFAULT_PATH)

# extract version from the include
IF (CUDNN_INCLUDE_DIR)
    FIND_PATH (_VERSION_PATH cudnn_version.h PATHS ${CUDNN_INCLUDE_DIR} DOC "location of cudnn_version.h" NO_DEFAULT_PATH)
    IF (_VERSION_PATH)
        FILE (READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_H_CONTENTS)
        UNSET (_VERSION_PATH)
    ELSE()
        FILE (READ "${CUDNN_INCLUDE_DIR}/cudnn.h" CUDNN_H_CONTENTS)
    ENDIF()

    STRING (REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    SET (CUDNN_MAJOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")
    STRING (REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    set(CUDNN_MINOR_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")
    STRING (REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
    SET (CUDNN_PATCH_VERSION ${CMAKE_MATCH_1} CACHE INTERNAL "")

    SET (
            CUDNN_VERSION
            "${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}.${CUDNN_PATCH_VERSION}"
            CACHE
            STRING
            "CUDNN version"
    )

    UNSET (CUDNN_H_CONTENTS)
ENDIF()


# find libcudnn.so or cudnn.lib
FIND_LIBRARY (
        CUDNN_LIBRARY
        NAMES
            libcudnn.so.${CUDNN_MAJOR_VERSION}
            libcudnn.so
            cudnn.lib
            # libcudnn_static.a
        PATHS
            ${_LIB_SEARCH_PATH}
        PATH_SUFFIXES
            lib
            lib32
            lib64
            lib/x32
            lib/x64
            cuda/lib
            cuda/lib64
        DOC
            "Path to cuDNN library."
)

# get library path
FIND_PATH (
    CUDNN_LIBRARY_DIR
        ${CUDNN_LIBRARY}
        NAMES
        libcudnn.so.${CUDNN_MAJOR_VERSION}
        libcudnn.so
        cudnn.lib
        # libcudnn_static.a
    PATHS
        ${_LIB_SEARCH_PATH}
    PATH_SUFFIXES
        lib
        lib32
        lib64
        lib/x32
        lib/x64
        cuda/lib
        cuda/lib64
    DOC
        "location of cudnn_version.h"
    NO_DEFAULT_PATH)

# make args
FIND_PACKAGE_HANDLE_STANDARD_ARGS (
    CUDNN
    FOUND_VAR
        CUDNN_FOUND
    REQUIRED_VARS
        CUDNN_INCLUDE_DIR
        CUDNN_LIBRARY
        CUDNN_LIBRARY_DIR
    VERSION_VAR
        CUDNN_VERSION
)


MARK_AS_ADVANCED (
    CUDNN_INCLUDE_DIR
    CUDNN_LIBRARY
    CUDNN_LIBRARY_DIR
    CUDNN_VERSION
)


IF (CUDNN_FOUND AND NOT TARGET cuDNN::cuDNN)
    IF (OpenCL_LIBRARY MATCHES "/([^/]+)\\.framework$")
        ADD_LIBRARY (cuDNN::cuDNN INTERFACE IMPORTED)
        SET_TARGET_PROPERTIES (cuDNN::cuDNN PROPERTIES INTERFACE_LINK_LIBRARIES   ${CUDNN_LIBRARY})
        SET_TARGET_PROPERTIES (cuDNN::cuDNN PROPERTIES INTERFACE_LINK_DIRECTORIES ${CUDNN_LIBRARY_DIR})
    ELSE()
        ADD_LIBRARY (cuDNN::cuDNN UNKNOWN IMPORTED)
        SET_TARGET_PROPERTIES (cuDNN::cuDNN PROPERTIES IMPORTED_LOCATION          ${CUDNN_LIBRARY})
        SET_TARGET_PROPERTIES (cuDNN::cuDNN PROPERTIES INTERFACE_LINK_DIRECTORIES ${CUDNN_LIBRARY_DIR})
    ENDIF()
    SET_TARGET_PROPERTIES (cuDNN::cuDNN PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
ENDIF()
