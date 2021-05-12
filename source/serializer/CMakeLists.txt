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

# unset for regeneration
UNSET (_SRL_HEADER_PATH)
UNSET (_TENGINE_SRL_SOURCE)


# set roots
SET (_SRL_SRC_ROOT                  "${CMAKE_SOURCE_DIR}/source/serializer")
SET (_SRL_BIN_ROOT                  "${CMAKE_BINARY_DIR}/source/serializer")


# set header searching path
LIST (APPEND _SRL_HEADER_PATH       "${_SRL_SRC_ROOT}")
LIST (APPEND _SRL_HEADER_PATH       "${_SRL_BIN_ROOT}")


# gather all serializer
FILE (GLOB   _SRL_LEVEL_SOURCE      "${_SRL_SRC_ROOT}/*.c")
FILE (GLOB   _SRL_TM2_SRL_SOURCE    "${_SRL_SRC_ROOT}/tmfile/*.c")
FILE (GLOB   _SRL_TM2_OPS_SOURCE    "${_SRL_SRC_ROOT}/tmfile/op/*.c")


# generate all serializer
GENERATE_REGISTER_HEADER_FILE("register_" "unregister_" "" "${_SRL_SRC_ROOT}/register.h.in" "${_SRL_BIN_ROOT}/register.h" "${_SRL_TM2_SRL_SOURCE}")

# generator ops for serializer
AUX_SOURCE_DIRECTORY("${_SRL_SRC_ROOT}/tmfile/op" _TENGINE_TM2_OPS_SOURCE)
GENERATE_REGISTER_HEADER_FILE("register_" "unregister_" "_op" "${_SRL_SRC_ROOT}/tmfile/tm2_ops.h.in" "${_SRL_BIN_ROOT}/tmfile/tm2_ops.h" "${_SRL_TM2_OPS_SOURCE}")


# gather
LIST (APPEND _TENGINE_SRL_SOURCE ${_SRL_LEVEL_SOURCE})
LIST (APPEND _TENGINE_SRL_SOURCE ${_SRL_TM2_SRL_SOURCE})
LIST (APPEND _TENGINE_SRL_SOURCE ${_SRL_TM2_OPS_SOURCE})


#
TENGINE_SOURCE_GROUP ("serializer"              ${_SRL_LEVEL_SOURCE})
TENGINE_SOURCE_GROUP ("serializer/tmfile"       ${_SRL_TM2_SRL_SOURCE})
TENGINE_SOURCE_GROUP ("serializer/tmfile/op"    ${_SRL_TM2_OPS_SOURCE})


# set all to cache
SET (TENGINE_SERIALIZER_HEADER_PATH ${_SRL_HEADER_PATH}     CACHE INTERNAL "Tengine serializer level header searching path" FORCE)
SET (TENGINE_SERIALIZER_SOURCE      ${_TENGINE_SRL_SOURCE}  CACHE INTERNAL "Tengine serializer level source files" FORCE)
