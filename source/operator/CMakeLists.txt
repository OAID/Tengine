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
UNSET (_OP_HEADER_PATH)
UNSET (_TENGINE_OP_SOURCE)


SET (_OP_ROOT "${CMAKE_SOURCE_DIR}/source/operator")

# generate header file searching path
LIST (APPEND _OP_HEADER_PATH "${CMAKE_SOURCE_DIR}/source/operator/prototype")


# config the operator prototype
AUX_SOURCE_DIRECTORY("${_OP_ROOT}/prototype" _OP_PROTOTYPE_SOURCE)
GENERATE_REGISTER_HEADER_FILE("register_" "unregister_" "_op" "${_OP_ROOT}/prototype.h.in" "${CMAKE_BINARY_DIR}/source/operator/prototype.h" "${_OP_PROTOTYPE_SOURCE}")


# operator level source files
FILE (GLOB   _OP_LEVEL_SOURCE   "${CMAKE_SOURCE_DIR}/source/operator/*.c")


# gather
LIST (APPEND _TENGINE_OP_SOURCE ${_OP_LEVEL_SOURCE})
LIST (APPEND _TENGINE_OP_SOURCE ${_OP_PROTOTYPE_SOURCE})


# add to src group
TENGINE_SOURCE_GROUP ("operator"            ${_OP_LEVEL_SOURCE})
TENGINE_SOURCE_GROUP ("operator/prototype"  ${_OP_PROTOTYPE_SOURCE})


# set all to cache
SET (TENGINE_OP_HEADER_PATH     ${_OP_HEADER_PATH}      CACHE INTERNAL "Tengine operator level header searching path"   FORCE)
SET (TENGINE_OPERATOR_SOURCE    ${_TENGINE_OP_SOURCE}   CACHE INTERNAL "Tengine operator level source files"            FORCE)
