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

#
UNSET (_TENGINE_SCHEDULER_SOURCE)


SET (_SCH_ROOT "${CMAKE_SOURCE_DIR}/source/scheduler")


# scheduler level source files
FILE (GLOB   _SCH_LEVEL_SOURCE  "${_SCH_ROOT}/*.c")


# gather
LIST (APPEND _TENGINE_SCHEDULER_SOURCE  ${_SCH_LEVEL_SOURCE})


#
TENGINE_SOURCE_GROUP ("scheduler"    ${_SCH_LEVEL_SOURCE})


# set all to cache
SET (TENGINE_SCHEDULER_SOURCE ${_TENGINE_SCHEDULER_SOURCE} CACHE INTERNAL "Tengine scheduler level source files" FORCE)
