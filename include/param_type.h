/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __PARAM_TYPE_H__
#define __PARAM_TYPE_H__

enum
{
    PE_GENERIC = 0,
    PE_INT32,
    PE_FP32,
    PE_INT_PTR,
    PE_CHAR_PTR,
    PE_VOID_PTR,
    PE_FP32_PTR
};

#define GET_PARAM_ENTRY_TYPE(var)                                               \
    ({                                                                          \
        int ret = PE_GENERIC;                                                   \
        if(__builtin_types_compatible_p(typeof(var), typeof(int)))              \
            ret = PE_INT32;                                                     \
        else if(__builtin_types_compatible_p(typeof(var), typeof(float)))       \
            ret = PE_FP32;                                                      \
        else if(__builtin_types_compatible_p(typeof(var), typeof(int*)))        \
            ret = PE_INT_PTR;                                                   \
        else if(__builtin_types_compatible_p(typeof(var), typeof(char*)) ||     \
                __builtin_types_compatible_p(typeof(var), typeof(const char*))) \
            ret = PE_CHAR_PTR;                                                  \
        else if(__builtin_types_compatible_p(typeof(var), typeof(void*)))       \
            ret = PE_VOID_PTR;                                                  \
        else if(__builtin_types_compatible_p(typeof(var), typeof(float*)))      \
            ret = PE_FP32_PTR;                                                  \
                                                                                \
        ret;                                                                    \
    })

#endif
