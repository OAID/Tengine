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

#ifndef __SYS_COMPILER_H__
#define __SYS_COMPILER_H__

#define DLLEXPORT __attribute__((visibility("default")))

#ifndef offsetof
#define offsetof(TYPE, MEMBER) ((size_t) & (( TYPE* )0)->MEMBER)
#endif

#define container_of(ptr, type, member)                      \
    ({                                                       \
        const typeof((( type* )0)->member)* __mptr = (ptr);  \
        ( type* )(( char* )__mptr - offsetof(type, member)); \
    })

#define UNIQ_DUMMY_NAME_WITH_LINE0(a, b) auto_dummy_##a##_##b

#define UNIQ_DUMMY_NAME_WITH_LINE(a, b) UNIQ_DUMMY_NAME_WITH_LINE0(a, b)

#define UNIQ_DUMMY_NAME(name) UNIQ_DUMMY_NAME_WITH_LINE(name, __LINE__)

#endif
