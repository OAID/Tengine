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
 * Copyright (c) 2018, Open AI Lab
 * Author: jingyou@openailab.com
 */
#include <string.h>
#include "tm_generate.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ALIGN(pos, alignbytes) (((pos) + ( alignbytes )-1) & ~(( alignbytes )-1))

uint32_t WriteTmFileAlign1(void* const start_ptr, uint32_t* cur_pos, const void* buf, const uint32_t buf_size)
{
    uint32_t buf_pos = *cur_pos;
    memcpy(start_ptr + *cur_pos, buf, buf_size);
    *cur_pos += buf_size;
    return buf_pos;
}

uint32_t WriteTmFileAlign4(void* const start_ptr, uint32_t* cur_pos, const void* buf, const uint32_t buf_size)
{
    *cur_pos = ALIGN(*cur_pos, 4);

    return WriteTmFileAlign1(start_ptr, cur_pos, buf, buf_size);
}

uint32_t WriteTmObject(void* const start_ptr, uint32_t* cur_pos, const void* buf, const uint32_t buf_size)
{
    return WriteTmFileAlign4(start_ptr, cur_pos, buf, buf_size);
}

#ifdef __cplusplus
}
#endif
