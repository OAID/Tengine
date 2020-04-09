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
 * Copyright (c) 2019, Open AI Lab
 * Author: jjzeng@openailab.com
 */

#ifndef __MD5_H__
#define __MD5_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    uint32_t total[2];          // number of bytes processed  
    uint32_t state[4];          // intermediate digest state
    unsigned char buffer[64];   // data block being processed 
}md5_context;

int get_md5(const unsigned char* dat,size_t len,unsigned char md5[16]);

int md5_start(md5_context* ctx);
int md5_update(md5_context* ctx,const unsigned char* dat,size_t len);
int md5_finish(md5_context* ctx,unsigned char md5[16]);

void md5_to_string(const unsigned char md5[16],char* str_md5);

#ifdef __cplusplus
}
#endif

#endif
