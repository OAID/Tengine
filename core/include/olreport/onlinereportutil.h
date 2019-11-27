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

#ifndef __ONLINE_REPORT_UTIL_H__
#define __ONLINE_REPORT_UTIL_H__

#include <stdio.h>
#include <stdint.h>

#ifdef ONLINE_REPORT_DEBUG
#define OL_RP_LOG(fmt, ...) printf(fmt,## __VA_ARGS__)
#else
#define OL_RP_LOG(fmt, ...) 
#endif

#ifdef __cplusplus
extern "C" {
#endif

int get_rand();
uint32_t get_arch();
uint32_t get_totoal_memory();
void get_cur_process_info(uint32_t* id,char *name);
void get_os_info(char* os,int maxLen);
void get_os_kernel_info(char* os,int maxlen);
void get_cpu_param_info(int query_cpuid, char* cpu_id,uint32_t* cpu_freq,uint32_t *total_cpu_nums);

#ifdef __cplusplus
}
#endif

#endif
