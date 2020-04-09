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

#ifndef __ONLINE_REPORT_MGR_H__
#define __ONLINE_REPORT_MGR_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* ONLINE_REPORT_CONTEXT_T;
typedef uint32_t TIMER_ID_T;
#define INVLIAD_TIMER_ID 0
#define PERIOD_ACTION 0x0acca

//implementor must guarantee the pointer of return value is valid 
typedef const char* (*get_online_report_dat_t)(int* out_len,int action);
 
ONLINE_REPORT_CONTEXT_T init_report_mgr(const char* host,uint16_t port,const char* request_url,get_online_report_dat_t get_dat_fun);
void free_report_mgr(ONLINE_REPORT_CONTEXT_T ctx);
int set_report_status(ONLINE_REPORT_CONTEXT_T ctx,int status);

TIMER_ID_T add_timer(ONLINE_REPORT_CONTEXT_T ctx,uint32_t timer,int is_loop); 
void remove_timer(ONLINE_REPORT_CONTEXT_T ctx,TIMER_ID_T id);              

int do_report(ONLINE_REPORT_CONTEXT_T ctx,int action);

#ifdef __cplusplus
}
#endif

#endif
