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

#ifndef __ONLINE_REPORT_FMTDAT_H__
#define __ONLINE_REPORT_FMTDAT_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    void* node;
}OLREPORT_FMTDAT_T;

void init_online_report_fmt_dat(OLREPORT_FMTDAT_T* fmt_dat);
void release_online_report_fmt_dat(OLREPORT_FMTDAT_T* fmt_dat);

//
void add_string_val(OLREPORT_FMTDAT_T* fmt_dat,const char* key,const char* val);
void add_int_val(OLREPORT_FMTDAT_T* fmt_dat,const char* key,uint32_t val);

void add_string_val_order(OLREPORT_FMTDAT_T* fmt_dat,const char* key,const char* val);
void add_int_val_order(OLREPORT_FMTDAT_T* fmt_dat,const char* key,uint32_t val);

int report_fmt_data_2_json(OLREPORT_FMTDAT_T* fmt_dat,char* output);
int report_fmt_data_2_url_keyval_string(OLREPORT_FMTDAT_T* fmt_dat,char* output);

void gc_free_node();

#ifdef __cplusplus
}
#endif

#endif
