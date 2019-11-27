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

#include "reportbasedata.h"
#include <string.h>

void strcpy_s(char* dst,const char* src,int maxlen )
{
    int src_len = strlen(src);
    if( src_len > maxlen )
    {
        src_len = maxlen - 1;
    }

    strncpy( dst,src,src_len );
    dst[src_len] = '\0';
}

#define IMP_UPDATE_INT_FUNC(NAME) \
void update_##NAME(REPORT_BASE_DATA_T* dat,uint32_t val) \
{\
    dat->NAME##_ = val; \
}\

#define IMP_UPDATE_STR_FUNC(NAME) \
void update_##NAME(REPORT_BASE_DATA_T* dat,const char* val) \
{\
    strcpy_s(dat->NAME##_,val,NAME##_LEN + 1); \
}\

IMP_UPDATE_STR_FUNC(API_VERSION);
IMP_UPDATE_STR_FUNC(KEY_TOKEN);
IMP_UPDATE_STR_FUNC(APPID);
IMP_UPDATE_STR_FUNC(IP);
IMP_UPDATE_STR_FUNC(OS);
IMP_UPDATE_STR_FUNC(ARCH);
IMP_UPDATE_STR_FUNC(KERNEL);
IMP_UPDATE_STR_FUNC(TENGINE_VERSION);
IMP_UPDATE_STR_FUNC(HCL_VERSION);
IMP_UPDATE_STR_FUNC(PROC);
IMP_UPDATE_STR_FUNC(UID);
IMP_UPDATE_STR_FUNC(CPUID);
IMP_UPDATE_STR_FUNC(ACTION);
IMP_UPDATE_STR_FUNC(SIGN);

IMP_UPDATE_INT_FUNC(CPUFREQ);
IMP_UPDATE_INT_FUNC(SESS);
IMP_UPDATE_INT_FUNC(MEM);
IMP_UPDATE_INT_FUNC(CPUNUM);
IMP_UPDATE_INT_FUNC(PID);
IMP_UPDATE_INT_FUNC(REPORT_DATE);
