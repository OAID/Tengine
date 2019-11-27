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

#ifndef __REPORT_BASE_DATA_H__
#define __REPORT_BASE_DATA_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define API_VERSION_LEN 16
#define KEY_TOKEN_LEN 20
#define KEY_LEN 20
#define IP_LEN 20
#define OS_LEN 32
#define ARCH_LEN 2
#define KERNEL_LEN 64
#define TENGINE_VERSION_LEN 64
#define HCL_VERSION_LEN 64
#define PROC_LEN 64
#define UID_LEN 40
#define CPUID_LEN 20
#define CPUFREQ_LEN 16
#define ACTION_LEN 16
#define SIGN_LEN 40
#define APPID_LEN 32
#define APP_KEN_LEN 32

#define API_VERSION "apiVersion"
#define KEY_TOKEN "keyToken"
#define IP "ip"
#define OS "os"
#define ARCH "arch"
#define KERNEL "kernelVersion"
#define TENGINE_VERSION "tengineVersion"
#define HCL_VERSION "hclVersion"
#define PID "tenginePid"
#define PROC "tengineProc"
#define UID "uid"
#define CPUNUM "cpuNum"
#define CPUID "cpuId"
#define CPUFREQ "cpuFreq"
#define MEM "mem"
#define SESS "sess"
#define ACTION "action"
#define SIGN "sign"
#define TIMESTAMP "timestamp"
#define REPORT_DATE "reportDate"
#define APPID "appid"
#define METHODNAME "methodname"

#define DEC_STR_VAR(NAME) char NAME##_ [NAME##_LEN + 1]
#define DEC_INT_VAR(NAME) uint32_t NAME##_

typedef struct
{
    DEC_STR_VAR(API_VERSION);
    DEC_STR_VAR(KEY_TOKEN);
    DEC_STR_VAR(APPID);
    DEC_STR_VAR(IP);
    DEC_STR_VAR(OS);
    DEC_STR_VAR(ARCH);
    DEC_STR_VAR(KERNEL);
    DEC_STR_VAR(TENGINE_VERSION);
    DEC_STR_VAR(HCL_VERSION);
    DEC_STR_VAR(PROC);
    DEC_STR_VAR(UID);
    DEC_STR_VAR(CPUID);
    DEC_STR_VAR(ACTION);
    DEC_STR_VAR(SIGN);

    DEC_INT_VAR(CPUFREQ);
    DEC_INT_VAR(SESS);
    DEC_INT_VAR(MEM);
    DEC_INT_VAR(CPUNUM);
    DEC_INT_VAR(PID);
    DEC_INT_VAR(REPORT_DATE);

}REPORT_BASE_DATA_T;

void strcpy_s(char* dst,const char* src,int maxlen );

#define DEC_UPDATE_INT_FUNC(NAME) void update_##NAME(REPORT_BASE_DATA_T* dat,uint32_t val)
#define DEC_UPDATE_STR_FUNC(NAME) void update_##NAME(REPORT_BASE_DATA_T* dat,const char* val)

DEC_UPDATE_STR_FUNC(API_VERSION);
DEC_UPDATE_STR_FUNC(KEY_TOKEN);
DEC_UPDATE_STR_FUNC(APPID);
DEC_UPDATE_STR_FUNC(IP);
DEC_UPDATE_STR_FUNC(OS);
DEC_UPDATE_STR_FUNC(ARCH);
DEC_UPDATE_STR_FUNC(KERNEL);
DEC_UPDATE_STR_FUNC(TENGINE_VERSION);
DEC_UPDATE_STR_FUNC(HCL_VERSION);
DEC_UPDATE_STR_FUNC(PROC);
DEC_UPDATE_STR_FUNC(UID);
DEC_UPDATE_STR_FUNC(CPUID);
DEC_UPDATE_STR_FUNC(ACTION);
DEC_UPDATE_STR_FUNC(SIGN);

DEC_UPDATE_INT_FUNC(CPUFREQ);
DEC_UPDATE_INT_FUNC(SESS);
DEC_UPDATE_INT_FUNC(MEM);
DEC_UPDATE_INT_FUNC(CPUNUM);
DEC_UPDATE_INT_FUNC(PID);
DEC_UPDATE_INT_FUNC(REPORT_DATE);


#ifdef __cplusplus
}
#endif

#endif
