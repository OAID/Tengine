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

#include "reportdata.h"
#include "reportbasedata.h"
#include "onlinereportfmtdat.h"
#include "md5.h"
#include "onlinereportutil.h"
#include "base_socket.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX_REPORT_DATA_LEN 1024 * 10
#define METHOD_NAME "methodname"
#define PARAM "param"
#define APP_TOEKN "token"
#define REQUEST_URL_LEN 512

#define ACTION_INIT_NAME "INIT"
#define ACTION_RELEASE_NAME "RELEASE"
#define ACTION_OTHER_NAME "PERIOD"

typedef struct
{
    char key_[KEY_LEN + 1];
    char app_key_[APP_KEN_LEN + 1];
    int is_init_all ;
    char report_dat_[MAX_REPORT_DATA_LEN];
    char REQ_URL[REQUEST_URL_LEN];
    REPORT_BASE_DATA_T data_;

}REPORT_DATA_T;

static REPORT_DATA_T gsData;
static char json_data[MAX_REPORT_DATA_LEN] = {'\0'};
static char app_data[MAX_REPORT_DATA_LEN] = {'\0'};

void init_report_data(const char* tengine_key,const char* tengine_key_token, const char* appid,const char* app_key,
                        const char* tengine_version,const char* hcl_version,const char* api_version,const char* request_url)
{
    OL_RP_LOG("tengine_ley : %s, tengine_key_token : %s, aapID : %s ,app_key : %s, tengine_version : %s, \
              hcl_version : %s,api_version : %s, request_url : %s \n " ,
              tengine_key,tengine_key_token,appid,app_key,tengine_version,hcl_version,api_version,request_url);

    strcpy_s( gsData.key_,tengine_key, KEY_LEN +1 );
    strcpy_s( gsData.app_key_,app_key, APP_KEN_LEN + 1);
    strcpy_s( gsData.REQ_URL,request_url,REQUEST_URL_LEN);

    update_TENGINE_VERSION( &(gsData.data_),tengine_version );
    update_HCL_VERSION( &(gsData.data_),hcl_version );
    update_API_VERSION( &(gsData.data_),api_version );
    update_KEY_TOKEN( &(gsData.data_),tengine_key_token );
    update_APPID( &(gsData.data_),appid);

    gsData.is_init_all = 0;

}

void release_report_data()
{
    gc_free_node();
}

uint32_t get_timestamp()
{
    time_t time_ = time(NULL);
    return time_;
}

void get_tengine_env_data(REPORT_DATA_T* dat)
{
    get_cpu_param_info(0,dat->data_.CPUID_,&(dat->data_.CPUFREQ_),&(dat->data_.CPUNUM_) );
    get_os_info(dat->data_.OS_,OS_LEN);
    get_os_kernel_info(dat->data_.KERNEL_,KERNEL_LEN);
    get_cur_process_info(&(dat->data_.PID_),dat->data_.PROC_);
    get_loacl_ip_and_mac( dat->data_.IP_,dat->data_.UID_ );
    dat->data_.MEM_ = get_totoal_memory();
    sprintf(dat->data_.ARCH_,"%u",get_arch());
}

#define ADD_STRING_VAL(NAME) add_string_val_order( &fmt_dat,NAME,gsData.data_.NAME##_ )
#define ADD_INT_VAL(NAME) add_int_val_order( &fmt_dat,NAME,gsData.data_.NAME##_ )

const char* general_report_data(int* len,int action)
{
    if( gsData.is_init_all == 0 )
    {
        gsData.is_init_all = 1;
        get_tengine_env_data( &(gsData) );
    }

    uint32_t tm = get_timestamp();
    update_REPORT_DATE (&(gsData.data_),tm );
    uint32_t rand_num = get_rand();
    update_SESS( &(gsData.data_),rand_num );
    if( action == ACTION_INIT )
    {
        update_ACTION( &(gsData.data_),ACTION_INIT_NAME );
    }
    else if( action == ACTION_RELEASE )
    {
        update_ACTION( &(gsData.data_),ACTION_RELEASE_NAME );
    }
    else
    {
        update_ACTION( &(gsData.data_),ACTION_OTHER_NAME );
    }

    OLREPORT_FMTDAT_T fmt_dat;
    init_online_report_fmt_dat(&fmt_dat);

    ADD_STRING_VAL(TENGINE_VERSION);
    ADD_STRING_VAL(HCL_VERSION);
    ADD_STRING_VAL(IP);
    ADD_STRING_VAL(OS);
    ADD_STRING_VAL(ARCH);
    ADD_STRING_VAL(KERNEL);
    ADD_STRING_VAL(PROC);
    ADD_STRING_VAL(UID);
    ADD_STRING_VAL(CPUID);
    ADD_STRING_VAL(ACTION);
    ADD_STRING_VAL(API_VERSION);
    ADD_STRING_VAL(KEY_TOKEN);

    ADD_INT_VAL(CPUFREQ);
    ADD_INT_VAL(CPUNUM);
    ADD_INT_VAL(MEM);
    ADD_INT_VAL(SESS);
    ADD_INT_VAL(PID);
    ADD_INT_VAL(REPORT_DATE);

    int offset = report_fmt_data_2_url_keyval_string(&fmt_dat,gsData.report_dat_);
    if( offset < 0 )
    {
        printf("format string failed!!!\n");
        return NULL;
    }

    offset += sprintf(gsData.report_dat_+offset,"%s",gsData.key_);
    OL_RP_LOG("report data string calc for md5 : %s\n",gsData.report_dat_);

    unsigned char tengine_data_md5[16];
    get_md5((unsigned char*)(gsData.report_dat_),offset,tengine_data_md5);
    md5_to_string(tengine_data_md5,gsData.data_.SIGN_);
    add_string_val(&fmt_dat,SIGN,gsData.data_.SIGN_);
    offset = report_fmt_data_2_json(&fmt_dat,json_data);

    //app key md5
    char tengine_md5_str[32] = {'\0'};

    offset = snprintf(app_data,MAX_REPORT_DATA_LEN,"%s=%s&%s=%s&%s=%s&%s=%u%s",
        APPID,gsData.data_.APPID_,METHOD_NAME,
        gsData.REQ_URL,PARAM,json_data,
        TIMESTAMP,tm,gsData.app_key_);
    if( offset > MAX_REPORT_DATA_LEN )    
    {
        offset = MAX_REPORT_DATA_LEN;
    }

	OL_RP_LOG("app md5 string :\n %s \n",app_data);
    get_md5((unsigned char*)(app_data),offset,tengine_data_md5);
    md5_to_string(tengine_data_md5,tengine_md5_str);

    //format to json
    gsData.report_dat_[0] = '0';
    *len = snprintf(gsData.report_dat_,1024*10,"{\"%s\":\"%s\",\"%s\":\"%s\",\"%s\":%s,\"%s\":%u,\"%s\":\"%s\"}",
            APPID,gsData.data_.APPID_ ,
            METHOD_NAME,gsData.REQ_URL ,
            PARAM,json_data,
            TIMESTAMP,tm,
            SIGN,tengine_md5_str);
    if( *len > 1024*10 )
    {
        *len = 1024*10;
    }
    //

    release_online_report_fmt_dat(&fmt_dat);

    return gsData.report_dat_;
}
