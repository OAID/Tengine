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

#include "tenginereportmgr.hpp"
#include "tengine_cfg_data.hpp"
#include "tengine_version.hpp"
#include "tengine_c_api.h"
#include "tengine_c_helper.hpp"

#include "onlinereportmgr.h"
#include "reportdata.h"

#include <string>

ONLINE_REPORT_CONTEXT_T report_mgr_context;

static int open_report_stat = 1;

void init_tengine_report_mgr()
{
    if( open_report_stat )
    {
        const char* tengine_report_host = std::getenv("tengine_report_host");
        if( tengine_report_host == NULL )
        {
            tengine_report_host = CFG_HOST;
        }

        uint16_t port = CFG_PORT;
        const char* tengine_report_port = std::getenv("tengine_report_port");
        if( tengine_report_port != NULL )
        {
            port = (uint16_t)(atoi(tengine_report_port));
        }
        const char* app_key = std::getenv("tengine_app_key");
        if( app_key == NULL )
        {
            app_key = CFG_APP_KEY;
        }
        init_report_data(CFG_TENGINE_KEY,CFG_TENGINE_TOKEN,CFG_APPID,app_key,get_tengine_version(),
                    get_tengine_hcl_version(),CFG_API_VERSION,CFG_REQ_URL);
        report_mgr_context = init_report_mgr(tengine_report_host,port,CFG_REQ_URL,general_report_data);
        add_timer(report_mgr_context,CFG_REPORT_TIMER,1);
    }
    
}

void release_tengine_report_mgr()
{
    if( open_report_stat )
    {
        free_report_mgr(report_mgr_context);
        release_report_data();
    }
}

void do_tengine_report(int action)
{
    if( open_report_stat )
    {
        do_report(report_mgr_context,action);
    }   
    
}

int set_tengine_report_stat(int stat)
{
    int res = open_report_stat;
    open_report_stat = stat;
    return res;
}