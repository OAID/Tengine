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

#ifndef __REPORT_DATA_H__
#define __REPORT_DATA_H__

#define ACTION_INIT 0x0101
#define ACTION_RELEASE 0x0202

#ifdef __cplusplus
extern "C" {
#endif

void init_report_data(const char* tengine_key,const char* tengine_key_token, const char* appid,const char* app_key,
                    const char* tengine_version,const char* hcl_version,const char* api_version,const char* request_url);


const char* general_report_data(int* len,int action);
void release_report_data();

#ifdef __cplusplus
}
#endif

#endif
