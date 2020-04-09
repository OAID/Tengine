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

#ifndef __CLIENT_SOCKET_H__ 
#define __CLIENT_SOCKET_H__

#include <stdint.h>

#define MAX_HTTP_DATA 4095
#define MAX_HOST_LEN 255

#ifdef __cplusplus
extern "C" {
#endif

typedef enum HttpReqType
{
    eHttp_Post,
    eHttp_Get,
} EHttpReqType;

enum
{
    eErHttp_CanNotConnectToHost = -1,
    eErHttp_HostResponseError = -2,
    eErHttp_HostResponseOverTime = -3,
};

typedef struct HttpReq
{
    EHttpReqType req_type;
    int status;
    uint16_t port;
    int len;
    int fix_len;
    int conect_timeout ;
    char host[MAX_HOST_LEN+1];
    char data[MAX_HTTP_DATA + 1];
}THttpReq;

/*
init req with Request Line and Host Header
host : server host
port : server port
request_url : Request-URL 
*/
void init_post_req(THttpReq* req,const char* host,uint16_t port,const char* request_url);

void free_req(THttpReq* req);

/*
add http headers
return : 0(sucess), other faield
*/
int add_header(THttpReq* req,const char* key,const char* val);
int add_header_fix(THttpReq* req,const char* key,const char* val);

/*
add data to request. support http post
return : 0(sucess), other faield
*/
int set_data(THttpReq* req,const char* dat,int len);

int get_staus(THttpReq* req);

/*
return : 0(sucess), other failed
function will block util http request finish or faield!!
*/
int http_post(THttpReq* req);
int finish_post(THttpReq* req);

#ifdef __cplusplus
}
#endif

#endif
