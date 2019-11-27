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

#ifndef __BASE_SOCKET_H__
#define __BASE_SOCKET_H__

#include <stddef.h>
#include <stdint.h>

#define NET_PROTO_TCP 0 
#define NET_PROTO_UDP 1 

#define NET_POLL_READ 1
#define NET_POLL_WRITE 2

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    ERR_NET_SOCKET_FAILED = -0x042,
    ERR_NET_RECV_FAILED = -0x043,
    ERR_NET_SEND_FAILED = -0x044,
    ERR_NET_CONNECT_FAILED = -0x045,
    ERR_NET_CONNECT_TIMEOUT = -0x046,
    ERR_NET_CONN_RESET = -0x050,
    ERR_NET_UNKNOWN_HOST = -0x052,
    ERR_NET_INVALID_CONTEXT = -0x062,
    ERR_NET_WANT_READ = -0x063,
    ERR_NET_WANT_WRITE = -0x064,
    ERR_NET_TIMEOUT = -0x065,
};

typedef struct
{
    int fd;
}
NET_CONTEXT;

void init_net_context(NET_CONTEXT* context);
void net_free(NET_CONTEXT* context);

int net_connect(NET_CONTEXT* context,const char *host,const char* port, int proto);
int net_connect_timeout(NET_CONTEXT* context,const char *host,const char* port, int timeout);

int net_send(NET_CONTEXT* context,const char* buf,int len);
int net_recv(NET_CONTEXT* context,char* buf,int len);
int net_recv_timeout(NET_CONTEXT* context,char *buf,int len,  uint32_t timeout);
int net_poll(NET_CONTEXT *context, uint32_t rw, uint32_t timeout);

int get_loacl_ip_and_mac(char* ip,char* mac);

#ifdef __cplusplus
}
#endif

#endif
