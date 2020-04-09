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

#include "base_socket.h"
#include "clientsocket.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define SPACE 0x20
#define HEADERS_END "\r\n\r\n"
#define SPACE_LINE "\r\n"
#define SPACE_LINE_LEN 2
#define POST "POST"
#define HTTP_VERSION "HTTP/1.1\r\n"
#define HTTPSTRING "HTTP"
#define CONTENTSTRING "Content-Length"
#define STATUS_POS 9

#ifdef NET_DEBUG
#define NET_LOG(fmt,...) printf(fmt,## __VA_ARGS__)
#else
#define NET_LOG(fmt,...)
#endif

void init_post_req(THttpReq* req,const char* host,uint16_t port,const char* request_url)
{
    req->status = 0;
    req->req_type = eHttp_Post;
    req->port = port;
    req->conect_timeout = 60 * 1000 ;
    strncpy(req->host,host,MAX_HOST_LEN);
    req->len = sprintf(req->data,"%s %s %sHost: %s:%d\r\nConnection: close\r\n",
                POST,request_url,HTTP_VERSION,host,port);
    req->fix_len = req->len;            
}

void free_req(THttpReq* req)
{

}

int add_header(THttpReq* req,const char* key,const char* val)
{
    int res = sprintf(&(req->data[req->len]),"%s: %s\r\n",key,val);
    if ( res < 0 )
    {
        NET_LOG("over httpreq data length\n");
        return -1;
    }

    req->len += res;
    return 0;
}

int add_header_fix(THttpReq* req,const char* key,const char* val)
{
    int res = sprintf(&(req->data[req->len]),"%s: %s\r\n",key,val);
    if ( res < 0 )
    {
        NET_LOG("over httpreq data length\n");
        return -1;
    }

    req->len += res;
    req->fix_len = req->len;
    return 0;
}

int set_data(THttpReq* req,const char* dat,int len)
{
    if( len + req->len + SPACE_LINE_LEN > MAX_HTTP_DATA )
    {
        NET_LOG("can not add moredatas\n");
        return -1;
    }
    sprintf(&(req->data[req->len]),SPACE_LINE);
    req->len += SPACE_LINE_LEN;

    memcpy( &(req->data[req->len]),dat,len);
    req->len += len;
    return 0;
}

int get_staus(THttpReq* req)
{
    return req->status;
}


int get_status(const char* resp,int len,THttpReq* req)
{
    const char* reader = resp;
    reader = strstr(resp,HTTPSTRING);
    if(reader == NULL)
    {
        NET_LOG("Http Headers is error. \'HTTP Field is null \' \n");
        return -1;
    }

    int status = atoi(reader + STATUS_POS);
    NET_LOG("current status : %d\n",status);
    req->status = status;

    reader = strstr(resp,HEADERS_END);
    if( reader == NULL )
    {
        NET_LOG("Http Headers is error . \'HTTP Header end flag is null\' \n");
        return -1;
    }

    const char* content = strstr(resp,CONTENTSTRING);
    if( content == NULL )
    {
        NET_LOG("Http Do not has any data . \'Content-Length Faield is null\' \n");
        return 0;
    }

#ifdef NET_DEBUG
    int dat_len = atoi(content+strlen(CONTENTSTRING) + 2);
    char tmpDat[4096] = {'\0'};
    memcpy(tmpDat,reader+strlen(HEADERS_END),dat_len);
    printf("Recv Data [%d]: %s\n",dat_len,tmpDat);
#endif
    return 0;
}

int http_response(THttpReq* req,const char* resp,int data_len)
{
    int ret = get_status(resp,data_len,req);
    if(ret < 0 )
    {
        return -1;
    }

    return 0;
}

#define MAX_RECV_COUNT 100

int http_post(THttpReq* req)
{
    NET_CONTEXT sock_fd;
    init_net_context(&sock_fd);

    char port[10] = {'\0'} ;    
    sprintf(port,"%d",req->port);
    int ret = 0;
    if( req->conect_timeout > 0 )
    {
        if( ( ret = net_connect_timeout(&sock_fd,req->host,port,req->conect_timeout) ) != 0 )
        {
            NET_LOG("can not connect to server[%s:%d] error:%d\n",req->host,req->port,ret);
            net_free( &sock_fd );
            return eErHttp_CanNotConnectToHost;
        }
    }
    else
    {
        if( ( ret = net_connect(&sock_fd,req->host,port,NET_PROTO_TCP) ) != 0 )
        {
            NET_LOG("can not connect to server[%s:%d] error:%d\n",req->host,req->port,ret);
            net_free( &sock_fd );
            return eErHttp_CanNotConnectToHost;
        }
    }

    if( ( ret = net_send(&sock_fd,req->data,req->len) ) != req->len )
    {
        NET_LOG("send data faield! error:%d\n",ret);
        net_free( &sock_fd );
        return ret;
    }

    NET_LOG("Wait Response\n");
    int recv_count = 0;
    int rpos = 0;
    char recv_buf[MAX_HTTP_DATA+1];
    while( ( ret = net_recv_timeout(&sock_fd,recv_buf + rpos,MAX_HTTP_DATA - rpos,1000) ) > 0 && recv_count++ < MAX_RECV_COUNT )
    {
        rpos += ret;
    }

    if( ret < 0 )
    {
        NET_LOG("recv data faield! error:%d\n",ret);
        net_free( &sock_fd );
        return eErHttp_HostResponseOverTime;
    }

    ret = 0;
    recv_buf[rpos] = '\0';
    ret = 0;
    if( http_response(req,recv_buf,rpos) < 0 )
    {
        NET_LOG("http response data invalid!!!\n");    
        ret = eErHttp_HostResponseError; 
    }

    net_free( &sock_fd );

    return ret;
}

int finish_post(THttpReq* req)
{
    req->len = req->fix_len;
    return 0;
}
