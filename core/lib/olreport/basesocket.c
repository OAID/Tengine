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

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <linux/if.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "base_socket.h"

void init_net()
{
    signal( SIGPIPE, SIG_IGN );
}

void init_net_context(NET_CONTEXT* context)
{
    context->fd = -1;
}

void net_free(NET_CONTEXT* context)
{
    if( context->fd == -1 )
        return;

    shutdown( context->fd, 2 );
    close( context->fd );

    context->fd = -1;
}

int net_connect(NET_CONTEXT* context,const char *host,const char* port, int proto)
{
    init_net();
    int ret;
    struct addrinfo hints, *addr_list, *cur;

    memset( &hints, 0, sizeof( hints ) );
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = proto == NET_PROTO_UDP ? SOCK_DGRAM : SOCK_STREAM;
    hints.ai_protocol = proto == NET_PROTO_UDP ? IPPROTO_UDP : IPPROTO_TCP;

    if( getaddrinfo( host, port, &hints, &addr_list ) != 0 )
    {
        return ERR_NET_UNKNOWN_HOST ;
    }

    ret = ERR_NET_UNKNOWN_HOST;

    for( cur = addr_list; cur != NULL; cur = cur->ai_next )
    {
        context->fd = (int) socket( cur->ai_family, cur->ai_socktype,
                            cur->ai_protocol );
        if( context->fd < 0 )
        {
            ret = ERR_NET_SOCKET_FAILED;
            close(context->fd);
            continue;
        }

        if( connect( context->fd, cur->ai_addr, cur->ai_addrlen ) == 0 )
        {
            ret = 0;
            break;
        }

        close( context->fd );
        ret = ERR_NET_CONNECT_FAILED;
    }

    freeaddrinfo( addr_list );

    return ret ;

}

int net_set_block(int fd)
{
    return fcntl( fd, F_SETFL, fcntl( fd, F_GETFL ) & ~O_NONBLOCK );
}

int net_set_nonblock(int fd)
{
    return fcntl( fd, F_SETFL, fcntl( fd, F_GETFL ) | O_NONBLOCK );
}

int is_connected(int fd,fd_set* rd,fd_set* wr)
{
    if( !FD_ISSET(fd,rd) && !FD_ISSET(fd,wr) )
    {
        return -1;
    }

    int err ;
    socklen_t len = sizeof(err);
    if( getsockopt(fd,SOL_SOCKET,SO_ERROR,&err,&len) < 0 )
    {
        return -1;
    }

    return err;
}

int net_connect_timeout(NET_CONTEXT* context,const char *host,const char* port, int timeout)
{
    init_net();
    int ret;
    struct addrinfo hints, *addr_list, *cur;

    memset( &hints, 0, sizeof( hints ) );
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    if( getaddrinfo( host, port, &hints, &addr_list ) != 0 )
    {
        return ERR_NET_UNKNOWN_HOST ;
    }

    ret = ERR_NET_UNKNOWN_HOST;
    struct timeval timeo = {timeout / 1000, (timeout % 1000) * 1000};
    socklen_t len = sizeof(timeo);

    for( cur = addr_list; cur != NULL; cur = cur->ai_next )
    {
        context->fd = (int) socket( cur->ai_family, cur->ai_socktype,
                            cur->ai_protocol );
        if( context->fd < 0 )
        {
            ret = ERR_NET_SOCKET_FAILED;
            close(context->fd);
            continue;
        }

        setsockopt(context->fd, SOL_SOCKET, SO_SNDTIMEO, &timeo, len);
        if( connect( context->fd, cur->ai_addr, cur->ai_addrlen ) == 0 )
        {
            ret = 0;
            break;
        }

        close( context->fd );
        ret = ERR_NET_CONNECT_FAILED;
    }

    freeaddrinfo( addr_list );

    return ret ;
}

int net_send(NET_CONTEXT* context,const char* buf,int len)
{
    int ret;
    int fd = context->fd;

    if( fd < 0 )
    {
        return ERR_NET_INVALID_CONTEXT ;
    }

    ret = write( fd, buf, len );

    if( ret < 0 )
    {

        if( errno == EPIPE || errno == ECONNRESET )
            return( ERR_NET_CONN_RESET );

        if( errno == EINTR )
            return( ERR_NET_WANT_WRITE );

        return ERR_NET_SEND_FAILED;
    }

    return ret ;
}

int net_recv(NET_CONTEXT* context,char* buf,int len)
{
    int ret;
    int fd = context->fd;

    if( fd < 0 )
    {
        return ERR_NET_INVALID_CONTEXT ;
    }

    ret = read( fd, buf, len );

    if( ret < 0 )
    {

        if( errno == EPIPE || errno == ECONNRESET )
        {
            return ERR_NET_CONN_RESET ;
        }

        if( errno == EINTR )
        {
            return ERR_NET_WANT_READ;
        }

        return( ERR_NET_RECV_FAILED );
    }

    return ret ;
}

int net_recv_timeout(NET_CONTEXT* context,char *buf,int len, unsigned int timeout)
{
    int ret;
    struct timeval tv;
    fd_set read_fds;
    int fd = context->fd;

    if( fd < 0 )
    {
        return ERR_NET_INVALID_CONTEXT ;
    }

    FD_ZERO( &read_fds );
    FD_SET( fd, &read_fds );

    tv.tv_sec  = timeout / 1000;
    tv.tv_usec = ( timeout % 1000 ) * 1000;

    ret = select( fd + 1, &read_fds, NULL, NULL, timeout == 0 ? NULL : &tv );

    // Zero fds ready means timed out 
    if( ret == 0 )
    {
        return( ERR_NET_TIMEOUT );
    }

    if( ret < 0 )
    {
        if( errno == EINTR )
        {
            return ERR_NET_WANT_READ ;
        }

        return ERR_NET_RECV_FAILED ;
    }

    return net_recv( context, buf, len ) ;
}

int get_loacl_ip_and_mac(char* ip,char* mac)
{
    struct ifconf if_conf;
    struct ifreq *if_req;
    char buf[1024] = {'\0'} ;
    if_conf.ifc_len = 1024;
    if_conf.ifc_buf = buf;

    struct ifreq ifr_mac;

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if( fd < 0 )
    {
        return -1;
    }

    ioctl(fd,SIOCGIFCONF,&if_conf);
    if_req = (struct ifreq*)(if_conf.ifc_buf);
    int nums = if_conf.ifc_len / sizeof(struct ifreq);
    for(int ii=0; ii<nums;++ii,++if_req)
    {
        if( if_req->ifr_name[0] == 'l' && if_req->ifr_name[1] == 'o' )
        {
            continue;
        }
        
        if( if_req->ifr_flags == AF_INET || if_req->ifr_flags == AF_INET6)
        {
            void *tmp_addr = &((struct sockaddr_in *)&(if_req->ifr_addr))->sin_addr;
            inet_ntop(AF_INET, tmp_addr, ip, INET_ADDRSTRLEN);

            strcpy(ifr_mac.ifr_name,if_req->ifr_name);
            if( ioctl(fd,SIOCGIFHWADDR,&ifr_mac) == 0 )
            {
                char temp_str[10] = { 0 };
                memcpy(temp_str, ifr_mac.ifr_hwaddr.sa_data, 6);
                sprintf(mac, "%02x-%02x-%02x-%02x-%02x-%02x", temp_str[0]&0xff, temp_str[1]&0xff, temp_str[2]&0xff, temp_str[3]&0xff, temp_str[4]&0xff, temp_str[5]&0xff);
            }

            break;
        }

    }

    close(fd);
    return 0;
}

/*int get_loacl_ip_and_mac(char* ip,char* mac)
{

    struct ifaddrs *if_addr_struct = NULL;
    struct ifaddrs *ifa = NULL;
 
    struct ifreq ifr;
    int fd = socket(AF_INET, SOCK_DGRAM, 0);

    getifaddrs(&if_addr_struct);

    for (ifa = if_addr_struct; ifa != NULL; ifa = ifa->ifa_next)
    {
        if (!ifa->ifa_addr || ( ifa->ifa_name[0] == 'l' && ifa->ifa_name[1] == 'o' ) )
        {
            continue;
        }
 
        strcpy(ifr.ifr_name,ifa->ifa_name);    
        // check if IPV4
        if (ifa->ifa_addr->sa_family == AF_INET)
        {
            void *tmp_addr = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char local_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmp_addr, local_ip, INET_ADDRSTRLEN);
 
            sprintf(ip,"%s", local_ip);
            
            if( ioctl(fd,SIOCGIFHWADDR,&ifr) == 0 )
            {
                char temp_str[10] = { 0 };
                memcpy(temp_str, ifr.ifr_hwaddr.sa_data, 6);
                sprintf(mac, "%02x-%02x-%02x-%02x-%02x-%02x", temp_str[0]&0xff, temp_str[1]&0xff, temp_str[2]&0xff, temp_str[3]&0xff, temp_str[4]&0xff, temp_str[5]&0xff);
            }

            break; 
        }
        else if(ifa->ifa_addr->sa_family==AF_INET6)
        {
            //if IPV6
            void *tmp_addr = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char local_ip[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, tmp_addr, local_ip, INET6_ADDRSTRLEN);

            sprintf(ip,"%s", local_ip);
            
            if( ioctl(fd,SIOCGIFHWADDR,&ifr) == 0 )
            {
                char temp_str[10] = { 0 };
                memcpy(temp_str, ifr.ifr_hwaddr.sa_data, 6);
                sprintf(mac, "%02x-%02x-%02x-%02x-%02x-%02x", temp_str[0]&0xff, temp_str[1]&0xff, temp_str[2]&0xff, temp_str[3]&0xff, temp_str[4]&0xff, temp_str[5]&0xff);
            }

            break;
        }

    }

    if (if_addr_struct)
    {
        freeifaddrs(if_addr_struct);                
    }

    if( fd )
    {
        close(fd);
    }

    return 0;

}*/
