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

#include "onlinereportmgr.h"
#include "thread.h"
#include "clientsocket.h"
#include "onlinereportutil.h"

#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>

#define MAX_ACTION_NUM 0x03

typedef struct
{
    TIMER_ID_T timer_id_;
    uint32_t over_timer_;
    long next_trigger_timer_;
    int is_loop_;
}TIMER;

typedef struct
{
    int action_;
}REPORT_TASK_T;

typedef struct
{
    THREAD_CONTEXT ctx_;
    THREAD_MUTEX mutex_;
    TIMER* timer_node_;
    get_online_report_dat_t get_dat_fun_;
    REPORT_TASK_T rp_task_[MAX_ACTION_NUM];
    int rp_task_mask_;
    int statu_;
    volatile int is_running;
    THttpReq http_req_;
}ONLIN_EREPOT_MGR_T;

long get_time()
{
    return time(NULL);
}

int trigger_timer(TIMER* timer)
{
    if( timer == NULL )
    {
        return 0;
    }

    long cur_time = get_time();
    if( cur_time >= timer->next_trigger_timer_ )
    {
        if( timer->is_loop_ )
        {
            timer->next_trigger_timer_ = cur_time + (long)(timer->over_timer_);
        }

        return 1;
    }

    return 0;
}

void do_real_online_report(ONLIN_EREPOT_MGR_T* ctx,int action)
{
    if( ctx->get_dat_fun_ == NULL )
    {
        printf("there is do not have get data function\n");            
        return;            
    }

    int len = 0;
    const char* dat = (ctx->get_dat_fun_)(&len,action);
    if( len <= 0 )
    {
        return ;
    }

    OL_RP_LOG("report datas : [%d], content : %s \n",len,dat);
    char datlen[50] = {'\0'};
    sprintf(datlen," %d",len);
    add_header(&(ctx->http_req_),"Content-Length",datlen);
    set_data(&(ctx->http_req_),dat,len);

    http_post(&(ctx->http_req_));
    finish_post(&(ctx->http_req_));

}   

int get_empty_task_slot(ONLIN_EREPOT_MGR_T* ctx)
{
    for(int ii=0; ii<MAX_ACTION_NUM; ++ii)
    {
        if( ( ctx->rp_task_mask_ & ( 1 << ii ) ) == 0 )
        {
            return ii;
        }
    }

    return -1;
}

int get_valid_task_slot(ONLIN_EREPOT_MGR_T* ctx)
{
    for(int ii=0; ii<MAX_ACTION_NUM; ++ii)
    {
        if( ( ctx->rp_task_mask_ & ( 1 << ii ) ) != 0 )
        {
            return ii;
        }
    }

    return -1;
}

static void* worker_run(void *ctx)
{
    ONLIN_EREPOT_MGR_T* context = (ONLIN_EREPOT_MGR_T*)ctx;

    //init headers will not changed
    add_header_fix( &(context->http_req_), "Content-Type","application/json" );
    add_header(&(context->http_req_),"Accept","text/plain");

    while( context->is_running )
    {
        if( context->statu_ )
        {
            if( trigger_timer( context->timer_node_ ) )
            {
                do_real_online_report(context,PERIOD_ACTION);
            }

            lock(&(context->mutex_));
            int idx = get_valid_task_slot(context);
            if( idx >=0 )
            {
                int action = context->rp_task_[idx].action_;
                context->rp_task_mask_ = ( context->rp_task_mask_ & ~( 1 << idx ) );
                unlock(&(context->mutex_));
                do_real_online_report(context,action);
            }
            else
            {
                unlock(&(context->mutex_));
            }
        }

        usleep( 500 * 1000 );
    }

    OL_RP_LOG("process pending task start : nums[%d] \n",( context->rp_task_mask_ & MAX_ACTION_NUM ));
    //finish pending task
    //context->http_req_.conect_timeout = 1 * 1000;
    while( ( context->rp_task_mask_ & MAX_ACTION_NUM ) != 0 )
    {
        lock(&(context->mutex_));
        int idx = get_valid_task_slot(context);
        if( idx >= 0 )
        {
            int action = context->rp_task_[idx].action_;
            do_real_online_report(context,action);
            context->rp_task_mask_ = ( context->rp_task_mask_ & ~( 1 << idx ) );
        }
        unlock(&(context->mutex_));                    
    }
    OL_RP_LOG("process pending task end \n");

    return NULL;
}

ONLINE_REPORT_CONTEXT_T init_report_mgr(const char* host,uint16_t port,const char* request_url,get_online_report_dat_t get_dat_fun)
{
    OL_RP_LOG("host : %s:%u\n",host,port);
    srand((unsigned)time(NULL));

    ONLIN_EREPOT_MGR_T* ctx = (ONLIN_EREPOT_MGR_T*)( malloc(sizeof(ONLIN_EREPOT_MGR_T)) );
    memset(ctx,0,sizeof(ONLIN_EREPOT_MGR_T));

    ctx->get_dat_fun_ = get_dat_fun;

    //init http
    init_post_req( &(ctx->http_req_),host,port,request_url );
    
    //init thread
    init_thread_context(&(ctx->ctx_));
    ctx->ctx_.runner_ = worker_run;
    init_thread_mutex(&(ctx->mutex_));

    //start working
    OL_RP_LOG("Start working threading\n");
    ctx->is_running = 1;
    ctx->statu_ = 1;
    start_thread( ctx );

    return ctx;
}

void free_report_mgr(ONLINE_REPORT_CONTEXT_T ctx)
{
    if(ctx != NULL)
    {
        ONLIN_EREPOT_MGR_T* context = (ONLIN_EREPOT_MGR_T*)ctx;
        if( context->ctx_.handle_ != 0 )
        {
            context->is_running = 0;
            stop_thread(context);
            free_thread_context(&(context->ctx_));
        }

        free_thread_mutex(&(context->mutex_));
        
        if( context->timer_node_ != 0 )
        {
            free( context->timer_node_ );
            context->timer_node_ = NULL;
        }

        free_req( &(context->http_req_) );
        context->get_dat_fun_ = NULL;

        free( context );
        context = NULL;
    }
}

int set_report_status(ONLINE_REPORT_CONTEXT_T ctx,int status)
{
    ONLIN_EREPOT_MGR_T* context = (ONLIN_EREPOT_MGR_T*)ctx;
    int old_stat = context->statu_;
    context->statu_ = status;
    return old_stat;
}

int do_report(ONLINE_REPORT_CONTEXT_T ctx,int action)
{

    ONLIN_EREPOT_MGR_T* context = (ONLIN_EREPOT_MGR_T*)ctx;
    if( (context->rp_task_mask_ & MAX_ACTION_NUM) >= MAX_ACTION_NUM )
    {
        OL_RP_LOG("can not add more report request\n ");
        return -1;            
    }

    lock(&(context->mutex_));
    int idx = get_empty_task_slot(context);
    if( idx >= 0 )
    {
        context->rp_task_[idx].action_ = action;
        context->rp_task_mask_ |= 1 << idx;
        OL_RP_LOG("add report cur task mask : %d\n",context->rp_task_mask_);
    }

    unlock(&(context->mutex_));

    return 0;
}


TIMER_ID_T add_timer(ONLINE_REPORT_CONTEXT_T ctx,uint32_t timer,int is_loop)
{
    ONLIN_EREPOT_MGR_T* context = (ONLIN_EREPOT_MGR_T*)ctx;
    if( context == NULL ) 
    {
        return INVLIAD_TIMER_ID; 
    }

    if( context->timer_node_ == NULL )
    {
        TIMER* tm = (TIMER*)malloc( sizeof(TIMER) );
        context->timer_node_ = tm;
        context->timer_node_->is_loop_ = is_loop;
        context->timer_node_->over_timer_ = timer;
        context->timer_node_->next_trigger_timer_ = timer + get_time();
        context->timer_node_->timer_id_ = get_rand();
        return tm->timer_id_;
    }

    //only support one timer now
    return INVLIAD_TIMER_ID;
} 

void remove_timer(ONLINE_REPORT_CONTEXT_T ctx,TIMER_ID_T id)
{

}              
