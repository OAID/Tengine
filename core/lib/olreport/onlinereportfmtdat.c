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

#include "onlinereportfmtdat.h"
#include "onlinereportutil.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>

enum 
{
    eValType_None = 0,
    eValType_INT = 1,
    eValType_STR = 2,
};

#define JSON_STR_FMT "\"%s\":\"%s\","
#define JSON_INT_FMT "\"%s\":%u,"

#define URL_KEYVAL_STR_FMT "%s=%s&"
#define URL_KEYVAL_INT_FMT "%s=%u&"

typedef struct OLREPORT_FMTNODE
{
    const char* json_key_;
    union 
    {
        const char* str_val_;
        uint32_t int_val_;
    }val_;

    int val_type_;
    struct OLREPORT_FMTNODE* next_;

}OLREPORT_FMTNODE_T;

static OLREPORT_FMTNODE_T* free_node_list = NULL;

 OLREPORT_FMTNODE_T* create_fmt_node()
 {
     OLREPORT_FMTNODE_T* res = NULL;
     if( free_node_list == NULL )
     {
        res = (OLREPORT_FMTNODE_T*)( malloc( sizeof( OLREPORT_FMTNODE_T ) ) );
     }
     else
     {
         res = free_node_list;
         free_node_list = free_node_list->next_;
     }

     memset(res,0,sizeof(OLREPORT_FMTNODE_T));
     return res;
 }

 void free_fmt_node(OLREPORT_FMTNODE_T* node)
 {
     if( free_node_list == NULL )
     {
         free_node_list = node;
         node->next_ = NULL;
     }
     else
     {
         node->next_ = free_node_list;
         free_node_list = node;
     }
 }

 void free_node(OLREPORT_FMTNODE_T* node)
 {
     OLREPORT_FMTNODE_T* cur_node = node;
     while( cur_node != NULL ) 
     {
         OLREPORT_FMTNODE_T* fr_node = cur_node;
         cur_node = cur_node->next_;
         free_fmt_node( fr_node );
     }
 }

 void gc_free_node()
 {
     while(free_node_list != NULL)
     {
         OLREPORT_FMTNODE_T* fr_node = free_node_list;
         free_node_list = free_node_list->next_;
         free( fr_node );
     }
}

void init_online_report_fmt_dat(OLREPORT_FMTDAT_T* fmt_dat)
{
    fmt_dat->node = NULL;
}

void release_online_report_fmt_dat(OLREPORT_FMTDAT_T* fmt_dat)
{
    if( fmt_dat == NULL || fmt_dat->node == NULL )
    {
        return ;
    }

    OLREPORT_FMTNODE_T* curNode = (OLREPORT_FMTNODE_T* )(fmt_dat->node);
    free_node( curNode );

}

//
void add_ftm_node(OLREPORT_FMTDAT_T* fmt_dat,OLREPORT_FMTNODE_T* node)
{
    if( fmt_dat->node == NULL )
    {
        fmt_dat->node = node;
    }
    else
    {
        OLREPORT_FMTNODE_T* curNode = (OLREPORT_FMTNODE_T* )(fmt_dat->node);
        node->next_ = curNode;
        fmt_dat->node = node;
    }       
}

void add_ftm_order_node(OLREPORT_FMTDAT_T* fmt_dat,OLREPORT_FMTNODE_T* node)
{
    if( fmt_dat->node == NULL )
    {
        fmt_dat->node = node;
        return ;
    }

    OLREPORT_FMTNODE_T* curNode = (OLREPORT_FMTNODE_T* )(fmt_dat->node);
    OLREPORT_FMTNODE_T* preNode = NULL;
    while( curNode != NULL )
    {
        if( strcmp( curNode->json_key_ ,node->json_key_) > 0 )
        {
            node->next_ = curNode;
            if( preNode != NULL )
            {
                preNode->next_ = node;
            }
            else
            {
                fmt_dat->node = node;
            }  
            return  ;              
        }   
        
        preNode = curNode;
        curNode = curNode->next_;
    }

    preNode->next_ = node;
    node->next_ = NULL;    
}

void add_interal_string_val(OLREPORT_FMTDAT_T* fmt_dat,const char* key,const char* val,int order)
{
    OLREPORT_FMTNODE_T* node = create_fmt_node();
    node->json_key_ = key;
    node->val_.str_val_ = val;
    node->val_type_ = eValType_STR;
    if( order != 0 )
    {
        add_ftm_order_node(fmt_dat,node);
    }
    else
    {
        add_ftm_node(fmt_dat,node);
    }
}

void add_interal_int_val(OLREPORT_FMTDAT_T* fmt_dat,const char* key,uint32_t val,int order)
{
    OLREPORT_FMTNODE_T* node = create_fmt_node();
    node->json_key_ = key;
    node->val_.int_val_ = val;    
    node->val_type_ = eValType_INT;

    if( order != 0 )
    {
        add_ftm_order_node(fmt_dat,node);
    }
    else
    {
        add_ftm_node(fmt_dat,node);
    }

}

//
void add_string_val(OLREPORT_FMTDAT_T* fmt_dat,const char* key,const char* val)
{
    add_interal_string_val(fmt_dat,key,val,0);
}

void add_int_val(OLREPORT_FMTDAT_T* fmt_dat,const char* key,uint32_t val)
{
    add_interal_int_val(fmt_dat,key,val,0);
}

void add_string_val_order(OLREPORT_FMTDAT_T* fmt_dat,const char* key,const char* val)
{
    add_interal_string_val(fmt_dat,key,val,1);
}

void add_int_val_order(OLREPORT_FMTDAT_T* fmt_dat,const char* key,uint32_t val)
{
    add_interal_int_val(fmt_dat,key,val,1);
}


int report_fmt_data_2_json(OLREPORT_FMTDAT_T* fmt_dat,char* output)
{
    if( fmt_dat->node == NULL )
    {
        return -1;
    }

    OLREPORT_FMTNODE_T* curNode = (OLREPORT_FMTNODE_T* )(fmt_dat->node);
    int offset = 0;
    offset += sprintf(output+offset,"{");

    while(curNode != NULL)
    {
        if( curNode->val_type_ == (int)eValType_INT )
        {
            offset += sprintf( output+offset,JSON_INT_FMT,curNode->json_key_,curNode->val_.int_val_ ) ;                 
        }
        else
        {
            offset += sprintf( output+offset,JSON_STR_FMT,curNode->json_key_,curNode->val_.str_val_ ) ;    
        }
        
        curNode = curNode->next_;
    }

    offset -= 1;
    offset += sprintf(output+offset,"}");
    
    return offset;
}

int report_fmt_data_2_url_keyval_string(OLREPORT_FMTDAT_T* fmt_dat,char* output)
{
    if( fmt_dat->node == NULL )
    {
        return -1;
    }

    OLREPORT_FMTNODE_T* curNode = (OLREPORT_FMTNODE_T* )(fmt_dat->node);
    int offset = 0;
    while(curNode != NULL)
    {
        if( curNode->val_type_==eValType_INT )
        {
            offset += sprintf(output+offset,URL_KEYVAL_INT_FMT,curNode->json_key_,curNode->val_.int_val_);
        }
        else if( strlen( curNode->val_.str_val_ ) > 0 )
        {
            offset += sprintf(output+offset,URL_KEYVAL_STR_FMT,curNode->json_key_,curNode->val_.str_val_);
        }

        curNode = curNode->next_;
    }

    output[offset-1] = '\0';
    return offset-1;
}
