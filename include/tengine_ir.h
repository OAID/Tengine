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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __TENGINE_IR_H__
#define __TENGINE_IR_H__

#include <stdint.h>

#include "tengine_c_api.h"
#include "module.h"
#include "vector.h"
#include "tengine_exec.h"

#define TENGINE_NODE_TYPE_INTER 1
#define TENGINE_NODE_TYPE_INPUT 2
#define TENGINE_NODE_TYPE_OUTPUT 4
#define MAX_CONSUMER_NUM 8

typedef int16_t fp16_t;

struct nn_device;
struct exec_context;
struct exec_attr;
struct dev_mem;
struct ir_node;

struct ir_tensor
{
    uint16_t idx;
    int16_t producer; /* node idx: -1 means no producer */
    int16_t consumer[MAX_CONSUMER_NUM]; /* idx of node */

    uint8_t reshaped; /* the tensor's shape has changed */
    uint8_t consumer_num;
    uint8_t tensor_type; /* const, input, var, dep */
    uint8_t data_type; /* int8,uint8,fp32,fp16,int32 */
    uint8_t dim_num;
    uint8_t elem_size; /* size of single element */
    uint8_t subgraph_num; /* subgraph_num wait this tensor ready */
    uint8_t free_host_mem; /* should free host memory ? */
    uint8_t internal_allocated; /* how memory is allocated? */
    uint8_t layout;

    uint16_t quant_param_num;
    uint32_t elem_num;
    int dims[MAX_SHAPE_DIM_NUM * 2];

    /* host cpu allocated memory */
    union
    {
        void* data;
        int8_t* i8;
        uint8_t* u8;
        float* f32;
        fp16_t* f16;
        int32_t* i32;
    };

    char* name;

    union
    {
        float* scale_list;
        float scale;
    };

    union
    {
        int zero_point;
        int* zp_list;
    };

    /* execution related fields */

    struct dev_mem* dev_mem;
    uint8_t* subgraph_list;
};

struct ir_op
{
    uint16_t op_type;
    uint8_t op_version;
    uint8_t same_shape;
    uint16_t param_size;
    void* param_mem;
    int (*infer_shape)(struct ir_node*);
};

struct ir_attr
{
    /* the whole memory block size of this attr
       the total size is sizeof(struct ir_attr)+data_size+strlen(attr_name)+1+strlen(type_name)+1 */
    uint16_t mem_size;
    uint16_t data_size;
    char* attr_name;
    char* type_name;
    int32_t mem_block[0]; /* aligned for 4 bytes */
    /* the content of mem_block is:
       data_val;
       attr_name;
       type_name;
    */
};

struct ir_node
{
    /* */
    uint16_t idx;
    uint8_t dynamic_shape;
    uint8_t input_num;
    uint8_t output_num;
    uint8_t attr_num;
    uint8_t node_type; /* subgraph_input, subgraph_output, intermediate */
    int8_t subgraph_idx;

    int16_t* input_tensors;
    int16_t* output_tensors;
    char* name;

    struct ir_op op;
    struct ir_attr* attr_mem;
    struct ir_graph* graph;
};

struct subgraph
{
    uint8_t idx;
    uint8_t input_ready_count;
    uint8_t input_wait_count;
    uint8_t input_num; /* input tensor number */
    uint8_t output_num;
    uint8_t status; /* subgraph execution status */
    uint16_t node_num;

    uint16_t* node_list; /* node idx list */
    uint16_t* input_tensor_list;
    uint16_t* output_tensor_list;

    struct ir_graph* graph;

    struct nn_device* nn_dev; /* the device to run the subgraph */
    void* exec_graph; /* the execution graph */
};

struct ir_graph
{
    struct ir_tensor** tensor_list;
    struct ir_node** node_list;
    int16_t* input_nodes;
    int16_t* output_nodes;

    uint16_t tensor_num;
    uint16_t node_num;
    uint16_t input_num;
    uint16_t output_num;

    int8_t graph_layout;
    int8_t model_layout;
    int8_t model_format;

    uint8_t attr_num;
    uint8_t status;

    struct serializer* serializer;
    void* serializer_priv; /* serializer saved content */
    void* dev_priv; /* DLA serializer may use this to pass some info to DLA device */

    struct nn_device * nn_dev; /* assigned nn_dev for this graph */
    struct exec_attr* exec_attr;
    struct ir_attr* attr_mem;
    struct vector* subgraph_list;
    struct vector* graph_list; /* for composed graph */
};

struct ir_graph* create_ir_graph(struct exec_context* context);
void init_ir_graph(struct ir_graph* ir_graph, struct exec_context* context);
void destroy_ir_graph(struct ir_graph* ir_graph);

int set_ir_graph_input_node(struct ir_graph* ir_graph, int16_t input_nodes[], int input_number);
int set_ir_graph_output_node(struct ir_graph* ir_graph, int16_t output_nodes[], int output_number);

static inline struct ir_tensor* get_ir_graph_tensor(struct ir_graph* ir_graph, int idx)
{
    return ir_graph->tensor_list[idx];
}

static inline struct ir_node* get_ir_graph_node(struct ir_graph* ir_graph, int idx)
{
    return ir_graph->node_list[idx];
}

static inline struct subgraph* get_ir_graph_subgraph(struct ir_graph* ir_graph, int idx)
{
    return *( struct subgraph** )get_vector_data(ir_graph->subgraph_list, idx);
}

static inline struct exec_context* get_ir_graph_context(struct ir_graph* ir_graph)
{
    return ir_graph->exec_attr->exec_context;
}

static inline struct exec_attr* get_ir_graph_exec_attr(struct ir_graph* ir_graph)
{
    return ir_graph->exec_attr;
}

int infer_shape_graph(struct ir_graph* ir_graph);

void dump_ir_graph(struct ir_graph* ir_graph);

/* subgraph related */

void init_subgraph(struct ir_graph* ir_graph, struct subgraph* subgraph, int subgraph_idx);
void release_subgraph(struct ir_graph* ir_graph, struct subgraph* subgraph);

/* Node related */

struct ir_node* create_ir_node(struct ir_graph* ir_graph, const char* node_name, int op_type, int op_version);
char* create_node_name_from_idx(int idx);
void destroy_ir_node(struct ir_graph* ir_graph, struct ir_node* node);

int get_node_idx_from_name(struct ir_graph* ir_graph, const char* node_name);
int set_ir_node_input_tensor(struct ir_node* node, int input_idx, struct ir_tensor* tensor);
int set_ir_node_output_tensor(struct ir_node* ir_node, int output_idx, struct ir_tensor* tensor);

void dump_ir_node(struct ir_graph* g, struct ir_node* n);

/* Tensor related */

struct ir_tensor* create_ir_tensor(struct ir_graph* ir_graph, const char* tensor_name, int data_type);

void destroy_ir_tensor(struct ir_graph* ir_graph, struct ir_tensor* ir_tensor);

int set_ir_tensor_shape(struct ir_tensor* ir_tensor, const int dims[], int dim_number);

char* create_tensor_name_from_idx(int idx);

int get_tensor_idx_from_name(struct ir_graph* ir_graph, const char* tensor_name);

int set_ir_tensor_quant_param(struct ir_tensor* ir_tensor, const float* scale, const int* zero_point, int number);
int get_ir_tensor_quant_param(struct ir_tensor* ir_tensor, float* scale, int* zero_point, int number);

void dump_ir_tensor(struct ir_graph* g, struct ir_tensor* t);

/* subgraph related */
void destroy_subgraph(struct ir_graph* g, struct subgraph* subgraph);

/* ir attr related */

/*
    if new attr entry added, return the new pointer of attr block.
    otherwise, return NULL.
*/

struct ir_attr* add_new_attr(struct ir_attr* attr_mem, int attr_num, const char* attr_name, const char* type_name,
                             int val_size);

struct ir_attr* remove_single_attr(struct ir_attr* attr_mem, int attr_num, const char* attr_name);

void remove_all_attr(struct ir_attr* attr_mem, int attr_num);

int set_attr_val(struct ir_attr* attr_mem, int attr_num, const char* attr_name, const char* type_name, const void* buf,
                 int size);

int get_attr_val(struct ir_attr* attr_mem, int attr_num, const char* attr_name, const char* type_name, void* buf,
                 int size);



/* simple pack and unpack */

int pack_ir_graph(struct ir_graph * ir_graph, void **mem, int * mem_size);
struct ir_graph* unpack_ir_graph(const void * mem, int mem_size);

#endif
