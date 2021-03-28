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

#include <stdlib.h>
#include <string.h>

#include "sys_port.h"
#include "tengine_c_api.h"
#include "tengine_errno.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "tengine_exec.h"
#include "tengine_log.h"
#include "tengine_utils.h"
#include "tengine_serializer.h"

#define TENGINE_DEFAULT_LAYOUT TENGINE_LAYOUT_NCHW

struct ir_graph* create_ir_graph(struct exec_context* context)
{
    struct ir_graph* g = ( struct ir_graph* )sys_malloc(sizeof(struct ir_graph));

    if (g == NULL)
    {
        set_tengine_errno(ENOMEM);
        return NULL;
    }

    g->exec_attr = ( struct exec_attr* )sys_malloc(sizeof(struct exec_attr));

    if (g->exec_attr == NULL)
    {
        sys_free(g);
        set_tengine_errno(ENOMEM);
        return NULL;
    }

    init_ir_graph(g, context);

    return g;
}

void init_ir_graph(struct ir_graph* g, struct exec_context* context)
{
    g->tensor_list = NULL;
    g->node_list = NULL;
    g->input_nodes = NULL;
    g->output_nodes = NULL;

    g->tensor_num = 0;
    g->node_num = 0;
    g->input_num = 0;
    g->output_num = 0;

    g->subgraph_list = create_vector(sizeof(struct subgraph*), NULL);

    g->attr_num = 0;
    g->attr_mem = NULL;

    g->graph_layout = TENGINE_DEFAULT_LAYOUT;
    g->model_layout = TENGINE_DEFAULT_LAYOUT;
    g->model_format = MODEL_FORMAT_TENGINE;

    g->nn_dev = NULL;
    g->dev_priv = NULL;
    g->serializer_priv = NULL;
    g->serializer = NULL;
    g->status = GRAPH_STAT_CREATED;

    init_exec_attr(g->exec_attr, context);
}

void destroy_ir_graph(struct ir_graph* g)
{
    struct serializer* serializer = g->serializer;

    /* subgraph is related with run */
    int subgraph_num = get_vector_num(g->subgraph_list);

    for (int i = 0; i < subgraph_num; i++)
        release_subgraph(g, *( struct subgraph** )get_vector_data(g->subgraph_list, i));

    release_vector(g->subgraph_list);

    if (serializer && serializer->unload_graph)
        serializer->unload_graph(serializer, g, g->serializer_priv, g->dev_priv);

    /* note: must destroy tensor first, then node */

    for (int i = 0; i < g->tensor_num; i++)
        destroy_ir_tensor(g, g->tensor_list[i]);

    for (int i = 0; i < g->node_num; i++)
        destroy_ir_node(g, g->node_list[i]);

    sys_free(g->tensor_list);
    sys_free(g->node_list);
    sys_free(g->input_nodes);
    sys_free(g->output_nodes);

    if (g->attr_num)
        remove_all_attr(g->attr_mem, g->attr_num);

    if (g->exec_attr)
        destroy_exec_attr(g, g->exec_attr);

    sys_free(g);
}

int set_ir_graph_input_node(struct ir_graph* ir_graph, int16_t input_nodes[], int input_number)
{
    int16_t* new_input_nodes = ( int16_t* )sys_malloc(input_number * sizeof(int16_t));

    if (new_input_nodes == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    if (ir_graph->input_nodes)
    {
        sys_free(ir_graph->input_nodes);
        ir_graph->input_nodes = NULL;
    }

    ir_graph->input_nodes = new_input_nodes;
    ir_graph->input_num = input_number;

    for (int i = 0; i < input_number; i++)
    {
        struct ir_node* ir_node = get_ir_graph_node(ir_graph, input_nodes[i]);
        ir_node->node_type = TENGINE_NODE_TYPE_INPUT;
        ir_graph->input_nodes[i] = input_nodes[i];
    }

    return 0;
}

int set_ir_graph_output_node(struct ir_graph* ir_graph, int16_t output_nodes[], int output_number)
{
    int16_t* new_output_nodes = ( int16_t* )sys_malloc(output_number * sizeof(int16_t));

    if (new_output_nodes == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    if (ir_graph->output_nodes)
    {
        sys_free(ir_graph->output_nodes);
        ir_graph->output_nodes = NULL;
    }

    ir_graph->output_nodes = new_output_nodes;
    ir_graph->output_num = output_number;

    for (int i = 0; i < output_number; i++)
    {
        struct ir_node* ir_node = get_ir_graph_node(ir_graph, output_nodes[i]);
        ir_node->node_type = TENGINE_NODE_TYPE_OUTPUT;

        ir_graph->output_nodes[i] = output_nodes[i];
    }

    return 0;
}

int infer_shape_graph(struct ir_graph* ir_graph)
{
    int node_num = ir_graph->node_num;

    for (int i = 0; i < node_num; i++)
    {
        struct ir_node* node = get_ir_graph_node(ir_graph, i);
        struct ir_op* op = &node->op;

        if (node->input_num == 0)
            continue;

        if (node->dynamic_shape)
        {
            // populate the dynamic_shape
            int output_num = node->output_num;

            for (int j = 0; j < output_num; j++)
            {
                struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, j);

                for (int l = 0; l < tensor->consumer_num; l++)
                {
                    struct ir_node* child_node = get_ir_graph_node(ir_graph, l);
                    child_node->dynamic_shape = 1;
                }
            }

            continue;
        }

        if (op->same_shape)
        {
            struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
            struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

            output->dim_num = input->dim_num;
            output->elem_num = input->elem_num;

            memcpy(output->dims, input->dims, sizeof(int32_t) * input->dim_num);
        }
        else
        {
            if (op->infer_shape(node) < 0)
            {
                TLOG_ERR("infer shape failed for node: %d op: %s\n", node->idx, get_op_name(node->op.op_type));
                return -1;
            }

            // dump_ir_node(ir_graph, node);
        }

        for (int j = 0; j < node->output_num; j++)
        {
            struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, j);

            tensor->reshaped = 0;
        }
    }

    return 0;
}

void dump_ir_graph(struct ir_graph* g)
{
    TLOG_INFO("graph node_num %u tensor_num: %u attr_num: %u  subgraph_num: %u\n", g->node_num, g->tensor_num,
              g->attr_num, get_vector_num(g->subgraph_list));

    TLOG_INFO("graph layout: %s model layout: %s model_format: %s\n", layout_string(g->graph_layout),
              layout_string(g->model_layout), model_format_string(g->model_format));

    for (int i = 0; i < g->node_num; i++)
        dump_ir_node(g, g->node_list[i]);

    TLOG_INFO("\ngraph inputs: %u\n", g->input_num);

    for (int i = 0; i < g->input_num; i++)
    {
        struct ir_node* ir_node = get_ir_graph_node(g, g->input_nodes[i]);

        if (ir_node->name)
            TLOG_INFO("\t%s\n", ir_node->name);
        else
            TLOG_INFO("\tnode_%d\n", ir_node->idx);
    }

    TLOG_INFO("graph outputs: %u\n", g->output_num);

    for (int i = 0; i < g->output_num; i++)
    {
        struct ir_node* ir_node = get_ir_graph_node(g, g->output_nodes[i]);

        if (ir_node->name)
            TLOG_INFO("\t%s\n", ir_node->name);
        else
            TLOG_INFO("\tnode_%d\n", ir_node->idx);
    }
}

/************************** subgraph ************************************/

void init_subgraph(struct ir_graph* ir_graph, struct subgraph* subgraph, int subgraph_idx)
{
    subgraph->idx = subgraph_idx;
    subgraph->input_ready_count = 0;
    subgraph->input_wait_count = 0;
    subgraph->input_num = 0;
    subgraph->output_num = 0;
    subgraph->node_num = 0;
    subgraph->node_list = NULL;
    subgraph->input_tensor_list = NULL;
    subgraph->output_tensor_list = NULL;
    subgraph->graph = ir_graph;
    subgraph->nn_dev = NULL;
    subgraph->exec_graph = NULL;
    subgraph->status = GRAPH_STAT_CREATED;
}

void release_subgraph(struct ir_graph* ir_graph, struct subgraph* subgraph)
{
    struct nn_device* nn_dev = subgraph->nn_dev;

    if (subgraph->exec_graph)
        release_nn_dev_exec_graph(nn_dev, subgraph->exec_graph);

    sys_free(subgraph->input_tensor_list);
    sys_free(subgraph->output_tensor_list);
    sys_free(subgraph->node_list);
    sys_free(subgraph);
}

/************************** node ************************************/

static void init_ir_node(struct ir_node* node, int op_type, int op_version, int node_idx)
{
    node->idx = node_idx;
    node->dynamic_shape = 0;
    node->input_num = 0;
    node->output_num = 0;
    node->attr_num = 0;
    node->node_type = TENGINE_NODE_TYPE_INTER;
    node->input_tensors = NULL;
    node->output_tensors = NULL;
    node->name = NULL;
    node->op.op_type = op_type;
    node->op.op_version = op_version;
    node->op.same_shape = 1;
    node->op.param_size = 0;
    node->op.param_mem = NULL;
    node->op.infer_shape = NULL;
    node->attr_mem = NULL;
    node->subgraph_idx = -1;
}

struct ir_node* create_ir_node(struct ir_graph* ir_graph, const char* node_name, int op_type, int op_version)
{
    struct ir_node* node = ( struct ir_node* )sys_malloc(sizeof(struct ir_node));

    if (node == NULL)
    {
        set_tengine_errno(ENOMEM);
        return NULL;
    }

    init_ir_node(node, op_type, op_version, ir_graph->node_num);

    /* check if any op param should be set */

    struct op_method* m = find_op_method(op_type, op_version);

    if (m && m->init_op && (m->init_op(&node->op) < 0))
    {
        sys_free(node);
        return NULL;
    }

    struct ir_node** new_node_list =
        sys_realloc(ir_graph->node_list, sizeof(struct ir_node*) * (ir_graph->node_num + 1));

    if (new_node_list == NULL)
    {
        sys_free(node);
        set_tengine_errno(ENOMEM);
        return NULL;
    }

    node->graph = ir_graph;

    if (node_name)
        node->name = strdup(node_name);

    new_node_list[ir_graph->node_num] = node;

    ir_graph->node_list = new_node_list;
    ir_graph->node_num++;

    return node;
}

void destroy_ir_node(struct ir_graph* ir_graph, struct ir_node* node)
{
    if (node->name)
        sys_free(node->name);

    if (node->attr_num > 0)
    {
        remove_all_attr(node->attr_mem, node->attr_num);
    }

    if (node->input_num)
        sys_free(node->input_tensors);

    if (node->output_num)
        sys_free(node->output_tensors);

    struct op_method* m = find_op_method(node->op.op_type, node->op.op_version);

    if (m && m->release_op)
        m->release_op(&node->op);

    sys_free(node);
}

char* create_node_name_from_idx(int idx)
{
    char* name = ( char* )sys_malloc(16);

    if (name == NULL)
        return NULL;

    sprintf(name, "node_%d", idx);

    return name;
}

int get_node_idx_from_name(struct ir_graph* ir_graph, const char* node_name)
{
    struct ir_node* node;

    /* first: try to get idx from suffix  */
    char* p = strrchr(node_name, '_');

    if (p)
    {
        int idx = atoi(++p);

        if (idx >= 0 && idx < ir_graph->node_num)
        {
            node = ir_graph->node_list[idx];
            /* Note:
               It is possible to match the wrong node if the suffix is a digital
               while the corresponding node has no name string
               But we leave this to the graph creator to avoid such case
            */
            if (!node->name || !strcmp(node->name, node_name))
                return idx;
        }
    }

    /* second: search all nodes to compare name */

    for (int i = 0; i < ir_graph->node_num; i++)
    {
        node = ir_graph->node_list[i];

        if (node->name && !strcmp(node->name, node_name))
            return i;
    }

    return -1;
}

int set_ir_node_input_tensor(struct ir_node* ir_node, int input_idx, struct ir_tensor* tensor)
{
    if (tensor->consumer_num >= MAX_CONSUMER_NUM)
    {
        set_tengine_errno(ENOSPC);
        return -1;
    }

    if (input_idx >= ir_node->input_num)
    {
        int16_t* new_tensor = ( int16_t* )sys_realloc(ir_node->input_tensors, sizeof(int16_t) * (input_idx + 1));

        if (new_tensor == NULL)
        {
            set_tengine_errno(ENOMEM);
            return -1;
        }

        for (int i = ir_node->input_num; i < input_idx + 1; i++)
        {
            new_tensor[i] = -1;
        }

        ir_node->input_tensors = new_tensor;
        ir_node->input_num = input_idx + 1;
    }

    ir_node->input_tensors[input_idx] = tensor->idx;
    tensor->consumer[tensor->consumer_num] = ir_node->idx;
    tensor->consumer_num++;

    return 0;
}

int set_ir_node_output_tensor(struct ir_node* ir_node, int output_idx, struct ir_tensor* tensor)
{
    if (output_idx >= ir_node->output_num)
    {
        int16_t* new_tensor = ( int16_t* )sys_realloc(ir_node->output_tensors, sizeof(int16_t) * (output_idx + 1));

        for (int i = ir_node->output_num; i < output_idx + 1; i++)
        {
            new_tensor[i] = -1;
        }

        ir_node->output_tensors = new_tensor;
        ir_node->output_num = output_idx + 1;
    }

    ir_node->output_tensors[output_idx] = tensor->idx;
    tensor->producer = ir_node->idx;

    return 0;
}

void dump_ir_node(struct ir_graph* g, struct ir_node* n)
{
    if (n->name)
        TLOG_INFO("\nnode: %d op: %s name: %s\n", n->idx, get_op_name(n->op.op_type), n->name);
    else
        TLOG_INFO("\nnode: %d op: %s name: node_%d\n", n->idx, get_op_name(n->op.op_type), n->idx);

    if (n->input_num > 0)
        TLOG_INFO("\tinput tensors: %d\n", n->input_num);

    for (int i = 0; i < n->input_num; i++)
    {
        struct ir_tensor* tensor = get_ir_graph_tensor(g, n->input_tensors[i]);

        TLOG_INFO("\t    %d: [id: %d] ", i, tensor->idx);

        dump_ir_tensor(g, tensor);
    }

    if (n->output_num > 0)
        TLOG_INFO("\toutput tensors: %d\n", n->output_num);

    for (int i = 0; i < n->output_num; i++)
    {
        struct ir_tensor* tensor = get_ir_graph_tensor(g, n->output_tensors[i]);

        TLOG_INFO("\t    %d: [id: %d] ", i, tensor->idx);

        dump_ir_tensor(g, tensor);
    }
}

/************************** tensor ************************************/

void init_ir_tensor(struct ir_tensor* tensor, int tensor_idx, int data_type)
{
    tensor->idx = tensor_idx;
    tensor->producer = -1;

    tensor->reshaped = 0;
    tensor->tensor_type = TENSOR_TYPE_VAR;
    tensor->data_type = data_type;
    tensor->dim_num = 0;
    tensor->elem_size = data_type_size(data_type);
    tensor->subgraph_num = 0;
    tensor->free_host_mem = 0;
    tensor->internal_allocated = 1;
    tensor->quant_param_num = 0;
    tensor->elem_num = 0;

    for (int i = 0; i < MAX_SHAPE_DIM_NUM; i++)
        tensor->dims[i] = 0;

    tensor->data = NULL;
    tensor->name = NULL;
    tensor->scale_list = NULL;
    tensor->zero_point = 0;
    tensor->dev_mem = NULL;
    tensor->subgraph_list = NULL;

    tensor->consumer_num = 0;
    for (int i = 0; i < MAX_CONSUMER_NUM; i++)
        tensor->consumer[i] = -1;
}

struct ir_tensor* create_ir_tensor(struct ir_graph* ir_graph, const char* tensor_name, int data_type)
{
    struct ir_tensor* tensor = ( struct ir_tensor* )sys_malloc(sizeof(struct ir_tensor));

    if (tensor == NULL)
    {
        set_tengine_errno(ENOMEM);
        return NULL;
    }

    init_ir_tensor(tensor, ir_graph->tensor_num, data_type);

    tensor->layout = ir_graph->graph_layout;

    struct ir_tensor** new_tensor_list =
        sys_realloc(ir_graph->tensor_list, sizeof(struct ir_tensor*) * (ir_graph->tensor_num + 1));

    if (new_tensor_list == NULL)
    {
        sys_free(tensor);
        set_tengine_errno(ENOMEM);
        return NULL;
    }

    if (tensor_name)
        tensor->name = strdup(tensor_name);

    new_tensor_list[ir_graph->tensor_num] = tensor;

    ir_graph->tensor_list = new_tensor_list;
    ir_graph->tensor_num++;

    return tensor;
}

char* create_tensor_name_from_idx(int idx)
{
    char* name = ( char* )sys_malloc(16);

    if (name == NULL)
        return NULL;

    sprintf(name, "tensor_%d", idx);

    return name;
}

void destroy_ir_tensor(struct ir_graph* ir_graph, struct ir_tensor* ir_tensor)
{
    if (ir_tensor->quant_param_num > 1)
    {
        sys_free(ir_tensor->scale_list);
        sys_free(ir_tensor->zp_list);
    }

    if (ir_tensor->dev_mem)
    {
        struct ir_node* node = get_ir_graph_node(ir_graph, ir_tensor->producer);
        struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, node->subgraph_idx);
        struct nn_device* nn_dev = subgraph->nn_dev;

//        release_dev_mem(nn_dev, ir_tensor->dev_mem);

        sys_free(ir_tensor->dev_mem);
    }

    if (ir_tensor->free_host_mem && ir_tensor->data)
        sys_free(ir_tensor->data);

    if (ir_tensor->subgraph_num)
        sys_free(ir_tensor->subgraph_list);

    if (ir_tensor->name)
        sys_free(ir_tensor->name);

    sys_free(ir_tensor);
}

int set_ir_tensor_shape(struct ir_tensor* ir_tensor, const int dims[], int dim_number)
{
    if (dim_number > MAX_SHAPE_DIM_NUM * 2)
    {
        set_tengine_errno(EINVAL);
        return -1;
    }

    int old_num = ir_tensor->elem_num;
    int new_num = 1;

    for (int i = 0; i < dim_number; i++)
    {
        ir_tensor->dims[i] = dims[i];
        new_num *= dims[i];
    }

    ir_tensor->dim_num = dim_number;
    ir_tensor->elem_num = new_num;

    if (new_num != old_num)
        ir_tensor->reshaped = ir_tensor->consumer_num;

    return 0;
}

int get_tensor_idx_from_name(struct ir_graph* ir_graph, const char* tensor_name)
{
    struct ir_tensor* tensor;

    char* p = strrchr(tensor_name, '_');

    if (p)
    {
        int idx = atoi(++p);

        if (idx >= 0 && idx < ir_graph->tensor_num)
        {
            tensor = ir_graph->tensor_list[idx];

            if (!tensor->name || !strcmp(tensor->name, tensor_name))
                return idx;
        }
    }

    /* search all tensors */

    for (int i = 0; i < ir_graph->tensor_num; i++)
    {
        tensor = ir_graph->tensor_list[i];

        if (tensor->name && !strcmp(tensor->name, tensor_name))
            return i;
    }

    return -1;
}

int set_ir_tensor_quant_param(struct ir_tensor* ir_tensor, const float* scale, const int* zero_point, int number)
{
    if (number == 1)
    {
        ir_tensor->scale = scale[0];
        ir_tensor->zero_point = zero_point[0];
        ir_tensor->quant_param_num = 1;
        return 0;
    }

    float* t_scale = ( float* )sys_malloc(sizeof(float) * number);
    int* t_zero = ( int* )sys_malloc(sizeof(int) * number);

    if (t_scale == NULL || t_zero == NULL)
    {
        sys_free(t_scale);
        sys_free(t_zero);

        set_tengine_errno(ENOMEM);
        return -1;
    }

    memcpy(t_scale, scale, sizeof(float) * number);
    memcpy(t_zero, zero_point, sizeof(int) * number);

    sys_free(ir_tensor->scale_list);
    sys_free(ir_tensor->zp_list);

    ir_tensor->scale_list = t_scale;
    ir_tensor->zp_list = t_zero;
    ir_tensor->quant_param_num = number;

    return 0;
}

int get_ir_tensor_quant_param(struct ir_tensor* ir_tensor, float* scale, int* zero_point, int number)
{
    if (number < ir_tensor->quant_param_num)
    {
        set_tengine_errno(ENOSPC);
        return -1;
    }

    if (ir_tensor->quant_param_num == 1)
    {
        scale[0] = ir_tensor->scale;
        zero_point[0] = ir_tensor->zero_point;

        return 1;
    }

    memcpy(scale, ir_tensor->scale_list, sizeof(float) * ir_tensor->quant_param_num);
    memcpy(zero_point, ir_tensor->zp_list, sizeof(int) * ir_tensor->quant_param_num);

    return ir_tensor->quant_param_num;
}

void dump_ir_tensor(struct ir_graph* g, struct ir_tensor* t)
{
    if (t->name)
        TLOG_INFO("%s type: %s/%s", t->name, data_type_string(t->data_type), tensor_type_string(t->tensor_type));
    else
        TLOG_INFO("tensor_%d type: %s/%s", t->idx, data_type_string(t->data_type), tensor_type_string(t->tensor_type));

    if (t->dim_num)
    {
        char shape_buf[64];
        sprintf(shape_buf, " shape: [");

        for (int i = 0; i < t->dim_num - 1; i++)
            sprintf(shape_buf + strlen(shape_buf), "%d,", t->dims[i]);

        sprintf(shape_buf + strlen(shape_buf), "%d]", t->dims[t->dim_num - 1]);

        TLOG_INFO("%s", shape_buf);
    }
    else
        TLOG_INFO(" shape: []");

    if (t->producer >= 0)
    {
        struct ir_node* node = g->node_list[t->producer];

        TLOG_INFO(" from node: %d", node->idx);
    }

    if (t->consumer_num > 0)
        TLOG_INFO(" (consumer: %d)", t->consumer_num);

    TLOG_INFO("\n");
}

/************************** attr ************************************/

static inline struct ir_attr* get_next_attr(struct ir_attr* p_attr)
{
    return ( struct ir_attr* )(( char* )p_attr + p_attr->mem_size);
}

struct ir_attr* add_new_attr(struct ir_attr* attr_mem, int attr_num, const char* attr_name, const char* type_name,
                             int val_size)
{
    struct ir_attr* p_attr = attr_mem;
    int mem_size = 0;
    /* first, check if already exists */
    for (int i = 0; i < attr_num; i++)
    {
        if (!strcmp(p_attr->attr_name, attr_name))
        {
            set_tengine_errno(EEXIST);
            return NULL;
        }

        mem_size += p_attr->mem_size;
        p_attr = get_next_attr(p_attr);
    }

    int new_attr_size = sizeof(struct ir_attr) + val_size + strlen(attr_name) + 1;

    if (type_name)
        new_attr_size += strlen(type_name) + 1;

    struct ir_attr* new_attr = sys_realloc(attr_mem, mem_size + new_attr_size);

    p_attr = ( struct ir_attr* )(( char* )new_attr + mem_size);

    char* mem_block = ( char* )(p_attr + 1);

    p_attr->mem_size = new_attr_size;
    p_attr->data_size = val_size;
    p_attr->attr_name = mem_block + val_size;
    strcpy(p_attr->attr_name, attr_name);

    if (type_name != NULL)
    {
        p_attr->type_name = p_attr->attr_name + strlen(attr_name) + 1;
        strcpy(p_attr->type_name, type_name);
    }
    else
    {
        p_attr->type_name = NULL;
    }

    return new_attr;
}

static int access_attr_val(struct ir_attr* attr_mem, int attr_num, const char* attr_name, const char* type_name,
                           void* buf, int size, int set)
{
    struct ir_attr* p_attr = attr_mem;
    int i = 0;

    while (i < attr_num)
    {
        if (!strcmp(attr_name, p_attr->attr_name))
            break;

        p_attr = get_next_attr(p_attr);

        i++;
    }

    if (i == attr_num)
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    /* do type check */

    if (type_name && p_attr->type_name && strcmp(type_name, p_attr->type_name))
    {
        set_tengine_errno(ENOTSUP);
        return -1;
    }

    /* do size check */

    if (p_attr->data_size != size)
    {
        set_tengine_errno(ENOSPC);
        return -1;
    }

    if (set)
        memcpy(p_attr + 1, buf, size);
    else
        memcpy(buf, p_attr + 1, size);

    return 0;
}

int set_attr_val(struct ir_attr* attr_mem, int attr_num, const char* attr_name, const char* type_name, const void* buf,
                 int size)
{
    return access_attr_val(attr_mem, attr_num, attr_name, type_name, ( void* )buf, size, 1);
}

int get_attr_val(struct ir_attr* attr_mem, int attr_num, const char* attr_name, const char* type_name, void* buf,
                 int size)
{
    return access_attr_val(attr_mem, attr_num, attr_name, type_name, ( void* )buf, size, 0);
}

struct ir_attr* remove_single_attr(struct ir_attr* attr_mem, int attr_num, const char* attr_name)
{
    struct ir_attr* p_attr = attr_mem;

    int i = 0;

    while (i < attr_num)
    {
        if (!strcmp(attr_name, p_attr->attr_name))
            break;

        p_attr = get_next_attr(p_attr);

        i++;
    }

    if (i == attr_num)
    {
        set_tengine_errno(ENOENT);
        return NULL;
    }

    struct ir_attr* p_next_attr = get_next_attr(p_attr);

    int left_mem_size = 0;

    for (i++; i < attr_num; i++)
    {
        left_mem_size += p_next_attr->mem_size;
        p_next_attr = get_next_attr(p_next_attr);
    }

    if (left_mem_size > 0)
    {
        memcpy(p_attr, get_next_attr(p_attr), left_mem_size);
    }

    return attr_mem;
}

void remove_all_attr(struct ir_attr* attr_mem, int attr_num)
{
    sys_free(attr_mem);
}
