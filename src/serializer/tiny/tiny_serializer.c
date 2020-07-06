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

#include <stdio.h>

#include "sys_port.h"
#include "module.h"
#include "vector.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_serializer.h"

#include "tiny_graph.h"
#include "tengine_op.h"
#include "tengine_ir.h"

#include "op/conv_param.h"
#include "op/pool_param.h"

#define LOAD_NOTHING ((tiny_loader_t)(0x1))

typedef int (*tiny_loader_t)(struct ir_node* ir_node, struct tiny_node* tiny_node);

struct tensor_map
{
    const struct tiny_tensor* tiny;
    struct ir_tensor* ir;
};

static tiny_loader_t* op_loader_map = NULL;

static tiny_loader_t find_op_loader(int tiny_op)
{
    if (tiny_op < 0 || tiny_op >= NN_OP_MAX)
        return NULL;

    return op_loader_map[tiny_op];
}

static int tiny_op_map(int tiny_op)
{
    switch (tiny_op)
    {
        case NN_OP_CONV:
            return OP_CONV;
        case NN_OP_FC:
            return OP_FC;
        case NN_OP_POOL:
            return OP_POOL;
        case NN_OP_RELU:
            return OP_RELU;
        case NN_OP_SOFTMAX:
            return OP_SOFTMAX;
        default:
            return -1;
    }
}

static int data_type_map(int tiny_type)
{
    switch (tiny_type)
    {
        case NN_DT_Q7:
            return TENGINE_DT_INT8;
        case NN_DT_Q15:
            return TENGINE_DT_INT16;
        case NN_DT_Q31:
            return TENGINE_DT_INT32;
        case NN_DT_FP32:
            return TENGINE_DT_FP32;
        default:
            return 0;
    }
}

static struct ir_tensor* find_tensor_map(struct vector* map_vector, const struct tiny_tensor* tiny_tensor)
{
    int n = get_vector_num(map_vector);

    for (int i = 0; i < n; i++)
    {
        struct tensor_map* map = ( struct tensor_map* )get_vector_data(map_vector, i);

        if (map->tiny == tiny_tensor)
            return map->ir;
    }

    return NULL;
}

static int add_tensor_map(struct vector* map_vector, const struct tiny_tensor* tiny, struct ir_tensor* ir)
{
    struct tensor_map map;

    map.tiny = tiny;
    map.ir = ir;

    return push_vector_data(map_vector, &map);
}

static void set_ir_tensor_from_tiny_tensor(struct ir_tensor* ir_tensor, const struct tiny_tensor* tiny_tensor)
{
    set_ir_tensor_shape(ir_tensor, tiny_tensor->dims, tiny_tensor->dim_num);

    if (tiny_tensor->data)
    {
        ir_tensor->data = ( void* )tiny_tensor->data;
        ir_tensor->internal_allocated = 0;
    }

    if (tiny_tensor->shift != 0)
    {
        float scale = 1 << tiny_tensor->shift;
        int zero_point = 0;

        set_ir_tensor_quant_param(ir_tensor, &scale, &zero_point, 1);
    }
}

struct ir_tensor* create_const_or_input_tensor(struct vector* tensor_map_vector, struct ir_graph* graph,
                                               const struct tiny_tensor* tiny_tensor)
{
    if (tiny_tensor->tensor_type != NN_TENSOR_INPUT && tiny_tensor->tensor_type != NN_TENSOR_CONST)
        return NULL;

    struct ir_tensor* ir_tensor = create_ir_tensor(graph, NULL, data_type_map(tiny_tensor->data_type));

    if (ir_tensor == NULL)
        return NULL;

    struct ir_node* ir_node;
    if (tiny_tensor->tensor_type == NN_TENSOR_INPUT)
    {
        ir_node = create_ir_node(graph, NULL, OP_INPUT, 1);
        ir_tensor->tensor_type = TENSOR_TYPE_INPUT;
    }
    else
    {
        ir_node = create_ir_node(graph, NULL, OP_CONST, 1);
        ir_tensor->tensor_type = TENSOR_TYPE_CONST;
    }

    if (ir_node == NULL)
        return NULL;

    set_ir_tensor_from_tiny_tensor(ir_tensor, tiny_tensor);
    add_tensor_map(tensor_map_vector, tiny_tensor, ir_tensor);

    set_ir_node_output_tensor(ir_node, 0, ir_tensor);

    return ir_tensor;
}

static int load_model(struct serializer* s, struct ir_graph* graph, const char* fname, va_list ap)
{
    /* indeed, the fname is the struct tiny_graph */
    struct tiny_graph* tiny_graph = ( struct tiny_graph* )fname;
    int node_number = tiny_graph->node_num;
    struct tiny_node** node_list = ( struct tiny_node** )tiny_graph->node_list;

    /* print out some debug info */

    TLOG_DEBUG("tiny graph name: %s, node num: %d\n", tiny_graph->name, node_number);

    /* set graph input and output nodes */
    int graph_input_num = 0;
    int graph_output_num = 0;

    int16_t* input_nodes = NULL;
    int16_t* output_nodes = NULL;

    struct vector* tensor_map_vector = create_vector(sizeof(struct tensor_map), NULL);

    if (tensor_map_vector == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /* set graph layout */

    graph->graph_layout = TENGINE_LAYOUT_NHWC;
    graph->model_layout = TENGINE_LAYOUT_NHWC;

    for (int n = 0; n < node_number; n++)
    {
        struct tiny_node* tiny_node = node_list[n];
        struct ir_node* ir_node = NULL;
        struct ir_tensor* input_tensor[MAX_NODE_INPUT_NUM];

        /* create input tensor if necessary */
        int input_num = tiny_node->input_num;

        if (tiny_node == NULL)
        {
            TLOG_ERR("tiny: why node %d 's ptr in list is NULL\n");
            goto error;
        }

        for (int i = 0; i < input_num; i++)
        {
            struct ir_tensor* ir_tensor = find_tensor_map(tensor_map_vector, tiny_node->input[i]);

            if (ir_tensor == NULL)
            {
                ir_tensor = create_const_or_input_tensor(tensor_map_vector, graph, tiny_node->input[i]);

                if (ir_tensor == NULL)
                {
                    TLOG_ERR("tiny: cannot find/create input tensor %d for node: %d\n", i, tiny_node->op_type);
                    set_tengine_errno(EFAULT);
                    goto error;
                }
            }

            input_tensor[i] = ir_tensor;
        }

        /* load node */

        int tiny_op = tiny_node->op_type;

        tiny_loader_t tiny_loader = find_op_loader(tiny_op);

        if (tiny_loader == NULL)
        {
            TLOG_ERR("tiny: cannot find loader for op: %d\n", tiny_op);
            set_tengine_errno(ENOTSUP);
            goto error;
        }

        int ir_op = tiny_op_map(tiny_op);

        if (ir_op < 0)
        {
            TLOG_ERR("tiny: cannot map op: %d to tengine-lite op\n", tiny_op);
            set_tengine_errno(EFAULT);
            goto error;
        }

        ir_node = create_ir_node(graph, NULL, ir_op, 1);

        if (ir_node == NULL)
        {
            TLOG_ERR("tiny: create ir node for tiny op: %d, ir op: %d failed\n", tiny_op, ir_op);
            set_tengine_errno(EFAULT);
            goto error;
        }

        if (tiny_loader != LOAD_NOTHING && tiny_loader(ir_node, tiny_node) < 0)
        {
            TLOG_ERR("tiny: load ir node for op: %d failed\n", tiny_op);
            goto error;
        }

        /* create output tensor */
        struct ir_tensor* output_tensor = create_ir_tensor(graph, NULL, data_type_map(tiny_node->output->data_type));

        if (output_tensor == NULL)
        {
            set_tengine_errno(ENOMEM);
            goto error;
        }

        set_ir_tensor_from_tiny_tensor(output_tensor, tiny_node->output);

        add_tensor_map(tensor_map_vector, tiny_node->output, output_tensor);

        /* bind node input tensor and output tensor */

        for (int i = 0; i < tiny_node->input_num; i++)
            set_ir_node_input_tensor(ir_node, i, input_tensor[i]);

        set_ir_node_output_tensor(ir_node, 0, output_tensor);
    }

    for (int i = 0; i < graph->node_num; i++)
    {
        struct ir_node* ir_node = graph->node_list[i];

        if (ir_node->op.op_type == OP_INPUT)
        {
            graph_input_num++;
            input_nodes = sys_realloc(input_nodes, graph_input_num * sizeof(int16_t));
            input_nodes[graph_input_num - 1] = ir_node->idx;
        }
        else
        {
            int j;
            for (j = 0; j < ir_node->output_num; j++)
            {
                struct ir_tensor* ir_tensor = get_ir_graph_tensor(graph, ir_node->output_tensors[j]);

                if (ir_tensor->consumer_num == 0)
                    break;
            }

            if (j != ir_node->output_num)
            {
                graph_output_num++;
                output_nodes = sys_realloc(output_nodes, graph_output_num * sizeof(int16_t));
                output_nodes[graph_output_num - 1] = ir_node->idx;
            }
        }
    }

    set_ir_graph_input_node(graph, input_nodes, graph_input_num);
    set_ir_graph_output_node(graph, output_nodes, graph_output_num);

    sys_free(input_nodes);
    sys_free(output_nodes);

    release_vector(tensor_map_vector);

    graph->serializer = s;
    graph->serializer_priv = NULL;
    graph->dev_priv = NULL;

    return 0;

error:
    release_vector(tensor_map_vector);
    return -1;
}

static int load_op_conv(struct ir_node* ir_node, struct tiny_node* tiny_node)
{
    struct tiny_conv_param* tiny_param = ( struct tiny_conv_param* )tiny_node->op_param;
    struct conv_param* conv_param = ( struct conv_param* )(ir_node->op.param_mem);

    conv_param->kernel_h = tiny_param->kernel_h;
    conv_param->kernel_w = tiny_param->kernel_w;
    conv_param->stride_h = tiny_param->stride_h;
    conv_param->stride_w = tiny_param->stride_w;
    conv_param->pad_h0 = conv_param->pad_h1 = tiny_param->pad_h;
    conv_param->pad_w0 = conv_param->pad_w1 = tiny_param->pad_w;

    /* input channel and output channel */
    const struct tiny_tensor* weight = tiny_node->input[1];

    conv_param->input_channel = weight->dims[2];
    conv_param->output_channel = weight->dims[3];

    return 0;
}

static int pool_method_map(int tiny_method)
{
    if (tiny_method == NN_POOL_MAX)
        return POOL_MAX;
    if (tiny_method == NN_POOL_AVG)
        return POOL_AVG;

    return -1;
}

static int load_op_pool(struct ir_node* ir_node, struct tiny_node* tiny_node)
{
    struct tiny_pool_param* tiny_param = ( struct tiny_pool_param* )tiny_node->op_param;
    struct pool_param* pool_param = ( struct pool_param* )(ir_node->op.param_mem);

    pool_param->kernel_h = tiny_param->kernel_h;
    pool_param->kernel_w = tiny_param->kernel_w;
    pool_param->stride_h = tiny_param->stride_h;
    pool_param->stride_w = tiny_param->stride_w;
    pool_param->pad_h0 = pool_param->pad_h1 = tiny_param->pad_h;
    pool_param->pad_w0 = pool_param->pad_w1 = tiny_param->pad_w;

    pool_param->pool_method = pool_method_map(tiny_param->pool_method);

    return 0;
}

static int init_tiny_serializer(struct serializer* s)
{
    op_loader_map = ( tiny_loader_t* )sys_malloc(sizeof(tiny_loader_t) * NN_OP_MAX);

    if (op_loader_map == NULL)
        return -1;

    for (int i = 0; i < NN_OP_MAX; i++)
        op_loader_map[i] = NULL;

    /* register op loader  */
    op_loader_map[NN_OP_CONV] = load_op_conv;
    op_loader_map[NN_OP_POOL] = load_op_pool;
    op_loader_map[NN_OP_FC] = LOAD_NOTHING;
    op_loader_map[NN_OP_RELU] = LOAD_NOTHING;
    op_loader_map[NN_OP_SOFTMAX] = LOAD_NOTHING;

    return 0;
}

static int release_tiny_serializer(struct serializer* s)
{
    sys_free(op_loader_map);

    return 0;
}

static const char* get_name(struct serializer* s)
{
    return "tiny";
}

static struct serializer tiny_serializer = {
    .get_name = get_name,
    .load_model = load_model,
    .load_mem = NULL,
    .unload_graph = NULL,
    .register_op_loader = NULL, /* do not export dynamic op extension */
    .unregister_op_loader = NULL,
    .init = init_tiny_serializer,
    .release = release_tiny_serializer,
};

static int reg_tiny_serializer(void* arg)
{
    return register_serializer(&tiny_serializer);
}

static int unreg_tiny_serializer(void* arg)
{
    return unregister_serializer(&tiny_serializer);
}

REGISTER_MODULE_INIT(MOD_DEVICE_LEVEL, "reg_tiny_serializer", reg_tiny_serializer);
REGISTER_MODULE_EXIT(MOD_DEVICE_LEVEL, "unreg_tiny_serializer", unreg_tiny_serializer);
