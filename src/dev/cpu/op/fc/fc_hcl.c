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
 * Author: haoluo@openailab.com
 */
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "fc_param.h"
#include "cortex_a/fc_kernel_arm.h"

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* filter_tensor;
    struct ir_tensor* output_tensor;

    struct fc_priv_info* priv_info = ( struct fc_priv_info* )exec_node->ops_priv;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* fc_param = ( struct fc_param* )ir_node->op.param_mem;

    /* prerun now */
    if (fc_kernel_prerun(input_tensor, filter_tensor, output_tensor, priv_info, fc_param) < 0)
    {
        TLOG_ERR("hcl fc prerun failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* weight_tensor;
    struct ir_tensor* bias_tensor = NULL;
    struct ir_tensor* output_tensor = NULL;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* fc_param = ( struct fc_param* )ir_node->op.param_mem;
    struct fc_priv_info* priv_info = ( struct fc_priv_info* )exec_node->ops_priv;

    if (fc_kernel_run(input_tensor, weight_tensor, bias_tensor, output_tensor, priv_info, fc_param, num_thread,
                      cpu_affinity) < 0)
    {
        TLOG_ERR("hcl fc run failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* node = exec_node->ir_node;
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int dim[4];

    int n = weight->dims[0];
    int k = weight->dims[1];

    int m = input->dims[0];
    int input_k = input->dims[1];

    if (input->dim_num == 2)
    {
        dim[0] = m;
        dim[1] = n;
    }
    else if (input->dim_num == 3)
    {
        input_k *= input->dims[2];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
        }
    }
    else if (input->dim_num == 4)
    {
        input_k *= input->dims[2] * input->dims[3];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = 1;
            dim[3] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
            dim[3] = 1;
        }
    }
    else
        return -1;

    if (k != input_k)
    {
        TLOG_ERR("fc: input tensor and weight tensor shape does not match, hidden_number: %d\n", k);
        set_tengine_errno(EFAULT);
        return -1;
    }

    int ret = set_ir_tensor_shape(output, dim, input->dim_num);

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct fc_priv_info* priv_info = ( struct fc_priv_info* )exec_node->ops_priv;

    if (fc_kernel_postrun(priv_info) < 0)
    {
        TLOG_ERR("hcl fc prerun failed\n");
        set_tengine_errno(EFAULT);
        return -1;
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    //    struct ir_node* ir_node   = exec_node->ir_node;
    //    struct ir_graph* ir_graph = ir_node->graph;
    //    struct ir_tensor* input_tensor  = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    //    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    //    struct fc_param* fc_param = ( struct fc_param* )ir_node->op.param_mem;

    struct fc_priv_info* priv_info = ( struct fc_priv_info* )sys_malloc(sizeof(struct fc_priv_info));
    if (priv_info == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    memset(priv_info, 0, sizeof(struct fc_priv_info));

    /* get shared memory size */
    exec_node->ops_priv = priv_info;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct fc_priv_info* priv_info = ( struct fc_priv_info* )exec_node->ops_priv;
    sys_free(priv_info);
    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_fc_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_FC, &hcl_node_ops);
}

static int unreg_fc_hcl_ops(void* arg)
{
    unregister_builtin_node_ops(OP_FC, &hcl_node_ops);
    return 0;
}

AUTO_REGISTER_OPS(reg_fc_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_fc_hcl_ops);
