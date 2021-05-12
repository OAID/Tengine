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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include "convolution_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/log.h"


static int conv_op_map(int op)
{
    return OP_CONV;
}


static int tm2_load_conv(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                         const TM2_Operator* tm_op)
{
    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_ConvParam* tm_param = ( TM2_ConvParam* )(mem_base + tm_op->offset_t_param);

    conv_param->kernel_h = tm_param->kernel_h;
    conv_param->kernel_w = tm_param->kernel_w;
    conv_param->stride_h = tm_param->stride_h;
    conv_param->stride_w = tm_param->stride_w;

    /* todo: using new TM2 model definition*/

    conv_param->pad_h0 = tm_param->pad_h0;
    conv_param->pad_h1 = tm_param->pad_h1;
    conv_param->pad_w0 = tm_param->pad_w0;
    conv_param->pad_w1 = tm_param->pad_w1;

    conv_param->dilation_h = tm_param->dilation_h;
    conv_param->dilation_w = tm_param->dilation_w;
    conv_param->output_channel = tm_param->output_channel;
    conv_param->activation = tm_param->activation;

    conv_param->group = tm_param->group;

    /* TODO: get input_channel from tm_param */

    struct tensor* weight = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    conv_param->input_channel = weight->dims[1] * conv_param->group;

    if (ir_node->input_num > 2)
    {
        struct tensor* bias = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        int shape = bias->elem_num;

        set_ir_tensor_shape(bias, &shape, 1);
    }

    return 0;
}


int register_tm2_conv_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_CONVOLUTION, 1, tm2_load_conv, conv_op_map, NULL);

    return 0;
}


int unregister_tm2_conv_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_CONVOLUTION, 1, tm2_load_conv);

    return 0;
}
