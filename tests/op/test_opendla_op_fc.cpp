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
 * Author: qtang@openailab.com
 */

#include "test_op.h"

#include "graph/graph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "operator/prototype/fc_param.h"

int float_mismatch(float* current, float* reference, int size)
{
    for (int i = 0; i < size; i++)
    {
        float tmp = fabs(current[i]) - fabs(reference[i]);
        fprintf(stderr, "index:%d, a:%f, b:%f\n", i, current[i], reference[i]);
        if (fabs(tmp) > 0.5)
        {
            fprintf(stderr, "test failed, index:%d, a:%f, b:%f\n", i, current[i], reference[i]);
            return -1;
        }
    }
    fprintf(stderr, "test pass\n");

    return 0;
}

int create_test_fc_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    (void)layout;
    (void)n;
    (void)c;
    (void)h;
    (void)w;

    /* create the test node */
    struct node* test_node = (struct node*)create_graph_node(graph, node_name, "FullyConnected");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if (NULL == input_tensor)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    /* create the sub node to product another input tensors which the test node is needed, such as weight/bias/slope tensor. */
    /* weight */
    node_t weight_node = create_graph_node(graph, "weight", "Const");
    tensor_t weight_tensor = create_graph_tensor(graph, "weight", TENGINE_DT_FP32);
    set_node_output_tensor(weight_node, 0, weight_tensor, TENSOR_TYPE_CONST);
    int weight_dims[4] = {1, 10, 1, 1}; // channel num
    set_tensor_shape(weight_tensor, weight_dims, 4);

    /* bias */
    // node_t bias_node = create_graph_node(graph, "bias", "Const");
    // tensor_t bias_tensor = create_graph_tensor(graph, "bias", TENGINE_DT_INT32);
    // set_node_output_tensor(bias_node, 0, bias_tensor, TENSOR_TYPE_CONST);
    // int bias_dims[1] = {1};  // channel num
    // set_tensor_shape(bias_tensor, bias_dims, 1);

    /* input tensors of test node */
    set_node_input_tensor(test_node, 0, input_tensor);
    set_node_input_tensor(test_node, 1, weight_tensor);
    // set_node_input_tensor(test_node, 2, bias_tensor);

    /* output tensors of test node */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    /* set params */
    struct fc_param* param = (struct fc_param*)(struct node*)test_node->op.param_mem;

    param->num_output = 1;

    return 0;
}

/* quant params */
/*
 * scale = (max(abs(min), abs(max))) / 127
 * zero_point = 0
 * float32 = int8 * scale
 */
float input_scale = 0.03937f;
float weight_scales[2] = {0.023622f, 0.007874f};
float output_scale = 0.2007874f;
int input_zero_point = 0;
int weight_zero_point[2] = {0, 0};
int output_zero_point = 0;

/* float32 data */
float reference_out[1] = {18};

float input_data[3] = {1, 2, 1};

float weight_data[10] = {3, 0, 3, 3, 0, 3, 3, 0, 3};

float bias_data[2] = {0.5, 0.1};

/* int8 data */
/* int8 = clip(round(float32 / scale), -127, 127) */
int8_t input_i8_data[10] = {25, 40, 25, 25, 40, 25, 25, 40, 25, 40};

int main(int argc, char* argv[])
{
    int n = 1, c = 10, h = 1, w = 1;
    const char* test_node_name = "conv";
    int data_type = TENGINE_DT_INT8;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed.\n");

    // create
    struct graph* ir_graph = (struct graph*)create_opendla_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_fc_node);
    if (NULL == ir_graph)
        return -1;

    set_log_level(LOG_INFO);
    dump_graph(ir_graph);

    // set quantize params
    struct tensor* input_tensor = (struct tensor*)get_graph_tensor(ir_graph, "input_node");
    struct tensor* weight_tensor = (struct tensor*)get_graph_tensor(ir_graph, "weight");
    struct tensor* output_tensor = (struct tensor*)get_graph_tensor(ir_graph, "conv");

    //    tensor_t weight_tesnor = get_graph_input_tensor(ir_graph, 1, 0);
    set_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
    //    set_tensor_quant_param(weight_tensor, weight_scales, &weight_zero_point, 2);
    set_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);

    // set input data
    set_tensor_buffer(input_tensor, input_i8_data, 10 * sizeof(int8_t));

    // set weight data
    set_tensor_buffer(weight_tensor, weight_data, 10 * sizeof(float));

    // set bias data
    //    set_tensor_buffer(bias_tensor, bias_data, 2 * sizeof(float));

    // graph run
    ret = test_graph_run(ir_graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(ir_graph);
        return -1;
    }

    /* get output and dequant int8 to fp32 */
    int output_size = output_tensor->elem_num;
    int8_t* output_int8 = (int8_t*)output_tensor->data;

    std::vector<float> output_fp32(output_size);
    for (int i = 0; i < output_size; i++)
        output_fp32[i] = (float)output_int8[i] * output_scale;

    /* check the result */
    ret = float_mismatch(output_fp32.data(), reference_out, output_size);

    if (ret == 0)
        fprintf(stderr, "test pass.\n");
    else
        fprintf(stderr, "test failed.\n");
    // exit
    test_graph_release(ir_graph);

    return ret;
}
