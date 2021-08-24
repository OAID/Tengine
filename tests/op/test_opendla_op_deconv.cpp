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
#include "operator/prototype/deconv_param.h"

float input_scale = 0.062992f;
int input_zero_point = 0;
float output_scale = 0.062992f;
int output_zero_point = 0;

int create_test_deconv_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    (void)layout;
    (void)n;
    (void)c;
    (void)h;
    (void)w;

    /* create the test node */
    struct node* test_node = (struct node*)create_graph_node(graph, node_name, "Deconvolution");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if (NULL == input_tensor)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    /* create the sub node to product another input tensors which the test node is needed, such as weight/bias/slope tensor. */
    /* weight */
    node_t weight_node = create_graph_node(graph, "weight", "Const");
    tensor_t weight_tensor = create_graph_tensor(graph, "weight", TENGINE_DT_INT8);
    set_node_output_tensor(weight_node, 0, weight_tensor, TENSOR_TYPE_CONST);
    int weight_dims[4] = {1, 1, 3, 3}; // channel num
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
    struct deconv_param* deconv_param = (struct deconv_param*)(struct node*)test_node->op.param_mem;

    deconv_param->num_output = 1;
    deconv_param->kernel_h = 3;
    deconv_param->kernel_w = 3;
    deconv_param->stride_h = 3;
    deconv_param->stride_w = 3;
    deconv_param->pad_h0 = 0;
    deconv_param->pad_w0 = 0;
    deconv_param->pad_h1 = 0;
    deconv_param->pad_w1 = 0;
    deconv_param->dilation_h = 1;
    deconv_param->dilation_w = 1;
    deconv_param->group = 1;
    deconv_param->activation = -1;
    deconv_param->output_pad_h0 = 0;
    deconv_param->output_pad_w0 = 0;

    return 0;
}

/*
 * scale = (max - min) / 255
 * zero_point = -min / scale
 * uint8   = clip(round(float32 / scale) + zero_point, 0, 255)
 * float32 = (uint8 - zero_point) * scale
 */
float input_fp32[18] = {
    3.0f,
    8.0f,
    1.0f,
    9.0f,
    5.0f,
    7.0f,
    3.0f,
    2.0f,
    3.0f,
    3.0f,
    8.0f,
    1.0f,
    9.0f,
    5.0f,
    7.0f,
    3.0f,
    2.0f,
    3.0f,
};

float weight_fp32[9] = {
    9.0f,
    0.0f,
    3.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    2.0f,
};

float reference_out[100] = {
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    50.000000,
    0.000000,
    50.000000,
    0.000000,
    23.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    3.000000,
    0.000000,
    8.000000,
    0.000000,
    7.000000,
    0.000000,
    6.000000,
};
float weight_scale = 0.023622f;
int weight_zero_point = 0;

void get_int8_data(float* data_fp32, int8_t* date_i8, int size, float scale, int zero_point)
{
    for (int i = 0; i < size; i++)
    {
        int idata = (round)(data_fp32[i] / scale + zero_point);
        if (idata > 255)
            idata = 255;
        else if (idata < 0)
            idata = 0;

        date_i8[i] = idata;
    }
}

int main(int argc, char* argv[])
{
    int n = 1, c = 1, h = 4, w = 4;
    const char* test_node_name = "deconv";
    int data_type = TENGINE_DT_INT8;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed.\n");

    // create
    struct graph* ir_graph = (struct graph*)create_cpu_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_deconv_node);
    if (NULL == ir_graph)
        return -1;

    set_log_level(LOG_INFO);
    dump_graph(ir_graph);

    // set quantize params
    struct tensor* input_tensor = (struct tensor*)get_graph_tensor(ir_graph, "input_node");
    struct tensor* weight_tensor = (struct tensor*)get_graph_tensor(ir_graph, "weight");
    struct tensor* output_tensor = (struct tensor*)get_graph_tensor(ir_graph, "deconv");

    //    tensor_t weight_tesnor = get_graph_input_tensor(ir_graph, 1, 0);
    set_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
    set_tensor_quant_param(weight_tensor, &weight_scale, &weight_zero_point, 1);
    set_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);

    // set input data
    int8_t input_i8[16] = {0};
    get_int8_data(input_fp32, input_i8, 16, input_scale, input_zero_point);
    set_tensor_buffer(input_tensor, input_i8, 16);

    // set weight data
    int8_t weight_i8[9] = {0};
    get_int8_data(weight_fp32, weight_i8, 9, weight_scale, weight_zero_point);
    set_tensor_buffer(weight_tensor, weight_i8, 9);

    // set bias data
    // fill_input_uint8_tensor_by_index(graph, 0, 0, 0.0f);

    // graph run
    ret = test_graph_run(ir_graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(ir_graph);
        return -1;
    }

    // get output and dequant
    int8_t* output_i8 = (int8_t*)output_tensor->data;
    int output_size = output_tensor->elem_num;

    get_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);
    float* output_data = (float*)malloc(output_size * sizeof(float));
    for (int i = 0; i < output_size; i++)
        output_data[i] = ((float)output_i8[i] - (float)output_zero_point) * output_scale;

    // check the result
    ret = 0;
    for (int i = 0; i < output_size; i++)
    {
        if (fabsf(output_data[i] - reference_out[i]) > 0.1)
        {
            fprintf(stderr, "index:%d, a:%f, b:%f\n", i, output_data[i], reference_out[i]);
            ret = -1;
        }
    }

    if (ret == 0)
        fprintf(stderr, "test pass.\n");
    else
        fprintf(stderr, "test failed.\n");

    // exit
    test_graph_release(ir_graph);

    return ret;
}
