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
#include "operator/prototype/eltwise_param.h"

int float_mismatch(float* current, float* reference, int size)
{
    for (int i = 0; i < size; i++)
    {
        float tmp = fabs(current[i]) - fabs(reference[i]);
        fprintf(stderr, "index:%d, a:%f, b:%f\n", i, current[i], reference[i]);
        if (fabs(tmp) > 0.1)
        {
            fprintf(stderr, "test failed, index:%d, a:%f, b:%f\n", i, current[i], reference[i]);
            return -1;
        }
    }
    fprintf(stderr, "test pass\n");

    return 0;
}

graph_t create_test_eltwise_graph(const char* node_name, int data_type, int layout, int n, int c, int h, int w, int dims_num = 4)
{
    (void)layout;
    (void)n;
    (void)c;
    (void)h;
    (void)w;

    context_t odla_context = create_context("odla", 1);
    int rtt = set_context_device(odla_context, "OPENDLA", NULL, 0);
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return NULL;
    }
    graph_t graph = create_graph(odla_context, nullptr, nullptr);
    if (nullptr == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return nullptr;
    }

    if (set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return nullptr;
    }

    /* create input left node  */
    if (create_input_node(graph, "input_left", data_type, layout, n, c, h, w, dims_num) < 0)
    {
        fprintf(stderr, "create input left node failed.\n");
        return nullptr;
    }

    /* create input right node  */
    if (create_input_node(graph, "input_right", data_type, layout, n, c, h, w, dims_num) < 0)
    {
        fprintf(stderr, "create input right node failed.\n");
        return nullptr;
    }

    /* create the eltwise node */
    struct node* test_node = (struct node*)create_graph_node(graph, node_name, "Eltwise");

    tensor_t input_left_tensor  = get_graph_tensor(graph, "input_left");
    tensor_t input_right_tensor = get_graph_tensor(graph, "input_right");

    /* create the sub node to product another input tensors which the test node is needed */
    /* input tensors of test node */
    set_node_input_tensor(test_node, 0, input_left_tensor);
    set_node_input_tensor(test_node, 1, input_right_tensor);

    /* output tensors of test node */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    /* set params */
    struct eltwise_param* eltwise_param = (struct eltwise_param*)(struct node*)test_node->op.param_mem;

    eltwise_param->type = ELT_SUM;

    /* set input/output node of graph */
    const char* inputs[] = {"input_left", "input_right"};
    const char* outputs[] = {"eltwise"};

    if (set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return nullptr;
    }

    if (set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return nullptr;
    }

    return graph;
}

/* fp32 data */
float reference_out_fp32[9] = {2, 2, 2,
                               3, 3, 3,
                               4, 4, 4};

float input_left_fp32[9] = {1, 1, 1,
                            2, 2, 2,
                            3, 3, 3};

float input_right_fp32[9] = {1, 1, 1,
                             1, 1, 1,
                             1, 1, 1};

/* int8 data */
float input_left_scale = 0.023622f;
int input_left_zero_point = 0;
float input_right_scale = 0.007874f;
int input_right_zero_point = 0;
float output_scale = 0.031496f;
int output_zero_point = 0;

int8_t reference_out_int8[9] = {2, 2, 2,
                                3, 3, 3,
                                4, 4, 4};

int8_t input_left_int8[9] = { 42,  42,  42,
                              87,  87,  87,
                              127, 127, 127};

int8_t input_right_int8[9] = {127, 127, 127,
                              127, 127, 127,
                              127, 127, 127};

int main(int argc, char* argv[])
{
    int n = 1;
    int c = 1;
    int h = 3;
    int w = 3;
    const char* test_node_name = "eltwise";
    int data_type = TENGINE_DT_INT8;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed.\n");

    // create
    graph_t graph = create_test_eltwise_graph(test_node_name, data_type, layout, n, c, h, w);
    if (nullptr == graph)
        return -1;

    /* fill test data */
    // set quantize params
    struct tensor* input_left_tensor  = (struct tensor*)get_graph_tensor(graph, "input_left");
    struct tensor* input_right_tensor = (struct tensor*)get_graph_tensor(graph, "input_right");
    struct tensor* output_tensor = (struct tensor*)get_graph_tensor(graph, "eltwise");

    set_tensor_quant_param(input_left_tensor, &input_left_scale, &input_left_zero_point, 1);
    set_tensor_quant_param(input_right_tensor, &input_right_scale, &input_right_zero_point, 1);
    set_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);

    // set input left data
    set_tensor_buffer(input_left_tensor, input_left_int8, 9 * sizeof(int8_t));

    // set input right data
    set_tensor_buffer(input_right_tensor, input_right_int8, 9 * sizeof(int8_t));

    // graph run
    ret = test_graph_run(graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(graph);
        return -1;
    }

    /* get output and dequant int8 to fp32 */
    int8_t* output_int8 = (int8_t*)output_tensor->data;
    int output_size = output_tensor->elem_num;
    get_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);

    std::vector<float> output_fp32(output_size);
    for (int i = 0; i < output_size; i++)
        output_fp32[i] = (float)output_int8[i] * output_scale;

    /* check the result */
    ret = float_mismatch(output_fp32.data(), reference_out_fp32, output_size);

    if (ret == 0)
        fprintf(stderr, "test pass.\n");
    else
        fprintf(stderr, "test failed.\n");

    // exit
    test_graph_release(graph);

    return ret;
}