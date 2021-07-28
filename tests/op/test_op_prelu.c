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

int create_test_prelu_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    (void)layout;
    (void)n;
    (void)c;
    (void)h;
    (void)w;

    /* create the test node */
    node_t test_node = create_graph_node(graph, node_name, "PReLU");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if (NULL == input_tensor)
    {
        fprintf(stderr, "create test node failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    /* create the sub node to product another input tensors which the test node is needed, such as weight/bias/slope tensor. */
    node_t slope_node = create_graph_node(graph, "slope", "Const");
    tensor_t slope_tensor = create_graph_tensor(graph, "slope", TENGINE_DT_FP32);
    set_node_output_tensor(slope_node, 0, slope_tensor, TENSOR_TYPE_CONST);

    int dims[4];
    get_tensor_shape(input_tensor, dims, 4);
    int slope_dims[1] = {dims[1]}; // channel num
    set_tensor_shape(slope_tensor, slope_dims, 1);

    /* input tensors of test node */
    set_node_input_tensor(test_node, 0, input_tensor);
    set_node_input_tensor(test_node, 1, slope_tensor);

    /* output tensors of test node */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    return 0;
}

float slope_value[3] = {0.1f, 0.2f, 0.3f};
float result_value[3] = {-1.f, -2.f, -3.f};

int main(int argc, char* argv[])
{
    int n = 1, c = 3, h = 6, w = 6;
    const char* test_node_name = "prelu";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed. ERRNO: %d.", get_tengine_errno());

    // create
    graph_t graph = create_common_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_prelu_node);
    if (NULL == graph)
        return -1;

    // set input data
    fill_input_float_tensor_by_index(graph, 0, 0, -10.0f);

    // set slope data
    fill_input_float_buffer_tensor_by_name(graph, test_node_name, 1, (void*)slope_value, 3 * sizeof(float));

    // graph run
    ret = test_graph_run(graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(graph);
        return -1;
    }

    // check the result
    struct tensor* output_tensor = get_graph_output_tensor(graph, 0, 0);
    int out_c = output_tensor->dims[1];
    int cstep = output_tensor->dims[2] * output_tensor->dims[3];

    ret = 0;
    for (int i = 0; i < out_c; i++)
    {
        float* output_data = (float*)output_tensor->data + i * cstep;
        for (int j = 0; j < cstep; j++)
        {
            if (output_data[j] != result_value[i])
            {
                fprintf(stderr, "Check result failed, current %f, expect %f\n", output_data[j], result_value[i]);
                ret = -1;
                break;
            }
        }
    }

    if (ret == 0)
        fprintf(stderr, "test pass.\n");
    else
        fprintf(stderr, "test failed.\n");

    // exit
    test_graph_release(graph);

    return ret;
}
