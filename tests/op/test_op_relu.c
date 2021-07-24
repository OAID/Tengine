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


int create_test_relu_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    (void)layout; (void)n; (void)c; (void)h; (void)w;

    /* create the test node */
    node_t test_node = create_graph_node(graph, node_name, "ReLU");
    if (NULL == test_node)
    {
        fprintf(stderr, "create test node failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    if (NULL == input_tensor)
    {
        fprintf(stderr, "get graph input tensor failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    /* create the sub node to product another input tensors which the test node is needed, such as weight/bias/slope tensor. */
    // None

    /* input tensors of test node */
    set_node_input_tensor(test_node, 0, input_tensor);

    /* output tensors of test node */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    if (NULL == output_tensor)
    {
        fprintf(stderr, "create graph output tensor failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    /* set the attr of test node */
    // None

    return 0;
}


int main(int argc, char* argv[])
{
    int n = 1, c = 3, h = 12, w = 12;
    const char* test_node_name = "relu";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Engine init failed. ERRNO: %d.", get_tengine_errno());

    // create
    graph_t graph = create_common_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_relu_node);
    if(NULL == graph)
        return -1;

    // set input data
    fill_input_float_tensor_by_index(graph, 0, 0, -10.0f);

    // graph run
    ret = test_graph_run(graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(graph);
        return -1;
    }

    // dump input node
    int input_node_count = get_graph_input_node_number(graph);
    for(int i = 0; i < input_node_count; i++)
    {
        node_t input = get_graph_input_node(graph, i);
        dump_node_output(input, 0);
    }

    // dump output node
    int output_node_count = get_graph_output_node_number(graph);
    for(int i = 0; i < output_node_count; i++)
    {
        node_t output = get_graph_output_node(graph, i);
        dump_node_output(output, 0);
    }

    // exit
    test_graph_release(graph);

    return 0;
}
