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
#include "operator/prototype/clip_param.h"

int create_test_clip_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    (void)layout;
    (void)n;
    (void)c;
    (void)h;
    (void)w;

    /* create the test node */
    struct node* test_node = (struct node*)create_graph_node(graph, node_name, "Clip");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if (NULL == input_tensor)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    /* input tensors of test node */
    set_node_input_tensor(test_node, 0, input_tensor);

    /* output tensors of test node */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    /* set params */
    struct clip_param* clip_param = (struct clip_param*)(struct node*)test_node->op.param_mem;

    clip_param->min = 0;
    clip_param->max = 6;

    return 0;
}

float input_fp32[5] = {-3.0f, 3.0f, 8.0f, 1.0f, -2.0f};

float reference_out[5] = {0.0f, 3.0f, 6.0f, 1.0f, 0.0f};

int main(int argc, char* argv[])
{
    int n = 1, c = 1, h = 5, w = 1;
    const char* test_node_name = "clip";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed.\n");

    // create
    graph_t graph = create_tensorrt_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_clip_node);
    if (NULL == graph)
        return -1;

    set_log_level(LOG_INFO);
    dump_graph(graph);

    // set quantize params
    struct tensor* input_tensor = (struct tensor*)get_graph_input_tensor(graph, 0, 0);
    struct tensor* output_tensor = (struct tensor*)get_graph_output_tensor(graph, 0, 0);

    // set input data
    set_tensor_buffer(input_tensor, input_fp32, 5 * 4);

    // graph run
    ret = test_graph_run(graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(graph);
        return -1;
    }

    // get output and dequant
    float* output_data = (float*)output_tensor->data;
    int output_size = output_tensor->elem_num;

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
    test_graph_release(graph);

    return ret;
}
