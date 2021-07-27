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
 * Author: qwang02@openailab.com
 */

#include "test_op.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "operator/prototype/reduction_param.h"

static void get_uint8_data(float* data_fp32, uint8_t* date_u8, int size, float scale, int zero_point)
{
    for (int i = 0; i < size; i++)
    {
        int udata = (round)(data_fp32[i] / scale + zero_point);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;

        date_u8[i] = udata;
    }
}

int create_test_reduction_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    (void)layout;
    (void)n;
    (void)c;
    (void)h;
    (void)w;

    /* create the test node */
    node_t test_node = create_graph_node(graph, node_name, "Reduction");

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
    struct reduction_param* reduction_param = (struct reduction_param*)((struct node*)test_node)->op.param_mem;
    reduction_param->type = 0;
    reduction_param->dim_0 = 1;
    reduction_param->dim_1 = -2;
    reduction_param->dim_2 = -2;
    reduction_param->dim_3 = -2;

    return 0;
}

/*
 * scale = (max - min) / 255
 * zero_point = -min / scale
 * uint8   = clip(round(float32 / scale) + zero_point, 0, 255)
 * float32 = (uint8 - zero_point) * scale
 */
int dims[5] = {1, 2, 2, 3, 2};

float input_fp32[24] = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 0.0f, 3.0f, 5.0f,
                        5.0f, 0.0f, 1.0f, 5.0f, 5.0f, 4.0f, 3.0f, 5.0f,
                        2.0f, 3.0f, 4.0f, 2.0f, 4.0f, 3.0f, 3.0f, 5.0f};
float input_scale = 1.0f;
int input_zero_point = 0;

float reference_out[12] = {8.0000, 12.0000, 4.0000, 14.0000, 7.0000, 3.0000, 7.0000, 7.0000, 9.0000, 3.0000, 4.0000, 10.0000};
float output_scale = 1.0f;
int output_zero_point = 0;

int main(int argc, char* argv[])
{
    int n = 1, c = 1, h = 3, w = 3;
    const char* test_node_name = "reduction";
    int data_type = TENGINE_DT_UINT8;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed.\n");

    // create
    graph_t graph = create_timvx_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_reduction_node);
    if (NULL == graph)
        return -1;

    // set quantize params
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    struct tensor* output_tensor = (struct tensor*)get_graph_output_tensor(graph, 0, 0);

    set_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
    set_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);

    if (set_tensor_shape(input_tensor, dims, 5) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }
    set_tensor_shape(output_tensor, dims, 24);
    // set input data
    uint8_t input_u8[24] = {0};
    get_uint8_data(input_fp32, input_u8, 24, input_scale, input_zero_point);
    set_tensor_buffer(input_tensor, input_u8, 24);

    // graph run
    ret = test_graph_run(graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(graph);
        return -1;
    }

    // get output and dequant
    uint8_t* output_u8 = (uint8_t*)output_tensor->data;
    int output_size = 12;

    get_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);
    float* output_data = (float*)malloc(output_size * sizeof(float));
    for (int i = 0; i < output_size; i++)
        output_data[i] = ((float)output_u8[i] - (float)output_zero_point) * output_scale;

    // check the result
    ret = 0;
    for (int i = 0; i < output_size; i++)
    {
        if (fabsf(output_data[i] - reference_out[i]) > 0.01)
        {
            fprintf(stderr, "index: %d, a:%.4f, b:%.4f\n", i, output_data[i], reference_out[i]);
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
