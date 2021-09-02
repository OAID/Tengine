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

#include <iostream>
#include "test_op.h"
#include "operator/prototype/eltwise_param.h"
#include "operator/prototype/concat_param.h"

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

float reference_out[50] = {
    1,
    2,
    2,
    2,
    2,
    3,
    1,
    3,
    3,
    3,
    3,
    3,
    1,
    3,
    3,
    4,
    4,
    4,
    1,
    4,
    4,
    4,
    4,
    4,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    1,
    3,
    3,
    3,
    3,
    3,
    1,
    3,
    3,
    4,
    4,
    4,
    1,
    4,
    4,
    4,
    4,
    4,
    1,
};

float input_array[25] = {1, 2, 2, 2, 2,
                         3, 1, 3, 3, 3,
                         3, 3, 1, 3, 3,
                         4, 4, 4, 1, 4,
                         4, 4, 4, 4, 1};

float input_scale = 0.062992f;
int input_zero_point = 0;
float output_scale = 0.062992f;
int output_zero_point = 0;

int create_test_concat_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    (void)layout;
    (void)n;
    (void)c;
    (void)h;
    (void)w;

    /* create the test node */
    node_t relu_1_node = create_graph_node(graph, "relu1", "ReLU");
    node_t relu_2_node = create_graph_node(graph, "relu2", "ReLU");
    node_t test_node = create_graph_node(graph, node_name, "Concat");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if (NULL == input_tensor)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    /* input tensors of relu node */
    set_node_input_tensor(relu_1_node, 0, input_tensor);

    /* output tensors of relu node */
    tensor_t relu_1_output_tensor = create_graph_tensor(graph, "relu_1_output", data_type);

    set_node_output_tensor(relu_1_node, 0, relu_1_output_tensor, TENSOR_TYPE_VAR);
    set_tensor_quant_param(relu_1_output_tensor, &output_scale, &output_zero_point, 1);

    set_node_input_tensor(relu_2_node, 0, relu_1_output_tensor);

    tensor_t relu_2_output_tensor = create_graph_tensor(graph, "relu_2_output", data_type);

    set_node_output_tensor(relu_2_node, 0, relu_2_output_tensor, TENSOR_TYPE_VAR);
    set_tensor_quant_param(relu_2_output_tensor, &output_scale, &output_zero_point, 1);

    set_node_input_tensor(test_node, 0, relu_1_output_tensor);
    set_node_input_tensor(test_node, 1, relu_2_output_tensor);
    struct concat_param* param = (struct concat_param*)((struct node*)test_node)->op.param_mem;
    param->axis = 1;
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    return 0;
}

int main(int argc, char* argv[])
{
    int n = 1, c = 1, h = 5, w = 5;
    const char* test_node_name = "concat";
    int data_type = TENGINE_DT_INT8;
    int layout = TENGINE_LAYOUT_NCHW;
    int img_size = n * c * h * w;
    int dims[] = {n, c, h, w}; // nchw
    std::vector<int8_t> input_i8(img_size);
    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed.\n");

    // create
    graph_t graph = create_opendla_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_concat_node);
    if (NULL == graph)
        return -1;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_i8.data(), img_size * sizeof(int8_t)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    // set quantize params
    tensor_t input_tesnor = get_graph_input_tensor(graph, 0, 0);
    tensor_t output_tesnor = get_graph_output_tensor(graph, 0, 0);
    set_tensor_quant_param(input_tesnor, &input_scale, &input_zero_point, 1);
    set_tensor_quant_param(output_tesnor, &output_scale, &output_zero_point, 1);

    // set input data
    /* prepare process input data, set the data mem to input tensor, quantize fp32 to int8 */

    for (int i = 0; i < img_size; i++)
    {
        int idata = (round)(input_array[i] / input_scale);
        if (idata > 127)
            idata = 127;
        else if (idata < -127)
            idata = -127;

        input_i8[i] = idata;
        std::cout << "input_i8 : " << i << " -> " << idata << std::endl;
    }

    // graph run
    ret = test_graph_run(graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(graph);
        return -1;
    }

    /* get output and dequant int8 to fp32 */
    struct tensor* output_tensor = (struct tensor*)get_graph_output_tensor(graph, 0, 0);

    int8_t* output_int8 = (int8_t*)output_tensor->data;
    int output_size = output_tensor->elem_num;
    get_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);

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
    test_graph_release(graph);

    return ret;
}