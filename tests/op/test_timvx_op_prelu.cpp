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
    (void)layout; (void)n; (void)c; (void)h; (void)w;

    /* create the test node */
    node_t test_node = create_graph_node(graph, node_name, "PReLU");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(NULL == input_tensor)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    /* create the sub node to product another input tensors which the test node is needed, such as weight/bias/slope tensor. */
    node_t slope_node = create_graph_node(graph, "slope", "Const");
    tensor_t slope_tensor = create_graph_tensor(graph, "slope", TENGINE_DT_FP16);
    set_node_output_tensor(slope_node, 0, slope_tensor, TENSOR_TYPE_CONST);

    int dims[4];
    get_tensor_shape(input_tensor, dims, 4);
    int slope_dims[1] = {dims[1]};  // channel num
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

/*
 * scale = (max - min) / 255
 * zero_point = -min / scale
 * uint8   = clip(round(float32 / scale) + zero_point, 0, 255)
 * float32 = (uint8 - zero_point) * scale
 */
float input_scale = 0.039216f;
int input_zero_point = 255;
float output_scale = 0.011764f;
int output_zero_point = 255;

int main(int argc, char* argv[])
{
    int n = 1, c = 3, h = 4, w = 5;
    const char* test_node_name = "prelu";
    int data_type = TENGINE_DT_UINT8;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed.\n");

    // create
    graph_t graph = create_timvx_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_prelu_node);
    if(NULL == graph)
        return -1;

    // set quantize params
    tensor_t input_tesnor = get_graph_input_tensor(graph, 0, 0);
    tensor_t output_tesnor = get_graph_output_tensor(graph, 0, 0);
    set_tensor_quant_param(input_tesnor, &input_scale, &input_zero_point, 1);
    set_tensor_quant_param(output_tesnor, &output_scale, &output_zero_point, 1);

    // set input data
    fill_input_uint8_tensor_by_index(graph, 0, 0, -10.0f);

    // set slope data, need cost fp32 to fp16
    __fp16* slope_fp16 = (__fp16*)malloc(3 * sizeof(__fp16));
    for (int k = 0; k < 3; k++)
        slope_fp16[k] = fp32_to_fp16(slope_value[k]);

    fill_input_float_buffer_tensor_by_name(graph, test_node_name, 1, (void*)slope_fp16, 3 * sizeof(__fp16));

    // graph run
    ret = test_graph_run(graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(graph);
        return -1;
    }

    // get output and dequant
    struct tensor* output_tensor = (struct tensor*)get_graph_output_tensor(graph, 0, 0);
    uint8_t* output_u8 = ( uint8_t* )output_tensor->data;
    int output_size = output_tensor->elem_num;
    int out_c = output_tensor->dims[1];
    int cstep = output_tensor->dims[2] * output_tensor->dims[3];

    get_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);
    float* output_data = ( float* )malloc(output_size * sizeof(float));
    for (int i = 0; i < output_size; i++)
        output_data[i] = (( float )output_u8[i] - ( float )output_zero_point) * output_scale;

    // check the result
    ret = 0;
    for (int i = 0; i< out_c; i++)
    {
        float* output_value =  (float *)output_data + i * cstep;
        for (int j = 0; j < cstep; j++)
        {
            if (fabsf(output_value[j] - result_value[i]) > 0.01)
            {
                fprintf(stderr, "index:%d, a:%f, b:%f\n", j, output_value[j], result_value[i]);
                ret = -1;
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
