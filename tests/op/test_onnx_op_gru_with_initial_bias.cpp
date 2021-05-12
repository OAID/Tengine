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
 * Author: qtang@openailab.com
 */


#include "test_onnx_op.h"

std::string node        = "test_gru_with_initial_bias";
std::string input_pb_0  = "../onnx_node/" + node + "/test_data_set_0/input_0.pb";
std::string input_pb_1  = "../onnx_node/" + node + "/test_data_set_0/input_1.pb";
std::string input_pb_2  = "../onnx_node/" + node + "/test_data_set_0/input_2.pb";
std::string input_pb_3  = "../onnx_node/" + node + "/test_data_set_0/input_3.pb";
std::string output_pb   = "../onnx_node/" + node + "/test_data_set_0/output_0.pb";
std::string model       = "../onnx_node/" + node + "/onnx.tmfile";

int main(int argc, char* argv[])
{
    int n_0 = 1;
    int c_0 = 1;
    int h_0 = 3;
    int w_0 = 3;

    int n_1 = 1;
    int c_1 = 1;
    int h_1 = 9;
    int w_1 = 3;

    int n_2 = 1;
    int c_2 = 1;
    int h_2 = 9;
    int w_2 = 3;

    int n_3 = 1;
    int c_3 = 1;
    int h_3 = 1;
    int w_3 = 18;    

    /* set runtime options */
    struct options opt;
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model.c_str());
    if (nullptr == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the shape, data buffer of input_tensor of the graph */
    /* input 0 */
    int input_size_0 = c_0 * h_0 * w_0;
    int dims_0[] = {c_0, h_0, w_0};
    std::vector<float> feature_in_0(input_size_0);

    tensor_t input_tensor_0 = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor_0 == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor_0, dims_0, 3) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor_0, feature_in_0.data(), input_size_0 * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* input 1 */
    int input_size_1 = c_1 * h_1 * w_1;
    int dims_1[] = {c_1, h_1, w_1};
    std::vector<float> feature_in_1(input_size_1);

    tensor_t input_tensor_1 = get_graph_input_tensor(graph, 1, 0);
    if (input_tensor_1 == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor_1, dims_1, 3) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor_1, feature_in_1.data(), input_size_1 * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }    

    /* input 2 */
    int input_size_2 = c_2 * h_2 * w_2;
    int dims_2[] = {c_2, h_2, w_2};
    std::vector<float> feature_in_2(input_size_2);

    tensor_t input_tensor_2 = get_graph_input_tensor(graph, 2, 0);
    if (input_tensor_2 == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor_2, dims_2, 3) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor_2, feature_in_2.data(), input_size_2 * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }      

    /* input 3 */
    int input_size_3 = h_3 * w_3;
    int dims_3[] = {h_3, w_3};
    std::vector<float> feature_in_3(input_size_3);

    tensor_t input_tensor_3 = get_graph_input_tensor(graph, 3, 0);
    if (input_tensor_3 == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor_3, dims_3, 2) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor_3, feature_in_3.data(), input_size_3 * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }            

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_pb_data(feature_in_0.data(), input_pb_0);
    get_pb_data(feature_in_1.data(), input_pb_1);
    get_pb_data(feature_in_2.data(), input_pb_2);
    get_pb_data(feature_in_3.data(), input_pb_3);

    /* run graph */
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "Run graph failed\n");
        return -1;
    }

    /* get the current result of inference */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* output_data = ( float* )get_tensor_buffer(output_tensor);
    int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

    /* get the reference result of inference */
    std::vector<float> reference_out(output_size);
    get_pb_data(reference_out.data(), output_pb);

    /* check the result */
    int ret = float_mismatch(output_data, reference_out.data(), output_size);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret;
}
