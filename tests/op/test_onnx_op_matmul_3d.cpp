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

std::string node      = "test_matmul_3d";
std::string input_pb_0  = "../onnx_node/" + node + "/test_data_set_0/input_0.pb";
std::string input_pb_1  = "../onnx_node/" + node + "/test_data_set_0/input_1.pb";
std::string output_pb = "../onnx_node/" + node + "/test_data_set_0/output_0.pb";
std::string model     = "../onnx_node/" + node + "/onnx.tmfile";

int main(int argc, char* argv[])
{
    int c_0 = 2;
    int m_0 = 3;
    int k_0 = 4;

    int c_1 = 2;
    int k_1 = 4;
    int n_1 = 3;

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
    int input_size_0 = c_0 * m_0 * k_0;
    int dims_0[] = {c_0, m_0, k_0};
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
    int input_size_1 = c_1 * k_1 * n_1;
    int dims_1[] = {c_1, k_1, n_1};
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

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_pb_data(feature_in_0.data(), input_pb_0);
    get_pb_data(feature_in_1.data(), input_pb_1);

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
