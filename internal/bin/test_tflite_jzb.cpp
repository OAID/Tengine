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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <time.h>

#include "tengine_c_api.h"

std::string model_name = "./models/converted_modelsx.tflite";

void inline dump_kernel_value(const tensor_t tensor, const char* dump_file_name)
{
    std::ofstream of(dump_file_name, std::ios::out);
    int kernel_dim[4];
    int dim_len = 0;
    dim_len = get_tensor_shape(tensor, kernel_dim, 4);
    int data_couts = 1;
    for(int ii = 0; ii < dim_len; ++ii)
    {
        data_couts *= kernel_dim[ii];
    }

    const float* tmp_data = ( const float* )get_tensor_buffer(tensor);
    char tmpBuf[1024];
    int iPos = 0;
    for(int ii = 0; ii < data_couts; ++ii)
    {
	printf("%f \n", tmp_data[ii]);
        iPos += sprintf(&tmpBuf[iPos], "%.18e", tmp_data[ii]);
        of << tmpBuf << std::endl;
        iPos = 0;
    }

    of.close();
}

int main(int argc, char* argv[])
{
    init_tengine();

    graph_t graph = create_graph(nullptr, "tflite", model_name.c_str());
    set_graph_layout(graph, TENGINE_LAYOUT_NHWC);
    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return 1;
    }

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int dim[4] = {1,160,160,3};

    set_tensor_shape(input_tensor, dim, 4);

    int input_size = get_tensor_buffer_size(input_tensor);
    float* input_data = ( float* )malloc(input_size*sizeof(float));
    //dump_graph(graph);
    for(int i = 0; i < input_size; i++)
        input_data[i] =1 ;

    set_tensor_buffer(input_tensor, input_data, input_size);

    prerun_graph(graph);

    //dump_graph(graph);

    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 2, 0);

    dump_kernel_value(output_tensor, "./models/est_jzb_out.txt");
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
