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

#include <cstdlib>
#include <cstdio>
#include <vector>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"



graph_t graph;
tensor_t unique_ids_raw_output;
tensor_t segment_ids;
tensor_t input_mask;
tensor_t input_ids;

tensor_t unique_ids;
tensor_t unstack_1;
tensor_t unstack_0;
int feature_len;

void init(const char* modelfile)
{
    int dims1[2] = {1, 256};
    int dims2[1] = {1};
    init_tengine();
    fprintf(stderr, "tengine version: %s\n", get_tengine_version());
    graph = create_graph(NULL, "tengine", modelfile);
    if (graph == NULL)
    {
        fprintf(stderr, "grph nullptr\n");
    }
    else
    {
        fprintf(stderr, "success init graph\n");
    }
    unique_ids_raw_output = get_graph_input_tensor(graph, 0, 0);
    segment_ids = get_graph_input_tensor(graph, 1, 0);
    input_mask = get_graph_input_tensor(graph, 2, 0);
    input_ids = get_graph_input_tensor(graph, 3, 0);

    set_tensor_shape(unique_ids_raw_output, dims2, 1);
    set_tensor_shape(segment_ids, dims1, 2);
    set_tensor_shape(input_mask, dims1, 2);
    set_tensor_shape(input_ids, dims1, 2);



    int rc = prerun_graph(graph);
    //dump_graph(graph);
    unique_ids = get_graph_output_tensor(graph, 0, 0);
    unstack_1 = get_graph_output_tensor(graph, 1, 0);
    unstack_0 = get_graph_output_tensor(graph, 2, 0);

    //get_tensor_shape(output_tensor, dims, 4);
    //feature_len = dims[1];
    fprintf(stderr, "bert prerun %d\n", rc);
}

int getResult()
{
   
    std::vector<float> input_data1(1,1.0f);
    std::vector<float> input_data2(256,1.0f);
    std::vector<float> input_data3(256,1.0f);
    std::vector<float> input_data4(256,1.0f);
    //get_input_data(imagefile, input_data.data(), height, width, means, scales);
    set_tensor_buffer(unique_ids_raw_output, input_data1.data(), 1 * sizeof(float));
    set_tensor_buffer(segment_ids, input_data2.data(), 256 * sizeof(float));
    set_tensor_buffer(input_mask, input_data3.data(), 256 * sizeof(float));
    set_tensor_buffer(input_ids, input_data4.data(), 256 * sizeof(float));

    //set_graph_layout(graph, 2);

    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "run_graph fail");
        return -1;
    }
    float* data1 = ( float* )get_tensor_buffer(unique_ids);
    float* data2 = ( float* )get_tensor_buffer(unstack_1);
    float* data3 = ( float* )get_tensor_buffer(unstack_0);
    
    printf ("data1: %f\n",*data1);
    printf ("data1: %f\n",data1[1]);
    printf ("data1: %f\n",data1[2]);

    printf ("data2: %f\n",*data2);
    printf ("data2: %f\n",data2[1]);
    printf ("data2: %f\n",data2[2]);

    printf ("data3: %f\n",data3[0]);
    printf ("data3: %f\n",data3[1]);
    printf ("data3: %f\n",data3[2]);

}

void release()
{
    release_graph_tensor(unique_ids);
    release_graph_tensor(unstack_1);
    release_graph_tensor(unstack_0);
    release_graph_tensor(unique_ids_raw_output);
    release_graph_tensor(segment_ids);
    release_graph_tensor(input_mask);
    release_graph_tensor(input_ids);
    destroy_graph(graph);
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-a person_a -b person_b]\n [-t thread_count]\n");
    fprintf(stderr, "\nmobilefacenet example: \n    ./mobilefacenet -m /path/to/mobilenet.tmfile -a "
                    "/path/to/person_a.jpg -b /path/to/person_b.jpg\n");
}

int main(int argc, char* argv[])
{
    char* model_file = NULL;
    //char* person_a = NULL;
    //char* person_b = NULL;

    int res;
    while ((res = getopt(argc, argv, "m:a:b:h")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (model_file == NULL)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file))
        return -1;

    init(model_file);



    getResult();



    release();
    return 0;
}