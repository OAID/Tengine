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
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "common.hpp"

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

void get_data(void* buffer, int datasize, const char* fname)
{
    // read data
    FILE* data_fp = fopen(fname, "rb");
    if (!data_fp) printf("data can not be open\n");

    size_t n=fread(buffer, sizeof(float), datasize, data_fp);
    if((int)n<datasize)
        printf("data read error\n");

    fclose(data_fp);
}

void maxerr(float*pred, float* gt, int size)
{
    float maxError = 0.f;
    for (int i = 0; i<size; i++)
    {
        maxError = MAX((float)fabs(gt[i] - *(pred+i)), maxError);
    }
    printf("====================================\n");
    printf("maxError is %f\n",maxError);
    printf("====================================\n");
}

int repeat_count=1;

int main(int argc, char *argv[])
{
    std::string model_dir = get_root_path() + "models";

    if(argc==1)
    {
        std::cout<<"[Usage]: "<<argv[0]<<" <model_dir>\n";
    }

    if(argc >1 ) 
        model_dir=argv[1];

    // init tengine
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;

    // load model
    const char *model_name = "lighten_cnn";
    std::string proto_name_ = model_dir+"/LightenedCNN_B.prototxt";
    std::string mdl_name_ = model_dir+"/LightenedCNN_B.caffemodel";
    if (load_model(model_name, "caffe", proto_name_.c_str(), mdl_name_.c_str()) < 0)
        return 1;
    std::cout << "load model done!\n";
    // create graph
    graph_t graph = create_runtime_graph("graph", model_name, NULL);

    if (!check_graph_valid(graph))
    {
        std::cout << "create graph0 failed\n";
        return 1;
    }
    std::cout << "create graph done!\n";

    // input
    int img_h = 128;
    int img_w = 128;
    int img_size = img_h * img_w ;
    float *input_data = (float *)malloc(sizeof(float) * img_size);
    //for(int i=0;i<img_size;i++) input_data[i]=(1%128)/255.f;

    std::string input_fname=model_dir+"/data_16384";

    get_data(input_data, img_size,input_fname.c_str());

    tensor_t input_tensor = get_graph_input_tensor(graph, 0 ,0);
    int dims[] = {1, 1, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        std::printf("set buffer for input tensor failed\n");
        return -1;
    }

    prerun_graph(graph);
    run_graph(graph, 1);

    free(input_data);

    int size1 = 256;
    tensor_t mytensor1 = get_graph_tensor(graph, "eltwise_fc1");
    float *data1 = (float *)get_tensor_buffer(mytensor1);
    float *out1 = (float *)malloc(sizeof(float) * size1);

    std::string out_data_file=model_dir+"/eltwise_fc1_256";
    get_data(out1, size1, out_data_file.c_str());
    maxerr(data1, out1, size1);
	
	free(out1);
	put_graph_tensor(mytensor1);

    postrun_graph(graph);

    destroy_runtime_graph(graph);
    remove_model(model_name);

    return 0;
}


   

