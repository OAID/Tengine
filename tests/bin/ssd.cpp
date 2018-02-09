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
#include <unistd.h>

#include <iostream> 
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "prof_utils.hpp"
#include "tengine_c_api.h"
#include "image_process.hpp"
#include "common_util.hpp"
#include "tengine_config.hpp"
#include "prof_utils.hpp"
#include "prof_record.hpp"

const char * text_file="../Tengine_models/ssd/VGG_VOC0712_SSD_300.prototxt";
const char * model_file="../Tengine_models/ssd/VGG_VOC0712_SSD_300.caffemodel";


using namespace TEngine;

int repeat_count=1;


#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif
void get_data(void* buffer, int datasize, char* fname)
{
	// read data	
	FILE* data_fp = fopen(fname, "rb");
	if (!data_fp) printf("data can not be open\n");
	fread(buffer, sizeof(float), datasize, data_fp);
	fclose(data_fp);
}
void maxerr(float*pred,float* gt,int size)
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

int main(int argc, char *argv[])
{
  

    int res;

   while((res=getopt(argc,argv,"er:"))!=-1)
   {
      switch(res)
      {
         case 'e':
            TEngineConfig::Set("exec.engine","event");
            break;
         case 'r':
            repeat_count=strtoul(optarg,NULL,10);
            break;
         default:
            break;
      }  
   }


    // init tengine
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;

    // load model
    const char* model_name="ssd_300";
    if (load_model(model_name, "caffe", text_file, model_file) < 0)
        return 1;
    std::cout << "load model done!\n";

    // create graph
    graph_t graph = create_runtime_graph("graph", model_name, NULL);
    if (!check_graph_valid(graph))
    {
        std::cout << "create graph0 failed\n";
        return 1;
    }

    // input
//     int img_h = 299;
//     int img_w = 299;
//     int img_size = img_h * img_w * 3;
//     float *input_data = (float *)malloc(sizeof(float) * img_size);
//     //get_input_data(image_file, input_data, img_h,  img_w);
//     char* in_data_file="./tests/data/inception-v4/incepv4_data/data_268203";
//     get_data(input_data, img_size,in_data_file );

//   /* set input and output node*/
//     const char *input_node_name = "input";
//     set_graph_input_node(graph, &input_node_name, 1);

//     const char *input_tensor_name = "data";
//     tensor_t input_tensor = get_graph_tensor(graph, input_tensor_name);
//     int dims[] = {1, 3, img_h, img_w};
//     set_tensor_shape(input_tensor, dims, 4);
//     set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4);

//     /* run the graph */
//     prerun_graph(graph);
//   //dump_graph(graph);
//    run_graph(graph,1); //warm up
//    char *out_data_file2 = "./tests/data/inception-v4/incepv4_data/inception_a1_pool_ave_470400";
//    int size2 = 470400;
//    tensor_t mytensor = get_graph_tensor(graph, "inception_a1_pool_ave");
//    float *data2 = (float *)get_tensor_buffer(mytensor);
//    float *out2 = (float *)malloc(sizeof(float) * size2);
//    get_data(out2, size2, out_data_file2);
//    maxerr(data2, out2, size2);

//    char *out_data_file1 = "./tests/data/inception-v4/incepv4_data/prob_1000";
//    int size1 = 1000;
//    mytensor = get_graph_tensor(graph, "prob");
//    float *data1 = (float *)get_tensor_buffer(mytensor);
//    float *out1 = (float *)malloc(sizeof(float) * size1);
//    get_data(out1, size1, out_data_file1);
//    maxerr(data1, out1, size1);







//    printf("REPEAT COUNT= %d\n",repeat_count);

//    unsigned long start_time=get_cur_time();

//    for(int i=0;i<repeat_count;i++)
//        run_graph(graph,1);

//    unsigned long end_time=get_cur_time();

//    printf("Repeat [%d] times %.2f per RUN, used [%lu] us\n",repeat_count,1.0f*(end_time-start_time)/repeat_count,
//                     end_time-start_time);

//     free(input_data);

//     tensor_t mytensor1 = get_graph_tensor(graph, "prob");
//     float *data1 = (float *)get_tensor_buffer(mytensor1);
//     float *end = data1 + 1000;

//     std::vector<float> result(data1, end);

//     std::vector<int> top_N = Argmax(result, 5);

//     std::vector<std::string> labels;

//     LoadLabelFile(labels, label_file);

//     for (unsigned int i = 0; i < top_N.size(); i++)
//     {
//         int idx = top_N[i];

//         std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
//         std::cout << labels[idx] << "\"\n";
//     }
    postrun_graph(graph);

    ProfRecord * prof=ProfRecordManager::Get("simple");

   if(prof)
      prof->Dump(1);

 

   destroy_runtime_graph(graph);
   remove_model(model_name);


   std::cout<<"ALL TEST DONE\n";


    return 0;
}


   

