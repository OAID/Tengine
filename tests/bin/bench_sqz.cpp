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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
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

const char * text_file="./tests/data/sqz.prototxt";
const char * model_file="./tests/data/squeezenet_v1.1.caffemodel";
const char * image_file="./tests/data/cat.jpg";
const char * mean_file="./tests/data/imagenet_mean.binaryproto";
const char * label_file="./tests/data/synset_words.txt";

using namespace TEngine;

int repeat_count=100;

void LoadLabelFile(std::vector<std::string>& result, const char * fname)
{
   std::ifstream labels(fname);

  std::string line;
  while (std::getline(labels, line))
     result.push_back(line);
}

   
int main(int argc, char * argv[])
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

   const char * model_name="squeezenet";
   int img_h=227;
   int img_w=227;

   /* prepare input data */
   float  * input_data=caffe_process_image(image_file,mean_file,img_h,img_w);
   //float  * input_data=caffe_process_image(image_file,NULL,img_h,img_w);

   init_tengine_library();

   if(request_tengine_version("0.1")<0)
       return 1;


   if(load_model(model_name,"caffe",text_file,model_file)<0)
       return 1; 

   std::cout<<"Load model successfully\n";

   //dump_model(model_name);

   graph_t graph=create_runtime_graph("graph0",model_name,NULL);

   if(!check_graph_valid(graph))
   {
       std::cout<<"create graph0 failed\n";
       return 1;
   }

   
   /* set input and output node*/

   const char * input_node_name="input";
   const char * output_node_name="prob";

   if(set_graph_input_node(graph,&input_node_name,1)<0)
   {
      std::printf("set input node: %s failed\n",input_node_name);
       return 1;
   }

   if(set_graph_output_node(graph,&output_node_name,1)<0)
   {
       std::printf("set output node: %s failed\n",output_node_name);
       return 1;
   }

   const char * input_tensor_name="data";

   tensor_t input_tensor=get_graph_tensor(graph,input_tensor_name);

   if(!check_tensor_valid(input_tensor))
   {
       std::printf("cannot find tensor: %s\n",input_tensor_name);
       return -1;
   }

   int dims[]={1,3,img_h,img_w};

   set_tensor_shape(input_tensor,dims,4);

   /* setup input buffer */

   if(set_tensor_buffer(input_tensor,input_data,3*img_h*img_w*4)<0)
   {
       std::printf("set buffer for tensor: %s failed\n",input_tensor_name);
       return -1;
   }

   const char * output_tensor_name="prob";

   tensor_t output_tensor=get_graph_tensor(graph,output_tensor_name);

  /* setup output buffer */

   void * output_data=malloc(sizeof(float)*1000);

   memset(output_data,0x0,4000);

   if(set_tensor_buffer(output_tensor,output_data,4*1000))
   {
       std::printf("set buffer for tensor: %s failed\n",output_tensor_name);
       return -1;
   }
   

   /* run the graph */
   prerun_graph(graph);

   int dim_size=get_tensor_shape(output_tensor,dims,4);

   if(dim_size<0)
   {
      printf("get output tensor shape failed\n");
      return -1;
   }

   
   printf("output tensor shape: [");
    
   for(int i=0;i<dim_size;i++)
      printf("%d ",dims[i]);

   printf("]\n");

   for(int i=0;i<10;i++)
      run_graph(graph,1); //warm up

   printf("REPEAT COUNT= %d\n",repeat_count);

   unsigned long start_time=get_cur_time();

   for(int i=0;i<repeat_count;i++)
        run_graph(graph,1);

   unsigned long end_time=get_cur_time();

   printf("Repeat [%d] times %.2f per RUN, used [%lu] us\n",repeat_count,1.0f*(end_time-start_time)/repeat_count,
                    end_time-start_time);

   int count=get_tensor_buffer_size(output_tensor)/4;

   float *  data=(float *)(output_data);
   float * end=data+count;
 
   std::vector<float> result(data, end);

   std::vector<int> top_N=Argmax(result,5);

   std::vector<std::string> labels;

   LoadLabelFile(labels,label_file);

   for(unsigned int i=0;i<top_N.size();i++)
   {
       int idx=top_N[i];

       std::cout<<std::fixed << std::setprecision(4)<<result[idx]<<" - \"";
       std::cout<< labels[idx]<<"\"\n";
   }
 
   postrun_graph(graph);  

   ProfRecord * prof=ProfRecordManager::Get("simple");

   if(prof)
      prof->Dump(1);

   put_graph_tensor(input_tensor);
   put_graph_tensor(output_tensor);

   destroy_runtime_graph(graph);
   remove_model(model_name);


   std::cout<<"ALL TEST DONE\n";

   return 0;
}
