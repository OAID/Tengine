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
#include <thread>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"
#include "graph_executor.hpp"
#include "common_util.hpp"
#include "image_process.hpp"

const char * text_file="./tests/data/sqz.prototxt";
const char * model_file="./tests/data/squeezenet_v1.1.caffemodel";
const char * image_file="./cat.jpg";
const char * mean_file="./tests/data/imagenet_mean.binaryproto";
const char * label_file="./tests/data/synset_words.txt";
const char * model_name="squeezenet";

using namespace TEngine;

ExecEngine* p_engine;

void LoadLabelFile(std::vector<std::string>& result, const char * fname)
{
   std::ifstream labels(fname);

  std::string line;
  while (std::getline(labels, line))
     result.push_back(line);
}

   
exec_handle_t create_nn(int img_h, int img_w, float * input_data, float * output_data)
{

   graph_t graph=create_runtime_graph("graph0",model_name,NULL);

   if(!check_graph_valid(graph))
   {
       std::cout<<"create graph0 failed\n";
       return nullptr;
   }

   /* set input and output node*/

   const char * input_node_name="input";
   const char * output_node_name="prob";

   if(set_graph_input_node(graph,&input_node_name,1)<0)
   {
      std::printf("set input node: %s failed\n",input_node_name);
       return nullptr;
   }

   if(set_graph_output_node(graph,&output_node_name,1)<0)
   {
       std::printf("set output node: %s failed\n",output_node_name);
       return nullptr;
   }

   const char * input_tensor_name="data";
   tensor_t input_tensor=get_graph_tensor(graph,input_tensor_name);
   if(!check_tensor_valid(input_tensor))
   {
       std::printf("cannot find tensor: %s\n",input_tensor_name);
       return nullptr;
   }

   int dims[]={1,3,img_h,img_w};
   set_tensor_shape(input_tensor,dims,4);

   GraphExecutor  * graph_executor=(GraphExecutor*)graph;
   exec_handle_t handle=p_engine->AddGraphExecutor(graph_executor);

   if(handle==nullptr)
   {
       std::cout<<"bind engine failed\n";
       return nullptr;
   }

   Tensor * tensor=graph_executor->FindTensor(input_tensor_name);

   if(!p_engine->SetTensorBuffer(tensor,input_data,3*img_h*img_w*4,handle))
   {
      std::cout<<"set tensor buffer for "<<tensor->GetName()<<" failed\n";
      return nullptr;
   }


   memset(output_data,0x0,4000);

   tensor=graph_executor->FindTensor(output_node_name);

   if(!p_engine->SetTensorBuffer(tensor,output_data,4000,handle))
   {
      std::cout<<"set tensor buffer for "<<tensor->GetName()<<" failed\n";
      return nullptr;
   }
    

   if(!p_engine->Prerun(handle))
   {
       std::cout<<"Prerun failed\n";
       return nullptr;
   }

   return handle;
}

void print_result(float * output_data, int output_num)
{
   float * end=output_data+output_num;
 
   std::vector<float> result(output_data, end);
   std::vector<int> top_N=Argmax(result,5);

   std::vector<std::string> labels;

   LoadLabelFile(labels,label_file);

   for(unsigned int i=0;i<top_N.size();i++)
   {
       int idx=top_N[i];

       std::cout<<std::fixed << std::setprecision(4)<<result[idx]<<" - \"";
       std::cout<< labels[idx]<<"\"\n";
   }

   sleep(5);
}

void run_nn_instance(exec_handle_t handle,  const char * image_file, int img_h, int img_w,float * input_data, float * output_data)
{
   int count=5;

   /* prepare input data */

   while(count-->0)
   {
      float  * input_image=caffe_process_image(image_file,mean_file,img_h,img_w);
      int input_size=img_h*img_w*3*sizeof(float);

      memcpy(input_data,input_image,input_size);

      free(input_image);

      exec_event_t event;

      if(!p_engine->Run(handle,event))
      {
          std::cout<<"Run failed\n";
      }

      p_engine->Wait(handle,event);

      //get result
      print_result(output_data,1000);

   }
}

int main(int argc, char * argv[])
{
   //Init library
   init_tengine_library();

   if(load_model(model_name,"caffe",text_file,model_file)<0)
       return 1; 

   std::cout<<"Load model successfully\n";

   int img_h=227;
   int img_w=227;

   float * input_data=(float *)malloc(3*img_h*img_w*sizeof(float));
   float * output_data=(float *)malloc(1000*sizeof(float));

   ExecEnginePtr  engine;

   if(!ExecEngineManager::SafeGet("event",engine))
   {
       std::cout<<"cannot get event engine\n";
       return 1;
   }

   p_engine=engine.get();

   exec_handle_t h1=create_nn(img_h,img_w,input_data,output_data);

   if(h1==nullptr)
   {
       std::cout<<"create h1 failed\n";
       return 1;
   }


   float * input_data2=(float *)malloc(3*img_h*img_w*sizeof(float));
   float * output_data2=(float *)malloc(1000*sizeof(float));
   exec_handle_t h2=create_nn(img_h,img_w,input_data2,output_data2);

   if(h2==nullptr)
   {
       std::cout<<"create h2 failed\n";
       return 1;
   }


   std::thread * tr1=new std::thread(run_nn_instance,h1, image_file,img_h,img_w,input_data,output_data);

   std::thread * tr2=new std::thread(run_nn_instance,h2, image_file,img_h,img_w,input_data2,output_data2);

   tr1->join();
   tr2->join();

   std::cout<<"ALL TEST DONE\n";
   sleep(1000);
   return 0;
  
}

