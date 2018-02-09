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
#include "image_process.hpp"
#include "common_util.hpp"
#include "tengine_config.hpp"


const char * text_file="./tests/data/sqz.prototxt";
const char * model_file="./tests/data/squeezenet_v1.1.caffemodel";
const char * image_file="./tests/data/cat.jpg";
const char * mean_file="./tests/data/imagenet_mean.binaryproto";
const char * label_file="./tests/data/synset_words.txt";
const char * model_name="squeezenet";
int img_h=227;
int img_w=227;

using namespace TEngine;

#include "graph_executor.hpp"
#include "graph.hpp"
#include "node.hpp"

void assign_dev_executor(graph_t graph)
{
    GraphExecutor * graph_executor=reinterpret_cast<GraphExecutor *>(graph);
    Graph * runtime_graph=graph_executor->GetGraph();

    //runtime_graph->DumpGraph();

    for(unsigned int i=0;i<runtime_graph->seq_nodes.size();i++)
    {
        Node * node=runtime_graph->seq_nodes[i];
        Operator * op=node->GetOp();

        if(op->GetName()=="Input" || op->GetName()=="Const")
            continue;

        if(op->GetName()=="ReLu")
            node->SetAttr("dev_id",std::string("cpu.rk3399.a53.all"));
        else if(op->GetName()=="Convolution")
            node->SetAttr("dev_id",std::string("cpu.rk3399.a72.all"));
        else
            node->SetAttr("dev_id",std::string("cpu.rk3399.cpu.all"));
    }

}


void LoadLabelFile(std::vector<std::string>& result, const char * fname)
{
   std::ifstream labels(fname);

  std::string line;
  while (std::getline(labels, line))
     result.push_back(line);
}

   
int common_init(void)
{

   init_tengine_library();

   if(request_tengine_version("0.1")<0)
       return 1;


   if(load_model(model_name,"caffe",text_file,model_file)<0)
       return 1; 

   std::cout<<"Load model successfully\n";

   dump_model(model_name);

   return 0;
}


int thread_func(void)
{
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

  float * input_data=(float *)malloc(3*img_h*img_w*4);

   if(set_tensor_buffer(input_tensor,input_data,3*img_h*img_w*4)<0)
   {
       std::printf("set buffer for tensor: %s failed\n",input_tensor_name);
       return -1;
   }

   const char * output_tensor_name="prob";
   tensor_t output_tensor=get_graph_tensor(graph,output_tensor_name);

   float * output_data=(float *)malloc(1000*4);
   memset(output_data,0x0,4000);

   if(set_tensor_buffer(output_tensor,output_data,1000*4)<0)
   {
       std::printf("set buffer for tensor: %s failed\n",output_tensor_name);
       return -1;
   }


   /* run the graph */

  assign_dev_executor(graph);

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


 int count=10;

  while (count-->0)
  {
      float  * input_image=caffe_process_image(image_file,mean_file,img_h,img_w);
      int input_size=img_h*img_w*3*sizeof(float);

      memcpy(input_data,input_image,input_size);

      run_graph(graph,1);

      int count=get_tensor_buffer_size(output_tensor)/4;

      float *  data=(float *)(output_data);
      float * end=data+count;
 
      std::vector<float> result(data, end);

      std::vector<int> top_N=Argmax(result,5);

      std::vector<std::string> labels;

      LoadLabelFile(labels,label_file);

      std::cout<<"RESULT:\n";

      for(unsigned int i=0;i<top_N.size();i++)
      {
          int idx=top_N[i];

          std::cout<<std::fixed << std::setprecision(4)<<result[idx]<<" - \"";
          std::cout<< labels[idx]<<"\"\n";
      }

      sleep(1);
   }
 
   postrun_graph(graph);  

   put_graph_tensor(input_tensor);
   put_graph_tensor(output_tensor);

   destroy_runtime_graph(graph);
   remove_model(model_name);

   std::cout<<"ALL TESTS DONE\n";
   return 0;
}


int main(int argc, char * argv[])
{
    int res;

   while((res=getopt(argc,argv,"e"))!=-1)
   {
      switch(res)
      {
         case 'e':
            TEngineConfig::Set("exec.engine","event");
            break;
         default:
            break;
      }
   }

   common_init();

   std::thread * tr1=new std::thread(thread_func);
   std::thread * tr2=new std::thread(thread_func);

   tr1->join();
   tr2->join();

   return 0;
}
