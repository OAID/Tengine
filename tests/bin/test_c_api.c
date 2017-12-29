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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <malloc.h>

#include "tengine_c_api.h"

const char * caffe_sqz_model_file="./tests/data/squeezenet_v1.1.caffemodel";
const char * caffe_sqz_text_file="./tests/data/sqz.prototxt";
const char * onnx_sqz_model_file="./tests/data/sqz.onnx.model";

int test_sqznet(const char * model_file, int caffe)
{
   const char * model_name="squeezenet";
   const char * graph_name="sqz0";
   user_context_t run_context;
   workspace_t ws;
   graph_t graph;

   const char * onnx_input_node="data";
   const char * onnx_input_tensor_name="data";
   const char * onnx_output_node="softmaxout";
   const char * onnx_output_tensor_name="softmaxout";

   const char * caffe_input_node="input";
   const char * caffe_input_tensor_name="data";
   const char * caffe_output_node="prob";
   const char * caffe_output_tensor_name="prob";

   const char * input_node;
   const char * input_tensor_name;
   const char * output_node;
   const char * output_tensor_name;

   if(caffe)
    {
       input_node=caffe_input_node;
       input_tensor_name=caffe_input_tensor_name;
       output_node=caffe_output_node;
       output_tensor_name=caffe_output_tensor_name;
    }
    else
    {
       input_node=onnx_input_node;
       input_tensor_name=onnx_input_tensor_name;
       output_node=onnx_output_node;
       output_tensor_name=onnx_output_tensor_name;
    }


   int dims[4]={10,3,227,227};
   int input_buf_size;
   void * input_buf;
   tensor_t input_tensor;
   tensor_t output_tensor;


   init_tengine_library();

   if(request_tengine_version("1.0.xz")<0)
      return 1;

   if(caffe)
   {
       if(load_model(model_name,"caffe",caffe_sqz_text_file,model_file)<0)
       {
           printf("load %s with method %s failed\n",model_file,"caffe");
           return -1;
       }
   }
   else
   {
       if(load_model(model_name,"onnx",model_file)<0)
       {
           printf("load %s with method %s failed\n",model_file,"onnx");
           return -1;
       }

   }

   printf("Dumping model: %s\n",model_name);

   dump_model(model_name);

   printf("prepare to run ...\n");

   run_context=create_user_context("user0");
   ws=create_workspace("ws0",run_context);

   graph=create_runtime_graph(graph_name,model_name,ws);

   if(!check_graph_valid(graph))
   {
       printf("create %s failed\n",graph_name);
       return -1;
   }

   //designate the input and output node

   if(set_graph_input_node(graph,&input_node,1)<0)
   {
       printf("set input node failed\n");
       return -1;
   }

   if(set_graph_output_node(graph,&output_node,1)<0)
   {
       printf("set output node failed\n");
       return -1;
   }

   //infer shape
   input_tensor=get_graph_tensor(graph,input_tensor_name);

   if(!check_tensor_valid(input_tensor))
   {
       printf("cannot find tensor: %s\n",input_tensor_name);
       return -1;
   }

   output_tensor=get_graph_tensor(graph,output_tensor_name);

   if(!check_tensor_valid(output_tensor))
   {
       printf("cannot find tensor: %s\n",output_tensor_name);
       return -1;
   }

   set_tensor_shape(input_tensor,dims,4);

   //prepare to run

   input_buf_size=get_tensor_buffer_size(input_tensor);

   printf("input buffer size: %d\n",input_buf_size);

   input_buf=malloc(input_buf_size);

   set_tensor_buffer(input_tensor,input_buf,input_buf_size);

   if(prerun_graph(graph)<0)
   {
      printf("prerun failed\n");
      return 1;
   }

   int dim_size=get_tensor_shape(output_tensor,dims,4);

   if(dim_size<0)
   {
      printf("get output tensor shape failed\n");
   }

   printf("output tensor shape: [");

   for(int i=0;i<dim_size;i++)
   {
      printf("%d ",dims[i]);
   }

   printf("]\n");

   run_graph(graph,1);

   //Get Output Result
   int output_size=get_tensor_buffer_size(output_tensor);

   float * output_buf=(float *)malloc(output_size);

   get_tensor_data(output_tensor,output_buf,output_size);

   postrun_graph(graph);

   /* resource release */
   put_graph_tensor(input_tensor);
   put_graph_tensor(output_tensor);

   destroy_runtime_graph(graph);
   remove_model(model_name);

   printf("RELEASE BUFFER\n");

   free(input_buf);
   free(output_buf);

   return 0;
}

int main(int argc, char * argv[])
{

   int caffe=1;

   int ret;

   while((ret=getopt(argc,argv,"o"))!=-1)
   {
        switch(ret)
        {
        case 'o':
             caffe=0;
             break;
        default:
             break;
        }
   }

   
   if(caffe)
   {
       test_sqznet(caffe_sqz_model_file,caffe);
   }
   else
   {
       test_sqznet(onnx_sqz_model_file,caffe);
   }

   printf("TEST DONE\n");

   return 0;
}
