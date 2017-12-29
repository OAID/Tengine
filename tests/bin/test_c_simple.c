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
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#include "tengine_c_api.h"


const char * caffe_sqz_text_file="./tests/data/sqz.prototxt";
const char * caffe_sqz_model_file="./tests/data/squeezenet_v1.1.caffemodel";
const char * onnx_sqz_model_file="./tests/data/sqz.onnx.model";


int main(int argc, char * argv[])
{

   graph_t graph;
   int input_shape[]={1,3,224,224};
   int input_size=1;
   char * input_buffer;
   char * outdata;
   int out_size;
   int ret;
   int onnx_model=0;


   if(onnx_model)
       graph=create_graph("squeeze","onnx", onnx_sqz_model_file);
   else
   {
       graph=create_graph("squeeze","caffe", caffe_sqz_text_file,caffe_sqz_model_file);
       input_shape[2]=227;
       input_shape[3]=227;
   }

   if(check_graph_valid(graph)<0)
   {
       printf("create graph failed\n");
       return -1;
   }

   if(!onnx_model)
   {
      //set shape
      set_input_shape(graph,input_shape,4);
   }

   for(int i=0;i<sizeof(input_shape)/sizeof(int);i++)
   {
       input_size*=input_shape[i];
   }

   input_size*=sizeof(float);

   input_buffer=malloc(input_size);

   ret=run_inference(graph,input_buffer,input_size);

   if(ret<0)
   {
       printf("run graph failed\n");
       return 1;
   }

    out_size=1000*4*input_shape[0];

    outdata=malloc(out_size);

   ret=get_graph_output(graph,outdata,out_size);

   if(ret<0)
   {
       printf("get output failed\n");
       return 1;
   }

   destroy_graph(graph);

   free(input_buffer);
   free(outdata);

   return 0;
}
