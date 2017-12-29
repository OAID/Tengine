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
#include <iomanip>

#include "share_lib_parser.hpp"
#include "caffe_serializer.hpp"
#include "graph.hpp"
#include "graph_executor.hpp"
#include "executor.hpp"

#include <opencv2/opencv.hpp>
#include "debug_utils.hpp"
#include "simple_executor.hpp"
#include "image_process.hpp"
#include "image_process.cpp"

const char * text_file="./tests/data/sqz.prototxt";
const char * model_file="./tests/data/squeezenet_v1.1.caffemodel";
const char * image_file="./tests/data/cat.jpg";
const char * mean_file="./tests/data/imagenet_mean.binaryproto";
const char * label_file="./tests/data/synset_words.txt";

using namespace caffe;
using namespace TEngine;


static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

void LoadLabelFile(std::vector<std::string>& result, const char * fname)
{
   std::ifstream labels(fname);

  std::string line;
  while (std::getline(labels, line))
     result.push_back(line);
}

   
int main(int argc, char * argv[])
{

   /* prepare input data */
   float  * input_data=caffe_process_image(image_file,mean_file,227,227);

   ShareLibParser p0("./build/operator/liboperator.so");
   p0.ExcecuteFunc<int()>("tengine_plugin_init");

   ShareLibParser p1("./build/serializer/libserializer.so");
   p1.ExcecuteFunc<int()>("tengine_plugin_init");

   ShareLibParser p2("./build/executor/libexecutor.so");
   p2.ExcecuteFunc<int()>("tengine_plugin_init");

   SerializerPtr p_caffe;

   if(!SerializerManager::SafeGet("caffe",p_caffe))
   {
      std::cout<<"No caffe registered in object manager\n";
      return 1;
   }

   StaticGraph * graph=CreateStaticGraph("test");

   std::vector<std::string> flist;

   flist.push_back(text_file);
   flist.push_back(model_file);

   if(!p_caffe->LoadModel(flist,graph))
   {
       std::cout<<"Load model failed\n";
       return 1;
   }

   std::cout<<"Load model successfully\n";

   DumpStaticGraph(graph);

  if( CheckGraphIntegraity(graph))
      std::cout<<"check passed\n";


   /* register the loaded static graph */
   StaticGraphPtr graph_ptr(graph);
   StaticGraphManager::SafeAdd(graph->name,graph_ptr);

   GraphExecutor executor;

   if(!executor.CreateGraph("runtime",graph->name))
   {
       std::cout<<"create graph from static graph: "<<graph->name<<"failed\n";

       return 1;
   }



   const std::string& node_name=executor.GetGraphInputNodeName(0);

   const std::string& tensor_name=executor.GetNodeOutputTensor(node_name,0);

   Tensor * tensor=executor.FindTensor(tensor_name);

   std::vector<int> dim={1,3,227,227};

   TShape& shape=tensor->GetShape();

   shape.SetDim(dim);


   if(!executor.InferShape())
   {
       std::cout<<"InferShape failed\n";
       return 1;
   }


   Graph * runtime_graph=executor.GetGraph();
   runtime_graph->DumpGraph();

   ExecEnginePtr p_engine(new SimpleExec());

   exec_handle_t handle=p_engine->AddGraphExecutor(&executor);


   if(!p_engine->SetTensorBuffer(tensor,input_data,3*227*227*4))
   {
      std::cout<<"set tensor buffer for "<<tensor->GetName()<<" failed\n";
      return 1;
   }

   if(!p_engine->Prerun(handle))
   {
       std::cout<<"Prerun failed\n";
   }

   exec_event_t event;

   if(!p_engine->Run(handle,event))
    {
        std::cout<<"Run failed\n";
    }

    p_engine->Wait(handle,event);



   /* get result */
   Node * node=runtime_graph->output_nodes[0];
   tensor=node->GetOutputTensor(0);

   float *  data=(float *)(p_engine->GetTensorBuffer(tensor));
   int count=tensor->GetTotalSize()/4;
   float * end=data+count;

   std::cout<<"Get result from: "<<(void *)data<<"\n";
 
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
   

   std::cout<<"ALL TEST DONE\n";
   return 0;
}
