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
#include <time.h>


#include "tengine_c_api.h"


const char * text_file="./models/sqz.prototxt";
const char * model_file="./models/squeezenet_v1.1.caffemodel";

   
int main(int argc, char * argv[])
{
  
   std::string model_name = "squeeze_net";

   init_tengine_library();
 
   if(request_tengine_version("0.1")<0)
       return 1;

   if(load_model(model_name.c_str(),"caffe",text_file,model_file)<0)
   {
       std::cout<<"load model failed\n";
       return 1; 
   }

   std::cout<<"Load model successfully\n";

   graph_t graph=create_runtime_graph("graph0",model_name.c_str(),NULL);

   if(!check_graph_valid(graph))
   {
       std::cout<<"Create graph0 failed\n";
       return 1;
   }

   /* 
      src_tm is the serializer name 
      squeezenet is the model name,
      which is used when load source model
   */

   if(save_model(graph,"src_tm","squeezenet")<0)
   {
       std::cout<<"Save graph failed\n";
       return 1;
   }

   /* free resource */

   destroy_runtime_graph(graph);
   remove_model(model_name.c_str());
   release_tengine_library();

   return 0;
}
