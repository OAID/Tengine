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
 * Author: jingyou@openailab.com
 */
#include <unistd.h>
#include <iostream>
#include "tengine_c_api.h"

int main(int argc, char *argv[]) {
  std::string proto_file;
  std::string model_file;
  std::string output_tmfile;

  int res;
  while ((res = getopt(argc, argv, "p:m:o:h")) != -1) {
    switch (res) {
      case 'p':
        proto_file = optarg;
        break;
      case 'm':
        model_file = optarg;
        break;
      case 'o':
        output_tmfile = optarg;
        break;
      case 'h':
        std::cout
            << "[Usage]: " << argv[0]
            << " [-h] [-p proto_file] [-m model_file] [-o output_tmfile]\n";
        return 0;
      default:
        break;
    }
  }

  if (proto_file.empty()) {
    std::cout
        << "Please specify the -p option to indicate the input proto file.\n";
    return -1;
  }
  if (model_file.empty()) {
    std::cout
        << "Please specify the -m option to indicate the input model file.\n";
    return -1;
  }
  if (output_tmfile.empty()) {
    std::cout << "Please specify the -o option to indicate the output tengine "
                 "model file.\n";
    return -1;
  }

  // init tengine
  init_tengine_library();
  if (request_tengine_version("0.1") < 0) return 1;

  // load caffe model
  std::string model_name = "temp_model";
  if (load_model(model_name.c_str(), "caffe", proto_file.c_str(),
                 model_file.c_str()) < 0) {
    std::cout << "Load caffe model failed.\n";
    return -1;
  }

  // create runtime graph
  graph_t graph = create_runtime_graph("graph", model_name.c_str(), NULL);
  if (!check_graph_valid(graph)) {
    std::cout << "Create graph0 failed.\n";
    return -1;
  }

  // Save the tengine model file
  if (save_model(graph, "tengine", output_tmfile.c_str()) == -1) {
    std::cout << "Create tengine model file failed.\n";
    return -1;
  }
  std::cout << "Create tengine model file done: " << output_tmfile << "\n";

  destroy_runtime_graph(graph);
  remove_model(model_name.c_str());
  release_tengine_library();
  return 0;
}
