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
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include "tengine_c_api.h"

int main(int argc, char *argv[]) {
  std::string file_format;
  std::string proto_file;
  std::string model_file;
  std::string output_tmfile;
  bool proto_file_needed = false;
  bool model_file_needed = false;
  int input_file_number = 0;

  int res;
  while ((res = getopt(argc, argv, "f:p:m:o:h")) != -1) {
    switch (res) {
      case 'f':
        file_format = optarg;
        break;
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
        std::cout << "[Usage]: " << argv[0]
                  << " [-h] [-f file_format] [-p proto_file] [-m model_file] "
                     "[-o output_tmfile]\n";
        return 0;
      default:
        break;
    }
  }

  // Check the input parameters

  if (file_format.empty()) {
    std::cout
        << "Please specify the -f option to indicate the input file format.\n";
    return -1;
  } else {
    if (file_format == "caffe" || file_format == "mxnet") {
      proto_file_needed = true;
      model_file_needed = true;
      input_file_number = 2;
    } else if (file_format == "caffe_single" || file_format == "onnx" ||
               file_format == "tensorflow") {
      model_file_needed = true;
      input_file_number = 1;
    } else {
      std::cout << "Allowed input file format: caffe, caffe_single, onnx, "
                   "mxnet, tensorflow\n";
      return -1;
    }
  }

  if (proto_file_needed) {
    if (proto_file.empty()) {
      std::cout
          << "Please specify the -p option to indicate the input proto file.\n";
      return -1;
    }
    if (access(proto_file.c_str(), 0) == -1) {
      std::cout << "Proto file does not exist: " << proto_file << "\n";
      return -1;
    }
  }

  if (model_file_needed) {
    if (model_file.empty()) {
      std::cout
          << "Please specify the -m option to indicate the input model file.\n";
      return -1;
    }
    if (access(model_file.c_str(), 0) == -1) {
      std::cout << "Model file does not exist: " << model_file << "\n";
      return -1;
    }
  }

  if (output_tmfile.empty()) {
    std::cout << "Please specify the -o option to indicate the output tengine "
                 "model file.\n";
    return -1;
  }
  if (output_tmfile.rfind("/") != std::string::npos) {
    std::string output_dir = output_tmfile.substr(0, output_tmfile.rfind("/"));
    if (access(output_dir.c_str(), 0) == -1) {
      std::cout << "The dir of output file does not exist: " << output_dir
                << "\n";
      return -1;
    }
  }

  // init tengine
  init_tengine_library();
  if (request_tengine_version("0.1") < 0) return 1;

  // load input model files
  std::string model_name = "temp_model";
  int ret = -1;
  if (input_file_number == 2)
    ret = load_model(model_name.c_str(), file_format.c_str(),
                     proto_file.c_str(), model_file.c_str());
  else if (input_file_number == 1)
    ret =
        load_model(model_name.c_str(), file_format.c_str(), model_file.c_str());
  if (ret < 0) {
    std::cout << "Load model failed.\n";
    return -1;
  }

  // create runtime graph
  graph_t graph = create_runtime_graph("graph0", model_name.c_str(), NULL);
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
