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
#include <sys/time.h>
#include <iomanip>
#include <iostream>
#include <string>
#include "tengine_c_api.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "[Usage]: " << argv[0] << " <model> <size> \n";
    return 0;
  }

  // init tengine
  init_tengine_library();
  if (request_tengine_version("0.1") < 0) return 1;

  // load model
  const char *model_name = "test";
  std::string mdl_name = argv[1];
  if (load_model(model_name, "tengine", mdl_name.c_str(), mdl_name.c_str()) < 0)
    return 1;
  std::cout << "load model done!\n";

  // create graph
  graph_t graph = create_runtime_graph("graph", model_name, NULL);
  if (!check_graph_valid(graph)) {
    std::cout << "create graph0 failed\n";
    return 1;
  }

  // input
  int img_h = atoi(argv[2]);
  int img_w = img_h;
  if (argc == 5) img_w = atoi(argv[3]);
  int img_size = img_h * img_w * 3;
  float *input_data = (float *)malloc(sizeof(float) * img_size);
  for (int i = 0; i < img_size; i++) input_data[i] = (i % 123) * 0.017;

  tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
  int dims[] = {1, 3, img_h, img_w};
  set_tensor_shape(input_tensor, dims, 4);
  prerun_graph(graph);

  int repeat_count = 1;
  const char *repeat = std::getenv("REPEAT_COUNT");

  if (repeat) repeat_count = std::strtoul(repeat, NULL, 10);

  struct timeval t0, t1;
  float avg_time = 0.f;
  set_tensor_buffer(input_tensor, input_data, img_size * 4);

  run_graph(graph, 1);

  gettimeofday(&t0, NULL);

  for (int i = 0; i < repeat_count; i++) run_graph(graph, 1);

  gettimeofday(&t1, NULL);
  float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) -
                         (t0.tv_sec * 1000000 + t0.tv_usec)) /
                 1000;
  avg_time += mytime;

  std::cout << "--------------------------------------\n";
  std::cout << "repeat " << repeat_count << " times, avg time per run is "
            << avg_time / repeat_count << " ms\n";
  free(input_data);
  postrun_graph(graph);
  destroy_runtime_graph(graph);

  remove_model(model_name);

  std::cout << "ALL TEST DONE\n";

  return 0;
}
