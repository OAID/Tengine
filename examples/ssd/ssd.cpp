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
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "common.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "tengine_c_api.h"

#define DEF_PROTO "models/VGG_VOC0712_SSD_300.prototxt"
#define DEF_MODEL "models/VGG_VOC0712_SSD_300.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

struct Box {
  float x0;
  float y0;
  float x1;
  float y1;
  int class_idx;
  float score;
};

void get_input_data_ssd(std::string &image_file, float *input_data, int img_h,
                        int img_w) {
  cv::Mat img = cv::imread(image_file);

  if (img.empty()) {
    std::cerr << "failed to read image file " << image_file << "\n";
    return;
  }
  cv::resize(img, img, cv::Size(img_h, img_w));
  img.convertTo(img, CV_32FC3);
  float *img_data = (float *)img.data;
  int hw = img_h * img_w;
  // float mean[3]={104.f,117.f,123.f};
  float mean[3] = {127.5, 127.5, 127.5};
  for (int h = 0; h < img_h; h++) {
    for (int w = 0; w < img_w; w++) {
      for (int c = 0; c < 3; c++) {
        input_data[c * hw + h * img_w + w] = (*img_data - mean[c]);
        img_data++;
      }
    }
  }
}

void post_process_ssd(std::string &image_file, float threshold, float *outdata,
                      int num, std::string &save_name) {
  const char *class_names[] = {
      "background", "aeroplane",   "bicycle", "bird",  "boat",
      "bottle",     "bus",         "car",     "cat",   "chair",
      "cow",        "diningtable", "dog",     "horse", "motorbike",
      "person",     "pottedplant", "sheep",   "sofa",  "train",
      "tvmonitor"};

  cv::Mat img = cv::imread(image_file);
  int raw_h = img.size().height;
  int raw_w = img.size().width;
  std::vector<Box> boxes;
  int line_width = raw_w * 0.005;
  printf("detect ruesult num: %d \n", num);
  for (int i = 0; i < num; i++) {
    if (outdata[1] >= threshold) {
      Box box;
      box.class_idx = outdata[0];
      box.score = outdata[1];
      box.x0 = outdata[2] * raw_w;
      box.y0 = outdata[3] * raw_h;
      box.x1 = outdata[4] * raw_w;
      box.y1 = outdata[5] * raw_h;
      boxes.push_back(box);
      printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
      printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
    }
    outdata += 6;
  }
  for (int i = 0; i < (int)boxes.size(); i++) {
    Box box = boxes[i];
    cv::rectangle(
        img, cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)),
        cv::Scalar(255, 255, 0), line_width);
    std::ostringstream score_str;
    score_str << box.score;
    std::string label =
        std::string(class_names[box.class_idx]) + ": " + score_str.str();
    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(
        img,
        cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        cv::Scalar(255, 255, 0), CV_FILLED);
    cv::putText(img, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 0, 0));
  }
  cv::imwrite(save_name, img);
  std::cout << "======================================\n";
  std::cout << "[DETECTED IMAGE SAVED]:\t" << save_name << "\n";
  std::cout << "======================================\n";
}

int main(int argc, char *argv[]) {
  const std::string root_path = get_root_path();
  std::string proto_file;
  std::string model_file;
  std::string image_file;
  std::string save_name = "save.jpg";

  int res;
  while ((res = getopt(argc, argv, "p:m:i")) != -1) {
    switch (res) {
      case 'p':
        proto_file = optarg;
        break;
      case 'm':
        model_file = optarg;
        break;
      case 'i':
        image_file = optarg;
        break;
      default:
        break;
    }
  }

  // init tengine
  init_tengine_library();
  if (request_tengine_version("0.1") < 0) return 1;

  const char *model_name = "vgg_300";
  if (proto_file.empty()) {
    proto_file = root_path + DEF_PROTO;
    std::cout << "proto file not specified,using " << proto_file
              << " by default\n";
  }
  if (model_file.empty()) {
    model_file = root_path + DEF_MODEL;
    std::cout << "model file not specified,using " << model_file
              << " by default\n";
  }
  if (image_file.empty()) {
    image_file = root_path + DEF_IMAGE;
    std::cout << "image file not specified,using " << image_file
              << " by default\n";
  }
  if (load_model(model_name, "caffe", proto_file.c_str(), model_file.c_str()) <
      0)
    return 1;
  std::cout << "load model done!\n";

  // create graph
  graph_t graph = create_runtime_graph("graph", model_name, NULL);
  if (!check_graph_valid(graph)) {
    std::cout << "create graph0 failed\n";
    return 1;
  }

  // input
  int img_h = 300;
  int img_w = 300;
  int img_size = img_h * img_w * 3;
  float *input_data = (float *)malloc(sizeof(float) * img_size);

  int node_idx = 0;
  int tensor_idx = 0;
  tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
  if (!check_tensor_valid(input_tensor)) {
    std::printf("Get input node failed : node_idx: %d, tensor_idx: %d\n",
                node_idx, tensor_idx);
    return 1;
  }
  int dims[] = {1, 3, img_h, img_w};
  set_tensor_shape(input_tensor, dims, 4);
  prerun_graph(graph);

  int repeat_count = 1;
  const char *repeat = std::getenv("REPEAT_COUNT");

  if (repeat) repeat_count = std::strtoul(repeat, NULL, 10);

  struct timeval t0, t1;
  float total_time = 0.f;
  for (int i = 0; i < repeat_count; i++) {
    get_input_data_ssd(image_file, input_data, img_h, img_w);

    gettimeofday(&t0, NULL);
    set_tensor_buffer(input_tensor, input_data, img_size * 4);
    run_graph(graph, 1);

    gettimeofday(&t1, NULL);
    float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) -
                           (t0.tv_sec * 1000000 + t0.tv_usec)) /
                   1000;
    total_time += mytime;
  }
  std::cout << "--------------------------------------\n";
  std::cout << "repeat " << repeat_count << " times, avg time per run is "
            << total_time / repeat_count << " ms\n";
  tensor_t out_tensor =
      get_graph_output_tensor(graph, 0, 0);  //"detection_out");
  int out_dim[4];
  get_tensor_shape(out_tensor, out_dim, 4);
  float *outdata = (float *)get_tensor_buffer(out_tensor);
  int num = out_dim[1];
  float show_threshold = 0.5;

  post_process_ssd(image_file, show_threshold, outdata, num, save_name);

  put_graph_tensor(input_tensor);
  postrun_graph(graph);
  free(input_data);
  destroy_runtime_graph(graph);
  remove_model(model_name);

  return 0;
}
