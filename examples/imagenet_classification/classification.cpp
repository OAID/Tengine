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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "cpu_device.h"
#include "model_config.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "tengine_c_api.h"

#define DEFAULT_MODEL_NAME "squeezenet"
#define DEFAULT_IMAGE_FILE "tests/images/cat.jpg"
#define DEFAULT_LABEL_FILE "models/synset_words.txt"
#define DEFAULT_IMG_H 227
#define DEFAULT_IMG_W 227
#define DEFAULT_SCALE 1.f
#define DEFAULT_MEAN1 104.007
#define DEFAULT_MEAN2 116.669
#define DEFAULT_MEAN3 122.679
#define DEFAULT_REPEAT_CNT 1
#define PRINT_TOP_NUM 5

void LoadLabelFile(std::vector<std::string> &result, const char *fname) {
  std::ifstream labels(fname);

  std::string line;
  while (std::getline(labels, line)) result.push_back(line);
}

static inline bool PairCompare(const std::pair<float, int> &lhs,
                               const std::pair<float, int> &rhs) {
  return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<float> &v, int N) {
  std::vector<std::pair<float, int>> pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i) result.push_back(pairs[i].second);
  return result;
}

void get_input_data(const char *image_file, float *input_data, int img_h,
                    int img_w, const float *mean, float scale) {
  cv::Mat sample = cv::imread(image_file, -1);
  if (sample.empty()) {
    std::cerr << "Failed to read image file " << image_file << ".\n";
    return;
  }
  cv::Mat img;
  if (sample.channels() == 4) {
    cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
  } else if (sample.channels() == 1) {
    cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
  } else {
    img = sample;
  }

  cv::resize(img, img, cv::Size(img_h, img_w));
  img.convertTo(img, CV_32FC3);
  float *img_data = (float *)img.data;
  int hw = img_h * img_w;
  for (int h = 0; h < img_h; h++) {
    for (int w = 0; w < img_w; w++) {
      for (int c = 0; c < 3; c++) {
        input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
        img_data++;
      }
    }
  }
}

void PrintTopLabels(const char *label_file, float *data) {
  // load labels
  std::vector<std::string> labels;
  LoadLabelFile(labels, label_file);

  float *end = data + 1000;
  std::vector<float> result(data, end);
  std::vector<int> top_N = Argmax(result, PRINT_TOP_NUM);

  for (unsigned int i = 0; i < top_N.size(); i++) {
    int idx = top_N[i];

    std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \""
              << labels[idx] << "\"\n";
  }
}

int sys_init(void) {
  init_tengine_library();
  if (request_tengine_version("0.1") < 0) return 1;
  //    const struct cpu_info * p_info=get_predefined_cpu("rk3399");
  //    int cpu_list[]={4,5};
  //    set_online_cpu((struct cpu_info
  //    *)p_info,cpu_list,sizeof(cpu_list)/sizeof(int));
  //    create_cpu_device("rk3399",p_info);

  return 0;
}

bool run_tengine_library(const char *model_name, const char *proto_file,
                         const char *model_file, const char *label_file,
                         const char *image_file, int img_h, int img_w,
                         const float *mean, float scale, int repeat_count) {
  // init tengine
  // init_tengine_library();
  // if (request_tengine_version("0.1") < 0)
  //   return false;
  sys_init();
  // load model
  if (load_model(model_name, "caffe", proto_file, model_file) < 0) return false;
  std::cout << "Load model done.\n";

  // create graph
  graph_t graph = create_runtime_graph("graph", model_name, NULL);
  if (!check_graph_valid(graph)) {
    std::cerr << "Create graph0 failed.\n";
    return false;
  }

  // input
  int img_size = img_h * img_w * 3;
  int dims[] = {1, 3, img_h, img_w};
  float *input_data = (float *)malloc(sizeof(float) * img_size);

  tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
  set_tensor_shape(input_tensor, dims, 4);

  // prerun
  prerun_graph(graph);

  struct timeval t0, t1;
  float avg_time = 0.f;
  for (int i = 0; i < repeat_count; i++) {
    get_input_data(image_file, input_data, img_h, img_w, mean, scale);
    set_tensor_buffer(input_tensor, input_data, img_size * 4);

    gettimeofday(&t0, NULL);
    run_graph(graph, 1);
    gettimeofday(&t1, NULL);

    float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) -
                           (t0.tv_sec * 1000000 + t0.tv_usec)) /
                   1000;
    avg_time += mytime;
  }
  std::cout << "\nModel name : " << model_name << "\n"
            << "Proto file : " << proto_file << "\n"
            << "Model file : " << model_file << "\n"
            << "label file : " << label_file << "\n"
            << "image file : " << image_file << "\n"
            << "img_h, imag_w, scale, mean[3] : " << img_h << " " << img_w
            << " " << scale << " " << mean[0] << " " << mean[1] << " "
            << mean[2] << "\n";
  std::cout << "\nRepeat " << repeat_count << " times, avg time per run is "
            << avg_time / repeat_count << " ms\n";
  std::cout << "--------------------------------------\n";

  // print output
  tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
  float *data = (float *)get_tensor_buffer(output_tensor);
  PrintTopLabels(label_file, data);
  std::cout << "--------------------------------------\n";

  put_graph_tensor(output_tensor);
  put_graph_tensor(input_tensor);

  free(input_data);
  postrun_graph(graph);
  destroy_runtime_graph(graph);
  remove_model(model_name);

  return true;
}

template <typename T>
static std::vector<T> ParseString(const std::string str) {
  typedef std::string::size_type pos;
  const char delim_ch = ',';
  std::string str_tmp = str;
  std::vector<T> result;
  T t;

  pos delim_pos = str_tmp.find(delim_ch);
  while (delim_pos != std::string::npos) {
    std::istringstream ist(str_tmp.substr(0, delim_pos));
    ist >> t;
    result.push_back(t);
    str_tmp.replace(0, delim_pos + 1, "");
    delim_pos = str_tmp.find(delim_ch);
  }
  if (str_tmp.size() > 0) {
    std::istringstream ist(str_tmp);
    ist >> t;
    result.push_back(t);
  }

  return result;
}

int main(int argc, char *argv[]) {
  int repeat_count = DEFAULT_REPEAT_CNT;
  const std::string root_path = get_root_path();
  std::string model_name;
  std::string proto_file;
  std::string model_file;
  std::string label_file;
  std::string image_file;
  std::vector<int> hw;
  std::vector<float> ms;
  int img_h = 0;
  int img_w = 0;
  float scale = 0;
  float mean[3] = {-1, -1, -1};

  int res;
  while ((res = getopt(argc, argv, "n:p:m:l:i:g:s:w:r:h")) != -1) {
    switch (res) {
      case 'n':
        model_name = optarg;
        break;
      case 'p':
        proto_file = optarg;
        break;
      case 'm':
        model_file = optarg;
        break;
      case 'l':
        label_file = optarg;
        break;
      case 'i':
        image_file = optarg;
        break;
      case 'g':
        hw = ParseString<int>(optarg);
        if (hw.size() != 2) {
          std::cerr << "Error -g parameter.\n";
          return -1;
        }
        img_h = hw[0];
        img_w = hw[1];
        break;
      case 's':
        scale = strtof(optarg, NULL);
        break;
      case 'w':
        ms = ParseString<float>(optarg);
        if (ms.size() != 3) {
          std::cerr << "Error -w parameter.\n";
          return -1;
        }
        mean[0] = ms[0];
        mean[1] = ms[1];
        mean[2] = ms[2];
        break;
      case 'r':
        repeat_count = std::strtoul(optarg, NULL, 10);
        break;
      case 'h':
        std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                  << "    [-n model_name] [-p proto_file] [-m model_file] [-l "
                     "label_file] [-i image_file]\n"
                  << "    [-g img_h,img_w] [-s scale] [-w "
                     "mean[0],mean[1],mean[2]] [-r repeat_count]\n";
        return 0;
      default:
        break;
    }
  }

  const Model_Config *mod_config;
  // if model files not specified
  if (proto_file.empty() || model_file.empty()) {
    // if model name not specified
    if (model_name.empty()) {
      // use default model
      model_name = DEFAULT_MODEL_NAME;
      std::cout << "Model name and model files not specified, run "
                << model_name << " by default.\n";
    }
    // get model config in predefined model list
    mod_config = get_model_config(model_name.c_str());
    if (!mod_config) return -1;

    // get proto file and model file
    proto_file = get_file(mod_config->proto_file);
    model_file = get_file(mod_config->model_file);
    if (proto_file.empty() || model_file.empty()) return -1;

    // if label file not specified
    if (label_file.empty()) {
      // get label file
      label_file = get_file(mod_config->label_file);
      if (label_file.empty()) return -1;
    }

    if (!hw.size()) {
      img_h = mod_config->img_h;
      img_w = mod_config->img_w;
    }
    if (!scale) scale = mod_config->scale;
    if (!ms.size()) {
      mean[0] = mod_config->mean[0];
      mean[1] = mod_config->mean[1];
      mean[2] = mod_config->mean[2];
    }
  }

  // if label file not specified, use default label file
  if (label_file.empty()) {
    label_file = root_path + DEFAULT_LABEL_FILE;
    std::cout << "Label file not specified, use " << label_file
              << " by default.\n";
  }

  // if image file not specified, use default image file
  if (image_file.empty()) {
    image_file = root_path + DEFAULT_IMAGE_FILE;
    std::cout << "Image file not specified, use " << image_file
              << " by default.\n";
  }

  if (!img_h) img_h = DEFAULT_IMG_H;
  if (!img_w) img_w = DEFAULT_IMG_W;
  if (!scale) scale = DEFAULT_SCALE;
  if (mean[0] == -1) mean[0] = DEFAULT_MEAN1;
  if (mean[1] == -1) mean[1] = DEFAULT_MEAN2;
  if (mean[2] == -1) mean[2] = DEFAULT_MEAN3;
  if (model_name.empty()) model_name = "unknown";

  // start to run
  if (run_tengine_library(model_name.c_str(), proto_file.c_str(),
                          model_file.c_str(), label_file.c_str(),
                          image_file.c_str(), img_h, img_w, mean, scale,
                          repeat_count))
    std::cout << "ALL TEST DONE\n";

  return 0;
}
