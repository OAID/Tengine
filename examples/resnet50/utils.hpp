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

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "tengine_config.hpp"
#include <sys/time.h>

static inline bool PairCompare(const std::pair<float, int> &lhs,
                               const std::pair<float, int> &rhs)
{
  return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<float> &v, int N)
{
  std::vector<std::pair<float, int>> pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

static inline unsigned long long get_cur_time(void)
{
  struct timeval tv;

  gettimeofday(&tv, NULL);

  return (unsigned long long)(tv.tv_sec * 1000000 + tv.tv_usec);
}

void get_input_data(std::string &image_file, float *data, int img_h, int img_w);

void LoadLabelFile(std::vector<std::string> &result, const char *fname);

#endif