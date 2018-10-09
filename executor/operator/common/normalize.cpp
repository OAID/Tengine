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
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>

#include <math.h>
#include "graph.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "operator/normalize.hpp"
#include "tensor_mem.hpp"

namespace TEngine {

namespace NormalizeImpl {

struct NormalizeOps : public NodeOps {
  bool Prerun(Node* node) {
    const Tensor* input_tensor = node->GetInputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> dims = shape.GetDim();

    float* buffer = (float*)std::malloc(sizeof(float) * dims[2] * dims[3]);
    (*node)["buffer"] = buffer;

    return true;
  }

  void norm_channel(float* input, float* output, float* buffer, float* scale,
                    int hw, int channel) {
    memset(buffer, 0, sizeof(float) * hw);
    //
    float* input_ptr = input;
    for (int c = 0; c < channel; c++) {
      for (int i = 0; i < hw; i++) {
        buffer[i] += (*input_ptr) * (*input_ptr);
        input_ptr++;
      }
    }
    // sqrt
    for (int i = 0; i < hw; i++) {
      buffer[i] = 1.f / sqrt(buffer[i]);
    }
    // div scale
    float* out_ptr = output;
    for (int c = 0; c < channel; c++) {
      for (int i = 0; i < hw; i++) {
        *out_ptr += (*input) * buffer[i] * scale[c];
        out_ptr++;
        input++;
      }
    }
  }

  bool Run(Node* node) {
    const Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const Tensor* scale_tensor = node->GetInputTensor(1);

    Normalize* normalize_op = dynamic_cast<Normalize*>(node->GetOp());
    NormalizeParam* param_ = normalize_op->GetParam();

    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> dims = shape.GetDim();

    int batch_number = dims[0];
    int channel_num = dims[1];
    int channel_size = dims[2] * dims[3];
    int img_size = channel_num * channel_size;

    float* input = (float*)get_tensor_mem(input_tensor);
    float* output = (float*)get_tensor_mem(output_tensor);
    float* scale = (float*)get_tensor_mem(scale_tensor);
    float* buffer = any_cast<float*>(node->GetAttr("buffer"));

    if (param_->channel_shared == 0 && param_->across_spatial == 0)
      for (int i = 0; i < batch_number; i++) {
        norm_channel(input, output, buffer, scale, channel_size, channel_num);
        input += img_size;
        output += img_size;
      }
    // other case to be support
    return true;
  }

  bool Postrun(Node* node) {
    float* addr;

    addr = any_cast<float*>(node->GetAttr("buffer"));
    std::free(addr);
    return true;
  }
};

}  // namespace NormalizeImpl

using namespace NormalizeImpl;

void RegisterNormalizeNodeExec(void) {
  NormalizeOps* ops = new NormalizeOps();

  NodeOpsRegistryManager::RegisterOPImplementor("common", "Normalize", ops);
}

}  // namespace TEngine
