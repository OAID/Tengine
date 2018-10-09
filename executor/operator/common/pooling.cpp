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
#include <stdlib.h>
#include <functional>
#include <iostream>

#include "graph.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "operator/pooling.hpp"
#include "tensor_mem.hpp"

namespace TEngine {

namespace PoolingImpl {

struct PoolOps : public NodeOps {
  void Generic_AvgPool(const float* input, float* output, int inc, int inh,
                       int inw, int outh, int outw, int k_h, int k_w,
                       int stride_h, int stride_w, int pad_h, int pad_w) {
    int in_hw = inh * inw;
    int out_hw = outh * outw;
    for (int c = 0; c < inc; c++) {
      int c_skip = c * in_hw;
      int oc_skip = c * out_hw;

      for (int ph = 0; ph < outh; ph++) {
        for (int pw = 0; pw < outw; pw++) {
          int h_start = ph * stride_h - pad_h;
          int h_end = std::min(h_start + k_h, inh + pad_h);
          int w_start = pw * stride_w - pad_w;
          int w_end = std::min(w_start + k_w, inw + pad_w);
          int pool_size = (h_end - h_start) * (w_end - w_start);

          h_start = std::max(h_start, 0);
          w_start = std::max(w_start, 0);
          h_end = std::min(h_end, inh);
          w_end = std::min(w_end, inw);

          const int out_index = oc_skip + ph * outw + pw;
          output[out_index] = 0.f;
          for (int h = h_start; h < h_end; h++) {
            for (int w = w_start; w < w_end; w++) {
              output[out_index] += input[c_skip + h * inw + w];
            }
          }  // end ksize_h,ksize_w
          output[out_index] /= pool_size;
        }
      }
    }
  }

  void Generic_MaxPool(const float* input, float* output, int inc, int inh,
                       int inw, int outh, int outw, int k_h, int k_w,
                       int stride_h, int stride_w, int pad_h, int pad_w) {
    int in_hw = inh * inw;
    int out_hw = outh * outw;
    for (int c = 0; c < inc; c++) {
      int c_skip = c * in_hw;
      int oc_skip = c * out_hw;

      for (int ph = 0; ph < outh; ph++) {
        int h_start = ph * stride_h - pad_h;
        int h_end = std::min(h_start + k_h, inh);
        h_start = std::max(h_start, 0);

        for (int pw = 0; pw < outw; pw++) {
          int w_start = pw * stride_w - pad_w;
          int w_end = std::min(w_start + k_w, inw);
          w_start = std::max(w_start, 0);

          const int out_index = oc_skip + ph * outw + pw;
          output[out_index] = input[c_skip + h_start * inw + w_start];
          for (int h = h_start; h < h_end; h++) {
            for (int w = w_start; w < w_end; w++) {
              int in_index = c_skip + h * inw + w;

              if (input[in_index] > output[out_index]) {
                output[out_index] = input[in_index];
              }
            }
          }  // end ksize_h,ksize_w
        }
      }
    }
  }

  void Global_MaxPool(float* input, float* output, int inc, int in_hw) {
    float* out_ptr = output;
    float* in_ptr = input;
    for (int c = 0; c < inc; c++) {
      float max_ = in_ptr[0];
      for (int j = 0; j < in_hw; j++) {
        max_ = std::max(max_, in_ptr[0]);
        in_ptr++;
      }
      *out_ptr = max_;
      out_ptr++;
    }
  }

  void Global_AvgPool(float* input, float* output, int inc, int in_hw) {
    float* out_ptr = output;
    float* in_ptr = input;
    for (int c = 0; c < inc; c++) {
      float sum = 0.f;
      for (int j = 0; j < in_hw; j++) {
        sum += in_ptr[0];
        in_ptr++;
      }
      *out_ptr = sum / in_hw;
      out_ptr++;
    }
  }

  bool Run(Node* node) {
    // operator, param
    Pooling* pooling_op = dynamic_cast<Pooling*>(node->GetOp());
    PoolParam* param_ = pooling_op->GetParam();

    // input, output, shape
    Tensor* itensor = node->GetInputTensor(0);
    const TShape& ishape = itensor->GetShape();
    Tensor* otensor = node->GetOutputTensor(0);
    TShape& oshape = otensor->GetShape();
    const std::vector<int>& in_dim = ishape.GetDim();
    const std::vector<int>& out_dim = oshape.GetDim();
    int in_hw = in_dim[3] * in_dim[2];
    int in_chw = in_dim[1] * in_hw;

    int out_hw = out_dim[2] * out_dim[3];
    int out_chw = out_dim[1] * out_hw;

    // data
    float* input_data = (float*)get_tensor_mem(itensor);
    float* output_data = (float*)get_tensor_mem(otensor);

    if (param_->global) {
      if (param_->alg == kPoolMax) {
        for (int n = 0; n < in_dim[0]; n++) {
          Global_MaxPool(input_data + n * in_chw, output_data + n * out_chw,
                         in_dim[1], in_hw);
        }
      } else if (param_->alg == kPoolAvg) {
        for (int n = 0; n < in_dim[0]; n++) {
          Global_AvgPool(input_data + n * in_chw, output_data + n * out_chw,
                         in_dim[1], in_hw);
        }
      } else {
        std::cout << " Pooling type Error\n";
      }
    } else if (param_->alg == kPoolMax) {
      for (int n = 0; n < in_dim[0]; n++) {
        Generic_MaxPool(input_data + n * in_chw, output_data + n * out_chw,
                        in_dim[1], in_dim[2], in_dim[3], out_dim[2], out_dim[3],
                        param_->kernel_shape[0], param_->kernel_shape[1],
                        param_->strides[0], param_->strides[1], param_->pads[0],
                        param_->pads[1]);
      }
    } else if (param_->alg == kPoolAvg) {
      for (int n = 0; n < in_dim[0]; n++) {
        Generic_AvgPool(input_data + n * in_chw, output_data + n * out_chw,
                        in_dim[1], in_dim[2], in_dim[3], out_dim[2], out_dim[3],
                        param_->kernel_shape[0], param_->kernel_shape[1],
                        param_->strides[0], param_->strides[1], param_->pads[0],
                        param_->pads[1]);
      }
    } else {
      std::cout << " Pooling type Error\n";
    }

    return true;
  }
};

}  // namespace PoolingImpl

using namespace PoolingImpl;

void RegisterPooling_NodeExec(void) {
  PoolOps* ops = new PoolOps();

  NodeOpsRegistryManager::RegisterOPImplementor("common", "Pooling", ops);
}

}  // namespace TEngine
