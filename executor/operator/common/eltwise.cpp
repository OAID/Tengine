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
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>

#include <math.h>
#include "graph.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "operator/eltwise.hpp"
#include "tensor_mem.hpp"
namespace TEngine {

namespace EltwiseImpl {

struct EltwiseOps : public NodeOps {
  bool Run(Node* node) {
    // input
    Tensor* input_tensor0 = node->GetInputTensor(0);
    const TShape& ishape = input_tensor0->GetShape();
    int input_count4 = ishape.GetSize();
    void* input0 = get_tensor_mem(input_tensor0);
    int input_chan = ishape.GetC();
    int input_hw = ishape.GetH() * ishape.GetW();

    Tensor* input_tensor1 = nullptr;
    void* input1 = nullptr;
    int input1_count4 = 0;

    if (node->GetInputNum() > 1) {
      input_tensor1 = node->GetInputTensor(1);
      input1 = get_tensor_mem(input_tensor1);
      input1_count4 = input_tensor1->GetTotalSize() / 4;
    }

    // this version only support for input_num=2
    // int input_number=node->GetInputNum();

    // output
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(output_tensor);
    float* out_ptr = (float*)output;
    float* in0 = (float*)input0;
    float* in1 = (float*)input1;
    Eltwise* eltwise_op = dynamic_cast<Eltwise*>(node->GetOp());
    EltwiseParam* param = eltwise_op->GetParam();

    switch (param->type) {
      case ELT_SUB:

        if (input_count4 == input1_count4) {
          for (int i = 0; i < input_count4; ++i) {
            *out_ptr++ = (*in0++) - (*in1++);
          }
        } else if (input_chan == input1_count4) {
          for (int i = 0; i < input_count4; ++i) {
            *out_ptr++ = in0[i] - in1[i / input_hw];
          }
        } else
          return false;
        break;
      case ELT_SUM:
        if (input1_count4 == 1) {
          for (int i = 0; i < input_count4; ++i) {
            *out_ptr++ = (*in0++) + in1[0];
          }
        } else if (input_count4 == input1_count4) {
          for (int i = 0; i < input_count4; ++i) {
            *out_ptr++ = (*in0++) + (*in1++);
          }
        } else if (input_chan == input1_count4) {
          for (int i = 0; i < input_count4; ++i) {
            *out_ptr++ = in0[i] + in1[i / input_hw];
          }
        } else
          return false;
        break;
      case ELT_MAX:
        for (int i = 0; i < input_count4; ++i) {
          *out_ptr++ = std::max(in0[i], in1[i]);
        }
        break;
      case ELT_PROD:
        if (input1_count4 == 1) {
          for (int i = 0; i < input_count4; ++i) {
            *out_ptr++ = (*in0++) * in1[0];
          }
        } else if (input_count4 == input1_count4) {
          for (int i = 0; i < input_count4; ++i) {
            *out_ptr++ = in0[i] * in1[i];
          }
        } else if (input_chan == input1_count4) {
          for (int i = 0; i < input_count4; ++i) {
            *out_ptr++ = in0[i] * in1[i / input_hw];
          }
        } else
          return false;
        break;
      case ELT_RSQRT:
        for (int i = 0; i < input_count4; ++i) {
          *out_ptr++ = 1 / sqrt(in0[i]);
        }
        break;
      case ELT_MIN_SCALAR:
        for (int i = 0; i < input_count4; ++i) {
          *out_ptr++ = std::min((*in0++), in1[0]);
        }
        break;
      case ELT_SUB_SCALAR:
        for (int i = 0; i < input_count4; ++i) {
          *out_ptr++ = (*in0++) - in1[0];
        }
        break;
      case ELT_PROD_SCALAR:
        for (int i = 0; i < input_count4; ++i) {
          *out_ptr++ = (*in0++) * in1[0];
        }
        break;
      default:
        return false;
    }
    return true;
  }  // Run

};  // struct EltwiseOps

}  // namespace EltwiseImpl

using namespace EltwiseImpl;

void RegisterEltwiseNodeExec(void) {
  EltwiseOps* ops = new EltwiseOps();

  NodeOpsRegistryManager::RegisterOPImplementor("common", "Eltwise", ops);
}

}  // namespace TEngine
