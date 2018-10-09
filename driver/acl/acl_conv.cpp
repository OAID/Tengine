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
 * Author: haoluo@openailab.com
 */
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "logger.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"

#include "acl_conv.hpp"
#include "acl_driver.hpp"

#include "CL/cl2.hpp"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

namespace TEngine {

struct ACLConvArg {
  CLTensor input;
  CLTensor weights;
  CLTensor biases;
  CLTensor out;

  CLConvolutionLayer clconv;
  int relu_fused;
};

bool ACLConvOps::Prerun(Node *node) {
  CLScheduler::get().default_init();
  Convolution *conv_op = dynamic_cast<Convolution *>(node->GetOp());
  ConvParam *param = conv_op->GetParam();

  ACLConvArg *arg = new ACLConvArg();

  if (node->ExistAttr("Fused.ReLu"))
    arg->relu_fused = true;
  else
    arg->relu_fused = false;

  /* input */
  Tensor *itensor = node->GetInputTensor(0);
  TShape &ishape = itensor->GetShape();
  unsigned int input_w = ishape.GetW();
  unsigned int input_h = ishape.GetH();
  unsigned int input_c = ishape.GetC();
  TensorShape _ishape(input_w, input_h, input_c);

  arg->input.allocator()->init(TensorInfo(_ishape, 1, DataType::F32));

  /* weights */
  Tensor *wtensor = node->GetInputTensor(1);
  TShape &wshape = wtensor->GetShape();
  float *weight_org = (float *)get_tensor_mem(wtensor);
  int wsize = wshape.GetSize();

  unsigned int kernel_w = wshape.GetW();
  unsigned int kernel_h = wshape.GetH();
  unsigned int kernel_n = wshape.GetN();
  TensorShape _wshape(kernel_w, kernel_h, input_c, kernel_n);

  arg->weights.allocator()->init(TensorInfo(_wshape, 1, DataType::F32));

  /* biases */
  Tensor *btensor = node->GetInputTensor(2);
  float *biases_org;
  int bsize = 0;
  if (btensor) {
    TShape &bshape = btensor->GetShape();
    biases_org = (float *)get_tensor_mem(btensor);
    bsize = bshape.GetSize();
    TensorShape _bshape(kernel_n);

    arg->biases.allocator()->init(TensorInfo(_bshape, 1, DataType::F32));
  }

  /* output */
  Tensor *otensor = node->GetOutputTensor(0);
  TShape &oshape = otensor->GetShape();
  int mem_size = otensor->GetTotalSize();
  void *addr = std::malloc(mem_size);
  set_tensor_mem(otensor, addr, mem_size, std::free);

  unsigned int out_w = oshape.GetW();
  unsigned int out_h = oshape.GetH();
  unsigned int out_c = oshape.GetC();
  TensorShape _oshape(out_w, out_h, out_c);

  arg->out.allocator()->init(TensorInfo(_oshape, 1, DataType::F32));

  int pad_x = param->pad_w;
  int pad_y = param->pad_h;
  int stride_x = param->stride_w;
  int stride_y = param->stride_h;

  arg->clconv.configure(&arg->input, &arg->weights,
                        btensor ? &arg->biases : nullptr, &arg->out,
                        PadStrideInfo(stride_x, stride_y, pad_x, pad_y));

  arg->input.allocator()->allocate();
  arg->weights.allocator()->allocate();
  arg->weights.map();
  float *wbuf = reinterpret_cast<float *>(arg->weights.buffer());
  memcpy(wbuf, weight_org, wsize * 4);
  arg->weights.unmap();

  if (btensor) {
    arg->biases.allocator()->allocate();
    arg->biases.map();
    float *bbuf = reinterpret_cast<float *>(arg->biases.buffer());
    memcpy(bbuf, biases_org, bsize * 4);
    arg->biases.unmap();
  }
  arg->out.allocator()->allocate();

  (*node)["ACLConvArg"] = arg;

  return true;
}

bool ACLConvOps::Run(Node *node) {
  /* input */
  Tensor *itensor = node->GetInputTensor(0);
  TShape &ishape = itensor->GetShape();
  float *input_org = (float *)get_tensor_mem(itensor);
  int isize = ishape.GetSize();

  ACLConvArg *arg = any_cast<ACLConvArg *>(node->GetAttr("ACLConvArg"));

  arg->input.map();

  float *ibuf = reinterpret_cast<float *>(arg->input.buffer());
  memcpy(ibuf, input_org, isize * 4);

  arg->input.unmap();

  /* output */
  Tensor *output_tensor = node->GetOutputTensor(0);
  float *output_buf = (float *)get_tensor_mem(output_tensor);
  TShape &output_shape = output_tensor->GetShape();
  int out_size = output_shape.GetSize();

  arg->clconv.run();
  const cl::Buffer &cl_buf = arg->out.cl_buffer();
  cl::copy<float *>(cl_buf, output_buf, output_buf + out_size);

  if (arg->relu_fused) {
    for (int i = 0; i < out_size; i++) {
      output_buf[i] = std::max(output_buf[i], 0.0f);
    }
  }

  return true;
}

bool ACLConvOps::Postrun(Node *node) {
  ACLConvArg *arg = any_cast<ACLConvArg *>(node->GetAttr("ACLConvArg"));

  arg->input.allocator()->free();
  arg->weights.allocator()->free();
  arg->out.allocator()->free();

  Tensor *btensor = node->GetInputTensor(2);
  Tensor *otensor = node->GetOutputTensor(0);
  free_tensor_mem(otensor);

  if (btensor) {
    arg->biases.allocator()->free();
  }

  delete arg;

  return true;
}

}  // namespace TEngine
