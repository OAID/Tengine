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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include "operator/fully_connected.hpp"
#include "static_graph.hpp"

namespace TEngine {

bool FullyConnected::InferShape(const std::vector<TEngine::TShape>& ishape,
                                std::vector<TEngine::TShape>& oshape) {
  const TShape& input = ishape[0];
  const TShape& weight = ishape[1];

  int m = input.GetN();
  int input_k = input.GetW() * input.GetH() * input.GetC();

  int n = weight.GetH();
  int k = weight.GetW();

  if (k != input_k) return false;

  TShape shape;

  std::vector<int> dim = {m, n, 1, 1};

  shape.SetDim(dim);
  shape.SetDataLayout("NCHW");

  oshape[0] = shape;

  return true;
}

float FullyConnected::GetFops(const std::vector<TShape>& inputs,
                              const std::vector<TShape>& outputs) {
  const TShape& input = inputs[0];
  const TShape& weight = inputs[1];

  int m = input.GetN();

  int n = weight.GetH();
  int k = weight.GetW();

  float fops = m * n * k * 2;

  return fops;
}

void FullyConnected::SetSchema(void) {
  Input({"input:float32", "weight:float32", "bias:float32"})
      .Output({"output:float32"})
      .SetLayout("NCHW")
      .SetAttr("num_output", 10)
      .SetDoc(R"DOC(Fully Connected Operator)DOC");
}

}  // namespace TEngine
