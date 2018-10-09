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
#include "operator/flatten.hpp"
#include "tensor_mem.hpp"

namespace TEngine {

namespace FlattenImpl {

struct FlattenOps : public NodeOps {
  bool Run(Node *node) {
    const Tensor *input_tensor = node->GetInputTensor(0);
    Tensor *output_tensor = node->GetOutputTensor(0);

    float *input = (float *)get_tensor_mem(input_tensor);
    float *output = (float *)get_tensor_mem(output_tensor);

    const TShape &shape = input_tensor->GetShape();
    int size = shape.GetSize();

    for (int i = 0; i < size; i++) {
      output[i] = input[i];
    }

    return true;
  }
};

}  // namespace FlattenImpl

using namespace FlattenImpl;

void RegisterFlattenNodeExec(void) {
  FlattenOps *ops = new FlattenOps();

  NodeOpsRegistryManager::RegisterOPImplementor("common", "Flatten", ops);
}

}  // namespace TEngine
