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
#include "operator/slice.hpp"

namespace TEngine {
bool Slice::InferShape(const std::vector<TEngine::TShape>& ishape,
                       std::vector<TEngine::TShape>& oshape) {
  // only support for slice_axis=1
  const TShape& input = ishape[0];

  int n = input.GetN();
  int c = input.GetC();
  int h = input.GetH();
  int w = input.GetW();

  if (c % 2 != 0) return false;

  TShape shape;

  std::vector<int> dim = {n, c / 2, h, w};

  shape.SetDim(dim);
  shape.SetDataLayout("NCHW");

  oshape[0] = shape;
  oshape[1] = shape;

  return true;
}
void Slice::SetSchema(void) {
  Input({"input:float32"})
      .Output({"output:float32"})
      .SetLayout("NCHW")
      .SetAttr("axis", 1)
      .SetDoc(R"DOC(Slice Operator)DOC");
}

}  // namespace TEngine
