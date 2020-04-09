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
#include "logger.hpp"
#include "operator/fused_operator.hpp"

namespace TEngine {

const std::string FusedBNScaleReLu::class_name("Fused.BNScaleReLu");

void FusedBNScaleReLu::SetSchema(void)
{
    Input({"input:float32", "gmma:float32", "beta:float32", "mean:float32", "var:float32"})
        .Output({"output:float32"})
        .SetAttr("eps", 1e-5f)
        .SetAttr("rescale_factor", 1.0f)
        .SetAttr("caffe_flavor", 0)
        .SetDoc(R"DOC(Fused Batch Normalizatoin/Scale/ReLu)DOC");
}

float FusedBNScaleReLu::GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs)
{
    return outputs[0].GetSize() * 5;
}

}    // namespace TEngine
