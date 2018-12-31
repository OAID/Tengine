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
#include "operator/scale.hpp"
#include "static_graph.hpp"

namespace TEngine {

void Scale::SetSchema(void)
{
    Input({"input:float32", "gamma:float32", "bias:float32"})
        .Output({"output:float32"})
        .SetAttr("axis", 1)
        .SetAttr("num_axes", 1)
        .SetAttr("bias_term", 0)
        .SetDoc(R"DOC(Scale: only caffe flavor scale)DOC");
}

}    // namespace TEngine
