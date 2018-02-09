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
 * Author: haitao@openailab.com
 */
#include "node_ops.hpp"
#include "arm64_registry.hpp"

namespace TEngine {


namespace arm64_ops {

static NodeOpsRegistry  arm64_ops_registry("arm64");

NodeOpsRegistry * GetRegistry(void)
{
    return &arm64_ops_registry;
}


} //namespace arm64


namespace a72_ops {

static NodeOpsRegistry  a72_ops_registry("A72");

NodeOpsRegistry * GetRegistry(void)
{
    return &a72_ops_registry;
}


}//namespace a72_ops


namespace a53_ops {

static NodeOpsRegistry  a53_ops_registry("A53");

NodeOpsRegistry * GetRegistry(void)
{
    return &a53_ops_registry;
}


}//namespace a53_ops


} //namespace TEngine
