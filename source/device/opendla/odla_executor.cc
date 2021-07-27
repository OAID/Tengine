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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#include "odla_executor.hpp"
#include "odla_define.h"

#ifdef ODLA_MODEL_CACHE
#include "defines.h"
#include "cstdlib"
#endif

#ifdef ODLA_MODEL_CACHE
#include <fstream>
#endif


ODLAEngine::ODLAEngine()
{
    std::cout << "ODLA Engine Init " << std::endl;
};


int ODLAEngine::ODLATensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type)
{
    std::cout << "ODLA TensorMap Entrance " << std::endl;

    return 0;
}

int ODLAEngine::Build(struct subgraph* subgraph)
{
    std::cout << "ODLA Build Entrance " << std::endl;

    return 0;
}


int ODLAEngine::ODLAEnginePreRun(struct subgraph* subgraph)
{
    std::cout << "ODLA PreRun Entrance " << std::endl;

    return 0;
};

int ODLAEngine::ODLAEngineRun(struct subgraph* subgraph)
{
    std::cout << "ODLA EngineRun Entrance " << std::endl;

    return 0;
}

void ODLAEngine::ODLAEnginePostRun()
{

};
