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

#pragma once

#include "torch_helper.hpp"

#include <map>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>

class TORCHEngine
{
public:
    TORCHEngine();
    ~TORCHEngine() = default;

    int TORCHEnginePreRun(struct subgraph* subgraph);
    int TORCHEngineRun(struct subgraph* subgraph);
    void TORCHEnginePostRun();

public:
    int TORCHTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type);

public:
    std::shared_ptr<Net> net;

private:
    dict_irt2vxt torch_tensor_map;
    dict_irt2vxo torch_node_map;
};
