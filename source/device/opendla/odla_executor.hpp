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

#include "priv/EngineAST.h"

extern "C"
{
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"
#include "utility/log.h"

#include "odla_dump.h"
}

#include <map>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>


#include "convolution_param.h"




class ODLAEngine
{
public:
    ODLAEngine();
    ~ODLAEngine() = default;

    int ODLAEnginePreRun(struct subgraph* subgraph);
    int ODLAEngineRun(struct subgraph* subgraph);
    void ODLAEnginePostRun();

private:
    int Build(struct subgraph* subgraph);
    int ODLATensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type);

    bool AddPoolingNode(struct node* ir_node);

public:



private:

};
