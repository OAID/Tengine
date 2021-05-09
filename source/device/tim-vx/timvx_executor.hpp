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

extern "C"
{
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"
#include "utility/log.h"
}

#include <map>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>


#include "convolution_param.h"

#include "tim/vx/tensor.h"

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"

#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/depth2space.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/fullyconnected.h"
#include "tim/vx/ops/gather.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/resize.h"
#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/softmax.h"
#include "tim/vx/ops/space2depth.h"
#include "tim/vx/ops/transpose.h"

#define SPEC_TYPE_OUTPUT    1
#define SPEC_TYPE_DWCONV    2
#define SPEC_TYPE_PRELU     3
#define SPEC_TYPE_INTERP    4
#define SPEC_TYPE_RESHAPE   5

typedef std::map<uint32_t, std::shared_ptr<tim::vx::Tensor>> dict_irt2vxt;
typedef std::map<uint32_t, std::shared_ptr<tim::vx::Operation>> dict_irt2vxo;


class VXEngine
{
public:
    VXEngine();
    ~VXEngine() = default;

    int VXEnginePreRun(struct subgraph* subgraph);
    int VXEngineRun(struct subgraph* subgraph);
    void VXEnginePostRun();

private:
    int Build(struct subgraph* subgraph);
    void VXTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type);

    bool AddClipNode(struct node* ir_node);
    bool AddConcatNode(struct node* ir_node);
    bool AddConvolutionNode(struct node* ir_node);
    bool AddDepthToSpaceNode(struct node* ir_node);
    bool AddDropoutNode(struct node* ir_node);
    bool AddEltwiseNode(struct node* ir_node);
    bool AddEluNode(struct node* ir_node);
    bool AddFlattenNode(struct node* ir_node);
    bool AddFullyConnectionNode(struct node* node);
    bool AddGatherNode(struct node* node);
    bool AddHardSwishNode(struct node* node);
    bool AddInterpNode(struct node* ir_node);
    bool AddPermuteNode(struct node* ir_node);
    bool AddPoolingNode(struct node* ir_node);
    bool AddPReluNode(struct node* ir_node);
    bool AddReluNode(struct node* ir_node);
    bool AddRelu1Node(struct node* ir_node);
    bool AddReshapeNode(struct node* ir_node);
    bool AddResizeNode(struct node* ir_node);
    bool AddScaleNode(struct node* ir_node);
    bool AddSigmoidNode(struct node* ir_node);
    bool AddSliceNode(struct node* ir_node);
    bool AddSoftmaxNode(struct node* ir_node);
    bool AddSpaceToDepthNode(struct node* ir_node);
    bool AddTanhNode(struct node* ir_node);
    bool AddTransposeNode(struct node* ir_node);
    bool AddUpsampleNode(struct node* ir_node);




public:
    std::shared_ptr<tim::vx::Context> context;
    std::shared_ptr<tim::vx::Graph> graph;
    std::shared_ptr<tim::vx::Operation> ops;

private:
    dict_irt2vxt     vx_tensor_map;
    dict_irt2vxo     vx_node_map;
};
