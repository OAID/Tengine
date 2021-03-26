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

#ifndef __TIMVX_TIMVX_EXECUTOR_HPP__
#define __TIMVX_TIMVX_EXECUTOR_HPP__

extern "C"
{
#include "tengine_ir.h"
#include "tengine_log.h"
}

#include <map>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>

#include <string.h>
#include <sys/stat.h>

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

#include "tim/vx/tensor.h"

#include "convolution_param.h"

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
    void VXTensorMap(struct ir_graph* ir_graph, int ir_tensor_idx, int spec_type);

    bool AddClipNode(struct ir_node* ir_node);
    bool AddConcatNode(struct ir_node* ir_node);
    bool AddConvolutionNode(struct ir_node* ir_node);
    bool AddDepthToSpaceNode(struct ir_node* ir_node);
    bool AddDropoutNode(struct ir_node* ir_node);
    bool AddEltwiseNode(struct ir_node* ir_node);
    bool AddEluNode(struct ir_node* ir_node);
    bool AddFlattenNode(struct ir_node* ir_node);
    bool AddFullyConnectionNode(struct ir_node* node);
    bool AddGatherNode(struct ir_node* node);
    bool AddHardSwishNode(struct ir_node* node);
    bool AddInterpNode(struct ir_node* ir_node);
    bool AddPermuteNode(struct ir_node* ir_node);
    bool AddPoolingNode(struct ir_node* ir_node);
    bool AddPReluNode(struct ir_node* ir_node);
    bool AddReluNode(struct ir_node* ir_node);
    bool AddRelu1Node(struct ir_node* ir_node);
    bool AddReshapeNode(struct ir_node* ir_node);
    bool AddSigmoidNode(struct ir_node* ir_node);
    bool AddSliceNode(struct ir_node* ir_node);
    bool AddSoftmaxNode(struct ir_node* ir_node);
    bool AddSpaceToDepthNode(struct ir_node* ir_node);
    bool AddTanhNode(struct ir_node* ir_node);
    bool AddTransposeNode(struct ir_node* ir_node);
    bool AddUpsampleNode(struct ir_node* ir_node);




public:
    std::shared_ptr<tim::vx::Context> context;
    std::shared_ptr<tim::vx::Graph> graph;
    std::shared_ptr<tim::vx::Operation> ops;

private:
    dict_irt2vxt     vx_tensor_map;
    dict_irt2vxo     vx_node_map;



};


















#endif
