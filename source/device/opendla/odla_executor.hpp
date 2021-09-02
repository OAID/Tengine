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
 * Copyright (c) 2021, Institute of Computing Technology
 * Author: wanglei21c@mails.ucas.ac.cn
 */

#pragma once

#include "nvdla/IRuntime.h"
#include "priv/EngineAST.h"
#include "priv/CanonicalAST.h"
#include "priv/Profiler.h"
#include "priv/Compiler.h"
#include "ErrorMacros.h"
#include "nvdla_os_inf.h"

extern "C" {
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"
#include "utility/log.h"
#include "utility/sys_port.h"

#include "odla_dump.h"
}

#include <map>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include "odla_dump.h"
#include "convolution_param.h"

#define SPEC_TYPE_CONV      1
#define SPEC_TYPE_CONV_BIAS 2
#define SPEC_TYPE_DWCONV    3
#define SPEC_TYPE_INTERP    4
#define SPEC_TYPE_OUTPUT    5
#define SPEC_TYPE_PRELU     6
#define SPEC_TYPE_SLICE     7
#define SPEC_TYPE_RESHAPE   8
#define SPEC_TYPE_INPUT     9

#define OPENDLA_DUMP_LAYER "TG_ODLA_DEBUG_DATA"

//#define OPENDLA_DEBUG_DATA

typedef std::map<uint32_t, nvdla::priv::Tensor*> dict_irt2odlat;
typedef std::map<nvdla::priv::canonical_ast::Node*, struct node*, nvdla::priv::canonical_ast::Graph::nodeCompareFn> dict_odlan2irtn;
typedef std::map<nvdla::priv::Tensor*, nvdla::priv::canonical_ast::Edge*> dict_odlat2edge;

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

    nvdla::priv::canonical_ast::Node* AddBatchNormalizationNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddConcatNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddDeconvlutionNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddConvolutionNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddEltwiseNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddFullyConnectionNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddPoolingNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddReluNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddScaleNode(struct node* ir_node);
    nvdla::priv::canonical_ast::Node* AddSplitNode(struct node* ir_node);

    NvDlaError ODLAConfigGenerate();

public:
    std::string tp_name = "fast-math";
    std::string target_config_name = "nv_small";
    nvdla::priv::Profile* profile{};
    nvdla::priv::TargetConfig* targetConfig{};
    nvdla::priv::canonical_ast::Graph* graph;

    nvdla::priv::CompilerFactory::CompilerPrivPair compiler;
    nvdla::IRuntime* runtime;
    nvdla::priv::LoadableFactory::LoadablePrivPair loadable;

private:
    nvdla::DataType precision = nvdla::DataType::INT8;
    nvdla::DataFormat inDataFormat = nvdla::DataFormat::NCHW;
    nvdla::TensorScalingMode scalingMode = nvdla::TensorScalingMode::PER_TENSOR;
    nvdla::QuantizationMode quantizationMode = nvdla::QuantizationMode::PER_FILTER;
    uint32_t numBatches = 1;
    std::vector<void*> inputBuffer;
    std::vector<void*> outputBuffer;
    std::vector<void*> host_buffer;
    dict_irt2odlat odla_tensor_map;
    dict_odlan2irtn odla_node_map;
    dict_odlat2edge odla_edge_map;
    void odla_input_data_convert(void* dst, const void* src, nvdla::IRuntime::NvDlaTensor tDesc) const;
    void odla_output_data_convert(void* dst, const void* src, nvdla::IRuntime::NvDlaTensor tDesc) const;
};
