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
 * Copyright (c) 2021, OPEN AI LAB
 * Author:   bzhang@openailab.com
 */

#ifndef __TENSORFLOW2TENGINE_HPP__
#define __TENSORFLOW2TENGINE_HPP__

#include <cstring>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <fstream>
#include <queue>
#include <stack>

#include "graph.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <algorithm>

extern "C" {
#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "executer/executer.h"
#include "module/module.h"
#include "utility/log.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "save_graph/op_include.h"
}

#define TF_RNN_LSTM       0
#define TF_RNN_GRU        1
#define TF_RNN_BASIC_LSTM 2
#define TF_RNN_BASIC_RNN  3
#define FUSE_NODE         10
static int NCHW_axis_swap[] = {0, 2, 3, 1};

struct TFNode
{
    int idx;
    std::string name;
    std::string op;
    std::vector<TFNode*> inputs;
    std::vector<TFNode*> outputs;
    std::vector<const tensorflow::NodeDef*> pb_defs;
    ir_node_t* ir_node;
    ir_tensor_t* ir_tensor;
    bool no_static_node;
    int BNAddType;
    std::vector<std::string> in_tensors;
    std::vector<std::string> out_tensors;
    int biasAdd;

    TFNode()
    {
        no_static_node = false;
    }

    virtual ~TFNode()
    {
    }
};

struct LSTMNode : public TFNode
{
    float clip;

    std::string direction;

    /* optional inputs */
    TFNode* kernel;
    TFNode* bias;
    TFNode* w_f_diag;
    TFNode* w_i_diag;
    TFNode* w_o_diag;
    TFNode* projection;
    TFNode* init_h;
    TFNode* init_c;
    TFNode* forget_bias;

    std::set<TFNode*> rnn_graph;

    LSTMNode()
    {
        kernel = nullptr;
        bias = nullptr;
        w_f_diag = nullptr;
        w_i_diag = nullptr;
        w_o_diag = nullptr;
        projection = nullptr;
        init_h = nullptr;
        init_c = nullptr;
        forget_bias = nullptr;
    }

    ~LSTMNode()
    {
        auto rnn_ir = rnn_graph.begin();
        auto rnn_end = rnn_graph.end();

        while (rnn_ir != rnn_end)
        {
            delete (*rnn_ir);
            rnn_ir++;
        }
    }
};

struct RNNNode : public TFNode
{
    float clip;

    std::string direction;

    /* optional inputs */
    TFNode* kernel;
    TFNode* bias;
    TFNode* init_h;

    std::set<TFNode*> rnn_graph;

    RNNNode()
    {
        kernel = nullptr;
        bias = nullptr;
        init_h = nullptr;
    }

    ~RNNNode()
    {
        auto rnn_ir = rnn_graph.begin();
        auto rnn_end = rnn_graph.end();

        while (rnn_ir != rnn_end)
        {
            delete (*rnn_ir);
            rnn_ir++;
        }
    }
};

struct GRUNode : public TFNode
{
    float clip;

    std::string direction;

    /* optional inputs */
    TFNode* kernel;
    TFNode* bias;
    TFNode* init_h;
    // gru kernel & bias
    TFNode* gate_kernel;
    TFNode* gate_bias;
    TFNode* candidate_kernel;
    TFNode* candidate_bias;

    std::set<TFNode*> rnn_graph;

    GRUNode()
    {
        kernel = nullptr;
        bias = nullptr;
        init_h = nullptr;
        gate_kernel = nullptr;
        gate_bias = nullptr;
        candidate_kernel = nullptr;
        candidate_bias = nullptr;
    }

    ~GRUNode()
    {
        auto rnn_ir = rnn_graph.begin();
        auto rnn_end = rnn_graph.end();

        while (rnn_ir != rnn_end)
        {
            delete (*rnn_ir);
            rnn_ir++;
        }
    }
};

struct TFGraph
{
    std::vector<TFNode*> seq_nodes;

    ~TFGraph()
    {
        for (auto node : seq_nodes)
            delete node;
    }
};

class tensorflow_serializer
{
public:
    graph_t tensorflow2tengine(std::string model_file);
    typedef int (*op_load_t)(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node);

private:
    std::unordered_map<std::string, std::pair<int, op_load_t> > op_load_map;
    int load_graph(ir_graph_t* graph);
    int load_model(ir_graph_t* graph, std::string model_file);
    int load_binary_file(std::string model_file);
    int load_graph_node(tensorflow::GraphDef& tf_net, ir_graph_t* graph);
    int load_tensor_data(TFNode* tf_node, ir_graph_t* graph);
    int optimize_graph();
    int set_graph_input(ir_graph_t* graph);
    int set_graph_output(ir_graph_t* graph);
    bool find_op_load_method(const std::string& op_name);
    int generate_graph(ir_graph_t* graph);
    int construct_graph();
    ir_tensor_t* find_tensor(ir_graph_t* graph, const std::string& tensor_name);
    void register_op_load();
    int FindRNNScope(std::string& rnn_scope);
    void ParseLSTMGraph(LSTMNode* lstm_node, std::set<TFNode*>& rnn_graph);
    void StripRNNScope(std::string& rnn_scope, int rnn_type);
    void MergeReluMinimum();
    int MergeChildNode(TFNode* base_node, TFNode* child_node);
    int MergeParentNode(TFNode* base_node, TFNode* child_node);
    int BNRecursiveInputMerge(TFNode* node);
    int FuseComposedBN(TFNode* cur_node);
    int optimize_rnn();
    void CleanupResizeNearestNeighbor();
    int DFSGraph(ir_graph_t* graph);

    tensorflow::GraphDef tf_net;
    TFGraph tf_graph;
    std::vector<std::string> input_tensors;
    std::vector<std::string> output_tensors;
    std::set<TFNode*> ck_graph;
    std::vector<TFNode*> out_graph;
    int fused_node_count;
};

#endif