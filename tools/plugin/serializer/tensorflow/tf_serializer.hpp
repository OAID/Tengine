
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

#ifndef __TF_SERIALIZER_HPP__
#define __TF_SERIALIZER_HPP__

#include <cstring>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <set>
#include <algorithm>

#include "graph.pb.h"
#include "logger.hpp"
#include "serializer.hpp"
#include "static_graph_interface.hpp"
 
namespace TEngine {

struct TFNode
{
    int idx;
    std::string name;
    std::string op;
    std::vector<TFNode*> inputs;
    std::vector<TFNode*> outputs;
    std::vector<const tensorflow::NodeDef*> pb_defs;
    StaticNode* static_node;
    StaticTensor* static_tensor;
    bool no_static_node;
    int BNAddType;

    TFNode()
    {
        no_static_node = false;
    }

    virtual ~TFNode() {}
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

        while(rnn_ir != rnn_end)
        {
            delete(*rnn_ir);
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

        while(rnn_ir != rnn_end)
        {
            delete(*rnn_ir);
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

        while(rnn_ir != rnn_end)
        {
            delete(*rnn_ir);
            rnn_ir++;
        }
    }
};

struct TFGraph
{
    std::vector<TFNode*> seq_nodes;

    ~TFGraph()
    {
        for(auto node : seq_nodes)
            delete node;
    }
};

#define TF_RNN_LSTM 0
#define TF_RNN_GRU 1
#define TF_RNN_BASIC_LSTM 2
#define TF_RNN_BASIC_RNN 3
class TFSerializer : public Serializer
{
public:
    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph) override;
    unsigned int GetFileNum(void) override
    {
        return 1;
    }
    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }

protected:
    bool LoadGraph(tensorflow::GraphDef& tf_net, StaticGraph* graph);
    bool LoadBinaryFile(const char* fname, tensorflow::GraphDef& tf_net);
    bool LoadTextFile(const char* fname, tensorflow::GraphDef& tf_net);
    bool ConstructGraph(tensorflow::GraphDef& tf_net, TFGraph& tf_graph);
    bool OptimizeGraph(TFGraph& tf_graph);
    bool GenerateStaticGraph(TFGraph& tf_graph, StaticGraph* graph);
    void CleanupResizeNearestNeighbor(TFGraph& tf_graph);
    void MergeReluMinimum(TFGraph& tf_graph);

    bool MergeChildNode(TFNode* base_node, TFNode* child_node);
    bool MergeParentNode(TFNode* base_node, TFNode* parent_node);
    void BNRecursiveInputMerge(TFNode* node);
    void FuseComposedBN(TFNode* cur_node);
    bool CheckComposedBNAdd(TFNode* node);

    void DisconnectNode(TFNode* node);

    void DumpTFGraph(TFGraph& tf_graph);

    bool OptimizeRNN(tensorflow::GraphDef& tf_net, TFGraph& tf_graph);

    int FindRNNScope(TFGraph& tf_graph, std::string& rnn_scope);

    void StripRNNScope(TFGraph& tf_graph, std::string& rnn_scope, int rnn_type);

    void ParseLSTMGraph(TFGraph& tf_graph, LSTMNode* lstm_node, std::set<TFNode*>& rnn_graph);

    void ParseRNNGraph(TFGraph& tf_graph, RNNNode* rnn_node, std::set<TFNode*>& rnn_graph);

    void ParseGRUGraph(TFGraph& tf_graph, GRUNode* gru_node, std::set<TFNode*>& rnn_graph);
};

}    // namespace TEngine

#endif
