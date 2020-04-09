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
#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include <vector>
#include <string>
#include <functional>

#include "base_object.hpp"
#include "operator.hpp"
#include "tensor.hpp"
#include "node.hpp"

#include "static_graph.hpp"

namespace TEngine {

class Graph;

using Subgraph = Graph;

class Graph : public BaseObject
{
public:
    using graph_visit_t = std::function<void(Graph*, Node*)>;

    Graph(const std::string& name)
    {
        name_ = name;
        model_format_ = -1;
        model_subformat_ = -1;
        model_layout_ = -1;
        layout_ = -1;
    }

    ~Graph()
    {
        for(auto node : owned_nodes_)
            delete node;
        for(auto e : owned_tensors_)
            delete e.second;
    }

    Node* FindNode(const std::string& node_name);
    Tensor* FindTensor(const std::string& tensor_name);

    void AddInputNode(Node* node)
    {
        input_nodes.push_back(node);
    }
    void AddOutputNode(Node* node)
    {
        output_nodes.push_back(node);
    }

    bool AddInputNode(const std::string& node_name);
    bool AddOutputNode(const std::string& node_name);

    void ResetInputNode(void)
    {
        input_nodes.clear();
    }
    void ResetOutputNode(void)
    {
        output_nodes.clear();
    }

    void SetName(const std::string& n)
    {
        name_ = n;
    };
    const std::string& GetName(void) const
    {
        return name_;
    }

    void DumpGraph(void);

    void SanitizeGraph(void);
    void StripGraph(void);
    void PopulateDynamicShape(void);

    bool AddNode(Node* node, bool set_owner = true);
    bool AddTensor(Tensor* tensor, bool set_owner = true);
    bool RemoveTensor(Tensor* tensor);
    bool RemoveNode(Node* node);

    bool CreateNodeFromStatic(Node*, const StaticGraph*, const StaticNode*);
    bool SetupConnection(Tensor*, const StaticGraph*, const StaticTensor*);

    bool RealCreateFromStatic(const StaticGraphPtr&);
    static Graph* CreateFromStatic(const std::string& graph_name, const StaticGraphPtr& static_graph);

    StaticGraphPtr& GetOrigGraph(void);

    std::vector<Node*> input_nodes;
    std::vector<Node*> output_nodes;
    std::vector<Node*> seq_nodes;

    static void BFSVisit(Graph* graph, std::vector<Node*>& starts, graph_visit_t func, bool backward = true,
                         bool input_ready = true);
    static void BackwardBFS(Graph* graph, std::vector<Node*>& starts, graph_visit_t func, bool input_ready);
    static void ForwardBFS(Graph* graph, std::vector<Node*>& starts, graph_visit_t func, bool input_ready);

    void RemoveNoChildTensor(void);
    void HandleNoChildTensor(void);

    bool IsOutputNode(Node* node);
    bool IsInputNode(Node* node);

    void SetNodeOwner(Node* node);
    void SetTensorOwner(Tensor* tensor);
    bool RemoveNodeOwner(Node* node);
    bool RemoveTensorOwner(Tensor* tensor);

    Tensor* GetInputTensor(const std::string& name);
    Tensor* GetOutputTensor(const std::string& name);

    bool Replace(Subgraph* orig_sub, Subgraph* new_sb);

    void AddTensorMap(const std::string& tensor_name, Tensor* tensor);
    Graph* GetViewCopy(void);

    bool NodeInGraph(Node*);

    void SetModelFormat(int model_format)
    {
        model_format_ = model_format;
    }

    int GetModelFormat(void)
    {
        return model_format_;
    }

    void SetModelSubFormat(int model_subformat)
    {
        model_subformat_ = model_subformat;
    }

    int GetModelSubFormat(void)
    {
        return model_subformat_;
    }

    void SetModelLayout(int model_layout)
    {
        model_layout_ = model_layout;
    }

    int GetModelLayout(void)
    {
        return model_layout_;
    }

    void SetLayout(int layout)
    {
        layout_ = layout;
    }
    int GetLayout(void)
    {
        return layout_;
    }
    void Lock(void)
    {
        graph_lock_.lock();
    }
    void Unlock(void)
    {
        graph_lock_.unlock();
    }

protected:
    std::string name_;
    std::string type_;

    std::vector<Node*> owned_nodes_;
    std::unordered_map<std::string, Tensor*> owned_tensors_;

    int model_format_;
    int model_subformat_;
    int model_layout_;
    int layout_;

    Attribute attrs_;

    std::unordered_map<std::string, Tensor*> tensor_map_;
    StaticGraphPtr orig_graph_;
    std::mutex graph_lock_;
};

}    // namespace TEngine

#endif
