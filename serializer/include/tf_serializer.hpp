
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

#include <iostream>
#include <functional>
#include <unordered_map>
#include <cstring>


#include "graph.pb.h"
#include "static_graph_interface.hpp"
#include "logger.hpp"
#include "serializer.hpp"


namespace TEngine {

struct TFNode {
   int idx;
   std::string name;
   std::string op;
   std::vector<TFNode *> inputs;
   std::vector<TFNode *> outputs;
   std::vector<const tensorflow::NodeDef *> pb_defs;
   StaticNode * static_node;
   StaticTensor * static_tensor;
   bool no_static_node;

   TFNode() { no_static_node=false;}

};

struct TFGraph {
   std::vector<TFNode *> seq_nodes;

   ~TFGraph() {
     for(auto node: seq_nodes)
            delete node;
   }
};

class TFSerializer: public Serializer {

public:
   bool LoadModel(const std::vector<std::string>& file_list, StaticGraph * graph) override;
   unsigned int GetFileNum(void) override { return 1;}
   bool LoadConstTensor(const std::string& fname, StaticTensor * const_tensor) override { return false;}
   bool LoadConstTensor(int fd, StaticTensor * const_tensor) override { return false;}

protected:

   bool LoadGraph(tensorflow::GraphDef& tf_net,StaticGraph * graph);
   bool LoadBinaryFile(const char * fname, tensorflow::GraphDef& tf_net);
   bool LoadTextFile(const char * fname, tensorflow::GraphDef& tf_net);
   bool ConstructGraph(tensorflow::GraphDef& tf_net, TFGraph& tf_graph);
   bool OptimizeGraph(TFGraph& tf_graph);
   bool GenerateStaticGraph(TFGraph& tf_graph, StaticGraph * graph);
   void CleanupResizeNearestNeighbor(TFGraph& tf_graph);
   void MergeReluMinimum(TFGraph & tf_graph);

   bool MergeChildNode(TFNode * base_node, TFNode * child_node);
   bool MergeParentNode(TFNode * base_node, TFNode * parent_node);
   void BNRecursiveInputMerge(TFNode * node);
   void FuseComposedBN(TFNode * cur_node);
   bool CheckComposedBNAdd(TFNode * node);


   void DisconnectNode(TFNode * node);

   void DumpTFGraph(TFGraph& tf_graph);

};

} //namespace TEngine

#endif
