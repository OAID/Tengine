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
#ifndef __STATIC_GRAPH_INTERFACE_HPP__
#define __STATIC_GRAPH_INTERFACE_HPP__

#include <string>
#include "attribute.hpp"

namespace TEngine {

struct StaticGraph;
struct StaticNode;
struct StaticTensor;
struct StaticOp;


//static graph level

StaticGraph * CreateStaticGraph(const std::string& name);
void DestroyStaticGraph(StaticGraph *);
void DumpStaticGraph(StaticGraph * graph);

//TODO: not available to user
void SetGraphInternalName(StaticGraph * graph, const std::string& name);

void SetGraphIdentity(StaticGraph * graph, const std::string& domain, const std::string&name, const std::string& version);

void SetGraphSource(StaticGraph * graph, const std::string& source);
void SetGraphSourceFormat(StaticGraph * graph, const std::string& format);

void SetGraphConstTensorFile(StaticGraph * graph, const std::string& fname);

//if attr_name exist, return false
bool AddGraphAttr(StaticGraph * graph, const std::string& attr_name, any&& value);



StaticNode * FindNode(StaticGraph * graph, const std::string& node_name);
StaticTensor * FindTensor(StaticGraph * graph, const std::string& tensor_name);
StaticTensor * FindConstTensor(StaticGraph * graph, const std::string& tensor_name);


/* those may not need to expose to serializer */
void AddGraphInputNode(StaticGraph * graph,StaticNode * node);
void AddGraphOutputNode(StaticGraph * graph, StaticNode * node);
bool CheckGraphIntegraity (StaticGraph *graph); 


//StaticNode
StaticNode * CreateStaticNode(StaticGraph * graph, const std::string& node_name);
int AddNodeInputTensor(StaticNode * node, StaticTensor * tensor);
int AddNodeOutputTensor(StaticNode * node, StaticTensor * tensor);
void SetNodeOp(StaticNode * node, StaticOp* op);
StaticOp * GetNodeOp(StaticNode * node);
const std::string& GetNodeName(StaticNode * node);
StaticTensor * GetNodeOutputTensor(StaticGraph * graph, StaticNode * node, int idx);

//StaticOp
StaticOp* CreateStaticOp(StaticGraph * graph, const std::string& op_name);
void SetOperatorParam(StaticOp*, any&& param);
void SetOperatorDynamicShape(StaticOp*);
void AddOperatorAttr(StaticOp*, const std::string& attr_name, any&& val);
any& GetOperatorParam(StaticOp *);

//StaticTensor

StaticTensor * CreateStaticTensor(StaticGraph * grap, const std::string& name);

void  SetTensorDim(StaticTensor * , const std::vector<int>& dims);
const std::vector<int>& GetTensorDim(StaticTensor *);
void  SetTensorDataType(StaticTensor *, const std::string& data_type);
void  SetTensorDataLayout(StaticTensor *, const std::string& data_layout);
void  SetTensorType(StaticTensor *, int type); 
int   SetTensorSize(StaticTensor *, int size);

void  SetTensorProducer(StaticTensor *, StaticNode * , int idx);
void  AddTensorConsumer(StaticTensor *, StaticNode *, int idx);

const std::string& GetTensorName(StaticTensor * tensor);

StaticNode*  GetTensorProducer(StaticGraph * graph, StaticTensor * tensor);
    
StaticTensor* CreateStaticConstTensor(StaticGraph * grap, const std::string& name);
void SetConstTensorBuffer(StaticTensor * tensor, void * addr);
void * GetConstTensorBuffer(StaticTensor * tensor);
void SetConstTensorFileLocation(StaticTensor * tensor, int offset, int file_size);


} //namespace TEngine

#endif
