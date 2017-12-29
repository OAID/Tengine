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
#ifndef __EXECUTOR_HPP__
#define __EXECUTOR_HPP__

#include <functional>
#include <unordered_map>


namespace TEngine {

class Node;
class ExecEngine;

#define ATTR_INPLACE "inplace"

using inplace_t=std::unordered_map<int,int>;
using run_node_t=std::function<bool(Node *,ExecEngine *)>;

struct NodeExec {
  run_node_t on_bind;
  run_node_t pre_run;
  run_node_t run;
  run_node_t post_run;
};


bool GetNodeExec(const std::string& name, NodeExec& node_exec);

bool RegisterNodeExec(const std::string& name, const NodeExec& node_exec);

bool PrerunNode(Node *, ExecEngine *);
bool RunNode(Node *, ExecEngine *);
bool PostrunNode(Node *, ExecEngine *);


} //namespace TEngine

#endif
