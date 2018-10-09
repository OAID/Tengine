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
#ifndef __NODE_OPS_HPP__
#define __NODE_OPS_HPP__

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpu_info.hpp"

namespace TEngine {

#define ATTR_NODE_OPS "node_ops"
#define ATTR_INPLACE "inplace"

class Node;
struct NodeOps;
struct sub_op_task;

using inplace_t = std::unordered_map<int, int>;

using task_exec_t = std::function<bool(int cpu, int seq, void *data)>;
using task_dispatch_t =
    std::function<bool(std::vector<sub_op_task> &tasks, int cpu)>;
using wait_done_t = std::function<void(void)>;

using mem_alloc_t = std::function<void *(int size)>;
using mem_free_t = std::function<void(void *)>;
using select_node_ops_t = std::function<NodeOps *(const CPUInfo *, Node *)>;
using node_ops_create_t = std::function<NodeOps *(const CPUInfo *, Node *)>;

struct sub_op_task {
  task_exec_t exec_func;
  int seq;
  void *data;
};

struct NodeOps {
  virtual bool OnBind(Node *) { return true; }
  virtual bool OnUnbind(Node *) { return true; }
  virtual bool Prerun(Node *) { return true; }
  virtual bool Postrun(Node *) { return true; }
  virtual bool Run(Node *) = 0;
  virtual bool GetSharedMemorySize(Node *, unsigned int &mem_size) {
    return false;
  }
  virtual bool SetSharedMemoryAddr(Node *, void *mem_addr, int mem_size) {
    return false;
  }
  virtual bool GetPrivateMemorySize(Node *, unsigned int &mem_size) {
    return false;
  }
  virtual bool SetPrivateMemoryAddr(Node *, void *mem_addr, int mem_size) {
    return false;
  }

  virtual bool DynPrerun(Node *) {
    return true;
  }  // used in dynamic cases: will be called before run

  /* note: the mem_addr will be released by caller */

  NodeOps(void) { need_free = false; }

  /* for delete this usage:
   * https://isocpp.org/wiki/faq/freestore-mgmt#delete-this */
  void Release(void) {
    if (need_free) delete this;
  }

  void SetHelper(mem_alloc_t alloc, mem_free_t free, task_dispatch_t disp,
                 wait_done_t wait) {
    mem_alloc = alloc;
    mem_free = free;
    task_dispatch = disp;
    wait_done = wait;
  }

  void SetCPUInfo(const CPUInfo *cpu) { cpu_info = cpu; }

  virtual ~NodeOps() {}

  bool need_free;
  mem_alloc_t mem_alloc;
  mem_free_t mem_free;
  task_dispatch_t task_dispatch;
  wait_done_t wait_done;

  const CPUInfo *cpu_info;
};

using MTNodeOps = NodeOps;

struct NodeOpsSelector {
  virtual NodeOps *Select(const CPUInfo *cpu_info, Node *node) = 0;
  virtual ~NodeOpsSelector(){};

  std::string op_name;
};

struct PrioSelector : public NodeOpsSelector {
  NodeOps *Select(const CPUInfo *cpu_info, Node *node) {
    auto begin = prio_list.begin();
    auto end = prio_list.end();

    for (auto ir = begin; ir != end; ir++) {
      auto match_func = ir->second;

      auto ops = match_func(cpu_info, node);

      if (ops) return ops;
    }

    return nullptr;
  }

  void Register(int priority, select_node_ops_t func) {
    prio_list[priority] = func;
  }

  std::map<int, select_node_ops_t> prio_list;
};

using NodeOpsSelectorPtr = std::shared_ptr<NodeOpsSelector>;

struct NodeOpsRegistry {
  NodeOpsRegistry(const std::string &name) { reg_name = name; }

  NodeOps *FindNodeOps(const CPUInfo *, Node *);

  NodeOpsSelector *FindSelector(const std::string &name);

  bool RegisterSelector(NodeOpsSelector *selector);

  std::unordered_map<std::string, NodeOpsSelectorPtr> registry;
  std::string reg_name;
};

#define REF_REGISTRY_NAME "reference"

class NodeOpsRegistryManager {
 public:
  using NodeOpsPtr = std::shared_ptr<NodeOps>;
  ~NodeOpsRegistryManager();

  static void RecordNodeOpsptr(NodeOps *ops);

  static NodeOps *RealFindNodeOps(const CPUInfo *, Node *);
  static NodeOps *FindNodeOps(const CPUInfo *, Node *);
  static NodeOps *FindNodeOps(const std::string &registry_name, const CPUInfo *,
                              Node *);

  static NodeOpsRegistryManager *GetInstance(void);

  static void AddRegistry(const std::string &name, NodeOpsRegistry *reg);
  static NodeOpsRegistry *FindRegistry(const std::string &name);

  template <typename T>
  static NodeOps *simple_select_function(T *ops, const CPUInfo *info,
                                         Node *node) {
    NodeOps *new_ops = new T(*ops);
    new_ops->need_free = true;

    return new_ops;
  }

  template <typename T>
  static bool RegisterOPImplementor(const std::string &registry_name,
                                    const std::string &op_name, T *ops) {
    auto f = std::bind(simple_select_function<T>, ops, std::placeholders::_1,
                       std::placeholders::_2);

    RecordNodeOpsptr(ops);

    return RegisterOPImplementor(registry_name, op_name, f, 1000);
  }

  static bool RegisterOPImplementor(const std::string &registry_name,
                                    const std::string &op_name,
                                    select_node_ops_t selec_func, int priority);

  std::unordered_map<std::string, NodeOpsRegistry *> registry_list;
  std::vector<NodeOpsPtr> ops_list;
};

}  // namespace TEngine

#endif
