
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
 * Author: haoluo@openailab.com
 */

#include "acl_graph_device.hpp"

using namespace arm_compute;

namespace TEngine {

static CLGraph* CreateACLGraph(Subgraph* graph, DataType type) {
  CLScheduler::get().default_init();
  CLGraph* acl_graph = new CLGraph(graph->GetName(), type);

  int node_size = graph->seq_nodes.size();
  int i = 0;
  for (i = 0; i < node_size; i++) {
    bool ret = false;
    Node* node = graph->seq_nodes[i];
    Operator* op = node->GetOp();
    std::string name = op->GetName();
    if (name == "Const") continue;
    // std::cout<<i<<" node name: "<<node->GetName()<<" ,op name: "<<name
    // <<"\n";

    if (name == "Input") {
      ret = acl_graph->AddInputLayer(node);
    } else if (name == "BatchNormalization") {
      Node* node_next = graph->seq_nodes[++i];
      if (node_next->GetOp()->GetName() != "Scale")
        ret = false;
      else
        ret = acl_graph->AddBNLayer(node, node_next);
    } else if (name == "Concat") {
      ret = acl_graph->AddConcatLayer(node);
    } else if (name == "Convolution") {
      ret = acl_graph->AddConvolutionLayer(node);
    } else if (name == "Dropout") {
      ret = acl_graph->AddDropoutLayer(node);
    } else if (name == "Eltwise") {
      ret = acl_graph->AddEltwiseLayer(node);
    } else if (name == "FullyConnected") {
      ret = acl_graph->AddFCLayer(node);
    } else if (name == "Pooling") {
      ret = acl_graph->AddPoolingLayer(node);
    } else if (name == "ReLu") {
      ret = acl_graph->AddReLuLayer(node);
    } else if (name == "ReLu6") {
      ret = acl_graph->AddReLu6Layer(node);
    } else if (name == "Resize") {
      ret = acl_graph->AddResizeLayer(node);
    } else if (name == "Softmax") {
      ret = acl_graph->AddSoftmaxLayer(node);
    }
    if (!ret) {
      LOG_INFO() << "Create ACL for Op " << name << " failed! \n";
      return nullptr;
    }
  }

  return acl_graph;
}

bool ACLDevice::RealOptimizeGraph(DevContext* context, Subgraph* graph) {
  context->optimized_graph = graph;

  GraphOptimizerManager::RunOpt("BNScale", context->optimized_graph);
  GraphOptimizerManager::RunOpt("ConvBN", context->optimized_graph);
#ifdef ACL_EXTENSTION
  GraphOptimizerManager::RunOpt("ConvReLu", context->optimized_graph);
  GraphOptimizerManager::RunOpt("ConvReLu6", context->optimized_graph);
#endif
  return true;
}

bool ACLDevice::RealPrerun(DevContext* context) {
  char* env = std::getenv("ACL_FP16");
  if (env)
    data_type_ = DataType::F16;
  else
    data_type_ = DataType::F32;

  CLGraph* graph = CreateACLGraph(context->sub_graph, data_type_);
  context->graph = graph;

  if (graph == nullptr) return false;

  auto ir_start = graph->tensors_map_.begin();
  auto ir_end = graph->tensors_map_.end();

  for (auto ir = ir_start; ir != ir_end; ir++) {
    CLTensor* tensor = ir->second;
    std::string name = ir->first;
    if (name.find("weight") != std::string::npos ||
        name.find("gamma") != std::string::npos ||
        name.find("beta") != std::string::npos ||
        name.find("means") != std::string::npos ||
        name.find("vars") != std::string::npos ||
        name.find("bias") != std::string::npos ||
        name.find("data") != std::string::npos)
      continue;
    if (tensor->allocator()->info().is_resizable())
      tensor->allocator()->allocate();
  }

  int output_node_size = context->sub_graph->output_nodes.size();
  for (int i = 0; i < output_node_size; i++) {
    Node* node = context->sub_graph->output_nodes[i];
    Tensor* output = node->GetOutputTensor(0);
    void* mem_addr = get_tensor_mem(output);
    if (mem_addr)
      continue;
    else {
      void* addr = std::malloc(output->GetTotalSize());
      set_tensor_mem(output, addr, output->GetTotalSize(), nullptr);
    }
  }

  return true;
}

bool ACLDevice::RealSyncRun(DevContext* context) { return true; }

bool ACLDevice::RealRun(DevContext* context) {
  CLGraph* graph = context->graph;

  Node* node = context->sub_graph->input_nodes[0];
  Tensor* out = node->GetOutputTensor(0);
#ifdef USE_CPU_CONVERT
  CLTensor* acl_input = graph->GetCLTensor(out->GetName());
#else
  CLTensor* acl_input = graph->GetCLTensor("start");
#endif
  acl_input->map();
  void* buf = get_tensor_mem(out);
  int size = out->GetTotalSize();
#ifdef USE_CPU_CONVERT
  copy_buffer(acl_input->buffer(), buf, size, data_type_, DataType::F32);
#else
  memcpy(acl_input->buffer(), buf, size);
#endif
  acl_input->unmap();

  graph->Run();

//#define DUMP_TENSOR
#ifdef DUMP_TENSOR
  int node_size = context->sub_graph->seq_nodes.size();

  for (int i = 0; i < node_size; i++) {
    Node* node = context->sub_graph->seq_nodes[i];
    Operator* op = node->GetOp();
    std::string name = op->GetName();
    Tensor* ooo = node->GetOutputTensor(0);
    CLTensor* cltensor = graph->GetCLTensor(ooo->GetName());
    cltensor->map();
    uint8_t* acl_buf = reinterpret_cast<uint8_t*>(cltensor->buffer());
    if (name != "Const" && name != "Input") {
      float* save32 = nullptr;
      __fp16* save16 = nullptr;
      int size = cltensor->info()->total_size();
      if (data_type_ == DataType::F16) {
        save16 = (__fp16*)acl_buf;
        size = size >> 1;
      } else {
        save32 = (float*)acl_buf;
        size = size >> 2;
      }
      std::cout << " out: " << node->GetName() << " ,out_size:" << size << "\n";
      const char* name_s = node->GetName().c_str();
      char name[100] = {0};
      for (unsigned int i = 0; i < strlen(name_s); i++) {
        if (name_s[i] == '/')
          name[i] = '_';
        else
          name[i] = name_s[i];
      }
      FILE* pf = fopen(name, "w");
      for (int j = 0; j < size; j++) {
        if (j % 16 == 0) fprintf(pf, "\n[%d]:", j);
        if (data_type_ == DataType::F16)
          fprintf(pf, "%g,", save16[j]);
        else
          fprintf(pf, "%g,", save32[j]);
      }
      fclose(pf);
    }

    cltensor->unmap();
  }
#endif

  int output_num = context->sub_graph->output_nodes.size();
  for (int i = 0; i < output_num; i++) {
    node = context->sub_graph->output_nodes[i];
    Tensor* output = node->GetOutputTensor(0);
    std::string output_name = output->GetName();
    CLTensor* cltensor = graph->GetCLTensor(output_name);
    if (data_type_ == DataType::F16) {
      int out_size = (output->GetTotalSize()) >> 1;
      cltensor->map();
      copy_buffer(get_tensor_mem(output), cltensor->buffer(), out_size,
                  DataType::F32, DataType::F16);
      cltensor->unmap();
    } else {
      int out_size = output->GetTotalSize() >> 2;
      float* output_buf = (float*)get_tensor_mem(output);
      cl::copy<float*>(cltensor->cl_buffer(), output_buf,
                       output_buf + out_size);
    }
  }

  return true;
}

bool ACLDevice::RealPostrun(DevContext* context) {
  CLGraph* graph = context->graph;
  auto ir_start = graph->tensors_map_.begin();
  auto ir_end = graph->tensors_map_.end();

  for (auto ir = ir_start; ir != ir_end; ir++) {
    CLTensor* tensor = ir->second;
    if (!tensor->allocator()->info().is_resizable())
      tensor->allocator()->free();
  }
  return true;
}

void ACLDevice::RunGraph(DevContext* context, dev_graph_cb_t graph_cb) {
  bool ret = RealRun(context);

  if (graph_cb) graph_cb(context->optimized_graph, ret);
}

void ACLDevice::Process(const acl_task& task, int acl_id) {
  RunGraph(task.context, task.context->graph_cb);
}

void ACLDevice::Launch(void) {
  auto f = std::bind(&ACLDevice::Process, this, std::placeholders::_1,
                     std::placeholders::_2);

  thread_ = new WorkerThread<acl_task>(f);

  thread_->SetQueue(&task_queue_, &queue_lock_, &queue_cv_);

  thread_->LaunchWorker();
}

void ACLDevice::IncRequest(int req_number) { request_ += req_number; }

void ACLDevice::IncDone(int done_number) {
  uint64_t prev_val = done_.fetch_add(done_number);

  if (prev_val + done_number == request_) {
    std::unique_lock<std::mutex> lock(wait_mutex_);

    wait_cv_.notify_all();

    lock.unlock();
  }
}

void ACLDevice::PushTask(std::vector<acl_task>& task_list) {
  thread_->PushTask(task_list);
}

void ACLDevice::WaitDone(void) {
  std::unique_lock<std::mutex> lock(wait_mutex_);

  if (done_ != request_)
    wait_cv_.wait(lock, [this] { return done_ == request_; });

  lock.unlock();
}

void ACLDevice::Kill(void) {
  if (thread_) {
    delete thread_;
    thread_ = nullptr;
  }
}

}  // namespace TEngine
