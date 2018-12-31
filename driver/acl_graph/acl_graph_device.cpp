
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

static void copy_itensor(CLTensor* cl_tensor, void* buf, int buf_size, bool to_tensor, DataType data_type)
{
    auto* cl_info = cl_tensor->info();

    const size_t slice_num = cl_info->tensor_shape().total_size_upper(2);
    const Strides strides = cl_info->strides_in_bytes();
    const PaddingSize padding = cl_info->padding();

    int slice_w = cl_info->dimension(0) + padding.left + padding.right;
    int slice_h = cl_info->dimension(1) + padding.bottom + padding.top;

    uint8_t* slice_ptr = cl_tensor->buffer();
    uint8_t* buf_ptr = ( uint8_t* )buf;

    for(unsigned int i = 0; i < slice_num; i++)
    {
        uint8_t* data_ptr = slice_ptr + padding.top * strides[1] + padding.left * strides[0];
        for(unsigned int h = 0; h < cl_info->dimension(1); h++)
        {
            int data_len = cl_info->dimension(0) * strides[0];
            int buf_len = data_len;

            if(data_type == DataType::F16)
                buf_len = data_len << 1;

            if(to_tensor)
            {
                copy_buffer(data_ptr, buf_ptr, buf_len, data_type, DataType::F32);
            }
            else
            {
                copy_buffer(buf_ptr, data_ptr, data_len, DataType::F32, data_type);
            }

            buf_ptr = buf_ptr + buf_len;

            data_ptr += slice_w * strides[0];
        }

        slice_ptr += slice_h * slice_w * strides[0];
    }
}
static void copy_to_itensor(CLTensor* cl_tensor, const void* buf, int buf_size, DataType tensor_dt)
{
    copy_itensor(cl_tensor, ( void* )buf, buf_size, true, tensor_dt);
}
void copy_from_itensor(const CLTensor* cl_tensor, void* buf, int buf_size, DataType tensor_dt)
{
    copy_itensor(( CLTensor* )cl_tensor, buf, buf_size, false, tensor_dt);
}

static CLGraph* CreateACLGraph(Subgraph* graph, DataType type)
{
    CLScheduler::get().default_init();
    CLGraph* acl_graph = new CLGraph(graph->GetName(), type);

    /* first, process input nodes' input tensor */
    int input_size = graph->input_nodes.size();

    for(int i = 0; i < input_size; i++)
    {
        Node* node = graph->input_nodes[i];

        for(unsigned j = 0; j < node->GetInputNum(); j++)
        {
            Tensor* tensor = node->GetInputTensor(j);

            if(tensor->GetType() != kConstTensor)
            {
                CLTensor* itensor = new CLTensor();
                const std::vector<int>& dims = tensor->GetShape().GetDim();
                const std::string& name = tensor->GetName();

                int dim_size = dims.size();

                if(dim_size == 4)
                {
                    itensor->allocator()->init(TensorInfo(TensorShape(dims[3], dims[2], dims[1], dims[0]), 1, type));
                }
                else if(dim_size == 3)
                {
                    itensor->allocator()->init(TensorInfo(TensorShape(dims[2], dims[1], dims[0], 1), 1, type));
                }
                else if(dim_size == 2)
                {
                    itensor->allocator()->init(TensorInfo(TensorShape(dims[1], dims[0], 1, 1), 1, type));
                }
                else if(dim_size == 1)
                {
                    itensor->allocator()->init(TensorInfo(TensorShape(dims[0], 1, 1, 1), 1, type));
                }
                else
                {
                    XLOG_ERROR() << "Bad shape dim: " << dim_size << "\n";
                }

                acl_graph->tensors_map_[name] = itensor;
            }
        }
    }

    /* now, let's scan all nodes! */

    int node_size = graph->seq_nodes.size();
    int i = 0;
    for(i = 0; i < node_size; i++)
    {
        bool ret = false;
        Node* node = graph->seq_nodes[i];
        Operator* op = node->GetOp();
        std::string name = op->GetName();
        if(name == "Const")
            continue;
        // std::cout<<i<<" node name: "<<node->GetName()<<" ,op name: "<<name <<"\n";

        if(name == "Input")
        {
            ret = acl_graph->AddInputLayer(node);
        }
        else if(name == "BatchNormalization")
        {
            Node* node_next = graph->seq_nodes[++i];
            if(node_next->GetOp()->GetName() != "Scale")
                ret = false;
            else
                ret = acl_graph->AddBNLayer(node, node_next);
        }
        else if(name == "Concat")
        {
            ret = acl_graph->AddConcatLayer(node);
        }
        else if(name == "Convolution")
        {
            ret = acl_graph->AddConvolutionLayer(node);
        }
        else if(name == "Dropout")
        {
            ret = acl_graph->AddDropoutLayer(node);
        }
        else if(name == "Eltwise")
        {
            ret = acl_graph->AddEltwiseLayer(node);
        }
        else if(name == "FullyConnected")
        {
            ret = acl_graph->AddFCLayer(node);
        }
        else if(name == "Pooling")
        {
            ret = acl_graph->AddPoolingLayer(node);
        }
        else if(name == "ReLu")
        {
            ret = acl_graph->AddReLuLayer(node);
        }
        else if(name == "ReLu6")
        {
            ret = acl_graph->AddReLu6Layer(node);
        }
        else if(name == "Resize")
        {
            ret = acl_graph->AddResizeLayer(node);
        }
        else if(name == "Softmax")
        {
            ret = acl_graph->AddSoftmaxLayer(node);
        }
        if(!ret)
        {
            LOG_INFO() << "Create ACL for Op " << name << " failed! \n";
            return nullptr;
        }
    }

    return acl_graph;
}

static void DestroyACLGraph(CLGraph* graph)
{
    auto ir_start = graph->tensors_map_.begin();
    auto ir_end = graph->tensors_map_.end();

    std::set<CLTensor*> ptr_set;

    for(auto ir = ir_start; ir != ir_end; ir++)
    {
        CLTensor* tensor = ir->second;
        // if(!tensor->allocator()->info().is_resizable())

        if(!ptr_set.count(tensor))
        {
            ptr_set.insert(tensor);
            tensor->allocator()->free();
            delete tensor;
        }
    }

    for(auto e : graph->functions_map_)
        delete e;

    delete graph;
}

bool ACLDevice::RealOptimizeGraph(DevContext* context, Subgraph* graph)
{
    context->optimized_graph = graph;

    GraphOptimizerManager::RunOpt("BNScale", context->optimized_graph);
    GraphOptimizerManager::RunOpt("ConvBN", context->optimized_graph);
#ifdef ACL_EXTENSTION
    GraphOptimizerManager::RunOpt("ConvReLu", context->optimized_graph);
    GraphOptimizerManager::RunOpt("ConvReLu6", context->optimized_graph);
#endif
    return true;
}

bool ACLDevice::RealPrerun(DevContext* context)
{
    char* env = std::getenv("ACL_FP16");
    if(env)
        data_type_ = DataType::F16;
    else
        data_type_ = DataType::F32;

    CLGraph* graph = CreateACLGraph(context->sub_graph, data_type_);
    context->graph = graph;

    if(graph == nullptr)
        return false;

    auto ir_start = graph->tensors_map_.begin();
    auto ir_end = graph->tensors_map_.end();

    for(auto ir = ir_start; ir != ir_end; ir++)
    {
        CLTensor* tensor = ir->second;
        std::string name = ir->first;
        /*
        if(name.find("weight")!= std::string::npos ||
            name.find("gamma") != std::string::npos ||
            name.find("beta") != std::string::npos ||
            name.find("means") != std::string::npos ||
            name.find("vars") != std::string::npos ||
            name.find("bias") != std::string::npos ||
            name.find("data") != std::string::npos  )
            continue;
        */

        if(tensor->allocator()->info().is_resizable())
            tensor->allocator()->allocate();
    }

    int output_node_size = context->sub_graph->output_nodes.size();
    for(int i = 0; i < output_node_size; i++)
    {
        Node* node = context->sub_graph->output_nodes[i];
        Tensor* output = node->GetOutputTensor(0);
        void* mem_addr = get_tensor_mem(output);
        if(mem_addr)
            continue;
        else
        {
            void* addr = std::malloc(output->GetTotalSize());
            set_tensor_mem(output, addr, output->GetTotalSize(), std::free);
        }
    }

    return true;
}

bool ACLDevice::RealSyncRun(DevContext* context)
{
    return true;
}

bool ACLDevice::RealRun(DevContext* context)
{
    CLGraph* graph = context->graph;

    int input_number = context->sub_graph->input_nodes.size();

    for(int i = 0; i < input_number; i++)
    {
        Node* node = context->sub_graph->input_nodes[i];

        if(node->GetInputNum())
        {
            /* only tensor input */

            for(unsigned int i = 0; i < node->GetInputNum(); i++)
            {
                Tensor* tensor_input = node->GetInputTensor(i);

                if(tensor_input->GetType() == kConstTensor)
                    continue;

                CLTensor* acl_input = graph->GetCLTensor(tensor_input->GetName());

                acl_input->map();
                void* buf = get_tensor_mem(tensor_input);
                int size = tensor_input->GetTotalSize();

                copy_to_itensor(acl_input, buf, size, data_type_);
                acl_input->unmap();
            }
        }
        else
        {
            /* normal Input Node */

            Tensor* out = node->GetOutputTensor(0);
            CLTensor* acl_input = graph->GetCLTensor(out->GetName());

            acl_input->map();
            void* buf = get_tensor_mem(out);
            int size = out->GetTotalSize();

            copy_to_itensor(acl_input, buf, size, data_type_);
            acl_input->unmap();
        }
    }

    graph->Run();

//#define DUMP_TENSOR
#ifdef DUMP_TENSOR
    int node_size = context->sub_graph->seq_nodes.size();

    for(int i = 0; i < node_size; i++)
    {
        Node* node = context->sub_graph->seq_nodes[i];
        Operator* op = node->GetOp();
        std::string name = op->GetName();
        // Tensor *ooo = node->GetOutputTensor(0);
        Tensor* ooo = node->GetInputTensor(0);
        // uint8_t* acl_buf = reinterpret_cast<uint8_t*>(cltensor->buffer());
        if(name != "Const" && name != "Input")
        {
            CLTensor* cltensor = graph->GetCLTensor(ooo->GetName());
            cltensor->map();
            float* save32 = nullptr;
            int size = cltensor->info()->total_size();

            int real_size = ooo->GetTotalSize();

            void* real_data = malloc(real_size);

            copy_from_itensor(cltensor, real_data, real_size, data_type_);

            size = real_size >> 2;
            save32 = ( float* )real_data;

            auto* op = node->GetOp();

            std::cout << " out: " << ooo->GetName() << " op: " << op->GetName() << " out_size:" << size << "\n";
            const char* name_s = ooo->GetName().c_str();
            char name[100] = {0};
            for(unsigned int i = 0; i < strlen(name_s); i++)
            {
                if(name_s[i] == '/')
                    name[i] = '_';
                else
                    name[i] = name_s[i];
            }
            std::string fname = "/tmp/dump/";

            fname += name;

            FILE* pf = fopen(fname.c_str(), "w");
            for(int j = 0; j < size; j++)
            {
                if(j % 16 == 0)
                    fprintf(pf, "\n[%d]:", j);

                fprintf(pf, "%g,", save32[j]);
            }
            fclose(pf);
            cltensor->unmap();
        }
    }
#endif

    int output_num = context->sub_graph->output_nodes.size();
    for(int i = 0; i < output_num; i++)
    {
        Node* node = context->sub_graph->output_nodes[i];
        Tensor* output = node->GetOutputTensor(0);
        std::string output_name = output->GetName();
        CLTensor* cltensor = graph->GetCLTensor(output_name);

        void* output_buf = get_tensor_mem(output);
        int out_size = output->GetTotalSize();

        cltensor->map();
        copy_from_itensor(cltensor, output_buf, out_size, data_type_);
        cltensor->unmap();
    }

    return true;
}

bool ACLDevice::RealPostrun(DevContext* context)
{
    CLGraph* graph = context->graph;

    DestroyACLGraph(graph);

    return true;
}

void ACLDevice::RunGraph(DevContext* context, dev_graph_cb_t graph_cb)
{
    bool ret = RealRun(context);

    if(graph_cb)
        graph_cb(context->optimized_graph, ret);
}

void ACLDevice::Process(const acl_task& task, int acl_id)
{
    RunGraph(task.context, task.context->graph_cb);
}

void ACLDevice::Launch(void)
{
    auto f = std::bind(&ACLDevice::Process, this, std::placeholders::_1, std::placeholders::_2);

    thread_ = new WorkerThread<acl_task>(f);

    thread_->SetQueue(&task_queue_, &queue_lock_, &queue_cv_);

    thread_->LaunchWorker();
}

void ACLDevice::IncRequest(int req_number)
{
    request_ += req_number;
}

void ACLDevice::IncDone(int done_number)
{
    uint64_t prev_val = done_.fetch_add(done_number);

    if(prev_val + done_number == request_)
    {
        std::unique_lock<std::mutex> lock(wait_mutex_);

        wait_cv_.notify_all();

        lock.unlock();
    }
}

void ACLDevice::PushTask(std::vector<acl_task>& task_list)
{
    thread_->PushTask(task_list);
}

void ACLDevice::WaitDone(void)
{
    std::unique_lock<std::mutex> lock(wait_mutex_);

    if(done_ != request_)
        wait_cv_.wait(lock, [this] { return done_ == request_; });

    lock.unlock();
}

void ACLDevice::Kill(void)
{
    if(thread_)
    {
        delete thread_;
        thread_ = nullptr;
    }
}

}    // namespace TEngine
