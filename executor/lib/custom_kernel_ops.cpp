#include <assert.h>

#include "logger.hpp"
#include "graph.hpp"
#include "tensor_mem.hpp"
#include "tengine_errno.hpp"
#include "node_ops.hpp"
#include "custom_kernel_ops.hpp"

namespace TEngine {

CustomKernelNodeOps::~CustomKernelNodeOps(void)
{
    if(k_ops_->release != nullptr)
        k_ops_->release(k_ops_);
}

bool CustomKernelNodeOps::OnBind(Node* node)
{
    if(!k_ops_->inplace_info)
        return true;

    int output_num = node->GetOutputNum();

    inplace_t io_map;

    for(int i = 0; i < output_num; i++)
    {
        int in_idx = k_ops_->inplace_info(k_ops_, i);

        if(in_idx < 0)
            continue;

        io_map[i] = in_idx;
    }

    return true;
}

bool CustomKernelNodeOps::Prerun(Node* node)
{
    if(k_ops_->prerun == nullptr)
        return true;

    assert(input_num_ == 0 && output_num_ == 0);
    assert(k_inputs_ == nullptr && k_outputs_ == nullptr);

    if(!PrepareTensors(node))
        return false;

    int dynamic_shape = node->IsDynamicShape() ? 1 : 0;

    int ret = k_ops_->prerun(k_ops_, k_inputs_, input_num_, k_outputs_, output_num_, dynamic_shape);

    if(ret < 0)
    {
        LOG_ERROR() << "custom kernel prerun failed on node: " << node->GetName() << "\n";
        set_tengine_errno(EFAULT);    // external error
        return false;
    }

    return true;
}

bool CustomKernelNodeOps::Reshape(Node* node)
{
    if(k_ops_->reshape == nullptr)
        return true;

    ReleaseTensors(node);

    if(!PrepareTensors(node))
        return false;

    int ret = k_ops_->reshape(k_ops_, k_inputs_, input_num_, k_outputs_, output_num_);

    if(ret < 0)
    {
        LOG_ERROR() << "custom kernel reshape failed on node: " << node->GetName() << "\n";
        set_tengine_errno(EFAULT);    // external error
        return false;
    }

    return true;
}

bool CustomKernelNodeOps::Run(Node* node)
{
    /* set mem in tensor */
    for(int i = 0; i < input_num_; i++)
    {
        Tensor* tensor = node->GetInputTensor(i);
        struct custom_kernel_tensor* t = k_inputs_[i];

        t->data = get_tensor_mem(tensor);
    }

    int ret = k_ops_->run(k_ops_, k_inputs_, input_num_, k_outputs_, output_num_);

    if(ret < 0)
    {
        LOG_ERROR() << "custom kernel run failed on node: " << node->GetName() << "\n";
        set_tengine_errno(EFAULT);    // external error
        return false;
    }

    return true;
}

bool CustomKernelNodeOps::Postrun(Node* node)
{
    if(k_ops_->postrun == nullptr)
        return true;

    int ret = k_ops_->postrun(k_ops_, k_inputs_, input_num_, k_outputs_, output_num_);

    ReleaseTensors(node);

    if(ret < 0)
    {
        LOG_ERROR() << "custom kernel postrun failed on node: " << node->GetName() << "\n";
        set_tengine_errno(EFAULT);    // external error
        return false;
    }

    return true;
}

bool CustomKernelNodeOps::PrepareTensors(Node* node)
{
    /*prepare inputs and outputs */
    input_num_ = node->GetInputNum();
    output_num_ = node->GetOutputNum();

    if(!CustomKernelManager::PrepareTensors(node, k_inputs_, input_num_, k_outputs_, output_num_))
        return false;

    /* set mem in tensor */
    for(int i = 0; i < input_num_; i++)
    {
        Tensor* tensor = node->GetInputTensor(i);
        struct custom_kernel_tensor* t = k_inputs_[i];

        t->data = get_tensor_mem(tensor);
    }

    for(int i = 0; i < output_num_; i++)
    {
        Tensor* tensor = node->GetOutputTensor(i);
        struct custom_kernel_tensor* t = k_outputs_[i];

        t->data = get_tensor_mem(tensor);
    }

    return true;
}

void CustomKernelNodeOps::ReleaseTensors(Node* node)
{
    CustomKernelManager::ReleaseTensors(node, k_inputs_, input_num_, k_outputs_, output_num_);

    input_num_ = 0;
    output_num_ = 0;

    k_inputs_ = nullptr;
    k_outputs_ = nullptr;
}

NodeOps* CustomKernelNodeOps::NewOps(Node* node, struct custom_kernel_ops* k_ops)
{
    NodeOps* ops = new CustomKernelNodeOps(node, k_ops);

    return ops;
}

}    // namespace TEngine
