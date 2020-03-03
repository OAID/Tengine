#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/zeros_like.hpp"

#include "kernel/zeros_like/ref_zeros_like_kernel.h"

namespace TEngine {
namespace RefZerosLikeOps {
struct ZerosLikeOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;

    void InitRegistry(void);

    ZerosLikeOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct zeros_like_param op_param;
    ref_zeros_like_t kernel_run;
    KernelRegistry<ref_zeros_like_t> kernel_registry;
};

bool ZerosLikeOps::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;

    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool ZerosLikeOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    auto in_dim = input_tensor->GetShape().GetDim();
    void* input = get_tensor_mem(input_tensor); 

    op_param.dim_size = (int)in_dim.size();

    for(int i = 0; i < op_param.dim_size; i++)
    {
        op_param.in_dim[i] = in_dim[i];
    }
    
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(output_tensor);

    int ret = kernel_run(input, output, &op_param);

    if (ret < 0)
    {
        return false;
    }
    else
        return true;
}

void ZerosLikeOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_zeros_like_t )ref_zeros_like_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_zeros_like_t )ref_zeros_like_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    ZerosLikeOps* ops = new ZerosLikeOps();

    LOG_DEBUG() << "ZerosLikeOps RefOp is selected\n";

    return ops;
}
}   //namespace ZerosLikeOps

void RegisterRefZerosLikeOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "ZerosLike", RefZerosLikeOps::SelectFunc, 1000);
}

}    // namespace TEngine