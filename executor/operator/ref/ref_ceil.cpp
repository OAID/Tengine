#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/ceil.hpp"

#include "kernel/ceil/ref_ceil_kernel.h"

namespace TEngine {
namespace RefCeilOps {
// const int default_prio = 1500;
struct CeilOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    // bool Postrun(Node* node) override;
    void InitRegistry(void);

    CeilOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct ceil_param op_param;
    ref_ceil_t kernel_run;
    KernelRegistry<ref_ceil_t> kernel_registry;
};

bool CeilOps::Prerun(Node* node)
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

bool CeilOps::Run(Node* node)
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

void CeilOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_ceil_t )ref_ceil_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_ceil_t )ref_ceil_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    CeilOps* ops = new CeilOps();

    LOG_DEBUG() << "CeilOps RefOp is selected\n";

    return ops;
}
}   //namespace RefCeilOps

void RegisterRefCeilOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Ceil", RefCeilOps::SelectFunc, 1000);
}

}    // namespace TEngine