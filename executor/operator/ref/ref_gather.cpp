#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/gather.hpp"

#include "kernel/gather/ref_gather_kernel.h"

namespace TEngine {
namespace RefGatherOps {
// const int default_prio = 1500;
struct GatherOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    // bool Postrun(Node* node) override;
    void InitRegistry(void);

    GatherOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct gather_param op_param;
    ref_gather_t kernel_run;
    KernelRegistry<ref_gather_t> kernel_registry;
};

bool GatherOps::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Gather* gather_op = dynamic_cast<Gather*>(node->GetOp());
    GatherParam* param = gather_op->GetParam();
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
#if 0
    auto in_dim = input_tensor->GetShape().GetDim();

    Tensor* input_tensor2 = node->GetInputTensor(1);
    auto in_dim2 = input_tensor2->GetShape().GetDim();
    for(int i = 0; i < (int)in_dim2.size();i++)
    {
        op_param.indices_num *= in_dim2[i];
    }
#endif
    op_param.axis = param->axis;
    op_param.indices_num = param->indices_num;

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool GatherOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    auto in_dim = input_tensor->GetShape().GetDim();
    void* input = get_tensor_mem(input_tensor);

    Tensor* indices_tensor = node->GetInputTensor(1);
    void* indices_data = get_tensor_mem(indices_tensor);
    
    op_param.dim_size = (int)in_dim.size();   

    for(int i = 0; i < op_param.dim_size; i++)
    {
        op_param.in_shape[i] = in_dim[i];
    }
    
    // int indices_num = op_param.indices_num;
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(output_tensor);

    int ret = kernel_run(input,indices_data, output, &op_param);

    if (ret < 0)
    {
        return false;
    }
    else
        return true;
}

void GatherOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_gather_t )ref_gather_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_gather_t )ref_gather_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

// #ifdef CONFIG_KERNEL_FP16
//    kernel_registry.Register(( ref_logical_t )ref_logical_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
//    kernel_registry.Register(( ref_logical_t )ref_logical_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
// #endif
// #ifdef CONFIG_KERNEL_INT8
//    kernel_registry.Register(( ref_logical_t )ref_logical_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
//    kernel_registry.Register(( ref_logical_t )ref_logical_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
// #endif

// #ifdef CONFIG_KERNEL_UINT8
//    kernel_registry.Register(( ref_logical_t )ref_logical_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
//    kernel_registry.Register(( ref_logical_t )ref_logical_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
// #endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    GatherOps* ops = new GatherOps();

    LOG_DEBUG() << "GatherOps RefOp is selected\n";

    return ops;
}

}    // namespace RefGatherOps
void RegisterRefGatherOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Gather", RefGatherOps::SelectFunc, 1000);
}

}    // namespace TEngine
