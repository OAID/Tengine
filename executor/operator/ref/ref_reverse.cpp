#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/reverse.hpp"

#include "kernel/reverse/ref_reverse_kernel.h"

namespace TEngine {
namespace RefReverseOps {
// const int default_prio = 1500;
struct ReverseOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    // bool Postrun(Node* node) override;
    void InitRegistry(void);

    ReverseOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct reverse_param op_param;
    ref_reverse_t kernel_run;
    KernelRegistry<ref_reverse_t> kernel_registry;
};

bool ReverseOps::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;

    // Reverse* reverse_op = dynamic_cast<Reverse*>(node->GetOp());
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool ReverseOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    auto in_dim = input_tensor->GetShape().GetDim();
    void* input = get_tensor_mem(input_tensor);
    
    Tensor* axis_tensor = node->GetInputTensor(1);
    void* axis_data = get_tensor_mem(axis_tensor);  

    op_param.dim_size = (int)in_dim.size();

    for(int i = 0; i < op_param.dim_size; i++)
    {
        op_param.in_shape[i] = in_dim[i];
    }
    
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(output_tensor);
    // int* f_axis=(int*)axis_data;
    // printf("axis_data: %d\n",f_axis[0]);
    int ret = kernel_run(input, axis_data, output, &op_param);

    if (ret < 0)
    {
        return false;
    }
    else
        return true;
}

void ReverseOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_reverse_t )ref_reverse_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_reverse_t )ref_reverse_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

    // #ifdef CONFIG_KERNEL_FP16
    // kernel_registry.Register(( ref_reverse_t )ref_reverse_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    // kernel_registry.Register(( ref_reverse_t )ref_reverse_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
    // #endif
    // #ifdef CONFIG_KERNEL_INT8
    // kernel_registry.Register(( ref_reverse_t )ref_reverse_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    // kernel_registry.Register(( ref_reverse_t )ref_reverse_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
    // #endif

    // #ifdef CONFIG_KERNEL_UINT8
    // kernel_registry.Register(( ref_reverse_t )ref_reverse_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    // kernel_registry.Register(( ref_reverse_t )ref_reverse_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
    // #endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    ReverseOps* ops = new ReverseOps();

    LOG_DEBUG() << "ReverseOps RefOp is selected\n";

    return ops;
}
}   //namespace RefReverseOps

void RegisterRefReverseOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Reverse", RefReverseOps::SelectFunc, 1000);
}

}    // namespace TEngine

