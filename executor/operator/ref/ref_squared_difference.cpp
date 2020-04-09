#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/squared_difference.hpp"
#include "kernel/squared_difference/ref_squared_difference_kernel.h"

namespace TEngine {

namespace RefSquaredDifferenceOps {

struct SquaredDifferenceOps: public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    squared_difference_param op_param;
    ref_squared_difference_t kernel_run;

    KernelRegistry<ref_squared_difference_t> kernel_registry;

    SquaredDifferenceOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool SquaredDifferenceOps::Prerun(Node* node)
{
    if(node->GetInputNum() != 2){
        return false;
    }

    Tensor* input_tensor1 = node->GetInputTensor(0);

    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor1->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool SquaredDifferenceOps::Run(Node* node)
{
    // SquaredDifference* squared_difference_op = dynamic_cast<SquaredDifference*>(node->GetOp());

    Tensor* input_tensor1 = node->GetInputTensor(0);
    Tensor* input_tensor2 = node->GetInputTensor(1);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& ishape = input_tensor1->GetShape();

    void* input1 = get_tensor_mem(input_tensor1);
    void* input2 = get_tensor_mem(input_tensor2);
    void* output = get_tensor_mem(output_tensor);
    
    op_param.in_dim_size = ishape.GetDim().size();
    auto in_dim = ishape.GetDim();
    int in_dim_size = ishape.GetDim().size();
    for(int i = 0; i < in_dim_size; i++){
        op_param.in_dim[i] = in_dim[i];
    }

    int ret = kernel_run(input1, input2, output, &op_param);
    if(ret < 0)
        return false;
    else
        return true;
}

void SquaredDifferenceOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_squared_difference_t )ref_squared_difference_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_squared_difference_t )ref_squared_difference_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    SquaredDifferenceOps* ops = new SquaredDifferenceOps();

    LOG_DEBUG() << "SquaredDifferenceOps RefOp is selected\n";

    return ops;
}
}   //namespace TEngine

void RegisterRefSquaredDifferenceOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "SquaredDifference", RefSquaredDifferenceOps::SelectFunc, 1000);
}

}    // namespace TEngine