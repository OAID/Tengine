#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/comparison.hpp"
#include "kernel/comparison/ref_comparison_kernel.h"

namespace TEngine{

namespace RefComparisonOps{

struct ComparisonOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    comparison_param op_param;
    ref_comparison_t kernel_run;

    KernelRegistry<ref_comparison_t> kernel_registry;

    ComparisonOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool ComparisonOps::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);

    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    // int elem_size=DataType::GetTypeSize(input->GetDataType());

    return true;
}

bool ComparisonOps::Run(Node* node)
{
    Comparison* comp_op = dynamic_cast<Comparison*>(node->GetOp());
    ComparisonParam* param = comp_op->GetParam();
    Tensor* input_tensor0 = node->GetInputTensor(0);
    const TShape& ishape = input_tensor0->GetShape();
    void* input0 = get_tensor_mem(input_tensor0);

    // output
    Tensor* output_tensor = node->GetOutputTensor(0);

    Tensor* input_tensor1 = node->GetInputTensor(1);
    const TShape& ishape1 = input_tensor1->GetShape();
    void* input1 = get_tensor_mem(input_tensor1);
    op_param.shape1[0] = ishape1.GetN();
    op_param.shape1[1] = ishape1.GetC();
    op_param.shape1[2] = ishape1.GetH();
    op_param.shape1[3] = ishape1.GetW();
    
    void* output = get_tensor_mem(output_tensor);
    op_param.shape0[0] = ishape.GetN();
    op_param.shape0[1] = ishape.GetC();
    op_param.shape0[2] = ishape.GetH();
    op_param.shape0[3] = ishape.GetW();
    op_param.layout = ishape.GetDataLayout();
    op_param.type = param->type;

    int ret = kernel_run(input0, input1, output, &op_param);

    if(ret < 0)
        return false;
    else
        return true;

}

void ComparisonOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_comparison_t )ref_comparison_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_comparison_t )ref_comparison_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    ComparisonOps* ops = new ComparisonOps();

    LOG_DEBUG() << "ComparisonOps RefOp is selected\n";

    return ops;
}

}

void RegisterComparisonOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Comparison", RefComparisonOps::SelectFunc, 5000);
}

}