#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/logical.hpp"
#include "kernel/logical/ref_logical_kernel.h"

namespace TEngine {

namespace RefLogicalOps {

struct LogicalOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    logical_param op_param;
    ref_logical_t kernel_run;

    KernelRegistry<ref_logical_t> kernel_registry;

    LogicalOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool LogicalOps::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);

    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool LogicalOps::Run(Node* node)
{
    Logical* logical_op = dynamic_cast<Logical*>(node->GetOp());
    LogicalParam* param = logical_op->GetParam();
    Tensor* input_tensor0 = node->GetInputTensor(0);
    const TShape& ishape = input_tensor0->GetShape();
    void* input0 = get_tensor_mem(input_tensor0);
    Tensor* input_tensor1 = nullptr;
    void* input1 = nullptr;

    // int x = node->GetInputNum();
    // printf("inputnum:%d\n", x);
    // this version only support for input_num=2
    // int input_number=node->GetInputNum();

    // output
    Tensor* output_tensor = node->GetOutputTensor(0);
    // if(input_tensor0->GetDataType() == TENGINE_DT_INT8 || input_tensor0->GetDataType() == TENGINE_DT_UINT8)
    // {
    //    if(get_scale_zero(input_tensor0, output_tensor, &op_param) < 0)
    //        return false;
    // }

    if(node->GetInputNum() > 1)
    {
        input_tensor1 = node->GetInputTensor(1);
        // printf("input_tensor1 %p\n", input_tensor1);
        const TShape& ishape1 = input_tensor1->GetShape();
        input1 = get_tensor_mem(input_tensor1);
        op_param.shape1[0] = ishape1.GetN();
        op_param.shape1[1] = ishape1.GetC();
        op_param.shape1[2] = ishape1.GetH();
        op_param.shape1[3] = ishape1.GetW();

        // if(input_tensor1->GetDataType() == TENGINE_DT_INT8 || input_tensor1->GetDataType() == TENGINE_DT_UINT8)
        // {
        //    if(get_scale_zero_1(input_tensor1, &op_param) < 0)
        //        return false;
        // }
    }
    void* output = get_tensor_mem(output_tensor);
    op_param.shape0[0] = ishape.GetN();
    op_param.shape0[1] = ishape.GetC();
    op_param.shape0[2] = ishape.GetH();
    op_param.shape0[3] = ishape.GetW();
    op_param.type = param->type;

    int ret = kernel_run(input0, input1, output, &op_param);

    // if(input_tensor0->GetDataType() == TENGINE_DT_INT8)
    // {
    //    auto* o_quant = output_tensor->GetQuantParam();
    //    QuantParam q_param;
    //    q_param.scale = op_param.scale[2];
    //    o_quant->resize(0);
    //    o_quant->push_back(q_param);
    // }

    if(ret < 0)
        return false;
    else
        return true;
}

void LogicalOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_logical_t )ref_logical_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_logical_t )ref_logical_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
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
    LogicalOps* ops = new LogicalOps();

    LOG_DEBUG() << "LogicalOps RefOp is selected\n";

    return ops;
}

}    // namespace RefLogicalOps
void RegisterRefLogicalOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Logical", RefLogicalOps::SelectFunc, 1000);
}

}    // namespace TEngine