#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/mvn.hpp"

#include "kernel/mvn/ref_mvn_kernel.h"

namespace TEngine{

namespace RefMVNOps
{

struct MVNOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    // bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);
    ref_mvn_param op_param;
    ref_mvn_kernel_t kernel_run;

    KernelRegistry<ref_mvn_kernel_t> kernel_registry;

    MVNOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

void MVNOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_mvn_kernel_t )ref_mvn_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_mvn_kernel_t )ref_mvn_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
/*
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_mvn_kernel_t )ref_mvn_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_mvn_kernel_t )ref_mvn_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_mvn_kernel_t )ref_mvn_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_mvn_kernel_t )ref_mvn_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_mvn_kernel_t )ref_mvn_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_mvn_kernel_t )ref_mvn_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
*/
}

bool MVNOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;
    op_param.layout = layout;
    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        // printf("errorno: %d\n",ENOENT);
        return false;
    }

    return true;
}

bool MVNOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);

    const TShape& shape = input_tensor->GetShape();

    op_param.input_n = shape.GetN();
    op_param.input_w = shape.GetW();
    op_param.input_h = shape.GetH();
    op_param.input_c = shape.GetC();

    MVN* mvn_op = dynamic_cast<MVN*>(node->GetOp());
    MVNParam* param = mvn_op->GetParam();

    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(output_tensor);
    op_param.normalize_variance = param->normalize_variance;
    op_param.across_channels = param->across_channels;
    op_param.eps = param->eps;

    if(TENGINE_DT_UINT8 == input_tensor->GetDataType() || TENGINE_DT_INT8 == input_tensor->GetDataType())
    {
        auto* in_quant = input_tensor->GetQuantParam();
        if(in_quant->size())
        {
            op_param.in_scale = (*in_quant)[0].scale;
            op_param.in_zero = (*in_quant)[0].zero_point;
        }
        if(node->GetInputNum() == 2)
        {
            Tensor* scale_tensor = node->GetInputTensor(1);
            auto* scale_quant = scale_tensor->GetQuantParam();
            if(scale_quant->size())
            {
                op_param.scale_scale = (*scale_quant)[0].scale;
                op_param.scale_zero = (*scale_quant)[0].zero_point;
            }
        }
    }
    if(TENGINE_DT_UINT8 == input_tensor->GetDataType())
    {
        auto* out_quant = output_tensor->GetQuantParam();
        if(out_quant->size())
        {
            op_param.out_scale = (*out_quant)[0].scale;
            op_param.out_zero = (*out_quant)[0].zero_point;
        }
    }

    int ret = kernel_run(in_data, out_data, &(this->op_param));

    if(ret < 0)
        return false;
    
    if(TENGINE_DT_INT8 == input_tensor->GetDataType())
    {
        auto* out_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.out_scale;
        q_param.zero_point = 0;
        out_quant->resize(0);
        out_quant->push_back(q_param);
    }

    return true;
}

NodeOps*SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    MVNOps* ops = new MVNOps();
    LOG_DEBUG() << "MVNOps RefOp is selected\n";

    return ops;
}

}

void RegisterRefMVNOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "MVN", RefMVNOps::SelectFunc, 1000);
}

}