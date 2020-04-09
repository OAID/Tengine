#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/instancenorm.hpp"

#include "kernel/instancenorm/ref_instancenorm_kernel.h"

namespace TEngine{

namespace RefInstanceNormOps
{
struct InstanceNormOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    // bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_instancenorm_t kernel_run;

    KernelRegistry<ref_instancenorm_t> kernel_registry;

    InstanceNormOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};
bool InstanceNormOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        // printf("errorno: %d\n",ENOENT);
        return false;
    }

    return true;
}
#if 0
bool InstanceNormOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}
#endif
bool InstanceNormOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* gamma_tensor = node->GetInputTensor(1);
    Tensor* beta_tensor  = node->GetInputTensor(2);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    int c = shape.GetC();
    int n = shape.GetN();
    int w = shape.GetW();
    int h = shape.GetH();
    int size = w*h;
    InstanceNorm* instancenorm_op = dynamic_cast<InstanceNorm*>(node->GetOp());
    InstanceNormParam* param = instancenorm_op->GetParam();
    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(output_tensor);
    void* beta_data = get_tensor_mem(beta_tensor);
    void* gamma_data = get_tensor_mem(gamma_tensor);
    int layout = exec_attr->graph_layout;
    float eps = param->eps;
    float scale = 1.f;
    int zero_point = 0;
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        scale = (*quant_param)[0].scale;
        zero_point = (*quant_param)[0].zero_point;
        auto out_quant_param = output_tensor->GetQuantParam();
        out_quant_param->resize(0);
        out_quant_param->push_back((*quant_param)[0]);
    }

    int ret = kernel_run(in_data, out_data, gamma_data, beta_data,size, c, n,eps, scale, zero_point, layout);
    if(ret < 0)
        return false;
    else
        return true;
}

void InstanceNormOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_instancenorm_t )ref_instancenorm_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_instancenorm_t )ref_instancenorm_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
/*
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_instancenorm_t )ref_instancenorm_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_instancenorm_t )ref_instancenorm_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_instancenorm_t )ref_instancenorm_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_instancenorm_t )ref_instancenorm_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_instancenorm_t )ref_instancenorm_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_instancenorm_t )ref_instancenorm_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
*/
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    InstanceNormOps* ops = new InstanceNormOps();

    LOG_DEBUG() << "InstanceNormOps RefOp is selected\n";

    return ops;
}
 
}

void RegisterRefInstanceNormOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "InstanceNorm", RefInstanceNormOps::SelectFunc, 1000);
}

}