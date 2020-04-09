#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/hardsigmoid.hpp"

#include "kernel/hardsigmoid/ref_hardsigmoid_kernel.h"

namespace TEngine {

namespace RefHardsigmoidOps {

struct HardsigmoidOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    // bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_hardsigmoid_t kernel_run;

    KernelRegistry<ref_hardsigmoid_t> kernel_registry;

    HardsigmoidOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool HardsigmoidOps::Prerun(Node* node)
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
bool HardsigmoidOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}
#endif
bool HardsigmoidOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = output_tensor->GetShape();
    int elem_num = shape.GetSize();
    Hardsigmoid* hardsigmoid_op = dynamic_cast<Hardsigmoid*>(node->GetOp());
    HardsigmoidParam* param = hardsigmoid_op->GetParam();
    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(output_tensor);
    float alpha = param->alpha;
    float beta = param->beta;

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

    int ret = kernel_run(in_data, out_data, elem_num, alpha, beta, scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

void HardsigmoidOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_hardsigmoid_t )ref_hardsigmoid_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_hardsigmoid_t )ref_hardsigmoid_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
/*
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_hardsigmoid_t )ref_hardsigmoid_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_hardsigmoid_t )ref_hardsigmoid_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_hardsigmoid_t )ref_hardsigmoid_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_hardsigmoid_t )ref_hardsigmoid_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_hardsigmoid_t )ref_hardsigmoid_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_hardsigmoid_t )ref_hardsigmoid_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
*/
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    HardsigmoidOps* ops = new HardsigmoidOps();

    LOG_DEBUG() << "HardsigmoidOps RefOp is selected\n";

    return ops;
}

}    // namespace RefHardsigmoidOps
void RegisterRefHardsigmoidOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Hardsigmoid", RefHardsigmoidOps::SelectFunc, 1000);
}
}    // namespace TEngine
