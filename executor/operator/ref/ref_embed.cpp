#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/embed.hpp"

#include "kernel/embed/ref_embed_kernel.h"

namespace TEngine {

namespace RefEmbedOps {

struct EmbedOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    // bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_embed_t kernel_run;

    KernelRegistry<ref_embed_t> kernel_registry;

    EmbedOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool EmbedOps::Prerun(Node* node)
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
bool EmbedOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;

#endif
bool EmbedOps::Run(Node* node)
{
    Embed* embed_op = dynamic_cast<Embed*>(node->GetOp());
    EmbedParam* param = embed_op->GetParam();
    int num_output = param->num_output;
    int input_dim = param->input_dim;
    int bias_term = param->bias_term;
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* weight_tensor = node->GetInputTensor(1);
    Tensor* bias_tensor = nullptr;
    if(bias_term)
        bias_tensor = node->GetInputTensor(2);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    int elem_num = shape.GetSize();
    void* in_data = get_tensor_mem(input_tensor);
    void* weight_data = get_tensor_mem(weight_tensor);
    void* bias_data = nullptr;
    if(bias_term)
        bias_data = get_tensor_mem(bias_tensor);
    void* out_data = get_tensor_mem(output_tensor);

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

    int ret = kernel_run(in_data, out_data, weight_data, bias_data, input_dim, num_output, elem_num, bias_term, scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

void EmbedOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_embed_t )ref_embed_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_embed_t )ref_embed_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
/*
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_embed_t )ref_embed_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_embed_t )ref_embed_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_embed_t )ref_embed_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_embed_t )ref_embed_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_embed_t )ref_embed_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_embed_t )ref_embed_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
*/
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    EmbedOps* ops = new EmbedOps();

    LOG_DEBUG() << "EmbedOps RefOp is selected\n";

    return ops;
}

}    // namespace RefEmbedOps
void RegisterRefEmbedOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Embedding", RefEmbedOps::SelectFunc, 1000);
}
}    // namespace TEngine
