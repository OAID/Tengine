#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/bias.hpp"

#include "kernel/bias/ref_bias_kernel.h"

namespace TEngine {

namespace RefBiasOps {

struct BiasOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    //bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_bias_t kernel_run;

    KernelRegistry<ref_bias_t> kernel_registry;

    BiasOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool BiasOps::Prerun(Node* node)
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
/*
bool BiasOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    std::cout<<"onbind down"<<std::endl;
    return true;
}
*/
bool BiasOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* bias_tensor = node->GetInputTensor(1);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = output_tensor->GetShape();
    int elem_num = shape.GetH()*shape.GetW();
    //Bias* bias_op = dynamic_cast<Bias*>(node->GetOp());
    //BiasParam* param = bias_op->GetParam();
    void* out_data = get_tensor_mem(output_tensor);
    void* in_data  = get_tensor_mem(input_tensor);
    void* bias_data = get_tensor_mem(bias_tensor);
    float scale = 1.f;
    int zero_point = 0;

    int ret = kernel_run(in_data, out_data, bias_data, elem_num, shape.GetC(), scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

void BiasOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_bias_t )ref_bias_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_bias_t )ref_bias_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    BiasOps* ops = new BiasOps();

    LOG_DEBUG() << "BiasOps RefOp is selected\n";

    return ops;
}

}    // namespace RefClipOps
void RegisterRefBiasOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Bias", RefBiasOps::SelectFunc, 1000);
}
}    // namespace TEngine
