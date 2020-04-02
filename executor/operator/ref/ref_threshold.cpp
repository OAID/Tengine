#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/threshold.hpp"

#include "kernel/threshold/ref_threshold_kernel.h"

namespace TEngine {

namespace RefThresholdOps {

struct ThresholdOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    //bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_threshold_t kernel_run;

    KernelRegistry<ref_threshold_t> kernel_registry;

    ThresholdOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool ThresholdOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        //printf("errorno: %d\n",ENOENT);
        return false;
    }

    return true;
}
/*
bool ThresholdOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    std::cout<<"onbind down"<<std::endl;
    return true;
}
*/
bool ThresholdOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = output_tensor->GetShape();
    int elem_num = shape.GetSize();
    Threshold* threshold_op = dynamic_cast<Threshold*>(node->GetOp());
    ThresholdParam* param = threshold_op->GetParam();
    void* out_data = get_tensor_mem(output_tensor);
    void* in_data  = get_tensor_mem(input_tensor);
    float scale = 1.f;
    int zero_point = 0;
    int ret = kernel_run(out_data, in_data, param->threshold, elem_num, scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

void ThresholdOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_threshold_t )ref_threshold_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_threshold_t )ref_threshold_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    ThresholdOps* ops = new ThresholdOps();

    LOG_DEBUG() << "ThresholdOps RefOp is selected\n";

    return ops;
}

}    // namespace RefClipOps
void RegisterRefThresholdOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Threshold", RefThresholdOps::SelectFunc, 1000);
}
}    // namespace TEngine
