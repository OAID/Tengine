#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/noop.hpp"

#include "kernel/noop/ref_noop_kernel.h"

namespace TEngine {

namespace RefNoopOps {

struct NoopOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_noop_t kernel_run;

    KernelRegistry<ref_noop_t> kernel_registry;

    NoopOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool NoopOps::Prerun(Node* node)
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

bool NoopOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    std::cout<<"onbind down"<<std::endl;
    return true;
}

bool NoopOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = output_tensor->GetShape();
    int elem_num = shape.GetSize();
    void* out_data = get_tensor_mem(output_tensor);
    void* in_data  = get_tensor_mem(input_tensor);
    float scale = 1.f;
    int zero_point = 0;
    memcpy(out_data, in_data, elem_num*sizeof(float));
    int ret = kernel_run(scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

void NoopOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_noop_t )ref_noop_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_noop_t )ref_noop_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    NoopOps* ops = new NoopOps();

    LOG_DEBUG() << "NoopOps RefOp is selected\n";

    return ops;
}

}    // namespace RefClipOps
void RegisterRefNoopOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Noop", RefNoopOps::SelectFunc, 1000);
}
}    // namespace TEngine
