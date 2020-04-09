#include <vector>
#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/bias.hpp"

#include "kernel/broadmul/ref_broadmul_kernel.h"

namespace TEngine {

namespace RefBroadMulOps {

struct BroadMulOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    //bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_broadmul_t kernel_run;
    ref_broadmul_param param;

    KernelRegistry<ref_broadmul_t> kernel_registry;

    BroadMulOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool BroadMulOps::Prerun(Node* node)
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
bool BroadMulOps::Run(Node* node)
{
    Tensor* input0_tensor = node->GetInputTensor(0);
    Tensor* input1_tensor = node->GetInputTensor(1);
    Tensor* output_tensor = node->GetOutputTensor(0);

    uint8_t* out_data = (uint8_t*)get_tensor_mem(output_tensor);
    uint8_t* in0  = (uint8_t*)get_tensor_mem(input0_tensor);
    uint8_t* in1 = (uint8_t*)get_tensor_mem(input1_tensor);

    const TShape& shape0 = input0_tensor->GetShape();
    const TShape& shape1 = input1_tensor->GetShape();
    
    int in_size = 1;
    int on_size = 1;
    int out_size = 1;
    
    const std::vector<int>& dim0 = shape0.GetDim();
    const std::vector<int>& dim1 = shape1.GetDim();

    int axis = 0;
   
    for(int i = 0; i < (int)dim1.size(); i++)
    {
        if(dim1[i] == 1)
        {
            out_size = out_size * dim0[i];
        }
        else
        {
            axis = i;
            break;
        }
    }
    on_size = dim0[axis];
    
    for(int i = axis+1; i< (int)dim0.size();i++)
    {
        in_size = in_size * dim0[i];
    }
    param.in_size = in_size;
    param.out_size = out_size;
    param.on_size = on_size;
 
    int ret = kernel_run(in0, in1, out_data, &param);

    if(ret < 0)
        return false;
    else
        return true;
}

void BroadMulOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_broadmul_t )ref_broadmul_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    BroadMulOps* ops = new BroadMulOps();

    LOG_DEBUG() << "BroadMulOps RefOp is selected\n";

    return ops;
}

}    // namespace RefClipOps
void RegisterRefBroadMulOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "BroadMul", RefBroadMulOps::SelectFunc, 1000);
}
}    // namespace TEngine
