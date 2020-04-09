#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/sparsetodense.hpp"

#include "kernel/sparsetodense/ref_sparsetodense_kernel.h"

namespace TEngine {
namespace RefSparseToDenseOps {
struct SparseToDenseOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    SparseToDenseOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct sparsetodense_param op_param;
    ref_sparsetodense_t kernel_run;
    KernelRegistry<ref_sparsetodense_t> kernel_registry;
};

bool SparseToDenseOps::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;

    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    SparseToDense* sparsetodense_op = dynamic_cast<SparseToDense*>(node->GetOp());
    SparseToDenseParam* param = sparsetodense_op->GetParam();

    op_param.default_value = param -> default_value;


    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool SparseToDenseOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* outout_shape_tensor = node->GetInputTensor(1);
    Tensor* sparse_values_tensor = node->GetInputTensor(2);

    void* input = get_tensor_mem(input_tensor);
    void* outout_shape = get_tensor_mem(outout_shape_tensor);
    void* sparse_values = get_tensor_mem(sparse_values_tensor);

    auto in_dim = input_tensor->GetShape().GetDim();
    op_param.indices_dim_size = (int)in_dim.size();

    for(int i = 0; i < op_param.indices_dim_size; i ++)
    {
        op_param.indices_shape[i] = in_dim[i];
    }

    auto output_dim = outout_shape_tensor->GetShape().GetDim();
    op_param.output_dim_size = (int)output_dim.size();

    auto sparse_values_dim = sparse_values_tensor->GetShape().GetDim();
    op_param.sparse_values_size = (int)sparse_values_dim.size();
    
    
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(output_tensor);

    int ret = kernel_run(input, outout_shape, sparse_values, output, &op_param);

    if (ret < 0)
    {
        return false;
    }
    else
        return true;
}

void SparseToDenseOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_sparsetodense_t )ref_sparsetodense_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_sparsetodense_t )ref_sparsetodense_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

}
NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    SparseToDenseOps* ops = new SparseToDenseOps();

    LOG_DEBUG() << "SparseToDenseOps RefOp is selected\n";

    return ops;
}
}

void RegisterRefSparseToDenseOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "SparseToDense", RefSparseToDenseOps::SelectFunc, 1000);
}   
}   // namespace TEngine
