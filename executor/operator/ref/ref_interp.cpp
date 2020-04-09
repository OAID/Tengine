#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/interp.hpp"
#include "kernel/interp/ref_interp_kernel.h"

namespace TEngine {

namespace RefInterpOps {

struct InterpOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    interp_param op_param;
    ref_interp_t kernel_run;

    KernelRegistry<ref_interp_t> kernel_registry;

    InterpOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool InterpOps::Prerun(Node* node)
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

bool InterpOps::Run(Node* node)
{
    Interp* fm_op = dynamic_cast<Interp*>(node->GetOp());
    InterpParam* param = fm_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);

    void* input = get_tensor_mem(input_tensor);
    void* output = get_tensor_mem(output_tensor);
    

    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> in_dims = shape.GetDim();
    const TShape& shape1 = output_tensor->GetShape();
    const std::vector<int> out_dims = shape1.GetDim();

    float width_scale = param->width_scale;
    float height_scale = param->height_scale;
    int batch_number =shape.GetN();
    int inc = shape.GetC();
    int inh = shape.GetH(); 
    int inw = shape.GetW();



    int output_width = static_cast<int>(inh * width_scale);
    int output_height = static_cast<int>(inw * height_scale);



    op_param.inc=inc;
    op_param.inh=inh;
    op_param.inw=inw;
    op_param.batch_number=batch_number;
    op_param.output_width=output_width;
    op_param.output_height=output_height;
    op_param.width_scale=width_scale;
    op_param.height_scale=height_scale;

    

    int ret = kernel_run(input, output, &op_param);

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

void InterpOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_interp_t )ref_interp_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_interp_t )ref_interp_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

// #ifdef CONFIG_KERNEL_FP16
//    kernel_registry.Register(( ref_interp_t )ref_interp_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
//    kernel_registry.Register(( ref_interp_t )ref_interp_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
// #endif
// #ifdef CONFIG_KERNEL_INT8
//    kernel_registry.Register(( ref_interp_t )ref_interp_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
//    kernel_registry.Register(( ref_interp_t )ref_interp_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
// #endif

// #ifdef CONFIG_KERNEL_UINT8
//    kernel_registry.Register(( ref_interp_t )ref_interp_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
//    kernel_registry.Register(( ref_interp_t )ref_interp_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
// #endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    InterpOps* ops = new InterpOps();

    LOG_DEBUG() << "InterpOps RefOp is selected\n";

    return ops;
}

}    // namespace RefInterpOps
void RegisterRefInterpOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Interp", RefInterpOps::SelectFunc, 1000);
}

}    // namespace TEngine