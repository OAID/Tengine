#include <vector>
#include <math.h>
#include <algorithm>
#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/yolov3detectionoutput.hpp"
#include "kernel/yolov3detectionoutput/ref_yolov3detectionoutput_kernel.h"

namespace TEngine{

namespace RefYOLOV3DetectionOutputOps{

struct RefYOLOV3DetectionOutput : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    YOLOV3_ddo_parm param;
    ref_YOLOV3DetectionOutput_kernel_t kernel_run;
    KernelRegistry<ref_YOLOV3DetectionOutput_kernel_t> kernel_registry;

    RefYOLOV3DetectionOutput(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }

};

bool RefYOLOV3DetectionOutput::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input = node->GetInputTensor(0);
    if(input->GetDataType() != TENGINE_DT_FP32 && input->GetDataType() != TENGINE_DT_FP16 &&
       input->GetDataType() != TENGINE_DT_UINT8 && input->GetDataType() != TENGINE_DT_INT8)
        return false;

    YOLOV3DetectionOutput* yolov3ddo_op = dynamic_cast<YOLOV3DetectionOutput*>(node->GetOp());
    YOLOV3DetectionOutputParam* param_ = yolov3ddo_op->GetParam();

    param.num_box = param_->num_box;
    param.num_classes = param_->num_classes;
    param.nms_threshold = param_->nms_threshold;
    param.mask_group_num = param_->mask_group_num;
    param.anchors_scale = param_->anchors_scale;
    param.confidence_threshold = param_->confidence_threshold;
    param.bias = param_->bias;
    param.mask = param_->mask;
    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefYOLOV3DetectionOutput::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;
    
    int mem_size = 0;
    std::vector<void*> input_datas;
    std::vector<std::vector<int>> input_dims;
    std::vector<float> input_scales;
    std::vector<int> data_len;
    for(int i = 0; i < node->GetInputNum(); i++)
    {
        Tensor* input_tensor = node->GetInputTensor(i);
        TShape& tensor_shape = input_tensor->GetShape();
        void* input_data = (void*) get_tensor_mem(input_tensor);
        std::vector<int>& dims = tensor_shape.GetDim();
        auto tensor_quant = input_tensor->GetQuantParam();
        mem_size += tensor_shape.GetSize();
        data_len.push_back(tensor_shape.GetSize()*sizeof(float));
        input_datas.push_back(input_data);
        input_dims.push_back(dims);
        input_scales.push_back((*tensor_quant)[0].scale);
    }
    param.input_dims = input_dims;
    void* input_ptr = mem_alloc(mem_size*sizeof(float));
    int offset = 0;
    for(int i = 0; i < node->GetInputNum(); i++)
    { 
        memcpy((float*)input_ptr+offset, input_datas[i], data_len[i]);
        offset += data_len[i];
        //printf("%d\n", data_len[i]);
    }
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output = (void*)get_tensor_buffer(output_tensor);

    if(kernel_run(input_ptr, &param) < 0)
    {
        return false;
    }
    //printf("run done.\n");
    int num_detected = param.output_box.size();
    int total_size = num_detected * 6 * 4;
    void* mem_addr = mem_alloc(total_size);
    set_tensor_mem(output_tensor, mem_addr, total_size, mem_free);
    if(output_tensor->GetDataType() == TENGINE_DT_FP32)
    {
        float* output = ( float* )get_tensor_mem(output_tensor);
        TShape& out_shape = output_tensor->GetShape();
        std::vector<int> outdim = {1, num_detected, 6, 1};
        out_shape.SetDim(outdim);
        for(int i = 0; i < num_detected; i++)
        {
            const Box& r = param.output_box[i];
            float* outptr = output + i * 6;
            outptr[0] = r.class_idx+1;
            outptr[1] = r.score;
            outptr[2] = r.x0;
            outptr[3] = r.y0;
            outptr[4] = r.x1;
            outptr[5] = r.y1;
        }
        std::vector<int> dims = out_shape.GetDim();
        //printf("%d %d %d %d %d.\n", dims[0], dims[1], dims[2], dims[3], out_shape.GetDataLayout());
    }
    if(output_tensor->GetDataType() == TENGINE_DT_FP16)
    {
        __fp16* output = ( __fp16* )get_tensor_mem(output_tensor);
        TShape& out_shape = output_tensor->GetShape();
        std::vector<int> outdim = {1, num_detected, 6, 1};
        out_shape.SetDim(outdim);
        for(int i = 0; i < num_detected; i++)
        {
            const Box& r = param.output_box[i];
            __fp16* outptr = output + i * 6;
            outptr[0] = fp32_to_fp16(r.class_idx);
            outptr[1] = fp32_to_fp16(r.score);
            outptr[2] = fp32_to_fp16(r.x0);
            outptr[3] = fp32_to_fp16(r.y0);
            outptr[4] = fp32_to_fp16(r.x1);
            outptr[5] = fp32_to_fp16(r.y1);
        }
    }
    if(output_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        int8_t* output = ( int8_t* )get_tensor_mem(output_tensor);
        TShape& out_shape = output_tensor->GetShape();
        std::vector<int> outdim = {1, 1, num_detected, 6};
        out_shape.SetDim(outdim);
        for(int i = 0; i < num_detected; i++)
        {
            const Box& r = param.output_box[i];
            int8_t* outptr = output + i * 6;

            outptr[0] = r.class_idx;
            outptr[1] = round(r.score / param.out_scale);
            outptr[2] = round(r.x0 / param.out_scale);
            outptr[3] = round(r.y0 / param.out_scale);
            outptr[4] = round(r.x1 / param.out_scale);
            outptr[5] = round(r.y1 / param.out_scale);
        }
        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = param.out_scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }
    free(input_ptr);
    return true;
}

void RefYOLOV3DetectionOutput::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_YOLOV3DetectionOutput_kernel_t)ref_YOLOV3DetectionOutput_fp32, TENGINE_LAYOUT_NCHW,
                             TENGINE_DT_FP32);
    kernel_registry.Register(( ref_YOLOV3DetectionOutput_kernel_t)ref_YOLOV3DetectionOutput_fp32, TENGINE_LAYOUT_NHWC,
                             TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_YOLOV3DetectionOutput_kernel_t)ref_YOLOV3DetectionOutput_fp16, TENGINE_LAYOUT_NCHW,
                             TENGINE_DT_FP16);
    kernel_registry.Register(( ref_YOLOV3DetectionOutput_kernel_t)ref_YOLOV3DetectionOutput_fp16, TENGINE_LAYOUT_NHWC,
                             TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_YOLOV3DetectionOutput_kernel_t)ref_YOLOV3DetectionOutput_int8, TENGINE_LAYOUT_NCHW,
                             TENGINE_DT_INT8);
    kernel_registry.Register(( ref_YOLOV3DetectionOutput_kernel_t)ref_YOLOV3DetectionOutput_int8, TENGINE_LAYOUT_NHWC,
                             TENGINE_DT_INT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefYOLOV3DetectionOutput* ops = new RefYOLOV3DetectionOutput();

    LOG_DEBUG() << "Demo RefYOLOV3DetectionOutput is selected.\n";

    return ops;
}

}
using namespace RefYOLOV3DetectionOutputOps;
void RegisterRefYOLOV3DetectionOutput(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "YOLOV3DetectionOutput",
                                                  RefYOLOV3DetectionOutputOps::SelectFunc, 1000);

}
}
