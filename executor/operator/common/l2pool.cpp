#include <iostream>
#include <functional>
#include <stdlib.h>
#include <cmath>
#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/l2pool.hpp"
#include "data_type.hpp"

namespace TEngine{

namespace L2PoolRef {

struct L2PoolOps : public NodeOps
{
    void do_L2Pool(float* input, float*output, int inc, int inh, int inw, int outh, int outw,
                int k_h, int k_w, int stride_h, int stride_w, int pad_h, int pad_w)
    {

        for(int c = 0; c < inc ; c++)
        {
            for(int ph = 0; ph < outh; ph++)
            {
                for(int pw = 0; pw < outw; pw++)
                {
                    int index = inc * (ph * outw + pw) + c;
                    int h_start = ph * stride_h - pad_h;
                    int h_end = std::min(h_start + k_h, inh + pad_h);
                    int w_start = pw * stride_w - pad_w;
                    int w_end = std::min(w_start + k_w, inw + pad_w);
                    h_start = std::max(0, ph * stride_h - pad_h);
                    w_start = std::max(0, pw * stride_w - pad_w);
                    h_end = std::min(h_end, inh);
                    w_end = std::min(w_end, inw);
                    int pool_size = 0;

                    float tmp = 0.0f;
                    float val = 0.0f;
                    for(int h = h_start; h < h_end; h++)
                    {
                        for(int w = w_start; w < w_end; w++)
                        {
                            val = input[h * inc * inw + w * inc +c];
                            tmp += val * val;
                            pool_size++;
                        }
                    }
                    if(tmp == 0)
                    {
                        output[index] = 0;
                    }
                    else
                    {
                        output[index] = std::sqrt(tmp / pool_size);    
                    }
                }
            }
        }

    }
    
    void ConvertPaddingStyleToParameters(float stride_h, float stride_w, 
                                         int in_height, int in_width, int filter_height, int filter_width, PaddingType paddingtype,
                                         int out_height, int out_width,
                                         int* padding_width, int* padding_height)
    {
        if(paddingtype == PaddingType::kNone || paddingtype == PaddingType::kValid)
        {
            *padding_width = 0;
            *padding_height = 0;
        }
        else if(paddingtype == PaddingType::kSame)
        {
            *padding_width = (int)(((out_width - 1) * stride_w + filter_width - in_width) / 2);
            *padding_height = (int)(((out_height - 1) * stride_h + filter_height - in_height)/2);
        }

        return;
    }

    bool Run(Node* node)
    {
        L2Pool* l2pool_op = dynamic_cast<L2Pool*>(node->GetOp());
        L2PoolParam* param_ = l2pool_op->GetParam();
        
        Tensor* inputTensor = node->GetInputTensor(0);
        const TShape& inputShape = inputTensor->GetShape();
        Tensor* outputTensor = node->GetOutputTensor(0);
        TShape& outputShape = outputTensor->GetShape();
        int input_c = inputShape.GetC();
        int input_h = inputShape.GetH();
        int input_w = inputShape.GetW();
        int input_n = inputShape.GetN();
        int output_h = outputShape.GetH();
        int output_w = outputShape.GetW();  
        int output_c = outputShape.GetC();
        int input_size = input_c * input_h * input_w;
        int output_size = output_h * output_w * output_c;
        int padding_w = 0;
        int padding_h = 0;

        float* input_data = (float*) get_tensor_mem(inputTensor);
        float* out_data = (float*) get_tensor_mem(outputTensor);
        ConvertPaddingStyleToParameters(param_->stride_h, param_->stride_w, 
                                        input_h, input_w, param_->kernel_h, param_->kernel_w, param_->padding,
                                        output_h, output_w, &padding_w, &padding_h);
        for(int i = 0; i < input_n; i++)
        {
            do_L2Pool(input_data + i * input_size, out_data + i * output_size, input_c, input_h, input_w,
                    output_h, output_w, param_->kernel_h, param_->kernel_w,
                    param_->stride_h, param_->stride_w, padding_h, padding_w);
        }
        
        return true;
    }
};

NodeOps* SelectionFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)
        return nullptr;

    L2PoolOps* ops = new L2PoolOps();

    return ops;
}

}

using namespace L2PoolRef;

void RegisterL2Pool_NodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "L2Pool", L2PoolRef::SelectionFunc, 1000);
}

}
