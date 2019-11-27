#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/l2normalization.hpp"
#include <math.h>

namespace TEngine{

namespace L2NormalizationImpl{

struct L2NormalizationOps : public NodeOps
{
    bool Prerun(Node* node)
    {
        const Tensor* input_tenosr = node->GetInputTensor(0);
        const TShape& input_shape = input_tenosr->GetShape();

        const std::vector<int> input_dims = input_shape.GetDim();

        int input_size = 1;
        for(unsigned int i = 0; i < input_dims.size(); i++)
        {
            input_size *= input_dims[i];
        }

        return true;
    }


    void l2norm(const float* input, const TShape& input_shape, float* output)
    {
        const std::vector<int> input_dims = input_shape.GetDim();
        int input_size = 1;
        int channel_size = input_dims[input_dims.size() - 1];
        
        for(unsigned int i = 0; i < input_dims.size(); i++)
        {
            input_size *= input_dims[i];
        }

        for(int i = 0; i < input_size; i++)
        {
            float sq_l2_norm = 0;
            for(int j = 0; j < channel_size; j++)
            {
                const float val = input[j];
                sq_l2_norm += val * val;
                //std::cout<<sq_l2_norm<<std::endl;
            }
            const float l2_norm = std::sqrt(sq_l2_norm);
            for(int j = 0; j < channel_size; j++)
            {
                *output = *input / l2_norm;
                output++;
                input++;
            }
        }
    }
    
    bool Run(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        
        //L2Normalization* l2normalization_op = dynamic_cast<L2Normalization*>(node->GetOp());

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();

        float* input = (float*)get_tensor_mem(input_tensor);
        float* output = (float*)get_tensor_mem(output_tensor);
        
        l2norm(input, shape, output);

        return true;        
    }

    bool Postrun(Node* node)
    {
        return true;
    }

};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)
        return nullptr;

    L2NormalizationOps* ops = new L2NormalizationOps();

    return ops;
}

}

using namespace L2NormalizationImpl;

void RegisterL2NormalizationNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "L2Normalization", L2NormalizationImpl::SelectFunc, 1000);
}

}
