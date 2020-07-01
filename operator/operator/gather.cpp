#include "operator/gather.hpp"
#include "operator/gather_param.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {

bool Gather::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    std::vector<int> input_dim = input.GetDim();
    std::vector<int> output_dim;

    if(param_.axis > ( int )input_dim.size())
    {
        return false;
    }    
    int indices_size = param_.indices_num;
    
    if(param_.is_onnx==true)
    {
        if(param_.axis == 0){
            for(int i = 0; i < (int)input_dim.size() - 1; i++){
                output_dim.push_back(input_dim[i+1]);
            }
        } else {
            for(int i = 0; i < (int)input_dim.size(); i++){
                if(i == param_.axis)
                    output_dim.push_back(indices_size);
                else
                {
                    output_dim.push_back(input_dim[i]);
                }
                
            }
        }
        oshape[0].SetDim(output_dim);
    }
    else{
        input_dim[param_.axis] = indices_size;
        oshape[0].SetDim(input_dim);
    }

    oshape[0].SetDataLayout(input.GetDataLayout());

    return true;
}

void Gather::SetSchema(void)
{   
    Input({"input:float32", "indices:float32"})
        .Output({"output:float32"})
        .SetAttr("axis", 0)
        .SetAttr("indices_size", 1)
        .SetAttr("is_onnx", false)
        .SetDoc(R"DOC(Gather Operator)DOC");
}

}    // namespace TEngine
