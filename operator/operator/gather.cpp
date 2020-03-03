#include "operator/gather.hpp"
#include "operator/gather_param.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {

bool Gather::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    std::vector<int> input_dim = input.GetDim();

    if(param_.axis > ( int )input_dim.size())
    {
        return false;
    }    
    int indices_size = param_.indices_num;
    
    input_dim[param_.axis] = indices_size;
    oshape[0].SetDim(input_dim);
    oshape[0].SetDataLayout(input.GetDataLayout());

    return true;
}

void Gather::SetSchema(void)
{   
    Input({"input:float32", "indices:float32"})
        .Output({"output:float32"})
        .SetAttr("axis", 0)
        .SetAttr("indices_size", 1)
        .SetDoc(R"DOC(Slice Operator)DOC");
}

}    // namespace TEngine
