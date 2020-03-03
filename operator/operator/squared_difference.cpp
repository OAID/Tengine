#include "operator/squared_difference.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine{

bool SquaredDifference::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    TShape input1 = ishape[0];
    TShape input2 = ishape[1];

    if(input1.GetDim().size() != input2.GetDim().size()){
        return false;
    }

    int dim_size = input1.GetDim().size();
    for(int i = 0; i < dim_size; i++){
        if(input1.GetDim()[i] != input2.GetDim()[i]){
            return false;
        }
    }

    oshape[0] = input1;

    return true;
}

void SquaredDifference::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetDoc(R"DOC(SquaredDifference Layer)DOC");
}

}   //namespace TEngine