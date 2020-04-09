#include "operator/broadmul.hpp"

namespace TEngine{

bool BroadMul::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input_shape = ishape[0];

    TShape shape;
    if(layout == TENGINE_LAYOUT_NCHW)
    {
        shape.SetDim(input_shape.GetDim());
        shape.SetDataLayout(TENGINE_LAYOUT_NCHW);
    }
    else
    {
        return false;
    }

    const TShape& input_shape1 = ishape[1];
    const std::vector<int>dim1 = input_shape1.GetDim();
 
    oshape[0] = shape;
    
    return true;
}

void BroadMul::SetSchema(void)
{
    Input({"input0:float","input1:float"})
    .Output({"output:float"});
}

}
