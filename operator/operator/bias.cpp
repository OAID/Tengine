#include "operator/bias.hpp"

namespace TEngine{

bool Bias::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input_shape = ishape[0];
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    int output_h = input_h;
    int output_w = input_w;

    TShape shape;
    if(layout == TENGINE_LAYOUT_NCHW)
    {
        std::vector<int> dim = {input_shape.GetN(), input_shape.GetC(), output_h, output_w};

        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NCHW);
    }
    else
    {
        std::vector<int> dim = {input_shape.GetN(), output_h, output_w, input_shape.GetC()};

        shape.SetDim(dim);
        shape.SetDataLayout(TENGINE_LAYOUT_NHWC);
    }
    oshape[0] = shape;
    
    return true;
}

void Bias::SetSchema(void)
{
    Input({"input:float","bias:float"})
    .Output({"output:float"})
    .SetAttr("bias_size",0);
}

}