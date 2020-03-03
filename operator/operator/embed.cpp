#include "operator/embed.hpp"
namespace TEngine {

bool Embed::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    const TShape input_shape = ishape[0];
    int word_szie = input_shape.GetSize();

    std::vector<int> dims(2);
    dims[0] = word_szie;
    dims[1] = param_.num_output;
    oshape[0].SetDim(dims);

    return true;
}

void Embed::SetSchema(void)
{
    Input({"input:float32", "weight:float32", "bias:float32"})
    .Output({"output:float32"})
    .SetAttr("num_output", 0)
    .SetAttr("input_dim", 0)
    .SetAttr("bias_term", 0)
    .SetAttr("weight_data_size", 0);
}

}