#include "operator/zeros_like.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {
bool ZerosLike::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const std::vector<int>& in_dim = input.GetDim();

    TShape shape;

    shape.SetDim(in_dim);
    shape.SetDataLayout(input.GetDataLayout());

    oshape[0] = shape;

    return true;
}
void ZerosLike::SetSchema(void)
{
    Input({"input:float32"})
    .Output({"output:float32"})
    .SetDoc(R"DOC(Zeros_like Operator)DOC");
}

}    // namespace TEngine