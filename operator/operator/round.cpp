#include "operator/round.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {
bool Round::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const std::vector<int>& in_dim = input.GetDim();

    TShape shape;

    shape.SetDim(in_dim);
    shape.SetDataLayout(input.GetDataLayout());

    oshape[0] = shape;

    return true;
}
void Round::SetSchema(void)
{
    Input({"input:float32"})
    .Output({"output:int32"})
    .SetDoc(R"DOC(Round Operator)DOC");
}

}    // namespace TEngine