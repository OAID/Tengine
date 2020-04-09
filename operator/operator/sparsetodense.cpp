#include "operator/sparsetodense.hpp"
#include "operator/sparsetodense_param.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {
bool SparseToDense::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];
    const TShape& output_shape = ishape[1];
    const std::vector<int>& in_dim = input.GetDim();
    const std::vector<int>& out_dim = output_shape.GetDim();
    const int output_shape_size0 = param_.output_shape_size0;
    const int output_shape_size1 = param_.output_shape_size1;
    std::vector<int> output_shape_dim;
    output_shape_dim.push_back(output_shape_size0);

    if(( int )in_dim.size() > 2)
    {
        return false;
    }

    TShape shape;

    if((( int )out_dim.size() == 2) & (( int )in_dim.size() == 2) & (output_shape_size1 != 0))
    {
        output_shape_dim.push_back(output_shape_size1);
        shape.SetDim(output_shape_dim);
        shape.SetDataLayout(input.GetDataLayout());
        oshape[0] = shape;
    }

    else if((( int )out_dim.size() == 1) & (( int )in_dim.size() == 1 || ( int )in_dim.size() == 0))
    {
        shape.SetDim(output_shape_dim);
        oshape[0] = shape;
        shape.SetDataLayout(input.GetDataLayout());
    }
    
    else
    {
        return false;
    }
    

    return true;
}
void SparseToDense::SetSchema(void)
{
    Input({"input:int32", "output_shape:int32", "sparse_values:float32"})
    .SetAttr("output_shape_size0", 1)
    .SetAttr("output_shape_size1", 0)
    .SetAttr("default_value", 0)
    .Output({"output:float32"})
    .SetDoc(R"DOC(SparseToDense Operator)DOC");
}

}    // namespace TEngine
