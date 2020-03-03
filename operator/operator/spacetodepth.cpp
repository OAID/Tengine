#include "operator/spacetodepth.hpp"
#include "static_graph.hpp"

namespace TEngine
{

bool SpaceToDepth::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    const TShape& input = ishape[0];

    std::vector<int> in_dims = input.GetDim();
    std::vector<int> out_dims;

    int batch, height, width, depth;

    if(TENGINE_LAYOUT_NCHW == layout)
    {
        batch = in_dims[0];
        depth = in_dims[1] * param_.block_size * param_.block_size;
        height = in_dims[2] / param_.block_size;
        width = in_dims[3] / param_.block_size;
        out_dims.push_back(batch);
        out_dims.push_back(depth);
        out_dims.push_back(height);
        out_dims.push_back(width);
    } 
    else if(TENGINE_LAYOUT_NHWC == layout)
    {
        batch = in_dims[0];
        depth = in_dims[3] * param_.block_size * param_.block_size;
        height = in_dims[1] / param_.block_size;
        width = in_dims[2] / param_.block_size;
        out_dims.push_back(batch);
        out_dims.push_back(height);
        out_dims.push_back(width);
        out_dims.push_back(depth);
    }

    TShape shape;
    shape.SetDim(out_dims);
    shape.SetDataLayout(input.GetDataLayout());
    oshape[0] = shape;   
    
    return true;
}

void SpaceToDepth::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("block_size", 1);
}

} // namespace TEngine
