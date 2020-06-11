#include "operator/yolov3detectionoutput.hpp"

namespace TEngine{
bool YOLOV3DetectionOutput::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape,
                                 int layout)
{
    const TShape& input = ishape[0];
    const std::vector<int>& in_dim = input.GetDim();
    std::vector<int> dims= input.GetDim();
    printf("%d %d %d %d", dims[0], dims[1], dims[2], dims[3]);
    TShape shape;
    std::vector<int> dim = {param_.num_box, 1, 6, 1};
    shape.SetDim(dim);
    shape.SetDataLayout(input.GetDataLayout());
    printf("layout is %d.\n", input.GetDataLayout());
    oshape[0] = shape;
    return true;   
}

void YOLOV3DetectionOutput::SetSchema(void)
{
    Input({"input:float32"})
    .Output({"output:float32"})
    .SetAttr("num_classes", 20)
    .SetAttr("confidence_threshold", 0.01f)
    .SetAttr("nms_threshold", 0.45f)
    .SetAttr("num_box", 5);
}

}