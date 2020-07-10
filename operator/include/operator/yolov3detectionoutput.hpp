#ifndef __YOLOV3_DETECTION_OUTPUT_HPP__
#define __YOLOV3_DETECTION_OUTPUT_HPP__

#include "operator.hpp"
#include "yolov3detectionoutput_param.hpp"

namespace TEngine {

class YOLOV3DetectionOutput : public OperatorWithParam<YOLOV3DetectionOutput, YOLOV3DetectionOutputParam>
{
public:
    YOLOV3DetectionOutput()
    {
        name_ = "YOLOV3DetectionOutput";
    }
    YOLOV3DetectionOutput(const YOLOV3DetectionOutput& src) = default;

    virtual ~YOLOV3DetectionOutput() {}
    bool InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape,
                    int layout) override;
    void SetSchema(void) override;
};

}    // namespace TEngine

#endif

