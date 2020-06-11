#ifndef __YOLOV3_DETECTION_OUTPUT_PARAM_HPP__
#define __YOLOV3_DETECTION_OUTPUT_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine{

struct YOLOV3DetectionOutputParam : public NamedParam
{
    int num_classes;
    int num_box;
    float confidence_threshold;
    float nms_threshold;
    std::vector<float> bias;
    std::vector<float> mask;
    std::vector<float> anchors_scale;
    int mask_group_num;

    DECLARE_PARSER_STRUCTURE(YOLOV3DetectionOutputParam)
    {
        DECLARE_PARSER_ENTRY(num_classes);
        DECLARE_PARSER_ENTRY(num_box);
        DECLARE_PARSER_ENTRY(nms_threshold);
        DECLARE_PARSER_ENTRY(bias);
        DECLARE_PARSER_ENTRY(mask);
        DECLARE_PARSER_ENTRY(anchors_scale);
        DECLARE_PARSER_ENTRY(confidence_threshold);
        DECLARE_PARSER_ENTRY(mask_group_num);
    }
};
}

#endif
