/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: jingyou@openailab.com
 */
#ifndef __DETECTION_POSTPROCESS_PARAM_HPP__
#define __DETECTION_POSTPROCESS_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct DetectionPostProcessParam : public NamedParam
{
    int max_detections;
    int max_classes_per_detection;
    float nms_score_threshold;
    float nms_iou_threshold;
    int num_classes;
    std::vector<float> scales;    // y_scale, x_scale, h_scale, w_scale

    DECLARE_PARSER_STRUCTURE(DetectionPostProcessParam)
    {
        DECLARE_PARSER_ENTRY(max_detections);
        DECLARE_PARSER_ENTRY(max_classes_per_detection);
        DECLARE_PARSER_ENTRY(nms_score_threshold);
        DECLARE_PARSER_ENTRY(nms_iou_threshold);
        DECLARE_PARSER_ENTRY(num_classes);
    };
};

}    // namespace TEngine

#endif
