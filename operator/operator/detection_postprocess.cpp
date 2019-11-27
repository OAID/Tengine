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
#include "operator/detection_postprocess.hpp"

namespace TEngine {

bool DetectionPostProcess::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape,
                                      int layout)
{
    int max_detections = param_.max_detections;
    int max_classes_per_detection = param_.max_classes_per_detection;
    int num_classes = param_.num_classes;
    int num_detected_boxes = max_detections * max_classes_per_detection;

    const std::vector<int>& in_dim1 = ishape[0].GetDim();
    const std::vector<int>& in_dim2 = ishape[1].GetDim();

    // Only support: batch_size == 1 && num_coord == 4
    if(in_dim1[0] != 1 || in_dim1[2] != 4 || in_dim2[0] != 1 || in_dim2[1] != in_dim1[1] ||
       in_dim2[2] != num_classes + 1)
        return false;

    TShape shape;
    std::vector<int> dim1 = {1, num_detected_boxes, 4};
    std::vector<int> dim2 = {1, num_detected_boxes};
    std::vector<int> dim3 = {1, num_detected_boxes};
    std::vector<int> dim4 = {1};

    shape.SetDataLayout(layout);
    shape.SetDim(dim1);
    oshape[0] = shape;
    shape.SetDim(dim2);
    oshape[1] = shape;
    shape.SetDim(dim3);
    oshape[2] = shape;
    shape.SetDim(dim4);
    oshape[3] = shape;

    return true;
}

void DetectionPostProcess::SetSchema(void)
{
    Input({"input:float32"}).Output({"output:float32"}).SetDoc(R"DOC(DetectionPostProcess Layer)DOC");
}

}    // namespace TEngine
