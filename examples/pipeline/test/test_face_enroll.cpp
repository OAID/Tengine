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
 * Copyright (c) 2021
 * Author: tpoisonooo
 */
#include "pipeline/actor/image_stream.h"
#include "pipeline/actor/face_detection.h"
#include "pipeline/actor/face_landmark.h"
#include "pipeline/actor/face_feature.h"
#include "pipeline/graph/graph.h"
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace pipeline;

namespace pipeline {

class SaveFeature : public Node<Param<Feature>, Param<void> >
{
public:
    SaveFeature()
    {
    }

    void exec() override
    {
        size_t idx = 0;
        Feature feat;
        auto suc = input<0>()->pop(feat);
        if (not suc)
        {
            return;
        }

        char filename[64] = {0};
        sprintf(filename, "feature%ld.bin", idx++);
        feat.serialize(std::string(filename));
    }
};

} // namespace pipeline

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: ./pipeline_enroll_face IMAGE_DIR \n");
        return -1;
    }

    Graph g;
    auto images = g.add_node<ImageStream>(argv[1]);
    auto detect_face = g.add_node<FaceDetection, std::string>("rfb-320.tmfile");
    auto landmark_face = g.add_node<FaceLandmark, std::string>("landmark.tmfile");
    auto feature_face = g.add_node<FaceFeature, std::string>("mobilefacenet.tmfile");
    auto save = g.add_node<SaveFeature>();

    auto image_det = g.add_edge<InstantEdge<cv::Mat> >(100);
    auto det_lmk = g.add_edge<InstantEdge<std::tuple<cv::Mat, std::vector<cv::Rect> > > >(100);
    auto lmk_feature = g.add_edge<InstantEdge<std::tuple<cv::Mat, std::vector<Feature> > > >(100);
    auto feature_save = g.add_edge<InstantEdge<Feature> >(100);

    images->set_output<0>(image_det);
    detect_face->set_input<0>(image_det);
    detect_face->set_output<0>(det_lmk);
    landmark_face->set_input<0>(det_lmk);
    landmark_face->set_output<0>(lmk_feature);
    feature_face->set_input<0>(lmk_feature);
    feature_face->set_output<0>(feature_save);
    save->set_input<0>(feature_save);

    g.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50000));
}
