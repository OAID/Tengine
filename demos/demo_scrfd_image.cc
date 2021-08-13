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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: lswang@openailab.com
 */

#include "algorithm/scrfd.hpp"

#include "utilities/cmdline.hpp"
#include "utilities/timer.hpp"

#include <algorithm>
#include <numeric>

#define VXDEVICE "VX"
const float DET_THRESHOLD   =   0.30f;
const float NMS_THRESHOLD   =   0.45f;

const int MODEL_WIDTH       =   640;
const int MODEL_HEIGHT      =   384;


int main(int argc, char* argv[])
{
    cmdline::parser cmd;

    cmd.add<std::string>("model", 'm', "model config file", true, "");
    cmd.add<std::string>("image", 'i', "image to infer", true, "");
    cmd.add<std::string>("device", 'd', "device", false, "");
    cmd.add<float>("score_threshold", 's', "score threshold", false, DET_THRESHOLD);
    cmd.add<float>("iou_threshold", 'o', "iou threshold", false, NMS_THRESHOLD);

    cmd.parse_check(argc, argv);

    auto model_path = cmd.get<std::string>("model");
    auto image_path = cmd.get<std::string>("image");
    auto device = cmd.get<std::string>("device");
    auto score_threshold = cmd.get<float>("score_threshold");
    auto iou_threshold = cmd.get<float>("iou_threshold");

    cv::Mat image = cv::imread(image_path);

    if (image.empty())
    {
        fprintf(stderr, "Reading image was failed.\n");
        return -1;
    }

    set_log_level(log_level::LOG_DEBUG);
    init_tengine();

    cv::Size input_shape(MODEL_WIDTH, MODEL_HEIGHT);

    SCRFD detector;
    auto ret = detector.Load(model_path, input_shape, device);
    if (!ret)
    {
        fprintf(stderr, "Load model(%s) failed.\n", model_path.c_str());
        return -1;
    }

    std::vector<Face> faces;

    detector.Detect(image, faces, score_threshold, iou_threshold);

    for (auto& face : faces)
    {
        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", face.confidence, face.box.x, face.box.y, face.box.width, face.box.height);

        // box
        cv::Rect2f rect(face.box.x, face.box.y, face.box.width, face.box.height);

        // draw box
        cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
        std::string box_confidence = "DET: " + std::to_string(face.confidence).substr(0, 5);
        cv::putText(image, box_confidence, rect.tl() + cv::Point2f(5, -10), cv::FONT_HERSHEY_TRIPLEX, 0.6f, cv::Scalar(255, 255, 0));

        cv::circle(image, cv::Point(face.landmark[0].x, face.landmark[0].y), 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(image, cv::Point(face.landmark[1].x, face.landmark[1].y), 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(image, cv::Point(face.landmark[2].x, face.landmark[2].y), 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(image, cv::Point(face.landmark[3].x, face.landmark[3].y), 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(image, cv::Point(face.landmark[4].x, face.landmark[4].y), 2, cv::Scalar(255, 255, 0), -1);
    }

    cv::imwrite("demo.png", image);

    release_tengine();

    return 0;
}
