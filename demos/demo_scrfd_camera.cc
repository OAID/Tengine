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

#include <opencv2/opencv.hpp>


const float DET_THRESHOLD   =   0.3f;
const float NMS_THRESHOLD   =   0.45f;

const int MODEL_WIDTH       =   384;
const int MODEL_HEIGHT      =   640;

#define MODEL_PATH  "models/scrfd_2.5g_bnkps_mm.tmfile"


int main(int argc, char* argv[])
{
    cmdline::parser cmd;

    cmd.add<std::string>("model", 'm', "model config file", false, MODEL_PATH);
    cmd.add<std::string>("device", 'd', "device", false, "TIMVX");
    cmd.add<float>("score_threshold", 's', "score threshold", false, DET_THRESHOLD);
    cmd.add<float>("iou_threshold", 'o', "iou threshold", false, NMS_THRESHOLD);

    cmd.parse_check(argc, argv);

    auto model_path = cmd.get<std::string>("model");
    auto device = cmd.get<std::string>("device");
    auto score_threshold = cmd.get<float>("score_threshold");
    auto iou_threshold = cmd.get<float>("iou_threshold");

    fprintf(stdout, "Init tengine...\n");
    init_tengine();
    fprintf(stdout, "Tengine was inited.\n");

    cv::VideoCapture vp(cv::CAP_ANY);
    if (!vp.isOpened())
    {
        printf("Open camera error.\n");

        return -1;
    }

    fprintf(stdout, "Camera is opened.\n");

#if CV_VERSION_MAJOR > 3
    vp.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, 1920);
    vp.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, 1080);
    cv::namedWindow("frame", cv::WindowFlags::WINDOW_NORMAL);
    cv::setWindowProperty("frame", cv::WindowPropertyFlags::WND_PROP_FULLSCREEN, cv::WindowFlags::WINDOW_FULLSCREEN);
#else
    vp.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    vp.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    vp.set(CV_CAP_PROP_FPS, 30);
    cv::namedWindow("frame", CV_WINDOW_NORMAL);
    cv::setWindowProperty("frame", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
#endif

    cv::Mat image_ori, image_flip;
    vp >> image_ori;

    if (image_ori.empty())
    {
        fprintf(stderr, "Reading image from camera was failed.\n");
        return -1;
    }


    fprintf(stdout, "Load detection model...\n");

    cv::Size input_shape(MODEL_WIDTH, MODEL_HEIGHT);

    SCRFD detector;
    auto ret = detector.Load(model_path, input_shape, device.c_str());
    if (!ret)
    {
        fprintf(stderr, "Load model(%s) failed.\n", model_path.c_str());
        return -1;
    }

    std::vector<Face> faces;

    while (true)
    {
        vp >> image_ori;
        //fprintf(stdout, "Get a frame(%d, %d).\n", image_ori.cols, image_ori.rows);

        cv::flip(image_ori, image_ori, 1);
        cv::transpose(image_ori, image_flip);
        // image_flip = image_ori.clone();
        // cv::rotate(image_ori, image_flip, -90);

        Timer det_timer;
        detector.Detect(image_flip, faces, score_threshold, iou_threshold);
        det_timer.Stop();

        for (auto& face : faces)
        {
            fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", face.confidence, face.box.x, face.box.y, face.box.width, face.box.height);

            // box
            cv::Rect2f rect(face.box.x, face.box.y, face.box.width, face.box.height);

            // draw box
            cv::rectangle(image_flip, rect, cv::Scalar(0, 0, 255), 2);
            std::string box_confidence = "DET: " + std::to_string(face.confidence).substr(0, 5);
            cv::putText(image_flip, box_confidence, rect.tl() + cv::Point2f(5, -10), cv::FONT_HERSHEY_TRIPLEX, 0.6f, cv::Scalar(255, 255, 0));

            cv::circle(image_flip, cv::Point(face.landmark[0].x, face.landmark[0].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_flip, cv::Point(face.landmark[1].x, face.landmark[1].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_flip, cv::Point(face.landmark[2].x, face.landmark[2].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_flip, cv::Point(face.landmark[3].x, face.landmark[3].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_flip, cv::Point(face.landmark[4].x, face.landmark[4].y), 2, cv::Scalar(255, 255, 0), -1);
        }

        cv::imshow("frame", image_flip);
        if (27 == cv::waitKey(1))
        {
            break;
        }
    }

    release_tengine();

    return 0;
}
