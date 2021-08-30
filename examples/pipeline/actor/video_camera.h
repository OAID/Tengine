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
#pragma once
#include <thread>
#include <mutex>
#include "pipeline/graph/node.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

namespace pipeline {

class VideoCamera : public Node<Param<void>, Param<cv::Mat> >
{
public:
    VideoCamera(const std::string video_path = "", const int h = 480, const int w = 640)
        : m_path(video_path), m_height(h), m_width(w)
    {
    }

    void exec() override
    {
        std::call_once(flag, [&]() {
            cv::VideoCapture cap;
            if (m_path.empty())
            {
                cap.open(0);
            }
            else
            {
                cap.open(m_path);
            }

            if (not cap.isOpened())
            {
                fprintf(stderr, "cannot open video %s\n", m_path.c_str());
                return;
            }

#if CV_VERSION_MAJOR < 4
            cap.set(CV_CAP_PROP_FRAME_WIDTH, m_width);
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, m_height);

            double rate = cap.get(CV_CAP_PROP_FPS);
            fprintf(stdout, "rate %lf\n", rate);
            fprintf(stdout, "pan %lf\n", cap.get(CV_CAP_PROP_PAN));
            fprintf(stdout, "width = %.2f\n", cap.get(CV_CAP_PROP_FRAME_WIDTH));
            fprintf(stdout, "height = %.2f\n", cap.get(CV_CAP_PROP_FRAME_HEIGHT));
            fprintf(stdout, "fbs = %.2f\n", cap.get(CV_CAP_PROP_FPS));
            fprintf(stdout, "brightness = %.2f\n", cap.get(CV_CAP_PROP_BRIGHTNESS));
            fprintf(stdout, "contrast = %.2f\n", cap.get(CV_CAP_PROP_CONTRAST));
            fprintf(stdout, "saturation = %.2f\n", cap.get(CV_CAP_PROP_SATURATION));
            fprintf(stdout, "hue = %.2f\n", cap.get(CV_CAP_PROP_HUE));
            fprintf(stdout, "exposure = %.2f\n", cap.get(CV_CAP_PROP_EXPOSURE));
#else
      double rate = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);
#endif

            while (true)
            {
                cv::Mat mat;
                if (not cap.read(mat))
                {
                    break;
                }

                if (mat.empty())
                {
                    break;
                }

                auto success = output<0>()->try_push(std::move(mat.clone()));
                if (not success)
                {
                    fprintf(stdout, "drop " __FILE__ "\n");
                }
            }

            cv::waitKey(1000 / std::max(1.0, rate));
        });
    }

private:
    std::string m_path;
    int m_height, m_width;
    std::once_flag flag;
};

} // namespace pipeline
