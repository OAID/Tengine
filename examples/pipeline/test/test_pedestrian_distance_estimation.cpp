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
#include "pipeline/actor/draw_video.h"
#include "pipeline/actor/pedestrian_detection.h"
#include "pipeline/actor/video_camera.h"
#include "pipeline/actor/spatial_distance_calculation.h"
#include "pipeline/graph/graph.h"
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace pipe;

#define HEIGHT (480)
#define WIDTH  (640)

int main()
{
    Graph g;
    auto cam = g.add_node<VideoCamera>();
    auto draw = g.add_node<DrawVideo>();

    // build preproc
    using preproc_func = std::function<void(const cv::Mat&, cv::Mat&)>;
    preproc_func preproc = [](const cv::Mat& in, cv::Mat& out) -> void {
        cv::Mat buf(out.rows, out.cols, CV_8UC3);
        cv::resize(in, buf, buf.size());
        cv::cvtColor(buf, buf, CV_BGR2RGB);

        buf.convertTo(buf, CV_32FC3);

        const float mean[3] = {127.5f, 127.5f, 127.5f};
        const float scale[3] = {0.007843f, 0.007843f, 0.007843f};

        float* img_data = reinterpret_cast<float*>(buf.data);
        float* out_ptr = reinterpret_cast<float*>(out.data);
        /* nhwc to nchw */
        for (int h = 0; h < out.rows; h++)
        {
            for (int w = 0; w < out.cols; w++)
            {
#pragma unroll(3)
                for (int c = 0; c < 3; c++)
                {
                    int in_index = h * out.cols * 3 + w * 3 + c;
                    int out_index = c * out.cols * out.rows + h * out.cols + w;
                    out_ptr[out_index] = (img_data[in_index] - mean[c]) * scale[c];
                }
            }
        }
    };

    // build postproc
    using postproc_func = std::function<std::vector<Box<int> >(const float* outdata, int num)>;
    postproc_func postproc = [](const float* outdata, int num) -> std::vector<Box<int> > {
        const char* class_names[] = {
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train",
            "tvmonitor"};

        int raw_h = HEIGHT;
        int raw_w = WIDTH;

        const int max_num = num;
        Box<int> boxes[max_num];
        for (int i = 0; i < max_num; ++i)
        {
            boxes[i] = {0};
        }
        int box_count = 0;

        fprintf(stderr, "detect result num: %d \n", num);
        for (int i = 0; i < num; i++)
        {
            if (outdata[1] >= 0.5f)
            {
                Box<int> box;

                box.class_idx = outdata[0];
                box.score = outdata[1];
                box.x0 = outdata[2] * raw_w;
                box.y0 = outdata[3] * raw_h;
                box.x1 = outdata[4] * raw_w;
                box.y1 = outdata[5] * raw_h;

                boxes[box_count] = box;
                box_count++;

                fprintf(stderr, "%s\t:%.1f%%\n", class_names[box.class_idx],
                        box.score * 100);
                fprintf(stderr, "BOX:( %d , %d ),( %d , %d )\n", box.x0, box.y0, box.x1,
                        box.y1);
            }
            outdata += 6;
        }

        Box<int> max = {0};
        for (int i = 0; i < box_count; i++)
        {
            if (boxes[i].score > max.score)
            {
                max = boxes[i];
            }
        }

        std::vector<Box<int> > ret = {max};
        return ret;
    };

    auto detect_ped = g.add_node<PedestrianDetection, std::string, preproc_func, postproc_func>("mobilenet_ssd.tmfile", std::move(preproc), std::move(postproc));
    auto dist_estimate = g.add_node<SpatialDistanceCalc>();

    auto cam_det = g.add_edge<InstantEdge<cv::Mat> >(100);
    auto det_dist = g.add_edge<InstantEdge<std::tuple<cv::Mat, cv::Rect> > >(100);
    auto dist_draw = g.add_edge<InstantEdge<cv::Mat> >(100);

    cam->set_output<0>(cam_det);
    detect_ped->set_input<0>(cam_det);
    detect_ped->set_output<0>(det_dist);
    dist_estimate->set_input<0>(det_dist);
    dist_estimate->set_output<0>(dist_draw);
    draw->set_input<0>(dist_draw);

    g.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(600000));
}
