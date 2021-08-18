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
using namespace pipeline;

#define HEIGHT (480)
#define WIDTH  (640)

int main()
{
    Graph g;
    auto cam = g.add_node<VideoCamera>();
    auto draw = g.add_node<DrawVideo>();

    auto detect_ped = g.add_node<PedestrianDetection, std::string>("mobilenet_ssd.tmfile");
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
