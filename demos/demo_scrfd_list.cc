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
#include <fstream>

#define VXDEVICE "VX"
const float DET_THRESHOLD   =   0.30f;
const float NMS_THRESHOLD   =   0.45f;

const int MODEL_WIDTH       =   640;
const int MODEL_HEIGHT      =   384;


int main(int argc, char* argv[])
{
    cmdline::parser cmd;

    cmd.add<std::string>("model", 'm', "model file", true, "");
    cmd.add<std::string>("image", 'i', "image folder", true, "");
    cmd.add<std::string>("proposal", 'p', "proposal folder", true, "");
    cmd.add<std::string>("list", 'l', "image list", true, "");
    cmd.add<std::string>("device", 'd', "device", false, "");
    cmd.add<float>("score_threshold", 's', "score threshold", true, DET_THRESHOLD);
    cmd.add<float>("iou_threshold", 'o', "iou threshold", true, NMS_THRESHOLD);

    cmd.parse_check(argc, argv);

    auto model_path = cmd.get<std::string>("model");
    auto image_folder = cmd.get<std::string>("image");
    auto proposal_folder = cmd.get<std::string>("proposal");
    auto list_path = cmd.get<std::string>("list");
    auto device = cmd.get<std::string>("device");
    auto score_threshold = cmd.get<float>("score_threshold");
    auto iou_threshold = cmd.get<float>("iou_threshold");

    std::vector<std::string> image_list;
    std::ifstream list_stream(list_path);
    if (!list_stream.is_open())
    {
        fprintf(stderr, "Open image list failed.\n");
        return -1;
    }

    std::string image_path;
    while (std::getline(list_stream, image_path))
    {
        image_list.push_back(image_path);
    }

    init_tengine();

    SCRFD detector;

    cv::Size input_shape(MODEL_WIDTH, MODEL_HEIGHT);
    auto ret = detector.Load(model_path, input_shape, device);
    if (!ret)
    {
        fprintf(stderr, "Load model(%s) failed.\n", model_path.c_str());
        return -1;
    }

    int i = 1;

    for (const auto& file : image_list)
    {
        std::string image_file_path = image_folder;
        image_file_path.append("/").append(file);

        cv::Mat image = cv::imread(image_file_path);
        if (image.empty())
        {
            fprintf(stderr, "Reading image was failed.\n");
            return -1;
        }

        std::vector<Face> faces;
        detector.Detect(image, faces, score_threshold, iou_threshold);

        std::string file_name(file.c_str(), file.size() - 3);
        file_name += "txt";

        std::string proposal_file_path = proposal_folder;
        proposal_file_path.append("/").append(file_name);

        std::ofstream proposal_stream(proposal_file_path);

        for (auto& face : faces)
        {
            cv::Point2f tl(face.box.x, face.box.y);
            cv::Point2f br(tl.x + face.box.width, tl.y + face.box.height);

            char line_buffer[128] = { 0 };
            sprintf(line_buffer, "%.4f %.8f %.8f %.8f %.8f\n", face.confidence, tl.x, tl.y, br.x, br.y);

            proposal_stream.write(line_buffer, (std::streamsize)strlen(line_buffer));
        }
        proposal_stream.close();

        fprintf(stdout, "\rProcessing: %4d/%d.", i++, (int)image_list.size());
    }

    fprintf(stdout, "\n");

    release_tengine();

    return 0;
}
