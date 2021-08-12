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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: lswang@openailab.com
 */

#include "tengine/c_api.h"

#include "algorithm/arcface.hpp"

#include "utilities/cmdline.hpp"
#include "utilities/distance.hpp"
#include "utilities/timer.hpp"

#include <fstream>
#include <opencv2/opencv.hpp>

const float DET_THRESHOLD   =   0.30f;
const float NMS_THRESHOLD   =   0.45f;

const int MODEL_WIDTH       =   640;
const int MODEL_HEIGHT      =   384;


int main(int argc, char* argv[])
{
    cmdline::parser cmd;

    cmd.add<std::string>("model", 'm', "recognition model file", true, "");
    cmd.add<std::string>("image", 'i', "image folder", true, "");
    cmd.add<std::string>("list", 'l', "image list", true, "");
    cmd.add<std::string>("save", 's', "save file", true, "");
    cmd.add<std::string>("device", 'd', "device", false, "CPU");

    cmd.parse_check(argc, argv);

    const auto model_path = cmd.get<std::string>("model");
    const auto image_folder = cmd.get<std::string>("image");
    const auto list_path = cmd.get<std::string>("list");
    const auto save_file = cmd.get<std::string>("save");
    const auto device = cmd.get<std::string>("device");

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


    recognition reg;
    auto ret = reg.load(model_path, device);
    if (!ret)
    {
        fprintf(stderr, "Load verify model(%s) failed.\n", model_path.c_str());
        return -1;
    }

    auto feature_file = std::fopen(save_file.c_str(), "w");
    for (int i = 0; i < (int)image_list.size(); i++)
    {

        std::vector<float> feature;

        cv::Mat image = cv::imread(image_folder + "/" + image_list[i]);
        if (image.empty())
        {
            fprintf(stderr, "Reading image was failed.\n");
            return -1;
        }

        ret = reg.get_feature_std(image, feature);
        if (!ret)
        {
            fprintf(stderr, "Get verify feature was failed.\n");
            return -1;
        }
        norm_feature(feature);

        for (const auto& val : feature)
        {
            std::fprintf(feature_file, "%.8f\t", val);
        }
        std::fprintf(feature_file, "\n");

        fprintf(stdout, "\r%d/%d.", i + 1, (int)image_list.size());
    }
    std::fclose(feature_file);

    fprintf(stdout, "\n");

    return 0;
}
