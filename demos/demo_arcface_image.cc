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

#include "algorithm/scrfd.hpp"
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

    cmd.add<std::string>("detect_model", 'd', "detection model file", true, "");
    cmd.add<std::string>("verify_model", 'v', "verify model file", true, "");
    cmd.add<std::string>("register_image", 'r', "register image file", true, "");
    cmd.add<std::string>("verify_image", 'i', "verify image file", true, "");
    cmd.add<std::string>("attack_image", 'a', "attack image file", true, "");
    cmd.add<std::string>("device", 'p', "device", false, "CPU");

    cmd.parse_check(argc, argv);

    auto detect_model = cmd.get<std::string>("detect_model");
    auto verify_model = cmd.get<std::string>("verify_model");
    auto register_image = cmd.get<std::string>("register_image");
    auto verify_image = cmd.get<std::string>("verify_image");
    auto attack_image = cmd.get<std::string>("attack_image");
    auto device = cmd.get<std::string>("device");


    //set_log_level(log_level::LOG_DEBUG);
    init_tengine();

    cv::Size detect_shape = { MODEL_WIDTH, MODEL_HEIGHT };

    SCRFD detector;
    auto ret = detector.Load(detect_model, detect_shape, device);
    if (!ret)
    {
        fprintf(stderr, "Load detection model(%s) failed.\n", detect_model.c_str());
        return -1;
    }

    recognition reg;
    ret = reg.load(verify_model, device);
    if (!ret)
    {
        fprintf(stderr, "Load verify model(%s) failed.\n", detect_model.c_str());
        return -1;
    }

    std::vector<Face> faces;
    std::vector<float> reg_feature, ver_feature, att_feature;

    cv::Mat image = cv::imread(register_image);
    if (image.empty())
    {
        fprintf(stderr, "Read register image was failed.\n");
        return -1;
    }

    detector.Detect(image, faces, DET_THRESHOLD, NMS_THRESHOLD);
    for (auto& face : faces)
    {
        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", face.confidence, face.box.x, face.box.y, face.box.width, face.box.height);
        ret = reg.get_feature(image, face.landmark, reg_feature);
        if (!ret)
        {
            fprintf(stderr, "Get register feature was failed.\n");
            return -1;
        }

        norm_feature(reg_feature);
    }

    image = cv::imread(verify_image);
    if (image.empty())
    {
        fprintf(stderr, "Read verify image was failed.\n");
        return -1;
    }

    detector.Detect(image, faces, DET_THRESHOLD, NMS_THRESHOLD);
    for (auto& face : faces)
    {
        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", face.confidence, face.box.x, face.box.y, face.box.width, face.box.height);
        ret = reg.get_feature(image, face.landmark, ver_feature);
        if (!ret)
        {
            fprintf(stderr, "Get verify feature was failed.\n");
            return -1;
        }

        norm_feature(ver_feature);
    }

    image = cv::imread(attack_image);
    if (image.empty())
    {
        fprintf(stderr, "Read attack image was failed.\n");
        return -1;
    }

    detector.Detect(image, faces, DET_THRESHOLD, NMS_THRESHOLD);
    for (auto& face : faces)
    {
        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", face.confidence, face.box.x, face.box.y, face.box.width, face.box.height);
        ret = reg.get_feature(image, face.landmark, att_feature);
        if (!ret)
        {
            fprintf(stderr, "Get attack feature was failed.\n");
            return -1;
        }

        norm_feature(att_feature);
    }

    auto dis_reg_ver = distance(reg_feature, ver_feature);
    auto dis_reg_att = distance(reg_feature, att_feature);
    auto dis_ver_att = distance(ver_feature, att_feature);
    fprintf(stderr, "distance: %.4f, %.4f, %.4f\n", dis_reg_ver, dis_reg_att, dis_ver_att);

    return 0;
}
