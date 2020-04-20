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
 * Author: qtang@openailab.com
 */

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>

#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "tengine_cpp_api.h"
#include "common_util.hpp"

#include "model_test.h"

#include "cmdline.h"
#include "json.hpp"


int main(int argc, char* argv[])
{
    std::string config_file = "mobilenet.json";

    // parse arg
    if (argc > 2)
    {
        cmdline::parser cmd;

        // parse input arg
        cmd.add<std::string>("config", 'c', "config for model level test", true, "");
        cmd.parse_check(argc, argv);

        config_file = cmd.get<std::string>("config");
    }

    // input param
    std::string model_path, model_input_name, model_output_name;
    float scale[3] = {0}, shift[3] = {0};
    int model_input_width = 0, model_input_height = 0;

    // model config buffer
    std::vector<uint8_t> config_buffer;
    read_file(config_file, config_buffer);
    if (config_buffer.empty())
    {
        printf("config file was empty.\n");
        return -1;
    }

    auto ret = parse_model(config_buffer, model_path, model_input_name, model_output_name, scale, shift, model_input_width, model_input_height);
    if (!ret)
    {
        printf("Parse model details failed.\n");
        return -1;
    }

    // image list
    std::vector<std::string> image_list;

    ret = parse_image_list(config_buffer, image_list);
    if (!ret)
    {
        printf("Parse image list failed.\n");
        return -1;
    }

    std::vector<std::vector<ClassScore>> image_topN(image_list.size()), image_topN_result(image_list.size());

    ret = parse_top_n(config_buffer, image_topN);
    if (!ret)
    {
        printf("Parse top N info failed.\n");
        return -1;
    }

    // sort first
    for (int i = 0; i < image_topN.size(); i++)
    {
        std::stable_sort(image_topN[i].begin(), image_topN[i].end(), sort_class_score);
    }

#ifdef _DEBUG
    printf("Model info:\n");
    printf("Path: %s, input: %s, output: %s.\n", model_path.c_str(), model_input_name.c_str(), model_output_name.c_str());
    printf("Input: w = %d, h = %d.\n", model_input_width, model_input_height);
    printf("Input: w = %d, h = %d.\n", model_input_width, model_input_height);

    for (auto& val : scale)
    {
        printf("Scale: %.4f.\n", val);
    }
    for (auto& val : shift)
    {
        printf("Shift: %.4f.\n", val);
    }

    for (auto& image : image_list)
    {
        printf("Image: %s.\n", image.c_str());
    }
#endif


    // run model
    tengine::Net net;
    tengine::Tensor input_tensor;

    /* prepare input data */
    input_tensor.create(model_input_width, model_input_height, 3);

    /* get result */
    tengine::Tensor output_tensor;

    /* load model */
    auto code = net.load_model(NULL, "tengine", model_path.c_str());
    if (0 != code)
    {
        printf("Load model(%s) failed.\n", model_path.c_str());
        return code;
    }

    /* set device */
    std::string device = "";
    net.set_device(device);

    for (int i = 0; i < image_list.size(); i++)
    {

        get_input_data(image_list[i].c_str(), ( float* )input_tensor.data, model_input_width, model_input_height, shift, scale);

        /* forward */
        net.input_tensor(model_input_name.c_str(), input_tensor);

        net.run();
        net.extract_tensor(model_output_name.c_str(), output_tensor);

        /* after process */
        std::vector<ClassScore> result;
        get_result( (float*)output_tensor.data, result);

        image_topN_result[i] = result;
    }

    for (int i = 0; i < image_list.size(); i++)
    {
        auto& gt = image_topN[i];
        auto& result = image_topN_result[i];

        for (int j = 0; j < 5; j++)
        {
            auto class_is_different = result[j].idx != gt[j].idx;
            auto score_is_different = std::abs(result[j].score - gt[j].score) > 0.01f;

            if (class_is_different)
            {
                printf("Class is not match(%d vs. %d).", gt[j].idx, result[j].idx);
                printf_class_score(gt);
                printf_class_score(result);

                return -1;
            }

            if (score_is_different)
            {
                printf("Score is not match(%.2f vs. %.2f).", gt[j].score, result[j].score);
                printf_class_score(gt);
                printf_class_score(result);

                return -1;
            }
        }
    }

    return 0;
}
