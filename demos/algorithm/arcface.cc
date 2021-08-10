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

#include "arcface.hpp"

#include "utilities/affine.hpp"
#include "utilities/permute.hpp"
#include "utilities/timer.hpp"

#include <cmath>

const int width = 112;
const int height = 112;

const float scale[] = { 1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f };
const float shift[] = { 127.5f, 127.5f, 127.5f };


bool recognition::load(const std::string &model, const std::string &device)
{
    int ret = init_tengine();
    if (0 != ret)
    {
        fprintf(stderr, "Init tengine failed(%d).\n", ret);
        return false;
    }

    this->context = create_context("ctx", 1);
    if (!device.empty())
    {
        ret = set_context_device(context, device.c_str(), nullptr, 0);
        if (0 != ret)
        {
            fprintf(stderr, "Set context device failed.\n");
            return false;
        }
    }

    this->graph = create_graph(this->context, "tengine", model.c_str());
    if (nullptr == this->graph)
    {
        fprintf(stderr, "Load model failed.\n");
        return false;
    }

    this->input_tensor = get_graph_input_tensor(this->graph, 0, 0);
    this->output_tensor = get_graph_output_tensor(this->graph, 0, 0);
    if (nullptr == this->input_tensor)
    {
        fprintf(stderr, "Get input tensor failed.\n");
        return false;
    }
    if (nullptr == this->output_tensor)
    {
        fprintf(stderr, "Get output tensor failed.\n");
        return false;
    }

    int input_dims[] = { 1, 3, height, width };
    ret = set_tensor_shape(this->input_tensor, input_dims, 4);
    if (0 != ret)
    {
        fprintf(stderr, "Set input tensor shape failed.\n");
        return false;
    }

    this->input_affine.resize(width * height * 3);
    this->input_uint8_buffer.resize(width * height * 3);
    this->input_float_buffer.resize(width * height * 3);

    const int tensor_type = get_tensor_data_type(this->input_tensor);
    if (TENGINE_DT_FP32 == tensor_type)
    {
        ret = set_tensor_buffer(this->input_tensor, this->input_float_buffer.data(), (int)(this->input_float_buffer.size() * sizeof(float)));
        if (0 != ret)
        {
            fprintf(stderr, "Set input tensor buffer failed.\n");
            return false;
        }
    }
    else
    {
        ret = set_tensor_buffer(this->input_tensor, this->input_uint8_buffer.data(), (int)(this->input_uint8_buffer.size() * sizeof(uint8_t)));
        if (0 != ret)
        {
            fprintf(stderr, "Set input tensor buffer failed.\n");
            return false;
        }
    }

    options_t opt = { 0 };
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.affinity = 0;
    opt.num_thread = 4;
    opt.precision = TENGINE_DT_FP32;

    ret = prerun_graph_multithread(this->graph, opt);
    if (0 != ret)
    {
        fprintf(stderr, "Pre-run graph failed.\n");
        return false;
    }

    int output_dims[MAX_SHAPE_DIM_NUM];
    int output_dim_count = get_tensor_shape(this->output_tensor, output_dims, MAX_SHAPE_DIM_NUM);
    if (0 > output_dim_count)
    {
        fprintf(stderr, "Get output tensor shape failed.\n");
        return false;
    }

    int output_element_count = 1;
    for (int i = 0; i < output_dim_count; i++)
    {
        output_element_count *= output_dims[i];
    }

    this->output_uint8_buffer.resize(output_element_count);
    this->output_float_buffer.resize(output_element_count);

    if (TENGINE_DT_FP32 != tensor_type)
    {
        get_tensor_quant_param(input_tensor, &this->input_scale, &this->input_zp, 1);
        get_tensor_quant_param(output_tensor, &this->output_scale, &this->output_zp, 1);
    }

    if (TENGINE_DT_FP32 == tensor_type)
    {
        ret = set_tensor_buffer(this->output_tensor, this->output_float_buffer.data(), (int)(this->output_float_buffer.size() * sizeof(float)));
        if (0 != ret)
        {
            fprintf(stderr, "Set output tensor buffer failed.\n");
            return false;
        }
    }
    else
    {
        ret = set_tensor_buffer(this->output_tensor, this->output_uint8_buffer.data(), (int)(this->output_uint8_buffer.size() * sizeof(uint8_t)));
        if (0 != ret)
        {
            fprintf(stderr, "Set output tensor buffer failed.\n");
            return false;
        }
    }

    return true;
}


recognition::~recognition()
{
    if (nullptr != this->graph)
    {
        destroy_graph(this->graph);
    }

    if (nullptr != this->context)
    {
        destroy_context(this->context);
    }
}


bool recognition::get_feature(const cv::Mat &image, const Coordinate *landmark, std::vector<float> &feature)
{
    cv::Mat image_affine(height, width, CV_8UC3, this->input_affine.data());

    auto ret = affine(image, image_affine, landmark);
    if (!ret)
    {
        fprintf(stderr, "Affine image failed.\n");
        return false;
    }

//    static int i = 0;
//    cv::imwrite("test_affine[" + std::to_string(i++) + "].bmp", image_affine);

    return get_feature_std(image_affine, feature);
}


bool recognition::get_feature_std(const cv::Mat &image, std::vector<float> &feature)
{
    cv::Mat image_permute(height, width, CV_8UC3, this->input_uint8_buffer.data());

    auto ret = permute(image, image_permute, true);
    if (!ret)
    {
        fprintf(stderr, "Permute image failed.\n");
        return false;
    }

    for (int i = 0; i < image.channels(); i++)
    {
        auto planar_size = width * height;

        uint8_t* input_ptr  = this->input_uint8_buffer.data() + planar_size * i;
        float*   output_ptr = this->input_float_buffer.data() + planar_size * i;

        for (int p = 0; p < planar_size; p++)
        {
            output_ptr[p] = ((float)input_ptr[p] - shift[i]) * scale[i];
        }
    }

    const int tensor_type = get_tensor_data_type(this->input_tensor);
    if (TENGINE_DT_UINT8 == tensor_type)
    {
        for (int i = 0; i < image.channels(); i++)
        {
            auto planar_size = width * height;

            float*   input_ptr  = this->input_float_buffer.data() + planar_size * i;
            uint8_t* output_ptr = this->input_uint8_buffer.data() + planar_size * i;

            for (int p = 0; p < planar_size; p++)
            {
                float val = input_ptr[p];

                int val_round = (int)(std::round(val / this->input_scale + (float)this->input_zp));
                if (0 <= val_round && 255 >= val_round)
                {
                    output_ptr[p] = val_round;
                }
                else
                {
                    if (255 < val_round) { output_ptr[p] = 255; }
                    if (  0 > val_round) { output_ptr[p] =   0; }
                }
            }
        }
    }

	Timer timer;
    int flag = run_graph(this->graph, 1);
    fprintf(stdout, "Cost %.2fms", timer.Cost());
    if (0 != flag)
    {
        fprintf(stderr, "Run graph failed.\n");
        return false;
    }

    if (TENGINE_DT_UINT8 == tensor_type)
    {
        for (size_t i = 0; i < this->output_uint8_buffer.size(); i++)
        {
            this->output_float_buffer[i] = (float)((int)this->output_uint8_buffer[i] - this->output_zp) * this->output_scale;
        }
    }

    if (feature.size() != this->output_float_buffer.size())
    {
        feature.resize(this->output_float_buffer.size());
    }

    for (size_t i = 0; i < this->output_float_buffer.size(); i++)
    {
        feature[i] = this->output_float_buffer[i];
    }

    return true;
}
