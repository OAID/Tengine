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

#include "scrfd.hpp"

#include "utilities/letterbox.hpp"
#include "utilities/permute.hpp"
#include "utilities/nms.hpp"
#include "utilities/timer.hpp"

#include <cmath>

//#define _DEBUG_


const float DEFAULT_SCORE_THRESHOLD = 0.45f;
const float DEFAULT_NMS_THRESHOLD = 0.3f;

const float MODEL_MEANS[]  = { 127.5f, 127.5f, 127.5f };
const float MODEL_SCALES[] = { 1 / 128.f, 1 / 128.f, 1 / 128.f };


SCRFD::SCRFD()
{
    this->context = nullptr;
    this->graph   = nullptr;

    this->is_quantization = false;

    this->input_scale       = 0.f;
    this->input_zp          = 0;

    this->score_threshold   = DEFAULT_SCORE_THRESHOLD;
    this->nms_threshold     = DEFAULT_NMS_THRESHOLD;

    this->canvas_width      = 0;
    this->canvas_height     = 0;

    this->canvas_width_gap  = 0.f;
    this->canvas_height_gap = 0.f;

    this->input_ratio       = 0.f;

    this->scales            = { 1.f, 2.f };
    this->ratios            = { 1.f };

    this->strides           = {  8, 16,  32 };
    this->bases             = { 16, 64, 256 };

    this->score_name        =  { "446", "466", "486" };
    this->bbox_name         =  { "449", "469", "489" };
    this->landmark_name     =  { "452", "472", "492" };
}


SCRFD::~SCRFD()
{
    postrun_graph(this->graph);
    destroy_context(this->context);
}


bool SCRFD::Load(const std::string& model, const cv::Size& input_shape, const std::string& device)
{
    this->canvas_width   = input_shape.width;
    this->canvas_height  = input_shape.height;

    this->score_threshold = DEFAULT_SCORE_THRESHOLD;
    this->nms_threshold   = DEFAULT_NMS_THRESHOLD;

    this->context = create_context("ctx", 1);

    if (!device.empty())
    {
        int ret = set_context_device(this->context, device.c_str(), nullptr, 0);
        if (0 != ret)
        {
            fprintf(stderr, "Set context device running in %s failed.\n", device.c_str());
            return false;
        }
    }

    this->graph = create_graph(this->context, "tengine", model.c_str());
    if (nullptr == this->graph)
    {
        fprintf(stderr, "Load model %s failed.\n", model.c_str());
        return false;
    }

    auto canvas_ret = init_canvas(this->canvas_width, this->canvas_height, 3);
    if (!canvas_ret)
    {
        fprintf(stderr, "Init canvas failed.\n");
        return false;
    }

    Timer timer;
    auto ret = prerun_graph(this->graph);
    if (0 != ret)
    {
        fprintf(stdout, "Prerun graph failed(%d).", ret);
        return false;
    }

    dump_graph(this->graph);

    auto buffer_ret = init_buffer();
    if (!buffer_ret)
    {
        fprintf(stderr, "Init buffer failed.\n");
        return false;
    }

    auto anchor_ret = init_anchor();
    if (!anchor_ret)
    {
        fprintf(stderr, "Init anchors failed.\n");
        return false;
    }

    return true;
}


bool SCRFD::init_canvas(int width, int height, int channel)
{
    this->canvas_width = width;
    this->canvas_height = height;

    this->canvas_uint8.resize(this->canvas_width * this->canvas_height * channel);
    this->canvas_float.resize(this->canvas_uint8.size());
    this->resized_container.resize(this->canvas_uint8.size());
    this->resized_permute_container.resize(this->canvas_uint8.size());

    tensor_t input_tensor = get_graph_input_tensor(this->graph, 0, 0);

    if (nullptr == input_tensor)
    {
        fprintf(stderr, "Get input tensor failed.\n");
        return false;
    }

    int input_dims[] = { 1, 3, this->canvas_height, this->canvas_width };
    int ret = set_tensor_shape(input_tensor, input_dims, 4);
    if (0 != ret)
    {
        fprintf(stderr, "Set input tensor shape failed.\n");
        return false;
    }

    const int tensor_type = get_tensor_data_type(input_tensor);
    if (TENGINE_DT_FP32 == tensor_type)
    {
        ret = set_tensor_buffer(input_tensor, this->canvas_float.data(), (int)(this->canvas_float.size() * sizeof(float)));
        if (0 != ret)
        {
            fprintf(stderr, "Set input tensor buffer failed.\n");
            return false;
        }
    }
    else
    {
        ret = set_tensor_buffer(input_tensor, this->canvas_uint8.data(), (int)(this->canvas_uint8.size() * sizeof(uint8_t)));
        if (0 != ret)
        {
            fprintf(stderr, "Set input tensor buffer failed.\n");
            return false;
        }
    }

    return true;
}


bool SCRFD::init_buffer()
{
    this->score_scale.resize(strides.size());
    this->bbox_scale.resize(strides.size());
    this->landmark_scale.resize(strides.size());

    this->score_zp.resize(strides.size());
    this->bbox_zp.resize(strides.size());
    this->landmark_zp.resize(strides.size());

    this->score_buffer.resize(strides.size());
    this->bbox_buffer.resize(strides.size());
    this->landmark_buffer.resize(strides.size());

    for (int i = 0; i < this->strides.size(); i++)
    {
        auto score_tensor    = get_graph_tensor(this->graph, this->score_name[i].c_str());
        auto bbox_tensor     = get_graph_tensor(this->graph, this->bbox_name[i].c_str());
        auto landmark_tensor = get_graph_tensor(this->graph, this->landmark_name[i].c_str());

        if (nullptr == score_tensor)
        {
            fprintf(stderr, "Get output score tensor(%s) at stride(%d) failed.\n", this->score_name[i].c_str(), this->strides[i]);
            return false;
        }
        if (nullptr == bbox_tensor)
        {
            fprintf(stderr, "Get output score tensor(%s) at stride(%d) failed.\n", this->bbox_name[i].c_str(), this->strides[i]);
            return false;
        }
        if (nullptr == landmark_tensor)
        {
            fprintf(stderr, "Get output score tensor(%s) at stride(%d) failed.\n", this->landmark_name[i].c_str(), this->strides[i]);
            return false;
        }

        auto tensor_type = get_tensor_data_type(score_tensor);
        if (TENGINE_DT_FP32 != tensor_type)
        {
            this->is_quantization = true;

            auto score_ret    = get_tensor_quant_param(score_tensor, &(this->score_scale[i]), &(this->score_zp[i]), 1);
            auto bbox_ret     = get_tensor_quant_param(bbox_tensor, &(this->bbox_scale[i]), &(this->bbox_zp[i]), 1);
            auto landmark_ret = get_tensor_quant_param(landmark_tensor, &(this->landmark_scale[i]), &(this->landmark_zp[i]), 1);

            if (0 > score_ret)
            {
                fprintf(stderr, "Get output score tensor(%s) quant param at stride(%d) failed.\n", this->score_name[i].c_str(), this->strides[i]);
                return false;
            }
            if (0 > bbox_ret)
            {
                fprintf(stderr, "Get output bbox tensor(%s) quant param at stride(%d) failed.\n", this->bbox_name[i].c_str(), this->strides[i]);
                return false;
            }
            if (0 > landmark_ret)
            {
                fprintf(stderr, "Get output landmark tensor(%s) quant param at stride(%d) failed.\n", this->landmark_name[i].c_str(), this->strides[i]);
                return false;
            }
        }

        auto scale_buffer_size    = get_tensor_buffer_size(score_tensor);
        auto bbox_buffer_size     = get_tensor_buffer_size(bbox_tensor);
        auto landmark_buffer_size = get_tensor_buffer_size(landmark_tensor);

        if (!this->is_quantization)
        {
            scale_buffer_size /= sizeof(float);
            bbox_buffer_size /= sizeof(float);
            landmark_buffer_size /= sizeof(float);
        }

        this->score_buffer[i].resize(scale_buffer_size);
        this->bbox_buffer[i].resize(bbox_buffer_size);
        this->landmark_buffer[i].resize(landmark_buffer_size);

        int score_shape[MAX_SHAPE_DIM_NUM] = { 0 };
        int bbox_shape[MAX_SHAPE_DIM_NUM] = { 0 };
        int landmark_shape[MAX_SHAPE_DIM_NUM] = { 0 };

        auto score_ret = get_tensor_shape(score_tensor, score_shape, MAX_SHAPE_DIM_NUM);
        auto bbox_ret = get_tensor_shape(bbox_tensor, bbox_shape, MAX_SHAPE_DIM_NUM);
        auto landmark_ret = get_tensor_shape(landmark_tensor, landmark_shape, MAX_SHAPE_DIM_NUM);

        if (0 > score_ret)
        {
            fprintf(stderr, "Get output score tensor(%s) shape at stride(%d) failed.\n", this->score_name[i].c_str(), this->strides[i]);
            return false;
        }
        if (0 > bbox_ret)
        {
            fprintf(stderr, "Get output bbox tensor(%s) shape at stride(%d) failed.\n", this->bbox_name[i].c_str(), this->strides[i]);
            return false;
        }
        if (0 > landmark_ret)
        {
            fprintf(stderr, "Get output landmark tensor(%s) shape at stride(%d) failed.\n", this->landmark_name[i].c_str(), this->strides[i]);
            return false;
        }

//        fprintf(stdout, "Score shape is [ %d, %2d, %2d, %2d ]; bbox shape is [ %d, %2d, %2d, %2d ], landmark shape is [ %d, %2d, %2d, %2d ].\n",
//                score_shape[0],    score_shape[1],    score_shape[2],    score_shape[3],
//                bbox_shape[0],     bbox_shape[1],     bbox_shape[2],     bbox_shape[3],
//                landmark_shape[0], landmark_shape[1], landmark_shape[2], landmark_shape[3]);
    }

    return true;
}


bool SCRFD::init_anchor()
{
    this->anchors.resize(this->strides.size());
    for (auto& val : this->anchors)
    {
        val.resize(this->ratios.size() * this->scales.size());
    }

    float x1, y1, x2, y2;
    const auto cx = 0, cy = 0;

    for (int i = 0; i < this->anchors.size(); i++)
    {
        auto& anchor = this->anchors[i];

        for (int j = 0; j < this->ratios.size(); j++)
        {
            float r = this->ratios[j];
            auto r_h = this->bases[j] * std::sqrt(r);
            auto r_w = this->bases[j] / std::sqrt(r);

            for (int k = 0; k < this->scales.size(); k++)
            {
                auto scale = this->scales[k];
                auto rs_w = r_w * scale;
                auto rs_h = r_h * scale;

                std::array<float, 4> point = { cx - rs_w * 0.5f, cx - rs_h * 0.5f, cx + rs_w * 0.5f, cx + rs_h * 0.5f };

                auto offset = j * this->scales.size() + k;
                anchor[offset] = point;
            }
        }
    }

    return true;
}


bool SCRFD::pre(const cv::Mat &image)
{
    cv::Mat image_resized(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), this->resized_container.data());
    letterbox(image, image_resized, cv::Scalar(0, 0, 0), this->canvas_width_gap, this->canvas_height_gap);

    if (this->canvas_width_gap > this->canvas_height_gap)
    {
        this->input_ratio = (float)image.rows / (float)this->canvas_height;
    }
    else
    {
        this->input_ratio = (float)image.cols / (float)this->canvas_width;
    }

    cv::Mat image_resized_permute(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), this->resized_permute_container.data());
    permute(image_resized, image_resized_permute, true);

#ifdef _DEBUG_
    cv::imwrite("test_resized_mat.bmp", image_resized);
    cv::Mat image_resized_test(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), this->resized_container.data());
    cv::imwrite("test_resized_buffer.bmp", image_resized_test);
#endif

#ifdef _DEBUG_
    std::vector<uint8_t> buffer(this->canvas_width * this->canvas_height * image.channels());
    for (int i = 0; i < image.channels(); i++)
    {
        for (int j = 0; j < this->canvas_height; j++)
        {
            for (int k = 0; k < this->canvas_width; k++)
            {
                buffer[j * this->canvas_width * image.channels() + k * image.channels() + i] = ((uint8_t*)image_resized_permute.data)[i * this->canvas_height * this->canvas_width + j * this->canvas_width + k];
            }
        }
    }
    cv::Mat image_resized_permute_test(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), buffer.data());
    cv::imwrite("test_permute_mat.bmp", image_resized_permute_test);
    cv::Mat image_resized_permute_test(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), this->resized_permute_container.data());
    cv::imwrite("test_permute_buffer.bmp", image_resized_permute_test);
#endif

    tensor_t input_tensor = get_graph_input_tensor(this->graph, 0, 0);
    const int tensor_type = get_tensor_data_type(input_tensor);
    if (TENGINE_DT_FP32 == tensor_type)
    {
        for (int i = 0; i < image.channels(); i++)
        {
            auto planar_size = this->canvas_width * this->canvas_height;

            uint8_t* input_ptr = this->resized_permute_container.data() + planar_size * i;
            float* output_ptr = this->canvas_float.data() + planar_size * i;

            for (int p = 0; p < planar_size; p++)
            {
                output_ptr[p] = ((float)input_ptr[p] - MODEL_MEANS[i]) * MODEL_SCALES[i];
            }
        }
    }
    else
    {
        for (int i = 0; i < image.channels(); i++)
        {
            auto planar_size = this->canvas_width * this->canvas_height;

            uint8_t* input_ptr = this->resized_permute_container.data() + planar_size * i;
            uint8_t* output_ptr = this->canvas_uint8.data() + planar_size * i;

            for (int p = 0; p < planar_size; p++)
            {
                float val = ((float)input_ptr[p] - MODEL_MEANS[i]) * MODEL_SCALES[i];

                int val_round = (int)(std::round(val / this->input_scale + (float)this->input_zp));

                if (0 <= val_round && 255 >= val_round)
                {
                    output_ptr[p] = val_round;
                }
                else
                {
                    if (255 < val_round) { output_ptr[p] = 255; }
                    if (0 > val_round) { output_ptr[p] = 0; }
                }
            }
        }
    }

#ifdef _DEBUG_
    if (TENGINE_DT_FP32 == tensor_type)
    {
        const auto model_input_ptr = (float*)get_tensor_buffer(input_tensor);

        cv::Mat model_input_buffer(this->canvas_height, this->canvas_width, CV_32FC(image.channels()), this->canvas_float.data());
        cv::Mat model_input_tensor(this->canvas_height, this->canvas_width, CV_32FC(image.channels()), model_input_ptr);
        cv::imwrite("test_input_buffer.bmp", model_input_buffer);
        cv::imwrite("test_input_tensor.bmp", model_input_tensor);
    }
    else
    {
        const auto model_input_ptr = (uint8_t*)get_tensor_buffer(input_tensor);

        cv::Mat model_input_buffer(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), this->canvas_uint8.data());
        cv::Mat model_input_tensor(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), model_input_ptr);
        cv::imwrite("test_input_buffer.bmp", model_input_buffer);
        cv::imwrite("test_input_tensor.bmp", model_input_tensor);
    }
#endif

    return true;
}


bool SCRFD::post(std::vector<Face>& boxes)
{
    Face proposal;

    for (int i = 0; i < this->strides.size(); i++)
    {
        auto score_tensor    = get_graph_tensor(this->graph, this->score_name[i].c_str());
        auto bbox_tensor     = get_graph_tensor(this->graph, this->bbox_name[i].c_str());
        auto landmark_tensor = get_graph_tensor(this->graph, this->landmark_name[i].c_str());

        if (nullptr == score_tensor)
        {
            fprintf(stderr, "Get output score tensor(%s) at stride(%d) failed.\n", this->score_name[i].c_str(), this->strides[i]);
            return false;
        }
        if (nullptr == bbox_tensor)
        {
            fprintf(stderr, "Get output score tensor(%s) at stride(%d) failed.\n", this->bbox_name[i].c_str(), this->strides[i]);
            return false;
        }
        if (nullptr == landmark_tensor)
        {
            fprintf(stderr, "Get output score tensor(%s) at stride(%d) failed.\n", this->landmark_name[i].c_str(), this->strides[i]);
            return false;
        }

        if (this->is_quantization)
        {
            const auto score_uint8_ptr     = (uint8_t*)get_tensor_buffer(score_tensor);
            const auto bbox_uint8_ptr      = (uint8_t*)get_tensor_buffer(bbox_tensor);
            const auto landmark_uint8_ptr  = (uint8_t*)get_tensor_buffer(landmark_tensor);

            for (size_t j = 0; j < this->score_buffer[i].size(); j++)
            {
                this->score_buffer[i][j] = (float)((int)score_uint8_ptr[j] - this->score_zp[i]) * this->score_scale[i];
            }

            for (size_t j = 0; j < this->bbox_buffer[i].size(); j++)
            {
                this->bbox_buffer[i][j] = (float)((int)bbox_uint8_ptr[j] - this->bbox_zp[i]) * this->bbox_scale[i];
            }

            for (size_t j = 0; j < this->landmark_buffer[i].size(); j++)
            {
                this->landmark_buffer[i][j] = (float)((int)landmark_uint8_ptr[j] - this->landmark_zp[i]) * this->landmark_scale[i];
            }
        }
        else
        {
            const auto score_float_ptr     = (float*)get_tensor_buffer(score_tensor);
            const auto bbox_float_ptr      = (float*)get_tensor_buffer(bbox_tensor);
            const auto landmark_float_ptr  = (float*)get_tensor_buffer(landmark_tensor);

            auto score_buffer_prt = (float*)this->score_buffer[i].data();
            auto bbox_buffer_prt = (float*)this->bbox_buffer[i].data();
            auto landmark_buffer_prt = (float*)this->landmark_buffer[i].data();

            auto scale_buffer_size    = get_tensor_buffer_size(score_tensor);
            auto bbox_buffer_size     = get_tensor_buffer_size(bbox_tensor);
            auto landmark_buffer_size = get_tensor_buffer_size(landmark_tensor);

            std::memcpy(score_buffer_prt, score_float_ptr, scale_buffer_size);
            std::memcpy(bbox_buffer_prt, bbox_float_ptr, bbox_buffer_size);
            std::memcpy(landmark_buffer_prt, landmark_float_ptr, landmark_buffer_size);
        }

        int score_shape[MAX_SHAPE_DIM_NUM] = { 0 };
        int bbox_shape[MAX_SHAPE_DIM_NUM] = { 0 };
        int landmark_shape[MAX_SHAPE_DIM_NUM] = { 0 };

        auto score_ret = get_tensor_shape(score_tensor, score_shape, MAX_SHAPE_DIM_NUM);
        auto bbox_ret = get_tensor_shape(bbox_tensor, bbox_shape, MAX_SHAPE_DIM_NUM);
        auto landmark_ret = get_tensor_shape(landmark_tensor, landmark_shape, MAX_SHAPE_DIM_NUM);

        if (0 > score_ret)
        {
            fprintf(stderr, "Get output score tensor(%s) shape at stride(%d) failed.\n", this->score_name[i].c_str(), this->strides[i]);
            return false;
        }
        if (0 > bbox_ret)
        {
            fprintf(stderr, "Get output bbox tensor(%s) shape at stride(%d) failed.\n", this->bbox_name[i].c_str(), this->strides[i]);
            return false;
        }
        if (0 > landmark_ret)
        {
            fprintf(stderr, "Get output landmark tensor(%s) shape at stride(%d) failed.\n", this->landmark_name[i].c_str(), this->strides[i]);
            return false;
        }
    }


    for (int i = 0; i < this->strides.size(); i++)
    {
        auto current_stride  = this->strides[i];
        auto fm_w = this->canvas_width / current_stride;
        auto fm_h = this->canvas_height / current_stride;
        auto fm_c = this->anchors[i].size();

        for (int h = 0; h < fm_h; h++)
        {
            for (int w = 0; w < fm_w; w++)
            {
                for (int c = 0; c < fm_c; c++)
                {
                    auto base_offset = h * fm_w * fm_c + w * fm_c + c;
                    auto score = this->score_buffer[i][base_offset];

                    if (score > this->score_threshold)
                    {
                        auto anchor = this->anchors[i][c];

                        auto bbox_offset = base_offset * 4;
                        auto landmark_offset = base_offset * 10;

                        auto dx = this->bbox_buffer[i][bbox_offset + 0] * (float)this->strides[i];
                        auto dy = this->bbox_buffer[i][bbox_offset + 1] * (float)this->strides[i];
                        auto dw = this->bbox_buffer[i][bbox_offset + 2] * (float)this->strides[i];
                        auto dh = this->bbox_buffer[i][bbox_offset + 3] * (float)this->strides[i];

                        auto cx = (anchor[0] + anchor[2]) * 0.5f + (float)(w * this->strides[i]);
                        auto cy = (anchor[1] + anchor[3]) * 0.5f + (float)(h * this->strides[i]);

                        auto x0 = cx - dx - this->canvas_width_gap;
                        auto y0 = cy - dy - this->canvas_height_gap;
                        auto x1 = cx + dw - this->canvas_width_gap;
                        auto y1 = cy + dh - this->canvas_height_gap;

                        proposal.confidence = score;
                        proposal.box.x = std::min(x0, x1) * this->input_ratio;
                        proposal.box.y = std::min(y0, y1) * this->input_ratio;
                        proposal.box.width = std::abs(x1 - x0) * this->input_ratio;
                        proposal.box.height = std::abs(y1 - y0) * this->input_ratio;

                        proposal.landmark[0].x = (cx + this->landmark_buffer[i][landmark_offset + 0] * (float)this->strides[i] - this->canvas_width_gap) * this->input_ratio;
                        proposal.landmark[0].y = (cy + this->landmark_buffer[i][landmark_offset + 1] * (float)this->strides[i] - this->canvas_height_gap) * this->input_ratio;
                        proposal.landmark[1].x = (cx + this->landmark_buffer[i][landmark_offset + 2] * (float)this->strides[i] - this->canvas_width_gap) * this->input_ratio;
                        proposal.landmark[1].y = (cy + this->landmark_buffer[i][landmark_offset + 3] * (float)this->strides[i] - this->canvas_height_gap) * this->input_ratio;
                        proposal.landmark[2].x = (cx + this->landmark_buffer[i][landmark_offset + 4] * (float)this->strides[i] - this->canvas_width_gap) * this->input_ratio;
                        proposal.landmark[2].y = (cy + this->landmark_buffer[i][landmark_offset + 5] * (float)this->strides[i] - this->canvas_height_gap) * this->input_ratio;
                        proposal.landmark[3].x = (cx + this->landmark_buffer[i][landmark_offset + 6] * (float)this->strides[i] - this->canvas_width_gap) * this->input_ratio;
                        proposal.landmark[3].y = (cy + this->landmark_buffer[i][landmark_offset + 7] * (float)this->strides[i] - this->canvas_height_gap) * this->input_ratio;
                        proposal.landmark[4].x = (cx + this->landmark_buffer[i][landmark_offset + 8] * (float)this->strides[i] - this->canvas_width_gap) * this->input_ratio;
                        proposal.landmark[4].y = (cy + this->landmark_buffer[i][landmark_offset + 9] * (float)this->strides[i] - this->canvas_height_gap) * this->input_ratio;

                        boxes.push_back(proposal);
                    }
                }
            }
        }
    }

    return true;
}


bool SCRFD::Detect(const cv::Mat& image, std::vector<Face>& boxes, const float& score_th, const float& nms_th)
{
    if (nullptr == this->graph)
    {
        fprintf(stderr, "Graph was not ready.\n");
        return false;
    }

    this->score_threshold = score_th;
    this->nms_threshold = nms_th;

    boxes.clear();

    std::vector<Face> proposals;

    auto pre_ret = this->pre(image);
    if (!pre_ret)
    {
        fprintf(stderr, "Pre-run failed.\n");
        return false;
    }

#ifdef _BENCHMARK_
    Timer timer;
#endif
    int run_ret = run_graph(this->graph, 1);
#ifdef _BENCHMARK_
    fprintf(stdout, "Run graph cost %.2fms.\n", timer.Cost());
#endif
    if (0 != run_ret)
    {
        fprintf(stderr, "Run graph failed.\n");
        return false;
    }

    auto pose_ret = this->post(proposals);
    if (!pose_ret)
    {
        fprintf(stderr, "Run graph failed.\n");
        return false;
    }

    auto region_compare = [](const Face& a, const Face& b)
    {
        return a.confidence > b.confidence;
    };

    std::sort(proposals.begin(), proposals.end(), region_compare);

    nms(proposals, boxes, this->nms_threshold);

    return true;
}
