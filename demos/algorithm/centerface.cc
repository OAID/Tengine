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

#include "centerface.hpp"

#include "utilities/letterbox.hpp"
#include "utilities/permute.hpp"
#include "utilities/nms.hpp"
#include "utilities/timer.hpp"

#include <cmath>

//#define _DEBUG_
//#define _BENCHMARK_

const int MODEL_WIDTH       =   640;
const int MODEL_HEIGHT      =   384;

const float DEFAULT_SCORE_THRESHOLD = 0.45f;
const float DEFAULT_NMS_THRESHOLD = 0.3f;

const float MODEL_MEANS[]   =   { 104.04f, 113.985f, 119.85f };
const float MODEL_SCALES[]  =   { 0.013569442f, 0.014312294f, 0.014106362f };


bool CenterFace::Init(int width, int height, int channel)
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

    ret = prerun_graph(this->graph);
    if (0 != ret)
    {
        fprintf(stderr, "Pre-run graph failed.\n");
        return false;
    }

    tensor_t score_tensor = get_graph_output_tensor(this->graph, 0, 0);
    tensor_t bbox_tensor = get_graph_output_tensor(this->graph, 1, 0);
    if (nullptr == score_tensor || nullptr == bbox_tensor)
    {
        fprintf(stderr, "Get output tensor failed.\n");
        return false;
    }

    int score_dims[MAX_SHAPE_DIM_NUM], bbox_dims[MAX_SHAPE_DIM_NUM];
    int score_ret = get_tensor_shape(score_tensor, score_dims, MAX_SHAPE_DIM_NUM);
    int bbox_ret = get_tensor_shape(bbox_tensor, bbox_dims, MAX_SHAPE_DIM_NUM);
    if (0 > score_ret || 0 > bbox_ret)
    {
        fprintf(stderr, "Get output tensor shape failed.\n");
        return false;
    }

    this->heatmap_width =  score_dims[3];
    this->heatmap_height = score_dims[2];

    this->score_heatmap.resize(score_dims[0] * score_dims[1] * score_dims[2] * score_dims[3]);
    this->bbox_heatmap.resize(bbox_dims[0] * bbox_dims[1] * bbox_dims[2] * bbox_dims[3]);
    this->region_heatmap.resize(this->score_heatmap.size());

    get_tensor_quant_param(input_tensor, &this->input_scale, &this->input_zp, 1);
    get_tensor_quant_param(score_tensor, &this->score_scale, &this->score_zp, 1);
    get_tensor_quant_param(bbox_tensor, &this->bbox_scale, &this->bbox_zp, 1);

    return true;
}


CenterFace::CenterFace()
{
    this->context = nullptr;

    this->graph = nullptr;

    this->heatmap_threshold = DEFAULT_SCORE_THRESHOLD;
    this->nms_threshold = DEFAULT_NMS_THRESHOLD;

    this->width_gap = 0;
    this->height_gap = 0;

    this->canvas_width = 0;
    this->canvas_height = 0;

    this->heatmap_width  = 0;
    this->heatmap_height = 0;

    this->input_scale = 0.f;
    this->input_zp = 0;

    this->score_scale = 0.f;
    this->bbox_scale = 0.f;
    this->score_zp = 0;
    this->bbox_zp = 0;

    this->context = create_context("ctx", 1);
}


bool CenterFace::Load(const std::string &detection_model, const cv::Size& input_shape, const std::string &device)
{
    this->context = create_context("ctx", 1);

    if (!device.empty())
    {
        int ret = set_context_device(this->context, device.c_str(), nullptr, 0);
        if (0 != ret)
        {
            fprintf(stderr, "Set context device failed.\n");
            return false;
        }
    }

    this->graph = create_graph(this->context, "tengine", detection_model.c_str());
    if (nullptr == this->graph)
    {
        fprintf(stderr, "Load model failed.\n");
        return false;
    }

    return this->Init(input_shape.width, input_shape.height, 3);
}


CenterFace::~CenterFace()
{
    postrun_graph(this->graph);
    destroy_context(this->context);
}


bool CenterFace::Detect(const cv::Mat& image, std::vector<Region>& boxes, const float& score_th, const float& nms_th)
{
    if (nullptr == this->graph)
    {
        fprintf(stderr, "Graph was not ready.\n");
        return -1;
    }

    boxes.clear();

    std::vector<Region> all_boxes;

    cv::Mat image_resized(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), this->resized_container.data());
    letterbox(image, image_resized, cv::Scalar_<uint8_t>(0, 0, 0), this->width_gap, this->height_gap);

#ifdef _DEBUG_
    cv::imwrite("test_resized_mat.bmp", image_resized);
    cv::Mat image_resized_test(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), this->resized_container.data());
    cv::imwrite("test_resized_buffer.bmp", image_resized_test);
#endif

    cv::Mat image_resized_permute(this->canvas_height, this->canvas_width, CV_8UC(image.channels()), this->resized_permute_container.data());
    permute(image_resized, image_resized_permute, false);

#ifdef _DEBUG_
    cv::imwrite("test_permute_mat.bmp", image_resized_permute);
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

//        std::ofstream file;
//        file.open("yf_input.bin", std::ios::binary | std::ios::out);
//        file.write((char*)(this->canvas_float.data()), this->canvas_float.size() * sizeof(float));
//        file.close();

//        std::ifstream file;
//        file.open("input_data.bin", std::ios::binary | std::ios::in);
//        file.read((char*)(this->canvas_float.data()), this->canvas_float.size() * sizeof(float));
//        file.close();
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

    int ret = run_graph(this->graph, 1);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph failed.\n");
        return -1;
    }

    if (TENGINE_DT_FP32 != tensor_type)
    {
        tensor_t score_tensor = get_graph_output_tensor(this->graph, 0, 0);
        tensor_t bbox_tensor = get_graph_output_tensor(this->graph, 1, 0);
        if (nullptr == score_tensor || nullptr == bbox_tensor)
        {
            fprintf(stderr, "Get output tensor failed.\n");
            return -1;
        }

        const auto score_uint8_ptr = (uint8_t*)get_tensor_buffer(score_tensor);
        for (size_t i = 0; i < this->score_heatmap.size(); i++)
        {
            this->score_heatmap[i] = (float)((int)score_uint8_ptr[i] - this->score_zp) * this->score_scale;
        }

        const auto bbox_uint8_ptr = (uint8_t*)get_tensor_buffer(bbox_tensor);
        for (size_t i = 0; i < this->bbox_heatmap.size(); i++)
        {
            this->bbox_heatmap[i] = (float)((int)bbox_uint8_ptr[i] - this->bbox_zp) * this->bbox_scale;
        }
    }
    else
    {
        tensor_t score_tensor = get_graph_output_tensor(this->graph, 0, 0);
        tensor_t bbox_tensor = get_graph_output_tensor(this->graph, 1, 0);
        if (nullptr == score_tensor || nullptr == bbox_tensor)
        {
            fprintf(stderr, "Get output tensor failed.\n");
            return -1;
        }

        const auto score_float_ptr = (float*)get_tensor_buffer(score_tensor);
        memcpy(this->score_heatmap.data(), score_float_ptr, this->score_heatmap.size() * sizeof(float));

        const auto bbox_float_ptr = (float*)get_tensor_buffer(bbox_tensor);
        memcpy(this->bbox_heatmap.data(), bbox_float_ptr, this->bbox_heatmap.size() * sizeof(float));
    }

#ifdef _DEBUG_
    if (TENGINE_DT_FP32 == tensor_type)
    {
        cv::Mat final_tensor(this->heatmap_height, this->heatmap_width, CV_32FC1, this->canvas_float.data());
        cv::imwrite("test_graph_output_score.bmp", final_tensor);
    }
    else
    {
        tensor_t score_tensor = get_graph_output_tensor(this->graph, 0, 0);
        if (nullptr == score_tensor)
        {
            fprintf(stderr, "Get output tensor failed.\n");
            return -1;
        }
        const auto score_uint8_ptr = (uint8_t*)get_tensor_buffer(score_tensor);
        cv::Mat final_tensor(this->heatmap_height, this->heatmap_width, CV_8UC1, score_uint8_ptr);
        cv::imwrite("test_output_score.bmp", final_tensor);
    }
#endif

    decode_heatmap(image.cols, image.rows, this->heatmap_threshold, boxes);

    return true;
}


void CenterFace::decode_heatmap(const int& image_width, const int& image_height, const float& threshold, std::vector<Region>& boxes)
{
    auto canvas_region_width = (float)this->canvas_width - this->width_gap * 2;
    auto canvas_region_height = (float)this->canvas_height - this->height_gap * 2;

    auto image_width_ratio = (float)image_width / canvas_region_width;
    auto image_height_ratio = (float)image_height / canvas_region_height;

    auto heatmap_width_ratio = (float)this->canvas_width / (float)this->heatmap_width;
    auto heatmap_height_ratio = (float)this->canvas_height / (float)this->heatmap_height;

    auto planar_size = this->heatmap_height * this->heatmap_width;

    for (int h = 0; h < this->heatmap_height; h++)
    {
        for (int w = 0; w < this->heatmap_width; w++)
        {
            auto offset = h * this->heatmap_width + w;

            auto& region = this->region_heatmap[offset];
            region.confidence = this->score_heatmap[offset];

            auto x1 = (float)w - this->bbox_heatmap[planar_size * 0 + offset];
            auto y1 = (float)h - this->bbox_heatmap[planar_size * 1 + offset];
            auto x2 = (float)w + this->bbox_heatmap[planar_size * 2 + offset];
            auto y2 = (float)h + this->bbox_heatmap[planar_size * 3 + offset];

            region.box.x = (x1 * heatmap_width_ratio - this->width_gap) * image_width_ratio;
            region.box.y = (y1 * heatmap_height_ratio - this->height_gap) * image_height_ratio;
            region.box.width = (x2 - x1) * heatmap_height_ratio * image_width_ratio;
            region.box.height = (y2 - y1) * heatmap_height_ratio * image_height_ratio;
        }
    }

    std::vector<Region> before_nms;
    for (const auto& region : this->region_heatmap)
    {
        if (region.confidence >= threshold)
        {
            before_nms.push_back(region);
        }
    }

    auto region_compare = [](const Region& a, const Region& b)
    {
        return a.confidence > b.confidence;
    };

    std::sort(before_nms.begin(), before_nms.end(), region_compare);


    //softer_nms(before_nms, boxes, this->nms_threshold, 1.f, this->heatmap_threshold);
    nms(before_nms, boxes, this->nms_threshold);
}
