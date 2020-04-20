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

#include "model_test.h"


bool read_file(const std::string &file_path, std::vector<uint8_t>& data)
{
    std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
    if (!ifs.is_open())
    {
        ifs.close();

        return false;
    }

    ifs.seekg(0, std::ifstream::end);
    const size_t length = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ifstream::beg);

    data.reserve(length);
    data.insert(data.begin(), std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    if (ifs.fail())
    {
        ifs.close();

        return false;
    }
    else
    {
        ifs.close();

        return true;
    }
}


void get_input_data(const char* const image_file, float* const input_data, int img_h, int img_w, const float* const mean, const float* const scale)
{
    image img = imread(image_file);
    image resImg = resize_image(img, img_w, img_h);
    resImg = rgb2bgr_premute(resImg);

    float* img_data = ( float* )resImg.data;
    int hw = img_h * img_w;
    for(int c = 0; c < 3; c++)
    {
        for(int h = 0; h < img_h; h++)
        {
            for(int w = 0; w < img_w; w++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale[c];
                img_data++;
            }
        }
    }
}


// sort class score
bool sort_class_score(const ClassScore& a, const ClassScore&b)
{
    return a.score > b.score;
};


void get_result(float* data, std::vector<ClassScore>& result)
{
    result.resize(5);

    float* end = data + 1000;
    std::vector<float> result_buffer(data, end);
    std::vector<int> top_N = TEngine::Argmax(result_buffer, 5);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        ClassScore class_score;
        class_score.idx = top_N[i];
        class_score.score = result_buffer[class_score.idx];

        result[i] = class_score;
    }
}


void printf_class_score(std::vector<ClassScore>& result)
{
    printf("--------------------------------\n");

    for (auto& pair : result)
    {
        printf("Class %d, Score %.4f;\n", pair.idx, pair.score);
    }

    printf("--------------------------------\n");
}


bool parse_model(const std::vector<uint8_t>& buffer, std::string& path, std::string& input_name, std::string& output_name, float* const scale, float* shift, int& width, int& height)
{
    // config json parser
    auto config_parser = nlohmann::json::parse(buffer);

    // get model
    auto model_field = config_parser.find("model");
    if (config_parser.end() != model_field)
    {
        for (auto& item : model_field->items())
        {
            if ("path" == item.key()) { path = item.value(); }
            if ("input" == item.key()) { input_name = item.value(); }
            if ("output" == item.key()) { output_name = item.value(); }

            if ("scale" == item.key() && item.value().is_array() && 3 == item.value().size())
            {
                for (int i = 0; i < 3; i++)
                {
                    scale[i] = item.value()[i];
                }
            }

            if ("shift" == item.key() && item.value().is_array() && 3 == item.value().size())
            {
                for (int i = 0; i < 3; i++)
                {
                    shift[i] = item.value()[i];
                }
            }

            if ("size" == item.key())
            {
                for (auto& val : item.value().items())
                {
                    if ("w" == val.key() || "width" == val.key()) { width = val.value(); }
                    if ("h" == val.key() || "height" == val.key()) { height = val.value(); }
                }
            }
        }
    }
    else
    {
        printf("Cannot find node'model' from json.\n");
        return false;
    }

    if (path.empty() || input_name.empty() || output_name.empty())
    {
        printf("Model detail is not finished.\n");
        return false;
    }

    if (0 == width || 0 == height)
    {
        printf("Input size is set to zero.\n");
        return false;
    }

    return true;
}


bool parse_image_list(const std::vector<uint8_t>& buffer, std::vector<std::string>& image_list)
{
    // config json parser
    auto config_parser = nlohmann::json::parse(buffer);

    // get image list
    auto image_field = config_parser.find("image");
    if (image_field != config_parser.end() && image_field.value().is_array())
    {
        image_list.resize(image_field.value().size());
        for (int i = 0; i < image_field.value().size(); i++)
        {
            image_list[i] = image_field.value()[i];
        }
    }
    else
    {
        printf("Cannot find node'image' from json.\n");
        return false;
    }

    // check image list
    if (image_list.empty())
    {
        printf("Image list was empty..\n");
        return false;
    }

    return true;
}


bool parse_top_n(const std::vector<uint8_t>& buffer, std::vector<std::vector<ClassScore>>& image_topN)
{
    // config json parser
    auto config_parser = nlohmann::json::parse(buffer);

    // get top 5
    auto result_field = config_parser.find("result");

    if (result_field != config_parser.end() && result_field.value().is_array() && result_field.value().size() == image_topN.size())
    {
        std::vector<ClassScore> top5;

        for (int i = 0; i < image_topN.size(); i++)
        {
            auto& image_top_node =  result_field.value()[i];
            if (5 != image_top_node.size())
            {
                printf("Size of top N field is not right(5 vs. %d).\n", int(image_top_node.size()));
                return false;
            }

            top5.clear();
            for (auto& top_list : image_top_node.items())
            {
                ClassScore score;
                score.idx = std::stoi(top_list.key());
                score.score = top_list.value();

                top5.push_back(score);
            }

            if (5 != top5.size())
            {
                printf("Size of top5 is not match(5 vs. %d).", (int)top5.size());
            }

            image_topN[i] = top5;
        }
    }
    else
    {
        printf("Cannot find node'result' from json.\n");
        return false;
    }

    return true;
}
