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
 * Author: hhchen@openailab.com
 */

#include "quant_utils.hpp"

void get_input_data_cv(const char* image_file, float* input_data, int img_h, int img_w, const float* mean,
                       const float* scale, int img_c = 1, int sw_RGB = 0, int center_crop = 0, int letterbox_rows = 0, int letterbox_cols = 0, int focus = 0)
{
    /* only for yolov5s */
    if (focus == 1 && letterbox_rows > 0 && letterbox_cols > 0)
    {
        cv::Mat sample = cv::imread(image_file, 1);
        cv::Mat img;

        if (sample.channels() == 4)
        {
            // printf(" ---> 4chan to BGR <---\n");
            cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
        }
        else if (sample.channels() == 1 && img_c == 3 && sw_RGB == 0)
        {
            // printf(" ---> GRAY2BGR <---\n");
            cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
        }
        else if (sample.channels() == 1 && img_c == 3 && sw_RGB == 1)
        {
            // printf(" ---> GRAY2RGB <---\n");
            cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
        }
        else if (sample.channels() == 3 && sw_RGB == 1 && img_c != 1)
        {
            // printf(" ---> BGR2RGB <---\n");
            cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
        }
        else if (sample.channels() == 3 && img_c == 1)
        {
            // printf(" ---> to GRAY <---\n");
            cv::cvtColor(sample, img, cv::COLOR_BGR2GRAY);
        }
        else
        {
            // printf(" ---> Ori BGR <---\n");
            img = sample;
        }

        /* letterbox process to support different letterbox size */
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols))
        {
            scale_letterbox = letterbox_rows * 1.0 / img.rows;
        }
        else
        {
            scale_letterbox = letterbox_cols * 1.0 / img.cols;
        }
        resize_cols = int(scale_letterbox * img.cols);
        resize_rows = int(scale_letterbox * img.rows);

        cv::resize(img, img, cv::Size(resize_cols, resize_rows));
        img.convertTo(img, CV_32FC3);

        // Generate a gray image for letterbox using opencv
        cv::Mat resize_img(letterbox_cols, letterbox_rows, CV_32FC3, cv::Scalar(0.5/scale[0] + mean[0], 0.5/scale[1] + mean[1], 0.5/ scale[2] + mean[2]));
        int top = (letterbox_rows - resize_rows) / 2;
        int bot = (letterbox_rows - resize_rows + 1) / 2;
        int left = (letterbox_cols - resize_cols) / 2;
        int right = (letterbox_cols - resize_cols + 1) / 2;

        // Letterbox filling
        cv::copyMakeBorder(img, resize_img, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0.5/scale[0] + mean[0], 0.5/scale[1] + mean[1], 0.5/ scale[2] + mean[2]));

        resize_img.convertTo(resize_img, CV_32FC3);
        float* img_data   = (float* )resize_img.data;
        float* input_temp = (float* )malloc(3 * letterbox_rows * letterbox_cols * sizeof(float));

        /* nhwc to nchw */
        for (int h = 0; h < letterbox_rows; h++)
        {
            for (int w = 0; w < letterbox_cols; w++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int in_index  = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    input_temp[out_index] = (img_data[in_index] - mean[c]) * scale[c];
                }
            }
        }

        /* focus process */
        for (int i = 0; i < 2; i++) // corresponding to rows
        {
            for (int g = 0; g < 2; g++) // corresponding to cols
            {
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < letterbox_rows/2; h++)
                    {
                        for (int w = 0; w < letterbox_cols/2; w++)
                        {
                            int in_index  = i + g * letterbox_cols + c * letterbox_cols * letterbox_rows +
                                            h * 2 * letterbox_cols + w * 2;
                            int out_index = i * 2 * 3 * (letterbox_cols/2) * (letterbox_rows/2) +
                                            g * 3 * (letterbox_cols/2) * (letterbox_rows/2) +
                                            c * (letterbox_cols/2) * (letterbox_rows/2) +
                                            h * (letterbox_cols/2) +
                                            w;

                            input_data[out_index] = input_temp[in_index];
                        }
                    }
                }
            }
        }

        free(input_temp);

        return;
    }

    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 4)
    {
        // printf(" ---> 4chan to BGR <---\n");
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if (sample.channels() == 1 && img_c == 3 && sw_RGB == 0)
    {
        // printf(" ---> GRAY2BGR <---\n");
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else if (sample.channels() == 1 && img_c == 3 && sw_RGB == 1)
    {
        // printf(" ---> GRAY2RGB <---\n");
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    }
    else if (sample.channels() == 3 && sw_RGB == 1 && img_c != 1)
    {
        // printf(" ---> BGR2RGB <---\n");
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
    }
    else if (sample.channels() == 3 && img_c == 1)
    {
        // printf(" ---> to GRAY <---\n");
        cv::cvtColor(sample, img, cv::COLOR_BGR2GRAY);
    }
    else
    {
        // printf(" ---> Ori BGR <---\n");
        img = sample;
    }

    if (center_crop == 1)
    {
        // printf("h:%d  w:%d\n",img.rows,img.cols);
        int h0 = 0;
        int w0 = 0;
        if ( img.rows < img.cols)
        {
            h0 = 256;
            w0 = int(img.cols*(256.0/img.rows));
        }
        else
        {
            h0 = int(img.rows*(256.0/img.cols));
            w0 = 256;
        }
        int center_h = int(h0/2);
        int center_w = int(w0/2);

        float* img_data = nullptr;

        // printf("h w %d %d\n",h0,w0);
        cv::resize(img, img, cv::Size(w0, h0));
        cv::Rect img_roi_box(center_w - 112, center_h - 112, 224, 224);
        cv::Mat img_crop = img(img_roi_box).clone();

        if (img_c == 3)
            img_crop.convertTo(img_crop, CV_32FC3);
        else if (img_c == 1)
            img_crop.convertTo(img_crop, CV_32FC1);
        img_data = ( float* )img_crop.data;

        int hw = img_h * img_w;
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                for (int c = 0; c < img_c; c++)
                {
                    input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale[c];
                    img_data++;
                }
            }
        }
    }
    else if (letterbox_rows > 0)
    {
        float letterbox_size = (float)letterbox_rows;
        int resize_h = 0;
        int resize_w = 0;
        if (img.rows > img.cols)
        {
            resize_h = letterbox_size;
            resize_w = int(img.cols * (letterbox_size / img.rows));
        }
        else
        {
            resize_h = int(img.rows * (letterbox_size / img.cols));
            resize_w = letterbox_size;
        }

        float* img_data = nullptr;

        cv::resize(img, img, cv::Size(resize_w, resize_h));
        img.convertTo(img, CV_32FC3);
        cv::Mat img_new(letterbox_size, letterbox_size, CV_32FC3,
                        cv::Scalar(0.5/scale[0] + mean[0], 0.5/scale[1] + mean[1], 0.5/ scale[2] + mean[2]));
        int dh = int((letterbox_size - resize_h) / 2);
        int dw = int((letterbox_size - resize_w) / 2);

        for (int h = 0; h < resize_h; h++)
        {
            for (int w = 0; w < resize_w; w++)
            {
                for (int c = 0; c < 3; ++c)
                {
                    int in_index  = h * resize_w * 3 + w * 3 + c;
                    int out_index = (dh + h) * letterbox_size * 3 + (dw + w) * 3 + c;

                    (( float* )img_new.data)[out_index] = (( float* )img.data)[in_index];
                }
            }
        }

        if (img_c == 3)
            img_new.convertTo(img_new, CV_32FC3);
        else if (img_c == 1)
            img_new.convertTo(img_new, CV_32FC1);
        img_data = ( float* )img_new.data;

        int hw = img_h * img_w;
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                for (int c = 0; c < img_c; c++)
                {
                    input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale[c];
                    img_data++;
                }
            }
        }
    }
    else
    {
        cv::resize(img, img, cv::Size(img_w, img_h));
        if (img_c == 3)
            img.convertTo(img, CV_32FC3);
        else if (img_c == 1)
            img.convertTo(img, CV_32FC1);
        float* img_data = ( float* )img.data;
        int hw = img_h * img_w;
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                for (int c = 0; c < img_c; c++)
                {
                    input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale[c];
                    img_data++;
                }
            }
        }
    }
}

void readFileList(std::string basePath, std::vector<std::string>& imgs)
{
    DIR *dir;
    struct dirent *ptr;
    std::string base;

    if ((dir=opendir(basePath.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1); 
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    ///file
        {
            base = basePath + "/" + ptr->d_name;
//            printf("str %s \n",base.c_str());
            imgs.push_back(base);
        }
        else if(ptr->d_type == 4)    ///dir
        {
            readFileList(basePath + "/" + ptr->d_name, imgs);
        }
    }
    closedir(dir);
}

double cosin_similarity(float** in_a,float** in_b, uint32_t imgs_num, uint32_t output_num)
{
    double norm_a=0;
    double norm_b=0;
    double a_b=0;

    uint32_t fnum = (output_num >> 4) << 4;
    uint32_t rnum = output_num - fnum;

    for (int i = 0; i < imgs_num; i++)
    {
        for (int j = 0; j < fnum; j=j+8)
        {
            for (int k = 0; k < 8; k=k+1)
            {
                norm_a += in_a[i][j+k] * in_a[i][j+k];

                norm_b += in_b[i][j+k] * in_b[i][j+k];

                a_b += in_a[i][j+k] * in_b[i][j+k];
            }
        }
    }

    for (int j = fnum; j < output_num; j++)
    {
        for (int i = 0; i < imgs_num; i++)
        {
            norm_a += in_a[i][j] * in_a[i][j];
            norm_b += in_b[i][j] * in_b[i][j];
            a_b += in_a[i][j] * in_b[i][j];
        }
    }

    
    double cosin;
    double _a_b_ = sqrt(norm_a) * sqrt(norm_b);
    if(_a_b_ < 0.0000001f && _a_b_ > -0.0000001f)
        cosin = a_b;
    else
        cosin = a_b/_a_b_;
    if (cosin < -999999 || cosin > 999999)
        cosin = 0;
    return cosin;
}

std::vector<uint32_t> histCount_int(float *data, uint32_t elem_num, float max_val, float min_val)
{
    float bin_scale = 1;
    if (abs(max_val) > abs(min_val))
        bin_scale = abs(max_val) / 2047.f;
    else
        bin_scale = abs(min_val) / 2047.f;
    int bin_zp = 0;
    std::vector<uint32_t> hist(2048);
    for (int i = 0; i < elem_num; i++)
        if (data[i] != 0)
        {
            uint32_t hist_idx = floor(abs(data[i]) / bin_scale);
            hist[hist_idx] ++;
        }
    return hist;
}

std::vector<uint32_t> histCount(float *data, uint32_t elem_num, float max_val, float min_val)
{
    float bin_scale = (max_val - min_val) / 2047.f;
    int bin_zp = int(-min_val / bin_scale);
    std::vector<uint32_t> hist(2048);
    for (int i = 0; i < elem_num; i++)
        if (data[i] != 0)
            hist[uint32_t(data[i] / bin_scale + bin_zp)] ++;
    return hist;
}

float compute_kl_divergence(std::vector<float> &dist_a, std::vector<float> &dist_b)
{
    const size_t length = dist_a.size();
    float result = 0;

    for (size_t i = 0; i < length; i++)
    {
        if (dist_a[i] != 0)
        {
            if (dist_b[i] == 0)
            {
                result += 1;
            }
            else
            {
                result += dist_a[i] * log(dist_a[i] / dist_b[i]);
            }
        }
    }

    return result;
}

std::vector<float> normalize_histogram(std::vector<uint32_t> &histogram)
{
    std::vector<float> histogram_out(histogram.size());
    const size_t length = histogram.size();
    float sum = 0;

    for (size_t i = 1; i < length; i++)
        sum += histogram[i];

    for (size_t i = 1; i < length; i++)
        histogram_out[i] = float(histogram[i] / sum);

    return histogram_out;
}

int threshold_distribution(std::vector<uint32_t> &distribution_in, const int target_bin) 
{
    int target_threshold = target_bin;
    float min_kl_divergence = 99999999999.9f;
    const int length = static_cast<int>(distribution_in.size());

    std::vector<float> distribution(distribution_in.size());
    std::vector<float> quantize_distribution(target_bin);
    distribution = normalize_histogram(distribution_in);

    float threshold_sum = 0;
    for (int threshold = target_bin; threshold < length; threshold++)
    {
        threshold_sum += distribution[threshold];
    }

    for (int threshold = target_bin; threshold < length; threshold++)
    {

        std::vector<float> t_distribution(distribution.begin(), distribution.begin() + threshold);

        t_distribution[threshold - 1] += threshold_sum;
        threshold_sum -= distribution[threshold];

        // get P
        fill(quantize_distribution.begin(), quantize_distribution.end(), 0.0f);

        const float num_per_bin = static_cast<float>(threshold) / static_cast<float>(target_bin);

        for (int i = 0; i < target_bin; i++)
        {
            const float start = static_cast<float>(i) * num_per_bin;
            const float end = start + num_per_bin;

            const int left_upper = static_cast<int>(ceil(start));
            if (static_cast<float>(left_upper) > start)
            {
                const float left_scale = static_cast<float>(left_upper) - start;
                quantize_distribution[i] += left_scale * distribution[left_upper - 1];
            }

            const int right_lower = static_cast<int>(floor(end));

            if (static_cast<float>(right_lower) < end)
            {

                const float right_scale = end - static_cast<float>(right_lower);
                quantize_distribution[i] += right_scale * distribution[right_lower];
            }

            for (int j = left_upper; j < right_lower; j++)
            {
                quantize_distribution[i] += distribution[j];
            }
        }


        // get Q
        std::vector<float> expand_distribution(threshold, 0);

        for (int i = 0; i < target_bin; i++)
        {
            const float start = static_cast<float>(i) * num_per_bin;
            const float end = start + num_per_bin;

            float count = 0;

            const int left_upper = static_cast<int>(ceil(start));
            float left_scale = 0;
            if (static_cast<float>(left_upper) > start)
            {
                left_scale = static_cast<float>(left_upper) - start;
                if (distribution[left_upper - 1] != 0)
                {
                    count += left_scale;
                }
            }

            const int right_lower = static_cast<int>(floor(end));
            float right_scale = 0;
            if (static_cast<float>(right_lower) < end)
            {
                right_scale = end - static_cast<float>(right_lower);
                if (distribution[right_lower] != 0)
                {
                    count += right_scale;
                }
            }

            for (int j = left_upper; j < right_lower; j++)
            {
                if (distribution[j] != 0)
                {
                    count++;
                }
            }

            const float expand_value = quantize_distribution[i] / count;

            if (static_cast<float>(left_upper) > start)
            {
                if (distribution[left_upper - 1] != 0)
                {
                    expand_distribution[left_upper - 1] += expand_value * left_scale;
                }
            }
            if (static_cast<float>(right_lower) < end)
            {
                if (distribution[right_lower] != 0)
                {
                    expand_distribution[right_lower] += expand_value * right_scale;
                }
            }
            for (int j = left_upper; j < right_lower; j++)
            {
                if (distribution[j] != 0)
                {
                    expand_distribution[j] += expand_value;
                }
            }
        }

        const float kl_divergence = compute_kl_divergence(t_distribution, expand_distribution);

        // the best num of bin
        if (kl_divergence < min_kl_divergence)
        {
            min_kl_divergence = kl_divergence;
            target_threshold = threshold;
        }
    }

    return target_threshold;
}
