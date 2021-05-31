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


#include <dirent.h>
#include <string.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _MSC_VER
#include "getopt.h"
#else
#include <unistd.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else    // _WIN32
#include <sys/time.h>
#endif    // _WIN32

#include "quant_utils.hpp"


#ifdef _WIN32
double get_current_time()
{
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
}
#else    // _WIN32

double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif    // _WIN32

void split(float* array, char* str, const char* del)
{
    char* s = nullptr;
    s = strtok(str, del);
    while (s != nullptr)
    {
        *array++ = atof(s);
        s = strtok(nullptr, del);
    }
}

void get_input_data_cv(const char* image_file, float* input_data, int img_c, int img_h, int img_w, const float* mean,
                       const float* scale, int sw_RGB = 0, int center_crop = 0, int letterbox_rows = 0, int letterbox_cols = 0, int focus = 0)
{
    /* only for yolov5s */
    if (focus == 1 && letterbox_rows > 0 && letterbox_cols > 0)
    {
        cv::Mat sample = cv::imread(image_file, 1);
        cv::Mat img;

        if (sample.channels() == 4)
        {
            cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
        }
        else if (sample.channels() == 1 && img_c == 3 && sw_RGB == 0)
        {
            cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
        }
        else if (sample.channels() == 1 && img_c == 3 && sw_RGB == 1)
        {
            cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
        }
        else if (sample.channels() == 3 && sw_RGB == 1 && img_c != 1)
        {
            cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
        }
        else if (sample.channels() == 3 && img_c == 1)
        {
            cv::cvtColor(sample, img, cv::COLOR_BGR2GRAY);
        }
        else
        {
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
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if (sample.channels() == 1 && img_c == 3 && sw_RGB == 0)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else if (sample.channels() == 1 && img_c == 3 && sw_RGB == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    }
    else if (sample.channels() == 3 && sw_RGB == 1 && img_c != 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
    }
    else if (sample.channels() == 3 && img_c == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGR2GRAY);
    }
    else
    {
        img = sample;
    }

    if (center_crop == 1)
    {
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
            imgs.push_back(base);
        }
        else if(ptr->d_type == 4)    ///dir
        {
            readFileList(basePath + "/" + ptr->d_name, imgs);
        }
    }
    closedir(dir);
}
