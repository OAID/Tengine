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
#include <string>

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <algorithm>

#include "common.h"
#include "tengine_cpp_api.h"
#include "tengine_operations.h"

#define DEFAULT_IMG_H        227
#define DEFAULT_IMG_W        227
#define DEFAULT_SCALE1       1.f
#define DEFAULT_SCALE2       1.f
#define DEFAULT_SCALE3       1.f
#define DEFAULT_MEAN1        104.007
#define DEFAULT_MEAN2        116.669
#define DEFAULT_MEAN3        122.679
#define DEFAULT_LOOP_COUNT   1
#define DEFAULT_THREAD_COUNT 1

using namespace std;

void show_usage()
{
    std::cout << "[Usage]: [-h]\n"
              << "    [-m model_file] [-l label_file] [-i image_file]\n"
              << "    [-g img_h,img_w] [-s scale] [-w mean[0],mean[1],mean[2]] [-r repeat_count]\n";

    std::cout << "\nmobilenet example: \n"
              << "    ./classification -m /path/to/mobilenet.tmfile -l /path/to/labels.txt -i /path/to/img.jpg -g 224,224 -s 0.017 -w 104.007,116.669,122.679" << std::endl;
}

int main(int argc, char* argv[])
{
    int loop_count = DEFAULT_LOOP_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    string model_file;
    string image_file;
    float img_hw[2] = {0.f};
    int img_h = 0;
    int img_w = 0;
    float mean[3] = {-1.f, -1.f, -1.f};
    float scale[3] = {0.f, 0.f, 0.f};

    int res;
    while ((res = getopt(argc, argv, "m:i:l:g:s:w:r:t:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'g':
            split(img_hw, optarg, ",");
            img_h = (int)img_hw[0];
            img_w = (int)img_hw[1];
            break;
        case 's':
            split(scale, optarg, ",");
            break;
        case 'w':
            split(mean, optarg, ",");
            break;
        case 'r':
            loop_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    if (model_file.empty())
    {
        std::cerr << "Error: Tengine model file not specified!" << std::endl;
        show_usage();
        return -1;
    }

    if (image_file.empty())
    {
        std::cerr << "Error: Image file not specified!" << std::endl;
        show_usage();
        return -1;
    }

    // check input files
    if (!check_file_exist(model_file.c_str()) || !check_file_exist(image_file.c_str()))
        return -1;

    if (img_h == 0)
    {
        img_h = DEFAULT_IMG_H;
        std::cout << "Image height not specified, use default [" << DEFAULT_IMG_H << "]" << std::endl;
    }
    if (img_w == 0)
    {
        img_w = DEFAULT_IMG_W;
        std::cout << "Image width not specified, use default [" << DEFAULT_IMG_W << "]" << std::endl;
    }
    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        scale[0] = DEFAULT_SCALE1;
        scale[1] = DEFAULT_SCALE2;
        scale[2] = DEFAULT_SCALE3;
        std::cout << "Scale value not specified, use default [" << scale[0] << ", " << scale[1] << ", " << scale[2] << "]" << std::endl;
    }
    if (mean[0] == -1.0 || mean[1] == -1.0 || mean[2] == -1.0)
    {
        mean[0] = DEFAULT_MEAN1;
        mean[1] = DEFAULT_MEAN2;
        mean[2] = DEFAULT_MEAN3;
        std::cout << "Mean value not specified, use default [" << mean[0] << ", " << mean[1] << ", " << mean[2] << "]" << std::endl;
    }

    init_tengine();
    {
        tengine::Net somenet;
        tengine::Tensor input_tensor;
        tengine::Tensor output_tensor;

        /* set runtime options of Net */
        somenet.opt.num_thread = num_thread;
        somenet.opt.cluster = TENGINE_CLUSTER_ALL;
        somenet.opt.precision = TENGINE_MODE_FP32;

        std::cout << "\ntengine model file : " << model_file << "\n"
                  << "image file : " << image_file << "\n"
                  << "img_h, imag_w, scale, mean[3] : " << img_h << " " << img_w << " " << scale[0] << " " << scale[1]
                  << " " << scale[2] << " " << mean[0] << " " << mean[1] << " " << mean[2] << "\n";

        /* load model */
        somenet.load_model(nullptr, "tengine", model_file.c_str());

        /* prepare input data */
        input_tensor.create(1, 3, img_h, img_w);
        get_input_data(image_file.c_str(), (float*)input_tensor.data, img_h, img_w, mean, scale);

        /* forward */
        somenet.input_tensor("data", input_tensor);

        double min_time, max_time, total_time;
        min_time = DBL_MAX;
        max_time = DBL_MIN;
        total_time = 0;
        for (int i = 0; i < loop_count; i++)
        {
            double start_time = get_current_time();
            somenet.run();
            double end_time = get_current_time();
            double cur_time = end_time - start_time;

            total_time += cur_time;
            max_time = std::max(max_time, cur_time);
            min_time = std::min(min_time, cur_time);
        }
        printf("Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", loop_count, min_time, max_time,
               total_time / loop_count);

        /* get result */
        somenet.extract_tensor("prob", output_tensor);

        /* after process */
        print_topk((float*)output_tensor.data, output_tensor.elem_num, 5);
        std::cout << "--------------------------------------\n";
        std::cout << "ALL TEST DONE\n";
    }

    release_tengine();

    return 0;
}
