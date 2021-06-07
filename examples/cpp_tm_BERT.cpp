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


#define DEFAULT_LOOP_COUNT 1
#define DEFAULT_THREAD_COUNT 1

using namespace std;

void show_usage()
{
    std::cout << "[Usage]: [-h]\n"
              << "    [-m model_file] [-l label_file] [-i image_file]\n"
              << "    [-g img_h,img_w] [-s scale] [-w mean[0],mean[1],mean[2]] [-r repeat_count]\n";

    std::cout << "\nmobilenet example: \n" << "    ./classification -m /path/to/mobilenet.tmfile -l /path/to/labels.txt -i /path/to/img.jpg -g 224,224 -s 0.017 -w 104.007,116.669,122.679" << std::endl;
}

int main(int argc, char* argv[])
{
    int loop_count = DEFAULT_LOOP_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    string model_file;


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
                img_h = ( int )img_hw[0];
                img_w = ( int )img_hw[1];
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



    // check input files
    if(!check_file_exist(model_file.c_str()))
        return -1;

   

    init_tengine();
    {
        tengine::Net somenet;
        tengine::Tensor unique_ids_raw_output;
        tengine::Tensor segment_ids;
        tengine::Tensor input_mask;
        tengine::Tensor input_ids;

        tengine::Tensor unique_ids;
        tengine::Tensor unstack_1;
        tengine::Tensor unstack_0;

        /* set runtime options of Net */
        somenet.opt.num_thread = num_thread;
        somenet.opt.cluster = TENGINE_CLUSTER_ALL;
        somenet.opt.precision = TENGINE_MODE_FP32;

        std::cout << "\ntengine model file : " << model_file << "\n";

        /* load model */
        somenet.load_model(nullptr, "tengine", model_file.c_str());

        /* prepare input data */
        unique_ids_raw_output.create(1);
        segment_ids.create(1,256);
        input_mask.create(1,256);
        input_ids.create(1,256);

        //get_input_data(image_file.c_str(), ( float* )input_tensor.data, img_h, img_w, mean, scale);
        for (int i=0; i<sizeof(unique_ids_raw_output.data)/sizeof(float*)); i++){
            unique_ids_raw_output.data=1;
        }
        for (int i=0; i<sizeof(segment_ids.data)/sizeof(float*)); i++){
            segment_ids.data=1;
        }
        for (int i=0; i<sizeof(input_mask.data)/sizeof(float*)); i++){
            input_mask.data=1;
        }
        for (int i=0; i<sizeof(input_ids.data)/sizeof(float*)); i++){
            input_ids.data=1;
        }        
        /* forward */
        somenet.input_tensor("data1", unique_ids_raw_output);
        somenet.input_tensor("data2", segment_ids);
        somenet.input_tensor("data3", input_mask);
        somenet.input_tensor("data4", input_ids);

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
        somenet.extract_tensor("out1", unique_ids);
        somenet.extract_tensor("out2", unstack_1);
        somenet.extract_tensor("out3", unstack_0);

        /* after process */
        print_topk(( float* )unique_ids.data, unique_ids.elem_num, 5);
        print_topk(( float* )unstack_1.data, unstack_1.elem_num, 5);
        print_topk(( float* )unstack_0.data, unstack_0.elem_num, 5);

        std::cout << "--------------------------------------\n";
        std::cout << "ALL TEST DONE\n";
    }

    release_tengine();

    return 0;
}
