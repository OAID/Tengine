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
 * Copyright (c) 2020, Open AI Lab
 * Author: zengjiejun@openailab.com
 */
#include <string>
#include <sys/time.h>
#include "mtcnn.hpp"
#include "mtcnn_utils.hpp"
#include "common.hpp"

int main(int argc, char** argv)
{
    const std::string root_path = get_root_path();
    std::string img_name = root_path + "tests/images/mtcnn_face4.jpg";
    std::string model_dir = root_path + "models/";
    std::string sv_name = "result.jpg";

    if(init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }
    if(request_tengine_version("0.9") < 0)
    {
        std::cout << " request tengine version failed\n";
        return 1;
    }

    if(argc == 1)
    {
        std::cout << "[usage]: " << argv[0] << " <test.jpg>  <model_dir> [save_result.jpg] \n";
    }
    if(argc >= 2)
        img_name = argv[1];
    if(argc >= 3)
        model_dir = argv[2];
    if(argc >= 4)
        sv_name = argv[3];

    image im = imread(img_name.c_str());
    if(im.data==NULL)
    {
        std::cerr << "cv::imread " << img_name << " failed\n";
        return -1;
    }

    int min_size = 40;

    float conf_p = 0.6;
    float conf_r = 0.7;
    float conf_o = 0.8;

    float nms_p = 0.5;
    float nms_r = 0.7;
    float nms_o = 0.7;

    mtcnn* det = new mtcnn(min_size, conf_p, conf_r, conf_o, nms_p, nms_r, nms_o);
    det->load_3model(model_dir);

    int repeat_count = 1;
    const char* repeat = std::getenv("REPEAT_COUNT");

    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    struct timeval t0, t1;
    float avg_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    std::vector<face_box> face_info;
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        det->detect(im, face_info);
        gettimeofday(&t1, NULL);

        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;

        std::cout << "i =" << i << " time is " << mytime << "\n";
        // <<face_info[0].x0<<","<<face_info[0].y0<<","<<face_info[0].x1<<","<<face_info[0].y1<<"\n";

        avg_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
    }
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n"
                  << "detected face num: " << face_info.size() << "\n";

    for(unsigned int i = 0; i < face_info.size(); i++)
    {
        face_box& box = face_info[i];
        std::printf("BOX:( %d , %d ),( %d , %d )\n", (int)box.x0, (int)box.y0, (int)box.x1, (int)box.y1);
        draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 0, 0, 0);
        for(int l = 0; l < 5; l++)
        {
            draw_circle(im, box.landmark.x[l], box.landmark.y[l], 3, 50, 125, 25);
        }
    }

    save_image(im, "tengine_example_out");
    free_image(im);
    delete det;

    release_tengine();

    return 0;
}
