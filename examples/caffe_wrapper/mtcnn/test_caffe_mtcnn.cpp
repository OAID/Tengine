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
 * Copyright (c) 2018, Open AI Lab
 * Author: jingyou@openailab.com
 */
#include "caffe_mtcnn.hpp"

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        std::cout << "[usage]: " << argv[0] << " <test.jpg>  <model_dir> [save_result.jpg] \n";
        return 0;
    }
    const char* fname = argv[1];
    std::string model_dir = argv[2];
    std::string sv_name = "result.jpg";
    if(argc == 4)
        sv_name = argv[3];

    cv::Mat frame = cv::imread(fname);

    if(!frame.data)
    {
        std::cerr << "failed to read image file: " << fname << std::endl;
        return 1;
    }

    std::vector<face_box> face_info;
    caffe_mtcnn* p_mtcnn = new caffe_mtcnn();

    p_mtcnn->load_model(model_dir);

    unsigned long start_time = get_cur_time();

    p_mtcnn->detect(frame, face_info);

    unsigned long end_time = get_cur_time();

    for(unsigned int i = 0; i < face_info.size(); i++)
    {
        face_box& box = face_info[i];

        printf("face %d: x0,y0 %2.5f %2.5f  x1,y1 %2.5f  %2.5f conf: %2.5f\n", i, box.x0, box.y0, box.x1, box.y1,
               box.score);
        printf("landmark: ");
        for(unsigned int j = 0; j < 5; j++)
            printf(" (%2.5f %2.5f)", box.landmark.x[j], box.landmark.y[j]);
        printf("\n");

        /*draw box */
        cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);

        /* draw landmark */
        for(int l = 0; l < 5; l++)
        {
            cv::circle(frame, cv::Point(box.landmark.x[l], box.landmark.y[l]), 1, cv::Scalar(0, 0, 255), 1.8);
        }
    }

    cv::imwrite(sv_name, frame);
    std::cout << "total detected: " << face_info.size() << " faces. used " << (end_time - start_time) << " us"
              << std::endl;

    delete p_mtcnn;

    return 0;
}
