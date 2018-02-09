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
 * Author: chunyinglv@openailab.com
 */
#include "mtcnn.hpp"
#include "mtcnn_utils.hpp"
#include <string>
#include <sys/time.h>


int main(int argc,char** argv)
{   
    init_tengine_library();
    if(request_tengine_version("0.1")<0)
        return 1;
    if(argc<3)
    {
        std::cout<<"[usage]: "<<argv[0]<<" <test.jpg>  <model_dir> [save_result.jpg] \n";
        return 0;
    }
    std::string img_name=argv[1];
    std::string sv_name="result.jpg";
    if(argc==4) sv_name=argv[3];

    cv::Mat image = cv::imread(img_name);
    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
        return -1;
    }
    std::string model_dir=argv[2];
    std::vector<face_box> face_info;


    int min_size=60;

    float conf_p=0.6;
    float conf_r=0.7;
    float conf_o=0.8;

	float nms_p=0.5;
    float nms_r=0.7;
    float nms_o=0.7;

    mtcnn det(min_size,conf_p,conf_r,conf_o,nms_p,nms_r,nms_o);
    det.load_3model(model_dir);

    struct timeval t0, t1;
    for(int i=0;i<1;i++)
    {
      gettimeofday(&t0, NULL);
	  
      det.detect(image,face_info);
      gettimeofday(&t1, NULL);
      float mytime=(float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
      std::cout<<"i ="<<i<<" time is "<<mytime<<"\n";
    }
 
    // struct timeval t0, t1;
    //std::vector<face_box> face_info1;
    // int  repeat=30;
    // float avg_time=0.f;
    // for(int i=0;i<repeat+1;i++)
    // {
    //   gettimeofday(&t0, NULL);
    //   det.detect(image,face_info1);
    //   gettimeofday(&t1, NULL);
    //   float mytime=(float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    //   std::cout<<"i ="<<i<<" time is "<<mytime<<"\n";
    //   if(i>0) avg_time+=mytime;
    // }
    // std::cout<<"repeat "<<repeat<<", avg time is "<<avg_time/repeat<<" \n ";
    //det.detect(image,face_info);

    int num_faces=face_info.size();
    std::cout<<"detected "<<num_faces<<" faces\n";

    for(unsigned int i=0;i<face_info.size();i++)
	{
		face_box& box=face_info[i];

        cv::rectangle(image, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);
		for(int l=0;l<5;l++)
		{
			cv::circle(image,cv::Point(box.landmark.x[l],box.landmark.y[l]),1,cv::Scalar(255, 0, 255),2);
		}
    }

    cv::imwrite(sv_name,image);
    return  0;
}