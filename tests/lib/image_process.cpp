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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "caffe.pb.h"
#include "image_process.hpp"
#include "logger.hpp"

using namespace caffe;

namespace TEngine {

static bool LoadBinaryFile(const char * fname, BlobProto& blob)
{
   std::ifstream is(fname, std::ios::in|std::ios::binary);

   if(!is.is_open())
   {
       LOG_ERROR()<<"cannot open file: "<<fname<<"\n";
       return false;
   }

   google::protobuf::io::IstreamInputStream input_stream(&is);
   google::protobuf::io::CodedInputStream coded_input(&input_stream);

   coded_input.SetTotalBytesLimit(512<<20, 64<<20);

   bool ret=blob.ParseFromCodedStream(&coded_input);

   is.close();

   if(!ret)
       LOG_ERROR()<<"parse file: "<<fname<<" failed\n";

   return ret;
}


static void GetMean(const std::string& mean_file, cv::Mat & mean_mat, int img_h, int img_w) 
{
  int num_channels=3;

  BlobProto blob;
  LoadBinaryFile(mean_file.c_str(), blob);

  /* Load BlobProto */

   std::vector<int> dims;

   if(blob.has_shape())
   {

        for(int i=0;i<blob.shape().dim_size();i++)
        {
            dims.push_back(blob.shape().dim(i));
        }
   }
   else
   {
        std::vector<int> temp;
        temp.push_back(blob.num());
        temp.push_back(blob.channels());
        temp.push_back(blob.height());
        temp.push_back(blob.width());

        int start=0;

        while(temp[start]==1)  start++;

        for(unsigned int i=start;i<temp.size();i++)
            dims.push_back(temp[i]);
   }

   int mem_size=blob.data_size()*4;

   float * data=(float *)std::malloc(mem_size);

   for(int i=0;i<blob.data_size();i++)
              data[i]=blob.data(i);

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  for (int i = 0; i < num_channels; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(dims[1], dims[2], CV_32FC1, data);
    channels.push_back(channel);
    data += dims[1]*dims[2];
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_mat=cv::Mat(cv::Size(img_h,img_w),mean.type(),channel_mean);
}



float * caffe_process_image(const char * image_file, const char * mean_file, int img_h, int img_w)
{

   cv::Mat img=cv::imread(image_file,-1);

   if(img.empty())
   {
       std::cerr<<"failed to read image file "<<image_file<<"\n";
       return nullptr;
   }

   cv::Mat resized;
   cv::resize(img,resized,cv::Size(img_h,img_w));

   cv::Mat float_img;

   resized.convertTo(float_img,CV_32FC3);

   cv::Mat normalized_img;

   if(mean_file)
   {
      cv::Mat mean;
      GetMean(mean_file,mean,img_h,img_w);

      cv::subtract(float_img,mean,normalized_img);
   }
   else
   {
      normalized_img=float_img;
   }

   std::vector<cv::Mat> input_channels;

   float * input_data=(float*)std::malloc(img_h*img_w*3*4);
 
   float * ptr=input_data;

   for(int i=0;i<3; ++i)
   {
      cv::Mat channel(img_h, img_w, CV_32FC1, ptr);
      input_channels.push_back(channel);
      ptr += img_h* img_w;
   }

   cv::split(normalized_img,input_channels);

   return input_data;
}

} //namespace TEngine

