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
#ifndef __CAFFE_MTCNN_HPP__
#define __CAFFE_MTCNN_HPP__

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#include "caffe_mtcnn_utils.hpp"

using namespace caffe;

class caffe_mtcnn
{
public:
    caffe_mtcnn()
    {
        min_size_ = 40;
        pnet_threshold_ = 0.6;
        rnet_threshold_ = 0.7;
        onet_threshold_ = 0.8;
        factor_ = 0.709;
    }

    int load_model(const std::string& model_dir);

    void detect(cv::Mat& img, std::vector<face_box>& face_list);

    ~caffe_mtcnn();

protected:
    void copy_one_patch(const cv::Mat& img, face_box& input_box, float* data_to, int width, int height);
    int run_PNet(const cv::Mat& img, scale_window& win, std::vector<face_box>& box_list);
    void run_RNet(const cv::Mat& img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes);
    void run_ONet(const cv::Mat& img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes);

public:
    int min_size_;
    float pnet_threshold_;
    float rnet_threshold_;
    float onet_threshold_;
    float factor_;

private:
    Net<float>* PNet_;
    Net<float>* RNet_;
    Net<float>* ONet_;
};

#endif    // __CAFFE_MTCNN_HPP__
