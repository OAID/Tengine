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

caffe_mtcnn::~caffe_mtcnn(void)
{
    delete PNet_;
    delete RNet_;
    delete ONet_;
}

int caffe_mtcnn::load_model(const std::string& proto_model_dir)
{
    Caffe::set_mode(Caffe::CPU);

    PNet_ = new Net<float>((proto_model_dir + "/det1.prototxt"), caffe::TEST);
    PNet_->CopyTrainedLayersFrom(proto_model_dir + "/det1.caffemodel");

    RNet_ = new Net<float>((proto_model_dir + "/det2.prototxt"), caffe::TEST);
    RNet_->CopyTrainedLayersFrom(proto_model_dir + "/det2.caffemodel");

    ONet_ = new Net<float>((proto_model_dir + "/det3.prototxt"), caffe::TEST);
    ONet_->CopyTrainedLayersFrom(proto_model_dir + "/det3.caffemodel");

    return 0;
}

void caffe_mtcnn::detect(cv::Mat& img, std::vector<face_box>& face_list)
{
    cv::Mat working_img;
    float alpha = 0.0078125;
    float mean = 127.5;

    img.convertTo(working_img, CV_32FC3);

    working_img = (working_img - mean) * alpha;

    working_img = working_img.t();

    cv::cvtColor(working_img, working_img, cv::COLOR_BGR2RGB);

    int img_h = working_img.rows;
    int img_w = working_img.cols;

    std::vector<scale_window> win_list;

    std::vector<face_box> total_pnet_boxes;
    std::vector<face_box> total_rnet_boxes;
    std::vector<face_box> total_onet_boxes;

    cal_pyramid_list(img_h, img_w, min_size_, factor_, win_list);

    for(unsigned int i = 0; i < win_list.size(); i++)
    {
        std::vector<face_box> boxes;

        run_PNet(working_img, win_list[i], boxes);

        total_pnet_boxes.insert(total_pnet_boxes.end(), boxes.begin(), boxes.end());
    }

    std::vector<face_box> pnet_boxes;

    process_boxes(total_pnet_boxes, img_h, img_w, pnet_boxes);

    if(!pnet_boxes.size())
        return;

    run_RNet(working_img, pnet_boxes, total_rnet_boxes);

    std::vector<face_box> rnet_boxes;
    process_boxes(total_rnet_boxes, img_h, img_w, rnet_boxes);

    if(!rnet_boxes.size())
        return;

    run_ONet(working_img, rnet_boxes, total_onet_boxes);

    // calculate the landmark
    for(unsigned int i = 0; i < total_onet_boxes.size(); i++)
    {
        face_box& box = total_onet_boxes[i];

        float h = box.x1 - box.x0 + 1;
        float w = box.y1 - box.y0 + 1;

        for(int j = 0; j < 5; j++)
        {
            box.landmark.x[j] = box.x0 + w * box.landmark.x[j] - 1;
            box.landmark.y[j] = box.y0 + h * box.landmark.y[j] - 1;
        }
    }

    // Get Final Result
    regress_boxes(total_onet_boxes);
    nms_boxes(total_onet_boxes, 0.7, NMS_MIN, face_list);

    // set_box_bound(face_list,img_h,img_w);

    // switch x and y, since working_img is transposed

    for(unsigned int i = 0; i < face_list.size(); i++)
    {
        face_box& box = face_list[i];

        std::swap(box.x0, box.y0);
        std::swap(box.x1, box.y1);

        for(int l = 0; l < 5; l++)
        {
            std::swap(box.landmark.x[l], box.landmark.y[l]);
        }
    }
}

int caffe_mtcnn::run_PNet(const cv::Mat& img, scale_window& win, std::vector<face_box>& box_list)
{
    cv::Mat resized;
    int scale_h = win.h;
    int scale_w = win.w;
    float scale = win.scale;

    cv::resize(img, resized, cv::Size(scale_w, scale_h), 0, 0, cv::INTER_NEAREST);

    Blob<float>* input_blob = PNet_->input_blobs()[0];
    input_blob->Reshape(1, 3, scale_h, scale_w);
    PNet_->Reshape();

    std::vector<cv::Mat> input_channels;
    float* input_data = PNet_->input_blobs()[0]->mutable_cpu_data();
    set_input_buffer(input_channels, input_data, scale_h, scale_w);

    cv::split(resized, input_channels);

    PNet_->Forward();

    Blob<float>* reg = PNet_->output_blobs()[0];
    Blob<float>* confidence = PNet_->output_blobs()[1];

    int feature_h = reg->shape(2);
    int feature_w = reg->shape(3);
    std::vector<face_box> candidate_boxes;

    generate_bounding_box(confidence->cpu_data(), confidence->count(), reg->cpu_data(), scale, pnet_threshold_,
                          feature_h, feature_w, candidate_boxes, true);

    nms_boxes(candidate_boxes, 0.5, NMS_UNION, box_list);

    return 0;
}

void caffe_mtcnn::copy_one_patch(const cv::Mat& img, face_box& input_box, float* data_to, int width, int height)
{
    std::vector<cv::Mat> channels;

    set_input_buffer(channels, data_to, height, width);

    cv::Mat chop_img = img(cv::Range(input_box.py0, input_box.py1), cv::Range(input_box.px0, input_box.px1));

    int pad_top = std::abs(input_box.py0 - input_box.y0);
    int pad_left = std::abs(input_box.px0 - input_box.x0);
    int pad_bottom = std::abs(input_box.py1 - input_box.y1);
    int pad_right = std::abs(input_box.px1 - input_box.x1);

    cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                       cv::Scalar(0));

    cv::resize(chop_img, chop_img, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::split(chop_img, channels);
}

void caffe_mtcnn::run_RNet(const cv::Mat& img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes)
{
    int batch = pnet_boxes.size();
    int channel = 3;
    int height = 24;
    int width = 24;

    std::vector<int> input_shape = {batch, channel, height, width};

    Blob<float>* input_blob = RNet_->input_blobs()[0];

    input_blob->Reshape(input_shape);

    RNet_->Reshape();

    float* input_data = input_blob->mutable_cpu_data();

    for(int i = 0; i < batch; i++)
    {
        int img_size = channel * height * width;

        copy_one_patch(img, pnet_boxes[i], input_data, height, width);
        input_data += img_size;
    }

    RNet_->Forward();

    const Blob<float>* reg = RNet_->output_blobs()[0];
    const Blob<float>* confidence = RNet_->output_blobs()[1];

    const float* confidence_data = confidence->cpu_data();
    const float* reg_data = reg->cpu_data();

    int conf_page_size = confidence->count(1);
    int reg_page_size = reg->count(1);

    for(int i = 0; i < batch; i++)
    {
        if(*(confidence_data + 1) > rnet_threshold_)
        {
            face_box output_box;
            face_box& input_box = pnet_boxes[i];

            output_box.x0 = input_box.x0;
            output_box.y0 = input_box.y0;
            output_box.x1 = input_box.x1;
            output_box.y1 = input_box.y1;

            output_box.score = *(confidence_data + 1);

            /*Note: regress's value is swaped here!!!*/

            output_box.regress[0] = reg_data[1];
            output_box.regress[1] = reg_data[0];
            output_box.regress[2] = reg_data[3];
            output_box.regress[3] = reg_data[2];

            output_boxes.push_back(output_box);
        }

        confidence_data += conf_page_size;
        reg_data += reg_page_size;
    }
}

void caffe_mtcnn::run_ONet(const cv::Mat& img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes)
{
    int batch = rnet_boxes.size();
    int channel = 3;
    int height = 48;
    int width = 48;

    std::vector<int> input_shape = {batch, channel, height, width};

    Blob<float>* input_blob = ONet_->input_blobs()[0];

    input_blob->Reshape(input_shape);

    ONet_->Reshape();

    float* input_data = input_blob->mutable_cpu_data();

    for(int i = 0; i < batch; i++)
    {
        copy_one_patch(img, rnet_boxes[i], input_data, height, width);
        input_data += channel * height * width;
    }

    ONet_->Forward();

    const Blob<float>* reg = ONet_->output_blobs()[0];
    const Blob<float>* confidence = ONet_->output_blobs()[2];
    const Blob<float>* points_blob = ONet_->output_blobs()[1];

    const float* confidence_data = confidence->cpu_data();
    const float* reg_data = reg->cpu_data();
    const float* points_data = points_blob->cpu_data();

    int conf_page_size = confidence->count(1);
    int reg_page_size = reg->count(1);
    int points_page_size = points_blob->count(1);

    for(int i = 0; i < batch; i++)
    {
        if(*(confidence_data + 1) > onet_threshold_)
        {
            face_box output_box;
            face_box& input_box = rnet_boxes[i];

            output_box.x0 = input_box.x0;
            output_box.y0 = input_box.y0;
            output_box.x1 = input_box.x1;
            output_box.y1 = input_box.y1;

            output_box.score = *(confidence_data + 1);

            output_box.regress[0] = reg_data[1];
            output_box.regress[1] = reg_data[0];
            output_box.regress[2] = reg_data[3];
            output_box.regress[3] = reg_data[2];

            /*Note: switched x,y points value too..*/

            for(int j = 0; j < 5; j++)
            {
                output_box.landmark.x[j] = *(points_data + j + 5);
                output_box.landmark.y[j] = *(points_data + j);
            }

            output_boxes.push_back(output_box);
        }

        confidence_data += conf_page_size;
        reg_data += reg_page_size;
        points_data += points_page_size;
    }
}
