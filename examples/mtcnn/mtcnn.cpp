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

mtcnn::mtcnn(int minsize, float conf_p, float conf_r, float conf_o, float nms_p, float nms_r, float nms_o)
{
    minsize_ = minsize;

    conf_p_threshold_ = conf_p;
    conf_r_threshold_ = conf_r;
    conf_o_threshold_ = conf_o;

    nms_p_threshold_ = nms_p;
    nms_r_threshold_ = nms_r;
    nms_o_threshold_ = nms_o;
}
int mtcnn::load_3model(const std::string& model_dir)
{
    std::string proto_name, mdl_name;

    // Pnet
    proto_name = model_dir + "/det1.prototxt";
    mdl_name = model_dir + "/det1.caffemodel";
    if(!check_file_exist(proto_name) or (!check_file_exist(mdl_name)))
    {
        return 1;
    }

    PNet_graph = create_graph(nullptr, "caffe", proto_name.c_str(), mdl_name.c_str());
    if(PNet_graph == nullptr)
    {
        std::cout << "Create Pnet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // Rnet
    proto_name = model_dir + "/det2.prototxt";
    mdl_name = model_dir + "/det2.caffemodel";
    if(!check_file_exist(proto_name) or (!check_file_exist(mdl_name)))
    {
        return 1;
    }
    RNet_graph = create_graph(nullptr, "caffe", proto_name.c_str(), mdl_name.c_str());
    if(RNet_graph == nullptr)
    {
        std::cout << "Create Rnet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // Onet
    proto_name = model_dir + "/det3.prototxt";
    mdl_name = model_dir + "/det3.caffemodel";
    if(!check_file_exist(proto_name) or (!check_file_exist(mdl_name)))
    {
        return 1;
    }
    ONet_graph = create_graph(nullptr, "caffe", proto_name.c_str(), mdl_name.c_str());
    if(ONet_graph == nullptr)
    {
        std::cout << "Create Onet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    return 0;
}

int mtcnn::run_PNet(const cv::Mat& img, scale_window& win, std::vector<face_box>& box_list)
{
    cv::Mat resized;
    int scale_h = win.h;
    int scale_w = win.w;
    float scale = win.scale;

    static bool first_run = true;

    cv::resize(img, resized, cv::Size(scale_w, scale_h), 0, 0, cv::INTER_NEAREST);
    /* input */

    tensor_t input_tensor = get_graph_tensor(PNet_graph, "data");
    int dims[] = {1, 3, scale_h, scale_w};
    set_tensor_shape(input_tensor, dims, 4);
    int in_mem = sizeof(float) * scale_h * scale_w * 3;
    // std::cout<<"mem "<<in_mem<<"\n";
    float* input_data = ( float* )malloc(in_mem);

    std::vector<cv::Mat> input_channels;
    set_cvMat_input_buffer(input_channels, input_data, scale_h, scale_w);
    cv::split(resized, input_channels);

    set_tensor_buffer(input_tensor, input_data, in_mem);

    if(first_run)
    {
        if(prerun_graph(PNet_graph) != 0)
        {
            std::cout << "Prerun PNet graph failed, errno: " << get_tengine_errno() << "\n";
            return 1;
        }
        first_run = false;
    }

    if(run_graph(PNet_graph, 1) != 0)
    {
        std::cout << "Run PNet graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    free(input_data);
    /* output */
    tensor_t tensor = get_graph_tensor(PNet_graph, "conv4-2");
    get_tensor_shape(tensor, dims, 4);
    float* reg_data = ( float* )get_tensor_buffer(tensor);
    int feature_h = dims[2];
    int feature_w = dims[3];
    // std::cout<<"Pnet scale h,w= "<<feature_h<<","<<feature_w<<"\n";

    tensor = get_graph_tensor(PNet_graph, "prob1");
    float* prob_data = ( float* )get_tensor_buffer(tensor);
    std::vector<face_box> candidate_boxes;
    generate_bounding_box(prob_data, reg_data, scale, conf_p_threshold_, feature_h, feature_w, candidate_boxes, true);

    release_graph_tensor(input_tensor);
    release_graph_tensor(tensor);

    nms_boxes(candidate_boxes, 0.5, NMS_UNION, box_list);

    // std::cout<<"condidate boxes size :"<<candidate_boxes.size()<<"\n";
    return 0;
}

int mtcnn::run_RNet(const cv::Mat& img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes)
{
    int batch = pnet_boxes.size();
    int channel = 3;
    int height = 24;
    int width = 24;
    static bool first_run = true;

    tensor_t input_tensor = get_graph_tensor(RNet_graph, "data");
    int dims[] = {batch, channel, height, width};
    set_tensor_shape(input_tensor, dims, 4);
    int img_size = channel * height * width;
    int in_mem = sizeof(float) * batch * img_size;
    float* input_data = ( float* )malloc(in_mem);
    float* input_ptr = input_data;
    set_tensor_buffer(input_tensor, input_ptr, in_mem);

    for(int i = 0; i < batch; i++)
    {
        copy_one_patch(img, pnet_boxes[i], input_ptr, width, height);
        input_ptr += img_size;
    }

    if(first_run)
    {
        if(prerun_graph(RNet_graph) != 0)
        {
            std::cout << "Prerun RNet graph failed, errno: " << get_tengine_errno() << "\n";
            return 1;
        }
        first_run = false;
    }

    if(run_graph(RNet_graph, 1) != 0)
    {
        std::cout << "Run RNet graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    free(input_data);
    // std::cout<<"run done ------\n";
    //
    /* output */
    tensor_t tensor = get_graph_tensor(RNet_graph, "conv5-2");
    float* reg_data = ( float* )get_tensor_buffer(tensor);

    tensor = get_graph_tensor(RNet_graph, "prob1");
    float* confidence_data = ( float* )get_tensor_buffer(tensor);

    int conf_page_size = 2;
    int reg_page_size = 4;

    for(int i = 0; i < batch; i++)
    {
        if(*(confidence_data + 1) > conf_r_threshold_)
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
            // std::cout<<"in ";
        }

        confidence_data += conf_page_size;
        reg_data += reg_page_size;
    }

    release_graph_tensor(input_tensor);
    release_graph_tensor(tensor);

    return 0;
}

int mtcnn::run_ONet(const cv::Mat& img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes)
{
    int batch = rnet_boxes.size();

    int channel = 3;
    int height = 48;
    int width = 48;
    tensor_t input_tensor = get_graph_tensor(ONet_graph, "data");
    int dims[] = {batch, channel, height, width};
    set_tensor_shape(input_tensor, dims, 4);
    int img_size = channel * height * width;
    int in_mem = sizeof(float) * batch * img_size;
    float* input_data = ( float* )malloc(in_mem);
    float* input_ptr = input_data;
    static bool first_run = true;

    set_tensor_buffer(input_tensor, input_ptr, in_mem);
    for(int i = 0; i < batch; i++)
    {
        copy_one_patch(img, rnet_boxes[i], input_ptr, width, height);
        input_ptr += img_size;
    }

    if(first_run)
    {
        if(prerun_graph(ONet_graph) != 0)
        {
            std::cout << "Prerun ONet graph failed, errno: " << get_tengine_errno() << "\n";
            return 1;
        }
        first_run = false;
    }

    if(run_graph(ONet_graph, 1) != 0)
    {
        std::cout << "Run ONet graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    free(input_data);
    /* output */
    tensor_t tensor = get_graph_tensor(ONet_graph, "conv6-3");
    float* points_data = ( float* )get_tensor_buffer(tensor);

    tensor = get_graph_tensor(ONet_graph, "prob1");
    float* confidence_data = ( float* )get_tensor_buffer(tensor);

    tensor = get_graph_tensor(ONet_graph, "conv6-2");
    float* reg_data = ( float* )get_tensor_buffer(tensor);

    int conf_page_size = 2;
    int reg_page_size = 4;
    int points_page_size = 10;
    for(int i = 0; i < batch; i++)
    {
        if(*(confidence_data + 1) > conf_r_threshold_)
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
    release_graph_tensor(input_tensor);
    release_graph_tensor(tensor);
    return 0;
}

void mtcnn::detect(cv::Mat& img, std::vector<face_box>& face_list)
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

    cal_scale_list(img_h, img_w, minsize_, win_list);
    for(unsigned int i = 0; i < win_list.size(); i++)
    {
        std::vector<face_box> boxes;
        if(run_PNet(working_img, win_list[i], boxes) != 0)
            return;
        total_pnet_boxes.insert(total_pnet_boxes.end(), boxes.begin(), boxes.end());
    }
    win_list.clear();
    std::vector<face_box> pnet_boxes;
    process_boxes(total_pnet_boxes, img_h, img_w, pnet_boxes, nms_p_threshold_);

    if(!pnet_boxes.size())
        return;
    // for(unsigned int i = 0;i < pnet_boxes.size(); i++)
    // {
    // 	face_box b=pnet_boxes[i];
    // 	std::cout<<i <<","<<b.x0<<" "<<b.x1<< " "<<b.y0<<" "<<b.y1<<"\t"<<b.score<<"\n";
    // }
    if(run_RNet(working_img, pnet_boxes, total_rnet_boxes) != 0)
        return;
    total_pnet_boxes.clear();

    std::vector<face_box> rnet_boxes;
    process_boxes(total_rnet_boxes, img_h, img_w, rnet_boxes, nms_r_threshold_);

    if(!rnet_boxes.size())
        return;
    // for(unsigned int i = 0;i < rnet_boxes.size(); i++)
    // {
    // 	face_box b=rnet_boxes[i];
    // 	std::cout<<i <<","<<b.x0<<" "<<b.x1<< " "<<b.y0<<" "<<b.y1<<"\t"<<b.score<<"\n";
    // }
    if(run_ONet(working_img, rnet_boxes, total_onet_boxes) != 0)
        return;
    total_rnet_boxes.clear();

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
        // std::cout<<"i="<<i<<"\t"<<box.x0<<" "<<box.y0<<" "<<box.x1<<" "<<box.y1<<" "<<box.landmark.x[3]<<"
        // "<<box.landmark.y[2]<<"\n";
    }
    regress_boxes(total_onet_boxes);
    nms_boxes(total_onet_boxes, nms_o_threshold_, NMS_MIN, face_list);
    total_onet_boxes.clear();

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
