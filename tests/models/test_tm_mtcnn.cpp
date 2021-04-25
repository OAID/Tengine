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
 * Copyright (c) 2018, OPEN AI LAB
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <string.h>
#include <iomanip>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "common.hpp"

#define NMS_UNION 1
#define NMS_MIN 2

struct scale_window
{
    int h;
    int w;
    float scale;
};

struct face_landmark
{
    float x[5];
    float y[5];
};

struct face_box
{
    float x0;
    float y0;
    float x1;
    float y1;

    /* confidence score */
    float score;

    /*regression scale */
    float regress[4];

    /* padding stuff*/
    float px0;
    float py0;
    float px1;
    float py1;

    face_landmark landmark;
};


class mtcnn
{
public:
    int minsize_;
    float conf_p_threshold_;
    float conf_r_threshold_;
    float conf_o_threshold_;

    float nms_p_threshold_;
    float nms_r_threshold_;
    float nms_o_threshold_;

    mtcnn(int minsize = 60, float conf_p = 0.6, float conf_r = 0.7, float conf_o = 0.8, float nms_p = 0.5,
          float nms_r = 0.7, float nms_o = 0.7);
    ~mtcnn()
    {
        if(postrun_graph(PNet_graph) != 0)
        {
            std::cout << "Postrun PNet graph failed, errno: " << get_tengine_errno() << "\n";
        }
        if(postrun_graph(RNet_graph) != 0)
        {
            std::cout << "Postrun RNet graph failed, errno: " << get_tengine_errno() << "\n";
        }
        if(postrun_graph(ONet_graph) != 0)
        {
            std::cout << "Postrun ONet graph failed, errno: " << get_tengine_errno() << "\n";
        }
        destroy_graph(PNet_graph);
        destroy_graph(RNet_graph);
        destroy_graph(ONet_graph);
    };

    int load_3model(const std::string& model_dir);

    void detect(image img, std::vector<face_box>& face_list);

protected:
    int run_PNet(image img, scale_window& win, std::vector<face_box>& box_list);
    int run_RNet(image img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes);
    int run_ONet(image img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes);

private:
    graph_t PNet_graph;
    graph_t RNet_graph;
    graph_t ONet_graph;
};

void cal_scale_list(int height, int width, int minsize, std::vector<scale_window>& list)
{
    float factor = 0.709;
    int MIN_DET_SIZE = 12;
    int minl = height < width ? height : width;
    float m = ( float )MIN_DET_SIZE / minsize;
    minl = minl * m;
    int factor_count = 0;
    while(minl > MIN_DET_SIZE)
    {
        if(factor_count > 0)
            m = m * factor;
        minl *= factor;
        factor_count++;

        scale_window win;
        win.h = ( int )ceil(height * m);
        ;
        win.w = ( int )ceil(width * m);
        ;
        win.scale = m;
        list.push_back(win);
    }
}


void generate_bounding_box(const float* confidence_data, const float* reg_data, float scale, float threshold,
                           int feature_h, int feature_w, std::vector<face_box>& output, bool transposed)
{
    int stride = 2;
    int cellSize = 12;

    int img_h = feature_h;
    int img_w = feature_w;
    int count = img_h * img_w;
    confidence_data += count;

    for(int i = 0; i < count; i++)
    {
        if(*(confidence_data + i) >= threshold)
        {
            int y = i / img_w;
            int x = i - img_w * y;

            float top_x = ( int )((x * stride + 1) / scale);
            float top_y = ( int )((y * stride + 1) / scale);
            float bottom_x = ( int )((x * stride + cellSize) / scale);
            float bottom_y = ( int )((y * stride + cellSize) / scale);

            face_box box;
            box.x0 = top_x;
            box.y0 = top_y;
            box.x1 = bottom_x;
            box.y1 = bottom_y;

            box.score = *(confidence_data + i);

            int c_offset = y * img_w + x;
            int c_size = img_w * img_h;

            if(transposed)
            {
                box.regress[1] = reg_data[c_offset];
                box.regress[0] = reg_data[c_offset + c_size];
                box.regress[3] = reg_data[c_offset + 2 * c_size];
                box.regress[2] = reg_data[c_offset + 3 * c_size];
            }
            else
            {
                box.regress[0] = reg_data[c_offset];
                box.regress[1] = reg_data[c_offset + c_size];
                box.regress[2] = reg_data[c_offset + 2 * c_size];
                box.regress[3] = reg_data[c_offset + 3 * c_size];
            }

            output.push_back(box);
        }
    }
}

void nms_boxes(std::vector<face_box>& input, float threshold, int type, std::vector<face_box>& output)
{
    output.clear();
    std::sort(input.begin(), input.end(), [](const face_box& a, const face_box& b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for(int i = 0; i < box_num; i++)
    {
        if(merged[i])
            continue;

        output.push_back(input[i]);

        float h0 = input[i].y1 - input[i].y0 + 1;
        float w0 = input[i].x1 - input[i].x0 + 1;

        float area0 = h0 * w0;

        for(int j = i + 1; j < box_num; j++)
        {
            if(merged[j])
                continue;

            float inner_x0 = std::max(input[i].x0, input[j].x0);
            float inner_y0 = std::max(input[i].y0, input[j].y0);

            float inner_x1 = std::min(input[i].x1, input[j].x1);
            float inner_y1 = std::min(input[i].y1, input[j].y1);

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if(inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y1 - input[j].y0 + 1;
            float w1 = input[j].x1 - input[j].x0 + 1;

            float area1 = h1 * w1;

            float score;

            if(type == NMS_UNION)
            {
                score = inner_area / (area0 + area1 - inner_area);
            }
            else
            {
                score = inner_area / std::min(area0, area1);
            }

            if(score > threshold)
                merged[j] = 1;
        }
    }
}
void regress_boxes(std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        face_box& box = rects[i];

        float h = box.y1 - box.y0 + 1;
        float w = box.x1 - box.x0 + 1;

        box.x0 = box.x0 + w * box.regress[0];
        box.y0 = box.y0 + h * box.regress[1];
        box.x1 = box.x1 + w * box.regress[2];
        box.y1 = box.y1 + h * box.regress[3];
    }
}
void square_boxes(std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        float h = rects[i].y1 - rects[i].y0 + 1;
        float w = rects[i].x1 - rects[i].x0 + 1;

        float l = std::max(h, w);

        rects[i].x0 = rects[i].x0 + (w - l) * 0.5;
        rects[i].y0 = rects[i].y0 + (h - l) * 0.5;
        rects[i].x1 = rects[i].x0 + l - 1;
        rects[i].y1 = rects[i].y0 + l - 1;
    }
}

void padding(int img_h, int img_w, std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        rects[i].px0 = std::max(rects[i].x0, 1.0f);
        rects[i].py0 = std::max(rects[i].y0, 1.0f);
        rects[i].px1 = std::min(rects[i].x1, ( float )img_w);
        rects[i].py1 = std::min(rects[i].y1, ( float )img_h);
    }
}
void process_boxes(std::vector<face_box>& input, int img_h, int img_w, std::vector<face_box>& rects, float nms_r_thresh)
{
    nms_boxes(input, nms_r_thresh, NMS_UNION, rects);

    regress_boxes(rects);

    square_boxes(rects);

    padding(img_h, img_w, rects);
}
void copy_one_patch(image img, face_box& input_box, float* data_to, int width, int height)
{
    double w = std::abs(input_box.px0 - input_box.px1);
    double h = std::abs(input_box.py0 - input_box.py1);
    image chop_im = make_image(( int )w, ( int )h, img.c);

    for(int c = 0; c < img.c; c++)
    {
        for(int i = 0; i < ( int )h; i++)
        {
            for(int j = 0; j < ( int )w; j++)
            {
                chop_im.data[c * ( int )h * ( int )w + i * ( int )w + j] =
                    img.data[c * img.h * img.w + (i + ( int )input_box.py0) * img.w + j + ( int )input_box.px0];
            }
        }
    }

    int pad_top = std::abs(input_box.py0 - input_box.y0);
    int pad_left = std::abs(input_box.px0 - input_box.x0);
    int pad_bottom = std::abs(input_box.py1 - input_box.y1);
    int pad_right = std::abs(input_box.px1 - input_box.x1);

    image copyMakerImage = copyMaker(chop_im, pad_top, pad_bottom, pad_left, pad_right, 0);

    image resImg = resize_image(copyMakerImage, width, height);

    for(int i = 0; i < 3 * width * height; i++)
    {
        *data_to = resImg.data[i];
        data_to++;
    }
}

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
    std::string mdl_name;

    // Pnet
    mdl_name = model_dir + "/det1.tmfile";
    if(!check_file_exist(mdl_name))
    {
        return 1;
    }

    PNet_graph = create_graph(nullptr, "tengine", mdl_name.c_str());
    if(PNet_graph == nullptr)
    {
        std::cout << "Create Pnet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // Rnet
    mdl_name = model_dir + "/det2.tmfile";
    if(!check_file_exist(mdl_name))
    {
        return 1;
    }
    RNet_graph = create_graph(nullptr, "tengine", mdl_name.c_str());
    if(RNet_graph == nullptr)
    {
        std::cout << "Create Rnet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // Onet
    mdl_name = model_dir + "/det3.tmfile";
    if(!check_file_exist(mdl_name))
    {
        return 1;
    }
    ONet_graph = create_graph(nullptr, "tengine", mdl_name.c_str());
    if(ONet_graph == nullptr)
    {
        std::cout << "Create Onet Graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    return 0;
}

int mtcnn::run_PNet(image img, scale_window& win, std::vector<face_box>& box_list)
{
    int scale_h = win.h;

    int scale_w = win.w;
    float scale = win.scale;
    static bool first_run = true;
    image resImg = resize_image(img, scale_w, scale_h);

    /* input */
    // tensor_t input_tensor = get_graph_tensor(PNet_graph, "data");
    tensor_t input_tensor = get_graph_input_tensor(PNet_graph, 0, 0);
    int dims[] = {1, 3, scale_h, scale_w};
    set_tensor_shape(input_tensor, dims, 4);
    int in_mem = sizeof(float) * scale_h * scale_w * 3;
    // std::cout<<"mem "<<in_mem<<"\n";
    float* input_data = ( float* )malloc(in_mem);

    memcpy(input_data, resImg.data, sizeof(float) * 3 * scale_h * scale_w);
    free_image(resImg);

    set_tensor_buffer(input_tensor, input_data, in_mem);

    if(first_run)
    {
	/* prerun the graph */
	struct options opt;
	opt.num_thread = 1;
	opt.cluster = TENGINE_CLUSTER_ALL;
	opt.precision = TENGINE_MODE_FP32;
	if(std::getenv("NumThreadLite"))
	    	opt.num_thread = atoi(std::getenv("NumThreadLite"));
	if(std::getenv("NumClusterLite"))
		    	opt.cluster = atoi(std::getenv("NumClusterLite"));
	if(std::getenv("DataPrecision"))
		    	opt.precision = atoi(std::getenv("DataPrecision"));

	std::cout<<"Number Thread  : [" << opt.num_thread <<"], use export NumThreadLite=1/2/4 set\n";
	std::cout<<"CPU Cluster    : [" << opt.cluster <<"], use export NumClusterLite=0/1/2/3 set\n";
	std::cout<<"Data Precision : [" << opt.precision <<"], use export DataPrecision=0/1/2/3 set\n";

        if(prerun_graph_multithread(PNet_graph, opt) < 0)
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
    // tensor_t tensor = get_graph_output_tensor(PNet_graph, 0, 0);
    get_tensor_shape(tensor, dims, 4);
    float* reg_data = ( float* )get_tensor_buffer(tensor);
    int feature_h = dims[2];
    int feature_w = dims[3];
    // std::cout<<"Pnet scale h,w= "<<feature_h<<","<<feature_w<<"\n";

    tensor = get_graph_tensor(PNet_graph, "prob1");
    // tensor = get_graph_output_tensor(PNet_graph, 1, 0);
    float* prob_data = ( float* )get_tensor_buffer(tensor);
    std::vector<face_box> candidate_boxes;
    generate_bounding_box(prob_data, reg_data, scale, conf_p_threshold_, feature_h, feature_w, candidate_boxes, true);

    release_graph_tensor(input_tensor);
    release_graph_tensor(tensor);

    nms_boxes(candidate_boxes, 0.5, NMS_UNION, box_list);

    // std::cout<<"condidate boxes size :"<<candidate_boxes.size()<<"\n";
    return 0;
}

int mtcnn::run_RNet(image img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes)
{
    int batch = pnet_boxes.size();
    int channel = 3;
    int height = 24;
    int width = 24;
    static bool first_run = true;

    // tensor_t input_tensor = get_graph_tensor(RNet_graph, "data");
    tensor_t input_tensor = get_graph_input_tensor(RNet_graph, 0, 0);
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
	/* prerun the graph */
	struct options opt;
	opt.num_thread = 1;
	opt.cluster = TENGINE_CLUSTER_ALL;
	opt.precision = TENGINE_MODE_FP32;
	if(std::getenv("NumThreadLite"))
	    	opt.num_thread = atoi(std::getenv("NumThreadLite"));
	if(std::getenv("NumClusterLite"))
		    	opt.cluster = atoi(std::getenv("NumClusterLite"));
	if(std::getenv("DataPrecision"))
		    	opt.precision = atoi(std::getenv("DataPrecision"));

	std::cout<<"Number Thread  : [" << opt.num_thread <<"], use export NumThreadLite=1/2/4 set\n";
	std::cout<<"CPU Cluster    : [" << opt.cluster <<"], use export NumClusterLite=0/1/2/3 set\n";
	std::cout<<"Data Precision : [" << opt.precision <<"], use export DataPrecision=0/1/2/3 set\n";

        if(prerun_graph_multithread(RNet_graph, opt) < 0)
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

int mtcnn::run_ONet(image img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes)
{
    int batch = rnet_boxes.size();

    int channel = 3;
    int height = 48;
    int width = 48;
    // tensor_t input_tensor = get_graph_tensor(ONet_graph, "data");
    tensor_t input_tensor = get_graph_input_tensor(ONet_graph, 0, 0);
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
	/* prerun the graph */
	struct options opt;
	opt.num_thread = 1;
	opt.cluster = TENGINE_CLUSTER_ALL;
	opt.precision = TENGINE_MODE_FP32;
	if(std::getenv("NumThreadLite"))
	    	opt.num_thread = atoi(std::getenv("NumThreadLite"));
	if(std::getenv("NumClusterLite"))
		    	opt.cluster = atoi(std::getenv("NumClusterLite"));
	if(std::getenv("DataPrecision"))
		    	opt.precision = atoi(std::getenv("DataPrecision"));

	std::cout<<"Number Thread  : [" << opt.num_thread <<"], use export NumThreadLite=1/2/4 set\n";
	std::cout<<"CPU Cluster    : [" << opt.cluster <<"], use export NumClusterLite=0/1/2/3 set\n";
	std::cout<<"Data Precision : [" << opt.precision <<"], use export DataPrecision=0/1/2/3 set\n";

        if(prerun_graph_multithread(ONet_graph, opt) < 0)
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

void mtcnn::detect(image img, std::vector<face_box>& face_list)
{
    image working_img = make_image(img.c, img.w, img.h);
    float alpha = 0.0078125;
    float mean = 127.5;
    for(int c = 0; c < img.c; c++)
    {
        for(int i = 0; i < img.h; i++)
        {
            for(int j = 0; j < img.w; j++)
            {
                working_img.data[c * img.h * img.w + i * img.w + j] =
                    (img.data[c * img.h * img.w + i * img.w + j] - mean) * alpha;
            }
        }
    }
    working_img.c = img.c;
    working_img.h = img.h;
    working_img.w = img.w;

    working_img = tranpose(working_img);

    int img_h = working_img.h;
    int img_w = working_img.w;
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
    free_image(working_img);
//    free_image(img);
}

int main(int argc, char** argv)
{
    const std::string root_path = get_root_path();
    std::string img_name = root_path + "./images/mtcnn_face4.jpg";
    std::string model_dir = root_path + "./models/";
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

    float* out_data = ( float* )malloc(face_info.size() * 4 * sizeof(float));
    for(unsigned int i = 0; i < face_info.size(); i++)
    {
        face_box& box = face_info[i];
        std::printf("BOX:( %d , %d ),( %d , %d )\n", (int)box.x0, (int)box.y0, (int)box.x1, (int)box.y1);
        out_data[i*4+0] = box.x0;
        out_data[i*4+1] = box.y0;
        out_data[i*4+2] = box.x1;
        out_data[i*4+3] = box.y1;
        draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 0, 0, 0);
        for(int l = 0; l < 5; l++)
        {
            draw_circle(im, box.landmark.x[l], box.landmark.y[l], 3, 50, 125, 25);
        }
    }

    // test output data
    float* out_data_ref = ( float* )malloc(face_info.size() * 4 * sizeof(float));
    FILE *fp;  
    fp=fopen("./data/mtcnn_face4_out.bin","rb");
    if(fread(out_data_ref, sizeof(float), face_info.size() * 4, fp)==0)
    {
        printf("read ref data file failed!\n");
        return false;
    }
    fclose(fp);
    if(float_mismatch(out_data_ref, out_data, face_info.size() * 4) != true)
        return -1;

    save_image(im, "tengine_example_out");
    free_image(im);
    delete det;

    release_tengine();

    return 0;
}
