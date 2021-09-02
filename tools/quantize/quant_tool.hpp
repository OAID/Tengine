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
 * Author: hhchen@openailab.com
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstring>
#include <algorithm>

extern "C" {
#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "utility/sys_port.h"
#include "utility/utils.h"
#include "utility/log.h"
#include "utility/vector.h"

#include "../source/device/cpu/cpu_node.h"
#include "../source/device/cpu/cpu_graph.h"

#include "convolution_param.h"
#include "fc_param.h"
#include "pooling_param.h"
#include "relu_param.h"
}

#include "quant_utils.hpp"
#include "quant_save_graph.hpp"

typedef std::unordered_map<std::string, int> dict_str2int;
typedef std::unordered_map<std::string, float> dict_str2float;
typedef std::unordered_map<uint32_t, uint32_t> dict_uint2uint;
typedef std::unordered_map<uint32_t, std::vector<uint32_t> > dict_uint2vecuint;
typedef std::unordered_map<uint32_t, std::string> dict_uint2str;
typedef std::unordered_map<uint32_t, std::vector<double> > dict_uint2doublex;

#define ALGORITHM_MIN_MAX 0
#define ALGORITHM_KL      1
#define ALGORITHM_ACIQ    2
#define ALGORITHM_DFQ     3
#define ALGORITHM_MM_EQ   4

struct node_graph
{
    int pass;
    std::vector<uint16_t> input_node_list;
    std::vector<uint16_t> output_node_list;
};

class QuantTool
{
public:
    QuantTool();
    ~QuantTool();

    int init();
    int activation_quant_tool();
    int assess_quant_loss(int gen);
    int quant_search();
    int data_free_quant();

private:
    void recursion_pass_through(struct graph* graphn, const char* layer_name, struct tensor* t,
                                dict_str2int& layer_used, dict_str2float& layer_scale,
                                dict_str2float& layer_zeropoint, dict_str2int& layer_pass);

    struct exec_graph* get_exec_graph(struct graph* graphn);
    void load_activation_scale(struct graph* graphn, const char* scale_file, int mode_sc);
    int prerun_for_get_ir_tensor(void* graph, struct options opt);
    void check_for_free();

    void check_for_interlearve();
    void weight_bias_requant(int search);
    void conv_hcl_interleave_pack4_fp32(int M, int K, float* pA, float* pA_t);
    void activation_requant(float* data, int elem_num, int bitcount, int symmetry, float scale, int zero_point = 0);
    void weight_requant(struct tensor* weight_tensor, float* data, int elem_num, int bitcount, int symmetry, int elem_channel);
    void weight_requant_search(struct tensor* weight_tensor, float* data, int elem_num, int bitcount, int symmetry, int elem_channel, float zoom);
    void weight_requant_search(struct tensor* weight_tensor, float* data, int elem_num, int bitcount, int symmetry, int elem_channel, float* zoom);
    void bias_requant(struct tensor* input_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor,
                      float* data, int elem_num, int elem_channel);
    void set_node_input_output_tensor(int idx, int imgi, int snum);
    double cosin_similarity(std::vector<float>* in_a, std::vector<float>* in_b, uint32_t imgs_num, uint32_t output_num);
    double cosin_similarity(std::vector<std::vector<float> >& in_a, std::vector<std::vector<float> >& in_b, uint32_t imgs_num, uint32_t output_num);
    void cosin_similarity(std::vector<double>& cosin, std::vector<std::vector<float> >& in_a, std::vector<std::vector<float> >& in_b, uint32_t imgs_num, uint32_t output_num, uint32_t output_channel); // cosin dis perchannel
    void weight_bias_reset();
    void free_used_layers(int idx);
    void gen_weight_scale(struct tensor* weight_tensor, float* data, int elem_num, int bitcount, int symmetry, int elem_channel);
    int get_exec_node_message(int exec_node_idx);

    void print_cosin(double* cosin, int idx, int output_channel);

public:
    struct options opt;

    std::string model_file;  // path to input float32 tmfile
    std::string scale_file;  // path to calibration scale file
    std::string output_file; // path to output int8/uint8 tmfile
    std::string image_dir;   // path to calibration images folder

    int num_thread;

    int img_c;
    int img_h;
    int img_w;
    float mean[3];   // value of mean (mean value, default is 104.0,117.0,123.0)
    float scale[3];  // value of normalize (scale value, default is 1.0,1.0,1.0)
    int center_crop; // flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)
    int letterbox_rows;
    int letterbox_cols;
    int sw_RGB;         // flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)
    int focus;          // flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)
    int inplace;        // process the inplace quant scale of activation in some types of op, such as max pooling, ReLU, Flatten, Reshape, Clip
    int algorithm_type; // the type of quant algorithm(0:min-max, 1:kl, default is 0)
    bool evaluate;      // evaluate quantitative losses

private: // system variable
    dict_uint2uint ir_exec;
    dict_uint2uint exec_ir;
    dict_uint2vecuint dict_free;
    dict_uint2uint execidx_elemnum;
    dict_uint2uint execidx_elemsize;
    dict_uint2str execidx_nodename;
    dict_uint2doublex execidx_loss;

    int max_search_img_num;

    std::vector<double> cosin;

private: // basic message
    int img_size;
    double cosin_max;
    float scale_acc;

private: // ir graph variable
    std::vector<std::vector<std::vector<float> > > fp32_out;
    std::vector<std::vector<std::vector<float> > > fake_quant_out;
    std::vector<std::vector<float> > input_datas_fp32;
    std::vector<std::vector<float> > input_datas_fake_quant;
    std::vector<std::vector<float> > out_imgs_fp32;
    std::vector<std::vector<float> > out_imgs_fake_quant;

    struct graph* graphn_fp32;
    struct graph* graphn_fake_quant;
    struct tensor* graph_input_tensor_fp32;
    struct tensor* graph_input_tensor_fake_quant;
    struct exec_graph* exec_graph_fp32;
    struct exec_graph* exec_graph_fake_quant;
    int exec_node_num;

private: // temp variable
    uint16_t op_name;

    struct exec_node* node_fp32;
    struct exec_node* node_fake_quant;
    struct node_ops* node_ops_fp32;
    struct node_ops* node_ops_fake_quant;

    struct tensor* input_tensor_fp32;
    struct tensor* input_tensor_fake_quant;
    struct tensor* weight_tensor_fp32;
    struct tensor* weight_tensor_fake_quant;
    struct tensor* bias_tensor_fp32;
    struct tensor* bias_tensor_fake_quant;
    struct tensor* output_tensor_fp32;
    struct tensor* output_tensor_fake_quant;

    float* weight_data_fp32;
    float* weight_data_fake_quant;
    uint32_t weight_size;
    float* interleave_buffer_fp32;
    float* interleave_buffer_fake_quant;
    uint32_t interleave_size_fake;
    float* bias_data_fp32;
    float* bias_data_fake_quant;
    uint32_t bias_size;
    uint32_t output_channel;

    struct conv_priv_info* conv_priv_info_fp32;
    struct conv_priv_info* conv_priv_info_fake_quant;
    struct conv_param* conv_param_fp32;
    struct conv_param* conv_param_fake_quant;
};
