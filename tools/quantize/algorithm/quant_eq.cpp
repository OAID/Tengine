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

#include "../quant_tool.hpp"

int QuantTool::init()
{
    // ir graph variable
    this->fp32_out.clear();
    this->fake_quant_out.clear();

    /* load fp32 graph and fake quant graph */
    this->graphn_fp32 = (struct graph*)create_graph(nullptr, "tengine", this->model_file.c_str());
    this->graphn_fake_quant = (struct graph*)create_graph(nullptr, "tengine", this->model_file.c_str());

    if (this->graphn_fp32 == nullptr || this->graphn_fake_quant == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    /* load activation scale to ir_tensor */
    this->load_activation_scale(this->graphn_fp32, this->scale_file.c_str(), this->inplace);
    this->load_activation_scale(this->graphn_fake_quant, this->scale_file.c_str(), this->inplace);

    /* get graph input tensor */
    this->graph_input_tensor_fp32 = (struct tensor*)get_graph_input_tensor((void*)this->graphn_fp32, 0, 0);
    this->graph_input_tensor_fake_quant = (struct tensor*)get_graph_input_tensor((void*)this->graphn_fake_quant, 0, 0);
    if (this->graph_input_tensor_fp32 == nullptr || this->graph_input_tensor_fake_quant == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    /* generate images list */
    std::vector<std::string> imgs_list;
    if (!this->image_dir.empty())
        readFileList(this->image_dir, imgs_list);
    uint32_t img_num = imgs_list.size();

    this->max_search_img_num = 50;
    if (img_num < this->max_search_img_num)
        this->max_search_img_num = img_num;

    //    fprintf(stderr, "# eq dataset num %d\n", this->max_search_img_num);

    /* set the shape, data buffer of input_tensor of the graph */
    this->img_size = this->img_h * this->img_w * this->img_c;
    int dims[] = {1, img_c, img_h, img_w}; // nchw
    float* input_data_fp32 = (float*)malloc(this->img_size * sizeof(float));
    float* input_data_fake_quant = (float*)malloc(this->img_size * sizeof(float));

    /* prepare process input data, set the data mem to input tensor */
    float scale_graph_input = this->graph_input_tensor_fake_quant->scale;
    int zero_point_graph_input = this->graph_input_tensor_fake_quant->zero_point;
    //    fprintf(stderr, "scale zp %f %d\n", scale_graph_input, zero_point_graph_input);

    this->input_datas_fp32.resize(this->max_search_img_num);
    this->input_datas_fake_quant.resize(this->max_search_img_num);

    for (int i = 0; i < this->max_search_img_num; i++)
    {
        this->input_datas_fp32[i].resize(this->img_size);
        this->input_datas_fake_quant[i].resize(this->img_size);

        get_input_data_cv(imgs_list[i].c_str(), this->input_datas_fp32[i].data(), img_c, img_h, img_w, mean, scale, sw_RGB, center_crop, letterbox_rows, letterbox_cols, focus);

        this->input_datas_fake_quant[i] = this->input_datas_fp32[i];
        this->activation_requant(this->input_datas_fake_quant[i].data(), this->img_size, 8, 1, scale_graph_input,
                                 zero_point_graph_input);
    }

    /* set graph input shape */
    int ret_fp32 = set_tensor_shape(this->graph_input_tensor_fp32, dims, 4);
    int ret_fake_quant = set_tensor_shape(this->graph_input_tensor_fake_quant, dims, 4);
    if (ret_fp32 < 0 || ret_fake_quant < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    /* set graph input buffer */
    ret_fp32 = set_tensor_buffer(this->graph_input_tensor_fp32, input_data_fp32, this->img_size * 4);
    ret_fake_quant = set_tensor_buffer(this->graph_input_tensor_fake_quant, input_data_fake_quant, this->img_size * 4);
    if (ret_fp32 < 0 || ret_fake_quant < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread((void*)this->graphn_fp32, this->opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }
    ret_fp32 = prerun_graph_multithread((void*)this->graphn_fp32, this->opt);
    ret_fake_quant = prerun_graph_multithread((void*)this->graphn_fake_quant, this->opt);
    if (ret_fp32 < 0 || ret_fake_quant < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* get exec graph */
    this->exec_graph_fp32 = this->get_exec_graph(this->graphn_fp32);
    this->exec_graph_fake_quant = this->get_exec_graph(this->graphn_fake_quant);
    this->exec_node_num = get_vector_num(this->exec_graph_fp32->exec_node_list);

    /* ir idx <<<->>> exec idx */
    for (int i = 0; i < this->exec_node_num; i++)
    {
        this->node_fp32 = (struct exec_node*)get_vector_data(this->exec_graph_fp32->exec_node_list, i);
        this->node_fake_quant = (struct exec_node*)get_vector_data(this->exec_graph_fake_quant->exec_node_list, i);

        int out_t = node_fp32->ir_node->output_tensors[0];
        this->ir_exec[graphn_fp32->tensor_list[out_t]->producer] = i; // ir idx --> exec idx
        this->exec_ir[i] = graphn_fp32->tensor_list[out_t]->producer; // exec idx --> ir idx
                                                                      //        printf(" %d : %d\n", graphn_fp32->tensor_list[out_t]->producer, i);
    }

    /* check for free node*/
    this->check_for_free();

    return 0;
}

void QuantTool::activation_requant(float* data, int elem_num, int bitcount, int symmetry, float scale, int zero_point)
{
    //    symmetry = 0;
    float fake_quant_max;
    float fake_quant_min;

    if (symmetry == 1)
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = -fake_quant_max;
    }
    else
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = 0;
    }

    for (int i = 0; i < elem_num; i++)
    {
        data[i] = round(data[i] / scale) + zero_point;
        data[i] = data[i] > fake_quant_max ? fake_quant_max : data[i];
        data[i] = data[i] < fake_quant_min ? fake_quant_min : data[i];
        data[i] = (data[i] - zero_point) * scale;
    }
}

void QuantTool::recursion_pass_through(struct graph* graphn, const char* layer_name, struct tensor* t,
                                       dict_str2int& layer_used, dict_str2float& layer_scale, dict_str2float& layer_zeropoint, dict_str2int& layer_pass)
{
    if (layer_pass[t->name] == 0 && layer_used[t->name] < 2)
    {
        t->scale = layer_scale[layer_name];
        t->zero_point = layer_zeropoint[layer_name];
        layer_scale[t->name] = layer_scale[layer_name];
        layer_zeropoint[t->name] = layer_zeropoint[layer_name];

        uint32_t ir_node_idx = t->producer;
        struct node* t_node = graphn->node_list[ir_node_idx];

        auto op_name = t_node->op.type;
        bool poolTrue = false;
        bool reluTrue = false;
        if (op_name == OP_POOL)
        {
            struct pool_param* pool_param = (struct pool_param*)t_node->op.param_mem;
            if (pool_param->pool_method == 0)
                poolTrue = true;
        }
        else if (op_name == OP_RELU)
        {
            struct relu_param* relu_param = (struct relu_param*)t_node->op.param_mem;
            if (relu_param->negative_slope == 0.f)
                reluTrue = true;
        }
        if (op_name == OP_FLATTEN || op_name == OP_RESHAPE || op_name == OP_SQUEEZE || op_name == OP_CLIP || poolTrue || reluTrue)
        {
            struct tensor* t_in_tensor = graphn->tensor_list[t_node->input_tensors[0]];
            if (layer_scale[t->name] != 0)
            {
                if (t_in_tensor->tensor_type == 1 || t_in_tensor->tensor_type == 3)
                {
                    QuantTool::recursion_pass_through(graphn, t->name, t_in_tensor, layer_used, layer_scale, layer_zeropoint, layer_pass);
                }
            }
        }
        layer_pass[t->name] = 1;
    }
}

struct exec_graph* QuantTool::get_exec_graph(struct graph* graphn)
{
    struct subgraph* subgraph = get_ir_graph_subgraph(graphn, 0);
    struct exec_graph* exec_graph = (struct exec_graph*)subgraph->device_graph;

    return exec_graph;
}

void QuantTool::check_for_free()
{
    dict_uint2uint nodeA2B;
    for (int i = 0; i < this->exec_node_num; i++)
    {
        this->node_fp32 = (struct exec_node*)get_vector_data(this->exec_graph_fp32->exec_node_list, i);
        this->op_name = this->node_fp32->ir_node->op.type;

        for (int j = 0; j < this->node_fp32->ir_node->input_num; j++)
        {
            struct tensor* t = graphn_fp32->tensor_list[node_fp32->ir_node->input_tensors[j]];
            if (t->tensor_type == 1)
            {
                uint32_t ir_idx = t->producer;
                nodeA2B[this->ir_exec[ir_idx]] = i;
            }
        }
    }

    for (auto iter = nodeA2B.begin(); iter != nodeA2B.end(); iter++)
    {
        this->dict_free[iter->second].push_back(iter->first);
        //        printf(" map %d %d\n", iter->first, iter->second);
    }
}

void QuantTool::check_for_interlearve()
{
    if (this->op_name == OP_CONV || this->op_name == OP_FC)
    {
        /* get weight tensor */
        this->weight_tensor_fp32 = this->graphn_fp32->tensor_list[this->node_fp32->ir_node->input_tensors[1]];
        this->weight_tensor_fake_quant = this->graphn_fake_quant->tensor_list[this->node_fake_quant->ir_node->input_tensors[1]];
        this->weight_size = this->weight_tensor_fp32->elem_num * this->weight_tensor_fp32->elem_size;

        this->weight_data_fp32 = (float*)this->weight_tensor_fp32->data;
        this->weight_data_fake_quant = (float*)this->weight_tensor_fake_quant->data;

        if (this->op_name == OP_CONV)
        {
            this->conv_param_fp32 = (struct conv_param*)this->node_fp32->ir_node->op.param_mem;
            this->conv_param_fake_quant = (struct conv_param*)this->node_fake_quant->ir_node->op.param_mem;

            if (this->conv_param_fp32->group != this->conv_param_fp32->output_channel)
            {
                this->conv_priv_info_fp32 = (struct conv_priv_info*)this->node_fp32->ops_priv;
                this->conv_priv_info_fake_quant = (struct conv_priv_info*)this->node_fake_quant->ops_priv;

                this->interleave_size_fake = this->conv_priv_info_fp32->interleave_buffer_pack4_size;

                this->interleave_buffer_fp32 = (float*)this->conv_priv_info_fp32->interleave_buffer_pack4;
                this->interleave_buffer_fake_quant = (float*)this->conv_priv_info_fake_quant->interleave_buffer_pack4;
            }
        }
        else
            this->interleave_size_fake = 0;
    }
}

void QuantTool::weight_bias_requant(int search)
{
    /* weight requant */
    //    printf("### 1.1 this->weight_tensor_fake_quant->scale %f\n",this->weight_tensor_fake_quant->scale);
    if (0 == search)
        this->weight_requant(this->weight_tensor_fake_quant, this->weight_data_fake_quant, this->weight_tensor_fake_quant->elem_num, 8, 1, this->weight_tensor_fake_quant->dims[0]);

    if (this->interleave_size_fake != 0)
    {
        int M = this->weight_tensor_fake_quant->dims[0];
        int K = this->weight_tensor_fake_quant->elem_num / weight_tensor_fake_quant->dims[0];
        this->conv_hcl_interleave_pack4_fp32(M, K, this->weight_data_fake_quant, this->interleave_buffer_fake_quant);
    }

    /* bias requant */
    if (this->node_fake_quant->ir_node->input_num > 2)
    {
        this->input_tensor_fake_quant = this->graphn_fake_quant->tensor_list[this->node_fake_quant->ir_node->input_tensors[0]];
        this->bias_tensor_fake_quant = this->graphn_fake_quant->tensor_list[this->node_fake_quant->ir_node->input_tensors[2]];
        this->bias_tensor_fp32 = this->graphn_fp32->tensor_list[this->node_fp32->ir_node->input_tensors[2]];
        this->bias_size = this->bias_tensor_fp32->elem_num * this->bias_tensor_fp32->elem_size;
        this->bias_data_fp32 = (float*)this->bias_tensor_fp32->data;
        this->bias_data_fake_quant = (float*)this->bias_tensor_fake_quant->data;
        this->bias_requant(this->input_tensor_fake_quant, this->weight_tensor_fake_quant, this->bias_tensor_fake_quant,
                           this->bias_data_fake_quant, this->bias_tensor_fake_quant->elem_num, this->bias_tensor_fake_quant->dims[0]);
        //        this->bias_tensor_fp32->scale = this->bias_tensor_fake_quant->scale;
    }
}

void QuantTool::set_node_input_output_tensor(int idx, int imgi, int snum)
{
    this->out_imgs_fp32[imgi].resize(this->output_tensor_fp32->elem_num);
    this->out_imgs_fake_quant[imgi].resize(this->output_tensor_fp32->elem_num);

    if (idx == 0)
    {
        set_tensor_buffer(this->graph_input_tensor_fp32, this->input_datas_fp32[imgi].data(), this->img_size * 4);
        set_tensor_buffer(this->graph_input_tensor_fake_quant, this->input_datas_fake_quant[imgi].data(), this->img_size * 4);
    }
    else
    {
        for (int inputi = 0; inputi < this->node_fp32->ir_node->input_num; inputi++)
        {
            uint32_t ir_input_tensor_idx = this->node_fp32->ir_node->input_tensors[inputi];
            this->input_tensor_fp32 = this->graphn_fp32->tensor_list[ir_input_tensor_idx];
            this->input_tensor_fake_quant = this->graphn_fake_quant->tensor_list[ir_input_tensor_idx];

            if (this->input_tensor_fp32->tensor_type == 1)
            {
                uint32_t ir_node_idx = this->input_tensor_fp32->producer;
                uint32_t input_size = this->input_tensor_fp32->elem_num * input_tensor_fp32->elem_size;

                uint32_t exec_node_idx = this->ir_exec[ir_node_idx];

                if (imgi == 0 && snum == 0)
                {
                    float* buf_fp32 = (float*)sys_malloc(32);
                    float* buf_fake_quant = (float*)sys_malloc(32);

                    set_tensor_buffer(this->input_tensor_fp32, buf_fp32, input_size);
                    set_tensor_buffer(this->input_tensor_fake_quant, buf_fake_quant, input_size);

                    set_tensor_buffer(this->input_tensor_fp32, this->fp32_out[exec_node_idx][imgi].data(), input_size);
                    set_tensor_buffer(this->input_tensor_fake_quant, this->fake_quant_out[exec_node_idx][imgi].data(), input_size);
                }
                else
                {
                    set_tensor_buffer(this->input_tensor_fp32, this->fp32_out[exec_node_idx][imgi].data(), input_size);
                    set_tensor_buffer(this->input_tensor_fake_quant, this->fake_quant_out[exec_node_idx][imgi].data(), input_size);
                }
            } // output tensor
        }     // node input number
    }         //  node i > 0

    /* init output buffer */
    set_tensor_buffer(this->output_tensor_fp32, this->out_imgs_fp32[imgi].data(), this->output_tensor_fp32->elem_num * this->output_tensor_fp32->elem_size);
    set_tensor_buffer(this->output_tensor_fake_quant, this->out_imgs_fake_quant[imgi].data(), this->output_tensor_fake_quant->elem_num * this->output_tensor_fake_quant->elem_size);
}

double QuantTool::cosin_similarity(std::vector<std::vector<float> >& in_a, std::vector<std::vector<float> >& in_b, uint32_t imgs_num, uint32_t output_num)
{
    double norm_a = 0;
    double norm_b = 0;
    double a_b = 0;

    uint32_t fnum = (output_num >> 4) << 4;
    uint32_t rnum = output_num - fnum;

#if 0 //__AVX__

    float _sumaa0[8] = {0.f};
    float _sumbb0[8] = {0.f};
    float _sumaabb0[8] = {0.f};
    float _sumaa1[8] = {0.f};
    float _sumbb1[8] = {0.f};
    float _sumaabb1[8] = {0.f};

    __m256 _suma_o0 = _mm256_set1_ps(0.0);
    __m256 _sumb_o0 = _mm256_set1_ps(0.0);
    __m256 _sumab_o0 = _mm256_set1_ps(0.0);
    __m256 _suma_o1 = _mm256_set1_ps(0.0);
    __m256 _sumb_o1 = _mm256_set1_ps(0.0);
    __m256 _sumab_o1 = _mm256_set1_ps(0.0);

    for (int i = 0; i < imgs_num; i++)
    {
        const float* in_a_addr = in_a[i].data();
        const float* in_b_addr = in_b[i].data();
        for (int j = 0; j < fnum; j=j+32)
        {
            __m256 _in_a0 = _mm256_loadu_ps(in_a_addr+j);
            __m256 _in_b0 = _mm256_loadu_ps(in_b_addr+j);
            __m256 _in_a1 = _mm256_loadu_ps(in_a_addr+j+8);
            __m256 _in_b1 = _mm256_loadu_ps(in_b_addr+j+8);

            _suma_o0 = _mm256_fmadd_ps(_in_a0, _in_a0, _suma_o0);
            _sumb_o0 = _mm256_fmadd_ps(_in_b0, _in_b0, _sumb_o0);
            _sumab_o0 = _mm256_fmadd_ps(_in_a0, _in_b0, _sumab_o0);
            _suma_o1 = _mm256_fmadd_ps(_in_a1, _in_a1, _suma_o1);
            _sumb_o1 = _mm256_fmadd_ps(_in_b1, _in_b1, _sumb_o1);
            _sumab_o1 = _mm256_fmadd_ps(_in_a1, _in_b1, _sumab_o1);
        }
    }
    _mm256_storeu_ps(_sumaa0, _suma_o0);
    _mm256_storeu_ps(_sumbb0, _sumb_o0);
    _mm256_storeu_ps(_sumaabb0, _sumab_o0);
    _mm256_storeu_ps(_sumaa1, _suma_o1);
    _mm256_storeu_ps(_sumbb1, _sumb_o1);
    _mm256_storeu_ps(_sumaabb1, _sumab_o1);

    for (int i = 0; i < 8; i++)
    {
        norm_a += _sumaa0[i] + _sumaa1[i];
        norm_b += _sumbb0[i] + _sumbb1[i];
        a_b += _sumaabb0[i] + _sumaabb1[i];

    }

#else // normal
    //    printf("AAAA DIRECT\n");
    for (int i = 0; i < imgs_num; i++)
    {
        for (int j = 0; j < fnum; j = j + 8)
        {
            for (int k = 0; k < 8; k = k + 1)
            {
                norm_a += in_a[i][j + k] * in_a[i][j + k];

                norm_b += in_b[i][j + k] * in_b[i][j + k];

                a_b += in_a[i][j + k] * in_b[i][j + k];
            }
        }
    }

#endif // __SSE__ __AVX__

    for (int j = fnum; j < output_num; j++)
    {
        for (int i = 0; i < imgs_num; i++)
        {
            norm_a += in_a[i][j] * in_a[i][j];
            norm_b += in_b[i][j] * in_b[i][j];
            a_b += in_a[i][j] * in_b[i][j];
        }
    }

    double cosin = 0.0;
    double _a_b_ = sqrt(norm_a) * sqrt(norm_b);
    if (_a_b_ < 0.0000001f && _a_b_ > -0.0000001f)
        cosin = a_b;
    else
        cosin = a_b / _a_b_;
    if (cosin < -999999 || cosin > 999999)
        cosin = 0;
    return cosin;
}

double QuantTool::cosin_similarity(std::vector<float>* in_a, std::vector<float>* in_b, uint32_t imgs_num, uint32_t output_num)
{
    uint32_t output_channel = 1;
    std::vector<double> norm_a(output_channel, 0.0);
    std::vector<double> norm_b(output_channel, 0.0);
    std::vector<double> a_b(output_channel, 0.0);

    int elem_perchannel = int(output_num / output_channel);

    for (int i = 0; i < imgs_num; i++)
    {
        for (int j = 0; j < output_channel; j++)
        {
            for (int k = 0; k < elem_perchannel; k++)
            {
                int elem_idx = j * elem_perchannel + k;
                norm_a[j] += in_a[i][elem_idx] * in_a[i][elem_idx];
                norm_b[j] += in_b[i][elem_idx] * in_b[i][elem_idx];
                a_b[j] += in_a[i][elem_idx] * in_b[i][elem_idx];
            }
        }
    }

    double cosin;
    for (int j = 0; j < output_channel; j++)
    {
        double _a_b_ = sqrt(norm_a[j]) * sqrt(norm_b[j]);
        if (_a_b_ < 0.0000001f && _a_b_ > -0.0000001f)
            cosin = a_b[j];
        else
            cosin = a_b[j] / _a_b_;
        if (cosin < -999999 || cosin > 999999)
            cosin = 0;
    }
    return cosin;
}

void QuantTool::weight_requant(struct tensor* weight_tensor, float* data, int elem_num, int bitcount, int symmetry, int elem_channel)
{
    float* scale_list = (float*)sys_malloc(elem_channel * 4);
    int* zero_point_list = (int*)sys_malloc(elem_channel * 4);

    int elem_perchannel = elem_num / elem_channel;

    float fake_quant_max;
    float fake_quant_min;

    if (symmetry == 1)
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = -fake_quant_max;
    }
    else
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = 0;
    }

    float scale = 1;
    int zero_point = 0;
    for (int c = 0; c < elem_channel; c++)
    {
        float weight_max = *std::max_element(data + c * elem_perchannel, data + (c + 1) * elem_perchannel);
        float weight_min = *std::min_element(data + c * elem_perchannel, data + (c + 1) * elem_perchannel);
        if (symmetry == 1)
        {
            if (std::abs(weight_max) > std::abs(weight_min))
                scale = std::abs(weight_max) / fake_quant_max;
            else
                scale = std::abs(weight_min) / fake_quant_max;
            zero_point = 0;
        }
        else
        {
            scale = (weight_max - weight_min) / fake_quant_max;
            zero_point = int(-weight_min / scale);
        }

        scale_list[c] = scale;
        zero_point_list[c] = zero_point;
    }

    if (weight_tensor->scale_list == NULL)
    {
        //        printf(" EMPTY\n ");
        weight_tensor->scale_list = scale_list;
        weight_tensor->zp_list = zero_point_list;
    }
    else
    {
        scale_list = weight_tensor->scale_list;
        zero_point_list = weight_tensor->zp_list;
    }

    int data_idx;
    for (int i = 0; i < elem_channel; i++)
    {
        for (int j = 0; j < elem_perchannel; j++)
        {
            data_idx = i * elem_perchannel + j;
            if (scale_list[i] == 0)
                data[data_idx] = 0;
            else
            {
                data[data_idx] = round(data[data_idx] / scale_list[i]) + zero_point_list[i];
                data[data_idx] = data[data_idx] > fake_quant_max ? fake_quant_max : data[data_idx];
                data[data_idx] = data[data_idx] < fake_quant_min ? fake_quant_min : data[data_idx];
                data[data_idx] = (data[data_idx] - zero_point_list[i]) * scale_list[i];
            }
        }
    }
}

void QuantTool::conv_hcl_interleave_pack4_fp32(int M, int K, float* pA, float* pA_t)
{
    int nn_outch = M >> 3;
    int remain_outch_start = nn_outch << 3;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        const float* k0 = pA + (p + 0) * K;
        const float* k1 = pA + (p + 1) * K;
        const float* k2 = pA + (p + 2) * K;
        const float* k3 = pA + (p + 3) * K;
        const float* k4 = pA + (p + 4) * K;
        const float* k5 = pA + (p + 5) * K;
        const float* k6 = pA + (p + 6) * K;
        const float* k7 = pA + (p + 7) * K;

        float* ktmp = pA_t + (p / 8) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp[4] = k4[0];
            ktmp[5] = k5[0];
            ktmp[6] = k6[0];
            ktmp[7] = k7[0];
            ktmp += 8;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
            k4 += 1;
            k5 += 1;
            k6 += 1;
            k7 += 1;
        }
    }

    nn_outch = (M - remain_outch_start) >> 2;
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        const float* k0 = pA + (p + 0) * K;
        const float* k1 = pA + (p + 1) * K;
        const float* k2 = pA + (p + 2) * K;
        const float* k3 = pA + (p + 3) * K;

        float* ktmp = pA_t + (p / 8 + (p % 8) / 4) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp += 4;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
        }
    }

    remain_outch_start += nn_outch << 2;

    for (int p = remain_outch_start; p < M; p++)
    {
        const float* k0 = pA + (p + 0) * K;

        float* ktmp = pA_t + (p / 8 + (p % 8) / 4 + p % 4) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp++;
            k0++;
        }
    }
}

void QuantTool::gen_weight_scale(struct tensor* weight_tensor, float* data, int elem_num, int bitcount, int symmetry, int elem_channel)
{
    float* scale_list = (float*)sys_malloc(elem_channel * 4);
    int* zero_point_list = (int*)sys_malloc(elem_channel * 4);

    int elem_perchannel = elem_num / elem_channel;

    float fake_quant_max;
    float fake_quant_min;

    if (symmetry == 1)
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = -fake_quant_max;
    }
    else
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = 0;
    }

    float scale = 1;
    int zero_point = 0;
    for (int c = 0; c < elem_channel; c++)
    {
        float weight_max = *std::max_element(data + c * elem_perchannel, data + (c + 1) * elem_perchannel);
        float weight_min = *std::min_element(data + c * elem_perchannel, data + (c + 1) * elem_perchannel);
        if (symmetry == 1)
        {
            if (std::abs(weight_max) > std::abs(weight_min))
                scale = std::abs(weight_max) / fake_quant_max;
            else
                scale = std::abs(weight_min) / fake_quant_max;
            zero_point = 0;
        }
        else
        {
            scale = (weight_max - weight_min) / fake_quant_max;
            zero_point = int(-weight_min / scale);
        }

        scale_list[c] = scale;
        zero_point_list[c] = zero_point;
    }

    weight_tensor->scale_list = scale_list;
    weight_tensor->zp_list = zero_point_list;
}

void QuantTool::bias_requant(struct tensor* input_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor,
                             float* data, int elem_num, int elem_channel)
{
    int elem_perchannel = elem_num / elem_channel;
    float* scale_list = (float*)sys_malloc(elem_channel * 4);

    for (int c = 0; c < elem_channel; c++)
    {
        float input_scale = input_tensor->scale;
        float weight_scale = weight_tensor->scale_list[c];
        float bias_scale = input_scale * weight_scale;
        scale_list[c] = bias_scale;
    }

    bias_tensor->scale_list = scale_list;

    int data_idx;
    for (int i = 0; i < elem_channel; i++)
    {
        for (int j = 0; j < elem_perchannel; j++)
        {
            data_idx = i * elem_perchannel + j;
            if (scale_list[i] == 0)
            {
                data[data_idx] = 0;
            }
            else
            {
                data[data_idx] = round(data[data_idx] / scale_list[i]);
                data[data_idx] = data[data_idx] * scale_list[i];
            }
        }
    }
}

void QuantTool::weight_bias_reset()
{
    if (this->op_name == OP_CONV || this->op_name == OP_FC)
    {
        std::memcpy(this->weight_data_fake_quant, this->weight_data_fp32, this->weight_size);
        std::memcpy(this->interleave_buffer_fake_quant, this->interleave_buffer_fp32, this->interleave_size_fake);
        if (this->node_fake_quant->ir_node->input_num > 2)
        {
            memcpy(this->bias_data_fake_quant, this->bias_data_fp32, this->bias_size);
        }
    }
}

void QuantTool::free_used_layers(int idx)
{
    //    printf("#### free 0 idx %d\n",idx);
    if (this->dict_free[idx].size() > 0)
    {
        //        printf("#### free 1 idx %d\n",idx);
        std::vector<std::vector<float> > freen_fp32;
        std::vector<std::vector<float> > freen_fake_quant;
        for (int fi = 0; fi < this->dict_free[idx].size(); fi++)
        {
            if (this->dict_free[idx][fi] != 0)
            {
                //                printf("---free---\n");
                this->fp32_out[this->dict_free[idx][fi]].clear();
                this->fake_quant_out[this->dict_free[idx][fi]].clear();
            }
        }
    }
}

void QuantTool::load_activation_scale(struct graph* graphn, const char* scale_file, int mode_sc)
{
    std::unordered_map<std::string, float> layer_scale;
    std::unordered_map<std::string, float> layer_zeropoint;
    bool parse_from_file = false;
    if (nullptr != scale_file)
    {
        std::ifstream scales(scale_file);
        std::string line;
        while (std::getline(scales, line))
        {
            std::string layer_name;
            float scale_val = 0.f;
            float zero_point = 0.f;
            size_t last = 0;
            size_t index = line.find_first_of(" ", last);
            size_t idx = line.find_last_of(" ", line.size());
            layer_name = line.substr(last, index - last);
            //            printf("layer_name : %s \n", layer_name.c_str());
            last = index + 1;
            scale_val = atof((line.substr(last, line.size() - last)).c_str());
            zero_point = atof((line.substr(idx + 1, line.size())).c_str());

            layer_scale[layer_name] = scale_val;
            layer_zeropoint[layer_name] = zero_point;
            //            fprintf(stderr, "quant value : %s %f %f \n", layer_name.c_str(), scale_val, zero_point);
        }
    }

    std::unordered_map<std::string, int> layer_used;
    for (int i = 0; i < graphn->node_num; i++)
    {
        struct node* noden = graphn->node_list[i];
        for (int j = 0; j < noden->input_num; j++)
        {
            std::string layern = graphn->tensor_list[noden->input_tensors[j]]->name;
            layer_used[layern]++;
        }
    }

    if (mode_sc == 0)
    {
        for (int i = 0; i < graphn->tensor_num; i++)
        {
            struct tensor* t = graphn->tensor_list[i];
            if (t->tensor_type == 1 || t->tensor_type == 3)
            {
                t->scale = layer_scale[t->name];
                t->zero_point = layer_zeropoint[t->name];
            }
        }
    }
    else
    {
        std::unordered_map<std::string, int> layer_pass;
        for (int i = graphn->tensor_num - 1; i >= 0; i--)
        {
            struct tensor* t = graphn->tensor_list[i];
            if (t->tensor_type == 1 || t->tensor_type == 3)
            {
                if (layer_pass[t->name] == 0)
                {
                    uint32_t ir_node_idx = t->producer;
                    struct node* t_node = graphn->node_list[ir_node_idx];

                    auto op_name = t_node->op.type;

                    bool poolTrue = false;
                    bool reluTrue = false;
                    if (op_name == OP_POOL)
                    {
                        struct pool_param* pool_param = (struct pool_param*)t_node->op.param_mem;
                        if (pool_param->pool_method == 0)
                            poolTrue = true;
                    }
                    else if (op_name == OP_RELU)
                    {
                        struct relu_param* relu_param = (struct relu_param*)t_node->op.param_mem;
                        if (relu_param->negative_slope == 0.f)
                            reluTrue = true;
                    }

                    if (op_name == OP_FLATTEN || op_name == OP_RESHAPE || op_name == OP_SQUEEZE || op_name == OP_CLIP || poolTrue || reluTrue)
                    {
                        struct tensor* t_in_tensor = graphn->tensor_list[t_node->input_tensors[0]];
                        if (layer_scale[t->name] != 0)
                        {
                            t->scale = layer_scale[t->name];
                            t->zero_point = layer_zeropoint[t->name];

                            if (t_in_tensor->tensor_type == 1 || t_in_tensor->tensor_type == 3)
                            {
                                this->recursion_pass_through(graphn, t->name, t_in_tensor, layer_used, layer_scale,
                                                             layer_zeropoint, layer_pass);
                            }
                        }
                    }
                    else
                    {
                        t->scale = layer_scale[t->name];
                        t->zero_point = layer_zeropoint[t->name];
                    }
                    layer_pass[t->name] = 1;
                }
            }
        }
    }

    //    for (int i = 0; i < graphn->tensor_num; i++)
    //    {
    //        struct ir_tensor* t = graphn->tensor_list[i];
    //        if (t->tensor_type == 1 || t->tensor_type == 3)
    //        {
    //            printf(" sz %s %f %d \n",t->name, t->scale, t->zero_point);
    //        }
    //    }
}

int QuantTool::get_exec_node_message(int exec_node_idx)
{
    /* get node */
    this->node_fp32 = (struct exec_node*)get_vector_data(this->exec_graph_fp32->exec_node_list, exec_node_idx);
    this->node_fake_quant = (struct exec_node*)get_vector_data(this->exec_graph_fake_quant->exec_node_list, exec_node_idx);

    /* get op type */
    this->op_name = this->node_fp32->ir_node->op.type;

    /* get exec ops */
    this->node_ops_fp32 = this->node_fp32->node_ops;
    this->node_ops_fake_quant = this->node_fake_quant->node_ops;

    /* handle the shape changed  and dynamic shape case */
    if (this->node_ops_fp32->reshape && this->node_ops_fp32->reshape(this->node_ops_fp32, this->node_fp32, this->exec_graph_fp32)
        && this->node_ops_fake_quant->reshape && this->node_ops_fake_quant->reshape(this->node_ops_fake_quant, this->node_fake_quant, this->exec_graph_fake_quant) < 0)
    {
        TLOG_ERR("failed to reshape node %d, %s\n", node_fp32->ir_node->index, node_fp32->ir_node->name);
        return -1;
    }

    /* get output tensor */
    this->output_tensor_fp32 = this->graphn_fp32->tensor_list[this->node_fp32->ir_node->output_tensors[0]];
    this->output_tensor_fake_quant = this->graphn_fake_quant->tensor_list[this->node_fake_quant->ir_node->output_tensors[0]];

    /* get exec ops */
    this->execidx_elemnum[exec_node_idx] = this->output_tensor_fp32->elem_num;   //exec idx --> output elem num
    this->execidx_elemsize[exec_node_idx] = this->output_tensor_fp32->elem_size; //exec idx --> output elem size
    this->execidx_nodename[exec_node_idx] = this->output_tensor_fp32->name;      //exec idx --> output tensor name

    return 0;
}

void QuantTool::cosin_similarity(std::vector<double>& cosin, std::vector<std::vector<float> >& in_a, std::vector<std::vector<float> >& in_b, uint32_t imgs_num, uint32_t output_num, uint32_t output_channel) // cosin dis perchannel
{
    //    fprintf(stderr, " in_a %f ",in_a[0][0]);
    //    fprintf(stderr, " in_b %f ",in_b[0][0]);

    std::vector<double> norm_a(output_channel, 0.0);
    std::vector<double> norm_b(output_channel, 0.0);
    std::vector<double> a_b(output_channel, 0.0);

    int elem_perchannel = int(output_num / output_channel);

    for (int i = 0; i < imgs_num; i++)
    {
        for (int j = 0; j < output_channel; j++)
        {
            for (int k = 0; k < elem_perchannel; k++)
            {
                int elem_idx = j * elem_perchannel + k;
                norm_a[j] += in_a[i][elem_idx] * in_a[i][elem_idx];
                norm_b[j] += in_b[i][elem_idx] * in_b[i][elem_idx];
                a_b[j] += in_a[i][elem_idx] * in_b[i][elem_idx];
            }
        }
    }

    cosin.resize(output_channel);
    for (int j = 0; j < output_channel; j++)
    {
        double _a_b_ = sqrt(norm_a[j]) * sqrt(norm_b[j]);
        //        fprintf(stderr, " %lf %f %f \n ", _a_b_, sqrt(norm_a[j]), sqrt(norm_b[j]) );
        if (_a_b_ < 0.0000001f && _a_b_ > -0.0000001f)
            cosin[j] = a_b[j];
        else
            cosin[j] = a_b[j] / _a_b_;
        if (cosin[j] < -999999 || cosin[j] > 999999)
            cosin[j] = 0;
    }
}

int QuantTool::assess_quant_loss(int gen)
{
    this->init();
    for (int i = 0; i < this->exec_node_num; i++)
    {
        this->get_exec_node_message(i);
        this->check_for_interlearve();

        this->out_imgs_fp32.resize(this->max_search_img_num);
        this->out_imgs_fake_quant.resize(this->max_search_img_num);
        if (this->op_name == OP_CONV || this->op_name == OP_FC)
            this->weight_bias_requant(gen);

        for (int imgi = 0; imgi < this->max_search_img_num; imgi++)
        {
            this->set_node_input_output_tensor(i, imgi, 0);

            /* op run */
            this->node_ops_fp32->run(this->node_ops_fp32, this->node_fp32, this->exec_graph_fp32);
            this->node_ops_fake_quant->run(this->node_ops_fake_quant, this->node_fake_quant, this->exec_graph_fake_quant);
            this->activation_requant(this->out_imgs_fake_quant[imgi].data(), this->output_tensor_fake_quant->elem_num, 8, 1, this->output_tensor_fake_quant->scale, this->output_tensor_fake_quant->zero_point);
        }

        if (this->op_name == OP_CONV || (this->op_name == OP_FC && this->max_search_img_num > 1))
            this->cosin_similarity(this->cosin, this->out_imgs_fp32, this->out_imgs_fake_quant, this->max_search_img_num, this->execidx_elemnum[i], this->weight_tensor_fp32->dims[0]);
        else
            this->cosin_similarity(this->cosin, this->out_imgs_fp32, this->out_imgs_fake_quant, this->max_search_img_num, this->execidx_elemnum[i], 1);

        if (this->op_name == OP_CONV || (this->op_name == OP_FC && this->max_search_img_num > 1))
            this->print_cosin(this->cosin.data(), i, this->weight_tensor_fp32->dims[0]);
        else
            this->print_cosin(this->cosin.data(), i, 1);
        //        fprintf(stderr, "cosin [%s] : %f\n", execidx_nodename[i].c_str(), cosin);

        this->weight_bias_reset();
        this->free_used_layers(i);

        /* save node output */
        this->fp32_out.push_back(this->out_imgs_fp32);
        this->fake_quant_out.push_back(this->out_imgs_fake_quant);
    }

    return 0;
}

void QuantTool::print_cosin(double* cosin, int idx, int output_channel)
{
    float avg_cosin = 0;
    float avg_num = 0;
    for (int c = 0; c < output_channel; c++)
    {
        if (cosin[c] != 0)
        {
            avg_cosin += cosin[c];
            avg_num++;
        }
    }
    fprintf(stderr, "cosin %3d  %4d  avg  %0.6f  ### ", idx, output_channel, avg_cosin / avg_num);
    for (int c = 0; c < output_channel; c++)
    {
        fprintf(stderr, "%0.6f ", cosin[c]);
    }
    fprintf(stderr, "\n");
}

void QuantTool::weight_requant_search(struct tensor* weight_tensor, float* data, int elem_num, int bitcount, int symmetry, int elem_channel, float zoom)
{
    float* scale_list = (float*)weight_tensor->scale_list;
    int* zero_point_list = (int*)weight_tensor->zp_list;

    int elem_perchannel = elem_num / elem_channel;

    float fake_quant_max;
    float fake_quant_min;

    if (symmetry == 1)
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = -fake_quant_max;
    }
    else
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = 0;
    }

    int data_idx;
    for (int i = 0; i < elem_channel; i++)
    {
        float scale = scale_list[i] * zoom;
        for (int j = 0; j < elem_perchannel; j++)
        {
            data_idx = i * elem_perchannel + j;
            if (scale_list[i] == 0)
                data[data_idx] = 0;
            else
            {
                data[data_idx] = round(data[data_idx] / scale) + zero_point_list[i];
                data[data_idx] = data[data_idx] > fake_quant_max ? fake_quant_max : data[data_idx];
                data[data_idx] = data[data_idx] < fake_quant_min ? fake_quant_min : data[data_idx];
                data[data_idx] = (data[data_idx] - zero_point_list[i]) * scale;
            }
        }
    }
}
void QuantTool::weight_requant_search(struct tensor* weight_tensor, float* data, int elem_num, int bitcount, int symmetry, int elem_channel, float* zoom)
{
    float* scale_list = (float*)weight_tensor->scale_list;
    int* zero_point_list = (int*)weight_tensor->zp_list;

    int elem_perchannel = elem_num / elem_channel;

    float fake_quant_max;
    float fake_quant_min;

    if (symmetry == 1)
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = -fake_quant_max;
    }
    else
    {
        fake_quant_max = pow(2, bitcount - symmetry) - 1;
        fake_quant_min = 0;
    }

    int data_idx;
    for (int i = 0; i < elem_channel; i++)
    {
        float scale = 1;
        if (zoom[i] > 5)
            scale = scale_list[i];
        else
            scale = scale_list[i] * zoom[i];
        for (int j = 0; j < elem_perchannel; j++)
        {
            data_idx = i * elem_perchannel + j;
            if (scale_list[i] == 0)
                data[data_idx] = 0;
            else
            {
                data[data_idx] = round(data[data_idx] / scale) + zero_point_list[i];
                data[data_idx] = data[data_idx] > fake_quant_max ? fake_quant_max : data[data_idx];
                data[data_idx] = data[data_idx] < fake_quant_min ? fake_quant_min : data[data_idx];
                data[data_idx] = (data[data_idx] - zero_point_list[i]) * scale;
            }
        }
    }
}

int QuantTool::quant_search()
{
    this->init();
    for (int i = 0; i < this->exec_node_num; i++)
    {
        this->get_exec_node_message(i);
        this->check_for_interlearve();

        this->out_imgs_fp32.resize(this->max_search_img_num);
        this->out_imgs_fake_quant.resize(this->max_search_img_num);

        if (this->op_name == OP_CONV || this->op_name == OP_FC)
        {
            this->gen_weight_scale(this->weight_tensor_fake_quant, this->weight_data_fake_quant, this->weight_tensor_fake_quant->elem_num, 8, 1, weight_tensor_fake_quant->dims[0]);
            this->gen_weight_scale(this->weight_tensor_fp32, this->weight_data_fp32, this->weight_tensor_fp32->elem_num, 8, 1, weight_tensor_fp32->dims[0]);

            std::vector<double> cosin_save(weight_tensor_fake_quant->dims[0], -1);
            std::vector<float> zoom_save(weight_tensor_fake_quant->dims[0], -1);
            for (int snum = 0; snum < 201; snum = snum + 20)
            {
                float zoom = 1.3 / 200 * (snum + 1);
                //                float zoom = 1.0;
                /* weight requant */
                if (snum < 200)
                    this->weight_requant_search(weight_tensor_fake_quant, weight_data_fake_quant, weight_tensor_fake_quant->elem_num, 8, 1, weight_tensor_fake_quant->dims[0], zoom);
                else
                {
                    this->weight_requant_search(weight_tensor_fake_quant, weight_data_fake_quant, weight_tensor_fake_quant->elem_num, 8, 1, weight_tensor_fake_quant->dims[0], zoom_save.data());
                    float* buf = (float*)sys_malloc(weight_tensor_fake_quant->dims[0] * 4);
                    memcpy(buf, zoom_save.data(), weight_tensor_fake_quant->dims[0] * 4);
                    //                    printf(" scale3 %f \n",weight_tensor_fp32->scale_list[0]);
                    for (int bi = 0; bi < weight_tensor_fake_quant->dims[0]; bi++)
                    {
                        buf[bi] *= weight_tensor_fp32->scale_list[bi];
                    }
                    //                    printf(" scale4 %f \n",buf[0]);
                    //                     weight_tensor_fake_quant->scale_list = buf;
                    weight_tensor_fp32->scale_list = buf;
                    weight_tensor_fp32->quant_param_num = weight_tensor_fp32->dims[0];
                    //                    printf(" scale5 %f \n",weight_tensor_fp32->scale_list[0]);
                }
                if (interleave_size_fake != 0)
                {
                    int M = weight_tensor_fake_quant->dims[0];
                    int K = weight_tensor_fake_quant->elem_num / weight_tensor_fake_quant->dims[0];
                    this->conv_hcl_interleave_pack4_fp32(M, K, weight_data_fake_quant, interleave_buffer_fake_quant);
                }

                /* bias requant */
                if (node_fake_quant->ir_node->input_num > 2)
                {
                    struct tensor* input_tensor_fake_quant = graphn_fake_quant->tensor_list[node_fake_quant->ir_node->input_tensors[0]];
                    struct tensor* bias_tensor_fake_quant = graphn_fake_quant->tensor_list[node_fake_quant->ir_node->input_tensors[2]];
                    struct tensor* bias_tensor_fp32 = graphn_fp32->tensor_list[node_fp32->ir_node->input_tensors[2]];

                    bias_size = bias_tensor_fp32->elem_num * bias_tensor_fp32->elem_size;

                    bias_data_fp32 = (float*)bias_tensor_fp32->data;
                    bias_data_fake_quant = (float*)bias_tensor_fake_quant->data;

                    this->bias_requant(input_tensor_fake_quant, weight_tensor_fake_quant, bias_tensor_fake_quant,
                                       bias_data_fake_quant, bias_tensor_fake_quant->elem_num, bias_tensor_fake_quant->dims[0]);
                }

                /* per image run */
                for (int imgi = 0; imgi < this->max_search_img_num; imgi++)
                {
                    this->set_node_input_output_tensor(i, imgi, snum);

                    /* FP32 op run */
                    if (snum == 0)
                    {
                        //                        set_tensor_buffer(output_tensor_fp32, out_imgs_fp32[imgi].data(), output_tensor_fp32->elem_num * output_tensor_fp32->elem_size);
                        node_ops_fp32->run(node_ops_fp32, node_fp32, exec_graph_fp32);

                        this->execidx_elemnum[i] = output_tensor_fp32->elem_num;   //exec idx --> output elem num
                        this->execidx_elemsize[i] = output_tensor_fp32->elem_size; //exec idx --> output elem size
                        this->execidx_nodename[i] = output_tensor_fp32->name;
                    }

                    /* fake quant op run */
                    //                    set_tensor_buffer(output_tensor_fake_quant, out_imgs_fake_quant[imgi].data(), output_tensor_fake_quant->elem_num * output_tensor_fake_quant->elem_size);
                    node_ops_fake_quant->run(node_ops_fake_quant, node_fake_quant, exec_graph_fake_quant);
                    this->activation_requant(out_imgs_fake_quant[imgi].data(), output_tensor_fake_quant->elem_num, 8, 1, output_tensor_fake_quant->scale, output_tensor_fake_quant->zero_point);
                } // image number

                output_channel = output_tensor_fp32->dims[1];

                if (this->op_name == OP_CONV || (this->op_name == OP_FC && this->max_search_img_num > 1))
                    this->cosin_similarity(this->cosin, this->out_imgs_fp32, this->out_imgs_fake_quant, this->max_search_img_num, this->execidx_elemnum[i], output_channel);
                else
                    this->cosin_similarity(this->cosin, this->out_imgs_fp32, this->out_imgs_fake_quant, this->max_search_img_num, this->execidx_elemnum[i], 1);

                //                this->cosin_similarity(this->cosin, out_imgs_fp32, out_imgs_fake_quant, this->max_search_img_num, this->execidx_elemnum[i], output_channel);

                for (int cosi = 0; cosi < output_channel; cosi++)
                {
                    if (cosin[cosi] > cosin_save[cosi])
                    {
                        cosin_save[cosi] = cosin[cosi];
                        zoom_save[cosi] = zoom;
                    }
                }
                if (snum == 200)
                {
                    if (this->op_name == OP_CONV || (this->op_name == OP_FC && this->max_search_img_num > 1))
                        this->print_cosin(this->cosin.data(), i, output_channel);
                    else
                        this->print_cosin(this->cosin.data(), i, 1);
                }

                if (op_name == OP_CONV || op_name == OP_FC)
                {
                    memcpy(weight_data_fake_quant, weight_data_fp32, weight_size);
                    //                    this->weight_correction(weight_data_fp32, weight_data_fake_quant, weight_tensor_fake_quant->elem_num, this->bitcount, this->symmetry, weight_tensor_fake_quant->dims[0]);
                    memcpy(interleave_buffer_fake_quant, interleave_buffer_fp32, interleave_size_fake);
                    if (node_fake_quant->ir_node->input_num > 2)
                    {
                        memcpy(bias_data_fake_quant, bias_data_fp32, bias_size);
                    }
                }
            }
        }
        else
        {
            /* per image run */
            for (int imgi = 0; imgi < this->max_search_img_num; imgi++)
            {
                this->set_node_input_output_tensor(i, imgi, 0);

                //                set_tensor_buffer(output_tensor_fp32, out_imgs_fp32[imgi].data(), output_tensor_fp32->elem_num * output_tensor_fp32->elem_size);
                node_ops_fp32->run(node_ops_fp32, node_fp32, exec_graph_fp32);

                /* fake quant op run */
                //                set_tensor_buffer(output_tensor_fake_quant, out_imgs_fake_quant[imgi].data(), output_tensor_fake_quant->elem_num * output_tensor_fake_quant->elem_size);
                node_ops_fake_quant->run(node_ops_fake_quant, node_fake_quant, exec_graph_fake_quant);
                this->activation_requant(out_imgs_fake_quant[imgi].data(), output_tensor_fake_quant->elem_num, 8, 1, output_tensor_fake_quant->scale, output_tensor_fake_quant->zero_point);

                this->execidx_elemnum[i] = output_tensor_fp32->elem_num;   //exec idx --> output elem num
                this->execidx_elemsize[i] = output_tensor_fp32->elem_size; //exec idx --> output elem size
                this->execidx_nodename[i] = output_tensor_fp32->name;
            }
            this->cosin_similarity(this->cosin, out_imgs_fp32, out_imgs_fake_quant, this->max_search_img_num, this->execidx_elemnum[i], 1);
            this->print_cosin(this->cosin.data(), i, 1);
            this->execidx_loss[i] = cosin;
        }

        this->free_used_layers(i);

        /* save node output */
        this->fp32_out.push_back(this->out_imgs_fp32);
        this->fake_quant_out.push_back(this->out_imgs_fake_quant);
    } // node number
      //    fprintf(stderr, "--------------------------------------\n");

    if (!save_graph(graphn_fp32, "save_i8_eq.tmfile"))
    {
        fprintf(stderr, "save graph failed.\n");
        return -1;
    }

    return 0;
}
