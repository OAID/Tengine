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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include "cpu_dump.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "utility/float.h"
#include "utility/vector.h"
#include "utility/utils.h"
#include "utility/log.h"

#include "cpu_node.h"
#include "cpu_graph.h"

#include "convolution_param.h"
#include "deconv_param.h"
#include "pooling_param.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/time.h>
#endif


char* replace_string_character(const char* src_str, char* dst_str, const char* target_char, const char* replaced_char)
{
    char* p;
    char* _out = dst_str;
    const char* _str = src_str;
    const char* _src = target_char;
    const char* _dst = replaced_char;
    size_t src_size = strlen(_src);
    size_t dst_size = strlen(_dst);
    size_t len = 0;

    do
    {
        p = strstr(_str, _src);
        if (p == 0)
        {
            strcpy(_out, _str);
            return dst_str;
        }
        len = p - _str;
        memcpy(_out, _str, len);
        memcpy(_out + len, _dst, dst_size);
        _str = p + src_size;
        _out = _out + len + dst_size;
    } while (p);

    return dst_str;
}


int get_tensor_cv_shape(const struct tensor* tensor, int* n, int* c, int* h, int* w)
{
    if (NULL == tensor || NULL == n || NULL == c || NULL == h || NULL ==w)
    {
        return -1;
    }

    *n = 0; *c = 0; *h = 0; *w = 0;
    const int* dims = tensor->dims;

    switch (tensor->dim_num)
    {
        case 4:
            *n = dims[0];
            *c = dims[1];
            *h = dims[2];
            *w = dims[3];
            break;
        case 3:
            *n = dims[0];
            *h = dims[1];
            *w = dims[2];
        case 2:
            *n = dims[0];
            *w = dims[1];
        default:
            return -1;
    }

    return 0;
}


float get_node_total_flops(struct node* node)
{
    float flops = 0.f;

    if (NULL != node)
    {
        if (OP_CONV == node->op.type || OP_DECONV == node->op.type || OP_FC == node->op.type)
        {
            // when calc i * w, calculation have two parts: multiplication, addition
            // flop(weight * kernel) of mul =  k_w * k_h * k_c      *  out_w * out_h * out_c;
            // flop(weight * kernel) of add = (k_w * k_h * k_c - 1) *  out_w * out_h * out_c;
            // flop(result + bias)          =  out_w * out_h * out_c;
            // so total calculation         =  k_w * k_h * k_c      *  out_w * out_h * out_c * 2;

            struct tensor* kernel = get_ir_graph_tensor(node->graph, node->input_tensors[1]);
            struct tensor* output = get_ir_graph_tensor(node->graph, node->output_tensors[0]);

            int kernel_volume = 1, feature_volume = 1;

            for (int i = 1; i < kernel->dim_num; i++)
            {
                kernel_volume *= kernel->dims[i];
            }
            for (int i = 1; i < output->dim_num; i++)
            {
                feature_volume *= output->dims[i];
            }

            flops = (float)feature_volume * (float)kernel_volume * 2.f;
        }
    }

    return flops;
}


int print_tensor_data_value(FILE* file, const struct tensor* tensor, int offset)
{
    switch (tensor->data_type)
    {
        case TENGINE_DT_FP32:
        {
            float* base_ptr = tensor->data;
            float val = base_ptr[offset];
            if (val < 0)
                fprintf(file, "%.4f ", val);
            else
                fprintf(file, " %.4f ", val);
            break;
        }
        case TENGINE_DT_FP16:
        {
            fp16_t* base_ptr = tensor->data;
            fp16_t val = base_ptr[offset];

            float val_fp32 = fp16_to_fp32(val);

            if (val_fp32 < 0)
                fprintf(file, "%.4f ", val_fp32);
            else
                fprintf(file, " %.4f ", val_fp32);
            break;
        }
        case TENGINE_DT_UINT8:
        {
            uint8_t* base_ptr = tensor->data;
            uint8_t val = base_ptr[offset];

            float scale = tensor->scale;
            int32_t zero_point = tensor->zero_point;

            float val_fp32 = (float)((int)val - (int)zero_point) * scale;
            if (val_fp32 < 0)
                fprintf(file, "%.4f ", val_fp32);
            else
                fprintf(file, " %.4f ", val_fp32);
            break;
        }
        case TENGINE_DT_INT8:
        {
            int8_t * base_ptr = tensor->data;
            int8_t val = base_ptr[offset];

            float scale = tensor->scale;

            float val_fp32 = (float)val * scale;
            if (val_fp32 < 0)
                fprintf(file, "%.4f ", val_fp32);
            else
                fprintf(file, " %.4f ", val_fp32);
        }
        case TENGINE_DT_INT32:
        {
            int32_t* base_ptr = tensor->data;
            int8_t val = base_ptr[offset];

            float scale = tensor->scale;
            float val_fp32 = (float)val * scale;

            if (val_fp32 < 0)
                fprintf(file, "%.6f ", val_fp32);
            else
                fprintf(file, " %.6f ", val_fp32);
        }
    }

    return 0;
}


void print_tensor_data_to_file(FILE* file, const struct tensor* tensor)
{
    switch (tensor->dim_num)
    {
        case 5:
        {
            int dim5 = tensor->dims[0], batch = tensor->dims[1], channel = 0, height = 0, width = 0;

            if (TENGINE_LAYOUT_NCHW == tensor->layout)
            {
                channel = tensor->dims[2];
                height = tensor->dims[3];
                width = tensor->dims[4];
            }
            if (TENGINE_LAYOUT_NHWC == tensor->layout)
            {
                height = tensor->dims[2];
                width = tensor->dims[3];
                channel = tensor->dims[4];
            }

            if (TENGINE_DT_FP32 == tensor->data_type)
            {
                fprintf(file, "Shape is {%d %d %d %d %d}, data type is fp32\n", dim5, batch, channel, height, width);
            }
            else
            {
                if (TENGINE_DT_FP16 == tensor->data_type)
                {
                    fprintf(file, "Shape is {%d %d %d %d %d}, data type is fp16, cast to fp32\n", dim5, batch, channel, height, width);
                }
                else
                {
                    const char* type_name = get_tensor_data_type_string(tensor->data_type);
                    fprintf(file, "Shape is {%d %d %d %d %d}, data type is %s, inverse quantization to fp32\n", dim5, batch, channel, height, width, type_name);
                }
            }

            for (int d5 = 0; d5 < dim5; d5++)
            {
                fprintf(file, "Dim5 %d:\n", d5);

                for (int n = 0; n < batch; n++)
                {
                    fprintf(file, "\tBatch %d:\n", n);

                    for (int ch = 0; ch < channel; ch++)
                    {
                        fprintf(file, "\t\tChannel %d:\n", ch);

                        for (int h = 0; h < height; h++)
                        {
                            fprintf(file, "\t\t\t");

                            for (int w = 0; w < width; w++)
                            {
                                int offset = 0;

                                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                                {
                                    offset += d5 * batch * channel * height * width;
                                    offset += n * channel * height * width;
                                    offset += ch * height * width;
                                    offset += h * width;
                                    offset += w;
                                }
                                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                                {
                                    offset += d5 * batch * channel * height * width;
                                    offset += n * channel * height * width;
                                    offset += ch;
                                    offset += h * width * channel;
                                    offset += w * channel;
                                }

                                print_tensor_data_value(file, tensor, offset);
                            }
                            fprintf(file, "\n");
                        }
                        fprintf(file, "\n");
                    }
                    fprintf(file, "\n");
                }
                fprintf(file, "\n");
            }

            break;
        }
        case 4:
        {
            int batch = tensor->dims[0], channel = 0, height = 0, width = 0;

            if (TENGINE_LAYOUT_NCHW == tensor->layout)
            {
                channel = tensor->dims[1];
                height = tensor->dims[2];
                width = tensor->dims[3];
            }
            if (TENGINE_LAYOUT_NHWC == tensor->layout)
            {
                height = tensor->dims[1];
                width = tensor->dims[2];
                channel = tensor->dims[3];
            }

            if (TENGINE_DT_FP32 == tensor->data_type)
            {
                fprintf(file, "Shape is {%d %d %d %d}, data type is fp32\n", batch, channel, height, width);
            }
            else
            {
                if (TENGINE_DT_FP16 == tensor->data_type)
                {
                    fprintf(file, "Shape is {%d %d %d %d}, data type is fp16, cast to fp32\n", batch, channel, height, width);
                }
                else
                {
                    const char* type_name = get_tensor_data_type_string(tensor->data_type);
                    fprintf(file, "Shape is {%d %d %d %d}, data type is %s, inverse quantization to fp32\n", batch, channel, height, width, type_name);
                }
            }

            for (int n = 0; n < batch; n++)
            {
                fprintf(file, "Batch %d:\n", n);

                for (int ch = 0; ch < channel; ch++)
                {
                    fprintf(file, "\tChannel %d:\n", ch);

                    for (int h = 0; h < height; h++)
                    {
                        fprintf(file, "\t\t");

                        for (int w = 0; w < width; w++)
                        {
                            int offset = 0;

                            if (TENGINE_LAYOUT_NCHW == tensor->layout)
                            {
                                offset += n * channel * height * width;
                                offset += ch * height * width;
                                offset += h * width;
                                offset += w;
                            }
                            if (TENGINE_LAYOUT_NHWC == tensor->layout)
                            {
                                offset += n * channel * height * width;
                                offset += ch;
                                offset += h * width * channel;
                                offset += w * channel;
                            }

                            print_tensor_data_value(file, tensor, offset);
                        }
                        fprintf(file, "\n");
                    }
                    fprintf(file, "\n");
                }
                fprintf(file, "\n");
            }

            break;
        }
        case 3:
        {
            int batch = 0, height = 0, width = 0;

            if (TENGINE_LAYOUT_NCHW == tensor->layout)
            {
                batch = tensor->dims[0];
                height = tensor->dims[1];
                width = tensor->dims[2];
            }
            if (TENGINE_LAYOUT_NHWC == tensor->layout)
            {
                height = tensor->dims[0];
                width = tensor->dims[1];
                batch = tensor->dims[2];
            }

            if (TENGINE_DT_FP32 == tensor->data_type)
            {
                fprintf(file, "Shape is {%d %d %d}, data type is fp32\n", batch, height, width);
            }
            else
            {
                if (TENGINE_DT_FP16 == tensor->data_type)
                {
                    fprintf(file, "Shape is {%d %d %d}, data type is fp16, cast to fp32\n", batch, height, width);
                }
                else
                {
                    const char* type_name = get_tensor_data_type_string(tensor->data_type);
                    fprintf(file, "Shape is {%d %d %d}, data type is %s, inverse quantization to fp32\n", batch, height, width, type_name);
                }
            }

            for (int n = 0; n < batch; n++)
            {
                for (int h = 0; h < height; h++)
                {
                    fprintf(file, "Channel %d:\n", h);
                    fprintf(file, "\t");

                    for (int w = 0; w < width; w++)
                    {
                        int offset = 0;

                        if (TENGINE_LAYOUT_NCHW == tensor->layout)
                        {
                            offset += n * height * width;
                            offset += h * width;
                            offset += w;
                        }
                        if (TENGINE_LAYOUT_NHWC == tensor->layout)
                        {
                            offset += h;
                            offset += n * width * height;
                            offset += w * height;
                        }

                        print_tensor_data_value(file, tensor, offset);
                    }
                    fprintf(file, "\n");
                }
                fprintf(file, "\n");
            }

            break;
        }
        case 2:
        {
            int batch = 0, width = 0;

            if (TENGINE_LAYOUT_NCHW == tensor->layout)
            {
                batch = tensor->dims[0];
                width = tensor->dims[1];
            }
            if (TENGINE_LAYOUT_NHWC == tensor->layout)
            {
                batch = tensor->dims[0];
                width = tensor->dims[1];
            }

            if (TENGINE_DT_FP32 == tensor->data_type)
            {
                fprintf(file, "Shape is {%d %d}, data type is fp32\n", batch, width);
            }
            else
            {
                if (TENGINE_DT_FP16 == tensor->data_type)
                {
                    fprintf(file, "Shape is {%d %d}, data type is fp16, cast to fp32\n", batch, width);
                }
                else
                {
                    const char* type_name = get_tensor_data_type_string(tensor->data_type);
                    fprintf(file, "Shape is {%d %d}, data type is %s, inverse quantization to fp32\n", batch, width, type_name);
                }
            }

            for (int n = 0; n < batch; n++)
            {
                for (int w = 0; w < width; w++)
                {
                    int offset = 0;

                    offset += n * width;
                    offset += w;

                    print_tensor_data_value(file, tensor, offset);
                }
                fprintf(file, "\n");
            }

            break;
        }
        case 1:
        {
            int width = tensor->dims[0];

            fprintf(file, "Shape is {%d}, data type is fp32\n", width);


            for (int w = 0; w < width; w++)
            {
                print_tensor_data_value(file, tensor, w);
            }

            break;
        }
        default:
            printf("Input dimension %d not to be supported.\n", tensor->dim_num);
    }
}


/*
 * Extract the blob feature map
 */
void extract_feature_from_tensor(const char* comment, const char* layer_name, const struct tensor* tensor)
{
    // 1. deal with saving path
    char save_dir[256] = { '0' };

    const char *env_path = getenv(TENGINE_DUMP_DIR);

    if (NULL != env_path && (256 - 2) > strlen(env_path))
    {
        strcpy(save_dir, env_path);

        if ('/' == save_dir[strlen(env_path)] || '\\' == save_dir[strlen(env_path)])
        {
#ifdef _MSC_VER
            save_dir[strlen(env_path)] = '\\';
            save_dir[strlen(env_path) + 1] = 0;
#else
            save_dir[strlen(env_path)] = '/';
            save_dir[strlen(env_path) + 1] = 0;
#endif
        }
    }
    else
    {
//        TLOG_WARNING("Tengine: Env var \"TENGINE_DUMP_DIR\" is too long(%d vs. 254). Using default path.\n", strlen(env_path));
        sprintf(save_dir, "./output/");
#ifdef _MSC_VER
        CreateDirectoryA(save_dir, NULL);
#else
        int ret = mkdir(save_dir, S_IRWXU | S_IRGRP | S_IWGRP | S_IROTH);
//        if (0 != ret)
//        {
//            TLOG_WARNING("Tengine: Create saving folder failed(%d), skip dump.\n", ret);
//            return;
//        }
#endif
    }

    // 2. deal with layer name
    char layer_short_name[64], layer_legal_name[64];

    if (64 < strlen(layer_name))
    {
        memcpy(layer_short_name, layer_name, 64 - 1);
        layer_short_name[64 - 1] = 0;
    }
    else
    {
        strcpy(layer_short_name, layer_name);
    }

    replace_string_character(layer_short_name, layer_legal_name, "/", "-");

    // 3. join path
    char output_file_path[512] = { '0' };

    if (strlen(layer_legal_name) + strlen(save_dir) + strlen(comment) > 256 - 16)
    {
        TLOG_WARNING("Tengine: Name of saving file is too long(%d vs. %d), skip dump.\n", strlen(layer_legal_name) + strlen(save_dir) + strlen(comment), 256 - 16);
        return;
    }

    sprintf(output_file_path, "%s%s_%s_blob_data.txt", save_dir, layer_legal_name, comment);

    FILE* file = fopen(output_file_path, "w");
    if (NULL == file)
    {
        fprintf(stderr, "Tengine: Open file(%s) failed, skip dump\n", output_file_path);
        return;
    }

    print_tensor_data_to_file(file, tensor);

    // close file
    fclose(file);
    file = NULL;
}


void extract_node_executed_time(struct subgraph* subgraph, int node_id)
{
    struct exec_graph* exec_graph = subgraph->device_graph;
    int node_num = get_vector_num(exec_graph->exec_node_list);
    int i = node_id;
    struct exec_node* node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);

    double* timer = (double*)exec_graph->timer;

    double sum_of_min_time = 0.0;
    for (int j = 0; j < node_num; j++)
    {
        sum_of_min_time += timer[j];
    }

    fprintf(stdout, "%2d [%5.2f%% : %4.1f ms] %13s idx: %2d ", i, timer[i] / sum_of_min_time * 100,
            timer[i], get_op_name_from_type(node->ir_node->op.type), node->ir_node->index);

    struct tensor* input_tensor = get_ir_graph_tensor(subgraph->graph, node->ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(subgraph->graph, node->ir_node->output_tensors[0]);
    int in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w;

    get_tensor_cv_shape(input_tensor, &in_n, &in_c, &in_h, &in_w);
    get_tensor_cv_shape(output_tensor, &out_n, &out_c, &out_h, &out_w);

    const char* in_data_type = get_tensor_data_type_string(input_tensor->data_type);
    const char* out_data_type = get_tensor_data_type_string(input_tensor->data_type);
    fprintf(stdout, "shape: {%d %3d %3d %3d} -> {%d %3d %3d %3d}\t %5s -> %5s ", in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w, in_data_type, out_data_type);

    switch (node->ir_node->op.type)
    {
        case OP_CONV:
        {
            struct conv_param* param = (struct conv_param*)node->ir_node->op.param_mem;
            fprintf(stdout, "K: %dx%d | S: %dx%d | P: %d %d %d %d", param->kernel_h, param->kernel_w, param->stride_h, param->stride_w,
                    param->pad_h0, param->pad_h1, param->pad_w0, param->pad_w1);
            if(param->group != 1)
            {
                fprintf(stdout, " DW(%3d) ", param->group);
            }
            else
            {
                fprintf(stdout, "         ");
            }
            break;
        }
        case OP_DECONV:
        {
            struct deconv_param* param = (struct deconv_param*)node->ir_node->op.param_mem;
            fprintf(stdout, "K: %dx%d | S: %dx%d | P: %d %d %d %d", param->kernel_h, param->kernel_w, param->stride_h, param->stride_w,
                    param->pad_h0, param->pad_h1, param->pad_w0, param->pad_w1);
            if(param->group != 1)
            {
                fprintf(stdout, " DW(%3d) ", param->group);
            }
            else
            {
                fprintf(stdout, "         ");
            }
            break;
        }
        case OP_POOL:
        {
            struct pool_param* param = (struct pool_param*)node->ir_node->op.param_mem;
            fprintf(stdout, "K: %dx%d | S: %dx%d | P: %d %d %d %d", param->kernel_h, param->kernel_w, param->stride_h, param->stride_w,
                    param->pad_h0, param->pad_h1, param->pad_w0, param->pad_w1);
            if(param->pool_method == 0)
            {
                fprintf(stdout, "         Max");
            }
            else
            {
                fprintf(stdout, "         Avg");
            }
            break;
        }
    }

    if (OP_CONV == node->ir_node->op.type || OP_DECONV == node->ir_node->op.type)
    {
        float mflops = get_node_total_flops(node->ir_node) / 1000000.0f;
        fprintf(stdout, "MFLOPS:%6.2f Rate:%3.0f", mflops, mflops / timer[i] * 1000.0f);
    }

    fprintf(stdout, "\n");

    if (node_id == node_num - 1)
    {
        fprintf(stdout, "total time: %.2f ms. avg time: %.2f ms. min time: %.2f ms.\n", timer[node_num + 1], timer[node_num + 1] / timer[node_num], sum_of_min_time);
    }
}




double get_current_time(void)
{
#ifdef _MSC_VER
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + (tv.tv_usec / 1000.0);
#endif
}
