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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#include "timvx_executor.hpp"

#include "timvx_define.h"

#ifdef TIMVX_MODEL_CACHE
#include "defines.h"
#include "cstdlib"
#endif

#ifdef TIMVX_MODEL_CACHE
#include "tim/vx/ops/nbg.h"
#include <fstream>
#endif

int print_tensor_data_value(FILE* file, const struct tensor* tensor, int offset)
{
    switch (tensor->data_type)
    {
        case TENGINE_DT_FP32:
        {
            float* base_ptr = (float*)tensor->data;
            float val = base_ptr[offset];
            if (val < 0)
                fprintf(file, "%.4f ", val);
            else
                fprintf(file, " %.4f ", val);
            break;
        }
//        case TENGINE_DT_FP16:
//        {
//            fp16_t* base_ptr = (fp16_t*)tensor->data;
//            fp16_t val = base_ptr[offset];
//
//            float val_fp32 = fp16_to_fp32(val);
//
//            if (val_fp32 < 0)
//                fprintf(file, "%.4f ", val_fp32);
//            else
//                fprintf(file, " %.4f ", val_fp32);
//            break;
//        }
        case TENGINE_DT_UINT8:
        {
            uint8_t* base_ptr = (uint8_t*)tensor->data;
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
            int8_t* base_ptr = (int8_t*)tensor->data;
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
            int32_t* base_ptr = (int32_t*)tensor->data;
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

const char* get_tensor_data_type_string(int data_type)
{
    switch (data_type)
    {
        case TENGINE_DT_FP32:
            return "fp32";
        case TENGINE_DT_FP16:
            return "fp16";
        case TENGINE_DT_INT8:
            return "int8";
        case TENGINE_DT_UINT8:
            return "uint8";
        case TENGINE_DT_INT32:
            return "int32";
        case TENGINE_DT_INT16:
            return "int16";
        default:
            return "unknown";
    }
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

char* replace_string_character(char* src_str, char* dst_str, char* target_char, char* replaced_char)
{
    char* p;
    char* _out = dst_str;
    char* _str = src_str;
    char* _src = target_char;
    char* _dst = replaced_char;
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

void dump_sub_graph(struct subgraph* sub_graph)
{
    TLOG_INFO("Sub graph[%d]: {%8s } has %d nodes, %d input tensors, %d output tensors.\n", sub_graph->index, sub_graph->device->name, sub_graph->node_num, sub_graph->input_num, sub_graph->output_num);
    TLOG_INFO("\tSub nodes: [ ");

    for (int j = 0; j < sub_graph->node_num - 1; j++)
    {
        int node_id = sub_graph->node_list[j];
        TLOG_INFO("%d, ", node_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->node_list[sub_graph->node_num - 1]);

    TLOG_INFO("\tSub input tensors: [ ");
    for (int j = 0; j < sub_graph->input_num - 1; j++)
    {
        int tensor_id = sub_graph->input_tensor_list[j];
        TLOG_INFO("%d, ", tensor_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->input_tensor_list[sub_graph->input_num - 1]);

    TLOG_INFO("\tSub output tensors: [ ");
    for (int j = 0; j < sub_graph->output_num - 1; j++)
    {
        int tensor_id = sub_graph->output_tensor_list[j];
        TLOG_INFO("%d, ", tensor_id);
    }
    TLOG_INFO("%d ].\n", sub_graph->output_tensor_list[sub_graph->output_num - 1]);
}

///////////////////////////////////////////////////////////////////////////////////////

VXEngine::VXEngine()
{
    this->context = tim::vx::Context::Create();
    this->graph = context->CreateGraph();
};


int VXEngine::VXTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type)
{
    auto iter = this->vx_tensor_map.find(ir_tensor_idx);

    if (this->vx_tensor_map.end() == iter)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        auto Dims = (unsigned int*)ir_tensor->dims;

        tim::vx::DataType datatype;
        switch(ir_tensor->data_type)
        {
            case (1):
                datatype = tim::vx::DataType::FLOAT16;
                break;
            case (3):
                datatype = tim::vx::DataType::UINT8;
                break;
            case (4):
                datatype = tim::vx::DataType::INT32;
                break;
            default:
                TLOG_ERR("FP32 Tensor: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
                return -1;
        }

        tim::vx::ShapeType vx_shape;

        struct node* ir_node = get_ir_graph_node(ir_graph, ir_tensor->producer);
        if (ir_node->op.type == OP_FC && ir_node->output_tensors[0] == ir_tensor_idx)
        {
            for (int i = 1; i >= 0; i--)
            {
                vx_shape.push_back(Dims[i]);
            }
        }
        else if (spec_type == SPEC_TYPE_PRELU)
        {
            vx_shape.push_back(1);
            vx_shape.push_back(1);
            vx_shape.push_back(Dims[0]);
        }
        else
        {
            for (int i = ir_tensor->dim_num - 1; i >= 0; i--)
            {
                vx_shape.push_back(Dims[i]);
            }
        }

        /* set quant params */
        tim::vx::Quantization vx_quant(tim::vx::QuantType::ASYMMETRIC, ir_tensor->scale,
                                       ir_tensor->zero_point);

        /* create the vx tesnor */
        std::shared_ptr<tim::vx::Tensor> vx_tensor;

        if (spec_type == SPEC_TYPE_OUTPUT)
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::OUTPUT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec);
        }
        else if (spec_type == SPEC_TYPE_DWCONV)
        {
            vx_shape[ir_tensor->dim_num - 2] = vx_shape[ir_tensor->dim_num - 1];
            vx_shape[ir_tensor->dim_num - 1] = 1;
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::CONSTANT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        else if (spec_type == SPEC_TYPE_PRELU)
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::CONSTANT);
            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_INPUT )
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::INPUT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec);
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_VAR)
        {
            const char* env = getenv(TENGINE_DUMP_LAYER);
            if (env && env[0] == '1')
            {
                tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                            tim::vx::TensorAttribute::OUTPUT, vx_quant);
                vx_tensor = this->graph->CreateTensor(vx_spec);
            }
            else
            {
                tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                            tim::vx::TensorAttribute::TRANSIENT, vx_quant);
                vx_tensor = this->graph->CreateTensor(vx_spec);
            }
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
                                        tim::vx::TensorAttribute::CONSTANT, vx_quant);
            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        this->vx_tensor_map[ir_tensor_idx] = vx_tensor;
    }

    return 0;
}

int VXEngine::Build(struct subgraph* subgraph)
{
//    dump_sub_graph(subgraph);
    struct graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;

        switch (op_type)
        {
            case OP_CLIP:
                this->AddClipNode(ir_node);
                break;
            case OP_CONCAT:
                this->AddConcatNode(ir_node);
                break;
            case OP_CONST:
            case OP_INPUT:
                continue;
            case OP_CONV:
                this->AddConvolutionNode(ir_node);
                break;
            case OP_DEPTHTOSPACE:
                this->AddDepthToSpaceNode(ir_node);
                break;
            case OP_DROPOUT:
                this->AddDropoutNode(ir_node);
                break;
            case OP_ELTWISE:
                this->AddEltwiseNode(ir_node);
                break;
            case OP_ELU:
                this->AddEluNode(ir_node);
                break;
            case OP_FC:
                this->AddFullyConnectionNode(ir_node);
                break;
            case OP_FLATTEN:
                this->AddFlattenNode(ir_node);
                break;
            case OP_GATHER:
                this->AddGatherNode(ir_node);
                break;
            case OP_HARDSWISH:
                this->AddHardSwishNode(ir_node);
                break;
            case OP_INTERP:
                this->AddInterpNode(ir_node);
                break;
            case OP_MISH:
                this->AddMishNode(ir_node);
                break;
            case OP_PERMUTE:
                this->AddPermuteNode(ir_node);
                break;
            case OP_POOL:
                this->AddPoolingNode(ir_node);
                break;
            case OP_PRELU:
                this->AddPReluNode(ir_node);
                break;
            case OP_RELU:
                this->AddReluNode(ir_node);
                break;
            case OP_RELU1:
                this->AddRelu1Node(ir_node);
                break;
            case OP_RESHAPE:
                this->AddReshapeNode(ir_node);
                break;
            case OP_RESIZE:
                this->AddResizeNode(ir_node);
                break;
            case OP_SCALE:
                this->AddScaleNode(ir_node);
                break;
            case OP_SIGMOID:
                this->AddSigmoidNode(ir_node);
                break;
            case OP_SLICE:
                this->AddSliceNode(ir_node);
                break;
            case OP_SOFTMAX:
                this->AddSoftmaxNode(ir_node);
                break;
            case OP_SPACETODEPTH:
                this->AddSpaceToDepthNode(ir_node);
                break;
            case OP_TANH:
                this->AddTanhNode(ir_node);
                break;
            case OP_TRANSPOSE:
                this->AddTransposeNode(ir_node);
                break;
            case OP_UPSAMPLE:
                this->AddUpsampleNode(ir_node);
                break;
            default:
                fprintf(stderr, "Tengine TIM-VX: Cannot support OP(%d).\n", ir_node->index);
                break;
        }
    }

    return 0;
}


int VXEngine::VXEnginePreRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

#ifdef TIMVX_MODEL_CACHE
    auto graph_node_count = subgraph->graph->node_num;
    auto graph_tensor_count = subgraph->graph->tensor_num;

    auto subgraph_node_count = subgraph->node_num;
    auto subgraph_tensor_count = 0;

    auto subgraph_input_count = subgraph->input_num;
    auto subgraph_output_count = subgraph->output_num;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        subgraph_tensor_count += ir_node->input_num;
        subgraph_tensor_count += ir_node->output_num;
    }

    std::string graph_name_field = std::to_string(graph_node_count) + "_" + std::to_string(graph_tensor_count);
    std::string subgraph_name_field = std::to_string(subgraph_node_count) + "_"
                                    + std::to_string(subgraph_input_count) + "_"
                                    + std::to_string(subgraph_output_count);

    std::string cache_file_name = "tm_" + graph_name_field + "_" + subgraph_name_field + ".tmcache";
    std::string full_cache_file_path;

    const char *env_cache_path = getenv(TE_MODEL_CACHE_PATH);
    if (nullptr != env_cache_path)
    {
        full_cache_file_path = std::string(env_cache_path) + "/" + full_cache_file_path;
    }
    else
    {
        full_cache_file_path = "./" + cache_file_name;
    }

    TLOG_INFO("Tengine: Model cache file for compiled is %s.", full_cache_file_path.c_str());

    bool cache_saved = false;
    std::ifstream read_stream;
    read_stream.open(full_cache_file_path, std::ios::in | std::ios::binary);
    if (read_stream.is_open())
    {
        cache_saved = true;
    }

    if (cache_saved)
    {
        /* Add TIM-VX Tensor */
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            this->VXTensorMap(ir_graph, ir_tensor_idx, SPEC_TYPE_OUTPUT);
        }

        /* Add TIM-VX Tensor */
        for (uint8_t i = 0; i < subgraph->input_num; i++)
        {
            int ir_tensor_idx = subgraph->input_tensor_list[i];
            this->VXTensorMap(ir_graph, ir_tensor_idx, 0);
        }

        read_stream.seekg(0, std::ifstream::beg);
        const auto start_length = read_stream.tellg();
        read_stream.seekg(0, std::ifstream::end);
        const auto end_length = read_stream.tellg();

        read_stream.seekg(0, std::ifstream::beg);

        auto file_size = end_length - start_length;

        nbg_buffer.reserve(file_size);
        nbg_buffer.insert(nbg_buffer.begin(), std::istreambuf_iterator<char>(read_stream), std::istreambuf_iterator<char>());
        read_stream.close();

        auto nbg_node = this->graph->CreateOperation<tim::vx::ops::NBG>(nbg_buffer.data(), subgraph_input_count, subgraph_output_count);

        std::vector<std::shared_ptr<tim::vx::Tensor>> inputs, outputs;
        for (uint8_t i = 0; i < subgraph->input_num; i++)
        {
            int ir_tensor_idx = subgraph->input_tensor_list[i];
            auto iter = this->vx_tensor_map[ir_tensor_idx];
            inputs.push_back(iter);
        }
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            auto iter = this->vx_tensor_map[ir_tensor_idx];
            outputs.push_back(iter);
        }
        (*nbg_node).BindInputs(inputs);
        (*nbg_node).BindOutputs(outputs);

        auto ret = this->graph->Compile();
        if (!ret)
        {
            TLOG_ERR("Tengine: Model compile from bin failed.");
            return -1;
        }
    }
    else
#endif
    {
        /* Add TIM-VX Tensor */
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            this->VXTensorMap(ir_graph, ir_tensor_idx, SPEC_TYPE_OUTPUT);
        }
        for (int i = 0; i < subgraph->node_num; i++)
        {
            uint16_t node_id = subgraph->node_list[i];
            struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
            if (ir_node->op.type == OP_CONV)
            {
                auto conv_param = (struct conv_param*)ir_node->op.param_mem;
                if (conv_param->group == conv_param->output_channel)
                {
                    this->VXTensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_DWCONV);
                }
            }
            else if (ir_node->op.type == OP_PRELU)
            {
                this->VXTensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_PRELU);
            }
        }
        for (int i = 0; i < subgraph->node_num; i++)
        {
            uint16_t node_id = subgraph->node_list[i];
            struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
            for (int j = 0; j < ir_node->input_num; j++)
            {
                int ir_tensor_idx = ir_node->input_tensors[j];
                this->VXTensorMap(ir_graph, ir_tensor_idx, 0);
            }
            for (int j = 0; j < ir_node->output_num; j++)
            {
                int ir_tensor_idx = ir_node->output_tensors[j];
                this->VXTensorMap(ir_graph, ir_tensor_idx, 0);
            }
        }

        /* Add TIM-VX Node */
        this->Build(subgraph);

#ifdef TIMVX_MODEL_CACHE
        size_t bin_size = -1;
        auto ret = graph->CompileToBinary(nullptr, &bin_size);
        if (-1 == bin_size || !ret)
        {
            TLOG_ERR("Tengine: Model compile to bin failed.");
            return -1;
        }

        this->nbg_buffer.resize(bin_size);
        ret = graph->CompileToBinary(nbg_buffer.data(), &bin_size);
        if (!ret)
        {
            TLOG_ERR("Tengine: Model compile to bin failed.");
            return -1;
        }

        std::ofstream nbg_stream;
        nbg_stream.open(full_cache_file_path, std::ios::out | std::ios::binary);
        if (nbg_stream.is_open())
        {
            TLOG_INFO("Tengine: Save compiled model to %s.", full_cache_file_path.c_str());
        }
        nbg_stream.write(this->nbg_buffer.data(), this->nbg_buffer.size());
        nbg_stream.close();
#else
        // fprintf(stderr,"subgraph->node_num %d\n",subgraph->node_num);
        if (subgraph->node_num > 0)
        {
            if (!this->graph->Compile()) {
                return -1;
            }
        }
#endif

    }

    return 0;
};

int VXEngine::VXEngineRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    /* upload data */
//    fprintf(stderr,"subgraph->input_num %d\n",subgraph->input_num);
    if (subgraph->input_num > 0)
    {
        for (uint8_t i = 0; i < subgraph->input_num; i++)
        {
            int ir_tensor_idx = subgraph->input_tensor_list[i];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            if (!this->vx_tensor_map[ir_tensor_idx]->CopyDataToTensor(ir_tensor->data, ir_tensor->elem_num * ir_tensor->elem_size)) {
                return -1;
            }
        }

        if (!this->graph->Run())
        {
            return -1;
        }

        /* download data */
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            if (nullptr == ir_tensor->data)
            {
                auto u8data = (uint8_t*)malloc(ir_tensor->elem_size * ir_tensor->elem_num);
                ir_tensor->data = u8data;

                ir_tensor->free_host_mem = 1;
                ir_tensor->internal_allocated = 0;
            }

            if (!this->vx_tensor_map[ir_tensor_idx]->CopyDataFromTensor(ir_tensor->data)) 
            {
                TLOG_INFO("Tengine: Copy output data from VX tensor to CPU failed.\n");
                return -1;
            }
        }


        const char* env = getenv(TENGINE_DUMP_LAYER);
        if (env && env[0] == '1')
        {
            for (uint8_t i = 0; i < ir_graph->tensor_num; i++)
            {
                if (ir_graph->tensor_list[i]->tensor_type == TENSOR_TYPE_VAR)
                {
                    if (ir_graph->tensor_list[i]->data == NULL)
                    {
                        TLOG_INFO("Log:download data is NULL\n");
                        uint8_t* u8data = (uint8_t*)malloc(ir_graph->tensor_list[i]->elem_size * ir_graph->tensor_list[i]->elem_num);
                        ir_graph->tensor_list[i]->data = u8data;
                    }
                    if (!this->vx_tensor_map[i]->CopyDataFromTensor(ir_graph->tensor_list[i]->data))
                    {
                        TLOG_INFO("Log:Copy output data fail\n");
                        return -1;
                    }
                }
            }

            for (uint8_t i = 0; i < ir_graph->tensor_num; i++)
            {
                fprintf(stderr,"tensor type %d\n",ir_graph->tensor_list[i]->tensor_type);
                if (ir_graph->tensor_list[i]->tensor_type == TENSOR_TYPE_VAR)
                {
                    char dir_str[32] = { 0 };
                    sprintf(dir_str, "out[%d]", i);

                    if (NULL != ir_graph->tensor_list[i]->data)
                    {
                        extract_feature_from_tensor(dir_str, ir_graph->tensor_list[i]->name, ir_graph->tensor_list[i]);
                    }
                }
            }
        }// End TENGINE_DUMP_LAYER
    }

    return 0;
}

void VXEngine::VXEnginePostRun()
{

};
