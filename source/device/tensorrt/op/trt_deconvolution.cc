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
 * Author: guanguojing1989@126.com
 */

#include "../trt_executor.hpp"

EXPORT_BEGIN
#include "deconv_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddDeConvolutionNode(struct graph* ir_graph, struct node *node)
{
    nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};

    struct tensor* deconv_data = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* deconv_weight = get_ir_graph_tensor(ir_graph, node->input_tensors[1]);

    if (nullptr == deconv_data || nullptr == deconv_weight)
    {
        fprintf(stderr, "Tengine: Get input data & weight for DeConvolution(id: %d, name: %s).\n", deconv_weight->index, deconv_weight->name);
        return false;
    }

    if (!check_if_input_in_map(deconv_data->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for DeConvolution(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* input = this->tensor_real_map[this->tensor_swap_map[deconv_data->index]];

    switch (deconv_weight->data_type)
    {
        case TENGINE_DT_FP32:
        {
            weight.values = deconv_weight->data;
            weight.count = deconv_weight->elem_num;
            weight.type = nvinfer1::DataType::kFLOAT;
            break;
        }
        case TENGINE_DT_INT8:
        {
            if (deconv_weight->quant_param_num != deconv_weight->dims[0])
            {
                fprintf(stderr, "Tengine: Unsupported weight quant channel of deconv(id: %d, name: %s).\n", node->index, node->name);
                return false;
            }

            float* weight_buffer = (float*)sys_malloc(deconv_weight->elem_num * deconv_weight->elem_size);
            this->host_buffer.push_back(weight_buffer);

            for (int ch = 0; ch < deconv_weight->quant_param_num; ch++)
            {
                int block_size = deconv_weight->dims[1] * deconv_weight->dims[2] * deconv_weight->dims[3];
                for (int i = 0; i < block_size; i++)
                {
                    int offset = block_size * ch;
                    weight_buffer[i] = (float)(((int8_t*)deconv_weight->data)[offset + i]) * deconv_weight->scale_list[ch];
                }
            }

            weight.values = weight_buffer;
            weight.count = deconv_weight->elem_num;
            weight.type = nvinfer1::DataType::kFLOAT;
            break;
        }
        case TENGINE_DT_UINT8:
        {
            float* weight_buffer = (float*)sys_malloc(deconv_weight->elem_num * deconv_weight->elem_size);
            this->host_buffer.push_back(weight_buffer);

            for (int i = 0; i < deconv_weight->elem_num; i++)
            {
                weight_buffer[i] = (float)(((uint8_t*)deconv_weight->data)[i] - deconv_weight->zero_point) * deconv_weight->scale;
            }

            weight.values = weight_buffer;
            weight.count = deconv_weight->elem_num;
            weight.type = nvinfer1::DataType::kFLOAT;
            break;
        }
        default:
            fprintf(stderr, "Tengine: Unsupported weight quant data type(%d) of conv(id: %d, name: %s).\n", deconv_weight->data_type, node->index, node->name);
            return false;
    }

    if(2 < node->input_num)
    {
        struct tensor* deconv_bias = get_ir_graph_tensor(ir_graph, node->input_tensors[2]);

        switch (deconv_bias->data_type)
        {
            case TENGINE_DT_FP32:
            {
                bias.values = deconv_bias->data;
                bias.count = deconv_bias->elem_num;
                bias.type = nvinfer1::DataType::kFLOAT;
                break;
            }
            case TENGINE_DT_INT32:
            {
                float* bias_buffer = (float*)sys_malloc(deconv_bias->elem_num * deconv_bias->elem_size);
                this->host_buffer.push_back(bias_buffer);

                if (1 == deconv_bias->quant_param_num)
                {
                    for (uint32_t i = 0; i < deconv_bias->elem_num; i++)
                    {
                        bias_buffer[i] = (float)(((int32_t*)deconv_bias->data)[i]) * deconv_bias->scale;
                    }
                }
                else
                {
                    for (uint32_t i = 0; i < deconv_bias->elem_num; i++)
                    {
                        bias_buffer[i] = (float)(((int32_t*)deconv_bias->data)[i]) * deconv_bias->scale_list[i];
                    }
                }

                break;
            }
            default:
                fprintf(stderr, "Tengine: Unsupported weight quant data type(%d) of conv(id: %d, name: %s).\n",
                        deconv_bias->data_type, node->index, node->name);
                return false;
        }
    }

    auto param = (struct deconv_param*)node->op.param_mem;

    nvinfer1::DimsHW  kernel_size{ param->kernel_h, param->kernel_w };
    int output_channel = param->num_output;

    nvinfer1::IDeconvolutionLayer* layer = this->network->addDeconvolutionNd(*input, output_channel, kernel_size, weight, bias);
    if(nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Deconvolution(id: %d, name: %s) layer failed.\n", deconv_data->index, deconv_weight->name);
        return false;
    }

    layer->setStride(nvinfer1::DimsHW(param->stride_h, param->stride_w));
    layer->setPrePadding(nvinfer1::DimsHW(param->pad_h0, param->pad_w0));
    layer->setPostPadding(nvinfer1::DimsHW(param->pad_h1, param->pad_w1));
    layer->setDilationNd(nvinfer1::DimsHW(param->dilation_h, param->dilation_w));
    layer->setNbGroups(param->group);

    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* deconv_tensor = layer->getOutput(0);

    this->SetRange(ir_graph, node->output_tensors[0], deconv_tensor);

    this->tensor_real_map[node->output_tensors[0]] = deconv_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    if (0 <= param->activation)
    {
        addReLUNode(ir_graph, node);
    }

    return true;
}
