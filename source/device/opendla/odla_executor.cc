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
 * Copyright (c) 2021, Institute of Computing Technology
 * Author: wanglei21c@mails.ucas.ac.cn
 */

#include "odla_executor.hpp"
#include "odla_define.h"
#include "priv/Check.h"
#include <thread>

void ODLAEngine::odla_input_data_convert(void * dst, const void * src, nvdla::IRuntime::NvDlaTensor tDesc) const{
#ifdef OPENDLA_DEBUG_DATA
    fprintf(stdout, "%s Entrance.\n", __func__ );
#endif
    uint32_t max_thread = std::thread::hardware_concurrency();
    uint32_t batch  = tDesc.dims.n;
    uint32_t channel = tDesc.dims.c;
    uint32_t height = tDesc.dims.h;
    uint32_t width = tDesc.dims.w;
    uint32_t atom_c_size = this->targetConfig->atomicCSize();
    uint32_t atom_k_size = this->targetConfig->atomicKSize();
    uint32_t line_stride = tDesc.stride[1];
    uint32_t surface_stride = tDesc.stride[2];


    for (size_t n = 0; n < batch; n++){
        #pragma omp parallel for num_threads(max_thread)
        for(size_t c = 0; c < channel; c++){
            uint32_t cquotient = c / atom_c_size;
            uint32_t cremainder = c % atom_c_size;

            for (size_t h = 0; h < height; ++h){
                for (size_t w = 0; w < width; ++w){
                    size_t idx = n * channel * height * width + c * height * width + h * width + w;
                    uint32_t _offset = (cquotient * surface_stride) + (h * line_stride ) + (w * atom_k_size) + cremainder + n * channel * height * width;
                    int8_t* _dst = (int8_t*)dst + _offset;
                    int8_t* _src = (int8_t*)src + idx;
                    *_dst = *_src;
                }
            }
        }
    }

}

void ODLAEngine::odla_output_data_convert(void * dst, const void * src, nvdla::IRuntime::NvDlaTensor tDesc) const
{
#ifdef OPENDLA_DEBUG_DATA
    fprintf(stdout, "%s Entrance.\n", __func__ );
#endif
    uint32_t max_thread = std::thread::hardware_concurrency();
    uint32_t batch = tDesc.dims.n;
    uint32_t channel = tDesc.dims.c;
    uint32_t height = tDesc.dims.h;
    uint32_t width = tDesc.dims.w;
    uint32_t atom_c_size = this->targetConfig->atomicCSize();
    uint32_t atom_k_size = this->targetConfig->atomicKSize();
    uint32_t line_stride = tDesc.stride[1];
    uint32_t surface_stride = tDesc.stride[2];

    // Copy contents
    for (size_t n = 0; n < batch; n++){
        #pragma omp parallel for num_threads(max_thread)
        for (size_t c = 0; c < channel; c++)
        {
            NvU32 cquotient = c / atom_c_size;
            NvU32 cremainder = c % atom_c_size;
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                {
                    size_t idx = n * channel * height * width + c * height * width + h * width + w;
                    int8_t* _dst = (int8_t*)dst + idx;
                    uint32_t _offset = (cquotient * surface_stride) + (h * line_stride) + (w * atom_k_size) + cremainder + n * c * h * w;
                    *_dst = *((int8_t*)src + _offset);
                };
        }
    }
}

NvDlaError ODLAEngine::ODLAConfigGenerate(){
#ifdef OPENDLA_DEBUG_DATA
    fprintf(stdout, "%s Entrance.\n", __func__ );
#endif
    NvDlaError e = NvDlaSuccess;

    nvdla::IProfiler* profiler = nvdla::priv::ProfilerFactory::newProfiler().priv();;
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No profiler available.");
    }

    this->profile = nvdla::priv::ProfileFactory::priv(profiler->getProfile(this->tp_name.c_str()));
    if ( !this->profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Couldn't find profile to compile.");
    }
    this->profile->setComputePrecision(this->precision);
    if(this->precision == nvdla::DataType::INT8){
        this->profile->setTensorScalingMode(this->scalingMode);
        this->profile->setQuantizationMode(this->quantizationMode);
        this->profile->setNetworkOutputSurfaceFormat(nvdla::PixelFormat::FEATURE_X8);
    }else{
        this->profile->setNetworkOutputSurfaceFormat(nvdla::PixelFormat::FEATURE);
    }
    this->profile->setNetworkOutputDataFormat(nvdla::DataFormat::NCxHWx);
    this->profile->setMultiBatchSize(this->numBatches);
    this->profile->setNetworkInputDataFormat(this->inDataFormat);
    switch(this->inDataFormat)
    {
        case nvdla::DataFormat::NHWC:
        if (this->precision == nvdla::DataType::HALF){
            PROPAGATE_ERROR_FAIL(this->profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::A16B16G16R16_F));
        }
        else if (this->precision == nvdla::DataType::INT8){
                PROPAGATE_ERROR_FAIL(this->profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::A8B8G8R8));
        }
        else{
            fprintf(stderr, "NHWC and compute precision %u is not yet supported", (uint32_t)this->precision);
        }
        break;
    case nvdla::DataFormat::NCxHWx:
    case nvdla::DataFormat::NCHW:
    case nvdla::DataFormat::UNKNOWN:    // atleast start the test with feature data format
    default:
        if (this->precision == nvdla::DataType::INT8)
            profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE_X8);
        else
            profile->setNetworkInputSurfaceFormat(nvdla::PixelFormat::FEATURE);
    }

    targetConfig = nvdla::priv::TargetConfigFactory::priv(profiler->getTargetConfig(this->target_config_name.c_str()));
    if ( !targetConfig )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Couldn't find target config to compile.");
    }

fail:
    return e;
}

ODLAEngine::ODLAEngine()
{
#ifdef OPENDLA_DEBUG_DATA
    fprintf(stdout, "%s Entrance.\n", __func__ );
#endif
    this->runtime = nvdla::createRuntime();
    this->compiler = nvdla::priv::CompilerFactory::newCompiler();
    this->loadable = nvdla::priv::LoadableFactory::LoadablePrivPair(0, 0);
    this->ODLAConfigGenerate();

    this->graph = new nvdla::priv::canonical_ast::Graph();
    if ( !this->graph )
    {
        fprintf(stderr, "Can't create a new Canonical AST.\n");
    }
}


int ODLAEngine::ODLATensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type)
{
#ifdef OPENDLA_DEBUG_DATA
    fprintf(stdout, "%s Entrance.\n", __func__ );
#endif
    auto iter = this->odla_tensor_map.find(ir_tensor_idx);

    if (this->odla_tensor_map.end() == iter)
    {
        if (spec_type == SPEC_TYPE_INTERP || spec_type == SPEC_TYPE_SLICE)
        {
            this->odla_tensor_map[ir_tensor_idx] = NULL;
            return 0;
        }
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        auto Dims = (unsigned int*)ir_tensor->dims;

        nvdla::DataType datatype;
        switch(ir_tensor->data_type)
        {
            case TENGINE_DT_FP32:
                // float32
                datatype = nvdla::DataType::FLOAT;
                break;
            case TENGINE_DT_FP16:
                // float16
                datatype = nvdla::DataType::HALF;
                break;
            case TENGINE_DT_INT8:
                datatype = nvdla::DataType::INT8;
                break;
            case TENGINE_DT_UINT8:
                datatype = nvdla::DataType::UINT8;
                break;
            case TENGINE_DT_INT16:
                datatype = nvdla::DataType::INT16;
                break;
            case TENGINE_DT_INT32:
//                TLOG_ERR("Tensor data type: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) not supported by opendla .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
                break;
            default:
                TLOG_ERR("Tensor data type: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
                break;
        }

        nvdla::Dims4 tensor_shape;

        struct node* ir_node = get_ir_graph_node(ir_graph, ir_tensor->producer);
        if (spec_type == SPEC_TYPE_PRELU)
        {
            tensor_shape.w = 1;
            tensor_shape.h = 1;
            tensor_shape.c = Dims[0];
            tensor_shape.n = 1;
        }
        else if (spec_type == SPEC_TYPE_CONV){
            tensor_shape.n = Dims[0];   // output channel
            tensor_shape.c = Dims[1];   // input channel
            tensor_shape.h = Dims[2];
            tensor_shape.w = Dims[3];
        }
        else if(spec_type == SPEC_TYPE_CONV_BIAS){
            // bias
            tensor_shape.n = 1;
            tensor_shape.c = Dims[0];
            tensor_shape.h = 1;
            tensor_shape.w = 1;
        }
        else
        {
            if(ir_tensor->dim_num == 4){
                tensor_shape.n = Dims[0];
                tensor_shape.c = Dims[1];
                tensor_shape.h = Dims[2];
                tensor_shape.w = Dims[3];
            } else if(ir_tensor->dim_num == 1){
                // bias op
                tensor_shape.n = 1;
                tensor_shape.c = Dims[0];
                tensor_shape.h = 1;
                tensor_shape.w = 1;
            } else if(ir_tensor->dim_num == 2){
                // bias op
                tensor_shape.n = Dims[0];
                tensor_shape.c = Dims[1];
                tensor_shape.h = 1;
                tensor_shape.w = 1;
            } else {
                fprintf(stderr, "Dims Number %d Not Supported. \n", ir_tensor->dim_num);
                return -1;
            }
        }

       /* create the odla tesnor */
        nvdla::priv::TensorFactory::TensorPrivPair t = nvdla::priv::TensorFactory::newTensor();
        nvdla::priv::Tensor* odla_tensor = NULL;
        if (spec_type == SPEC_TYPE_OUTPUT)
        {
            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kNW_OUTPUT);
            t.i()->setDataType(datatype);
            t.i()->setName(ir_tensor->name);
            if(ir_tensor->quant_param_num == 1){
                float tensor_min_val = ir_tensor->scale * -127.0f;
                float tensor_max_val = ir_tensor->scale * +127.0f;
                t.i()->setChannelDynamicRange(-1, tensor_min_val, tensor_max_val);
            }
            odla_tensor = t.priv();
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_INPUT || spec_type == SPEC_TYPE_INPUT)
        {
            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kNW_INPUT);
            t.i()->setDataType(datatype);
            t.i()->setName(ir_tensor->name);
            if(ir_tensor->quant_param_num == 1){
                float tensor_min_val = ir_tensor->scale * -127.0f;
                float tensor_max_val = ir_tensor->scale * +127.0f;
                t.i()->setChannelDynamicRange(-1, tensor_min_val, tensor_max_val);
            }
            odla_tensor = t.priv();
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_VAR)
        {
            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kIO);       // May Not be Right
            t.i()->setDataType(datatype);
            t.i()->setName(ir_tensor->name);
            if(ir_tensor->quant_param_num == 1){
                float tensor_min_val = ir_tensor->scale * -127.0f;
                float tensor_max_val = ir_tensor->scale * +127.0f;
                t.i()->setChannelDynamicRange(-1, tensor_min_val, tensor_max_val);
            }
            odla_tensor = t.priv();
        }
        else if (spec_type == SPEC_TYPE_CONV_BIAS)
        {
            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kBIAS);
            t.i()->setDataType(datatype);
            t.i()->setName(ir_tensor->name);
            if(1 == ir_tensor->quant_param_num){
                float tensor_min_val = ir_tensor->scale * -127.0f;
                float tensor_max_val = ir_tensor->scale * +127.0f;
                t.i()->setChannelDynamicRange(-1, tensor_min_val, tensor_max_val);
            }else if (1 < ir_tensor->quant_param_num){
                for (int ch = 0; ch < ir_tensor->quant_param_num; ++ch)
                {
                    float tensor_min_val = ir_tensor->scale_list[ch] * -127.0f;
                    float tensor_max_val = ir_tensor->scale_list[ch] * +127.0f;
                    t.i()->setChannelDynamicRange(ch, tensor_min_val, tensor_max_val);
                }
            }
            odla_tensor = t.priv();
        }
        else if (spec_type == SPEC_TYPE_CONV)
        {
            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kIO);
            t.i()->setDataType(datatype);
            t.i()->setName(ir_tensor->name);
            if(1 == ir_tensor->quant_param_num){
                float tensor_min_val = ir_tensor->scale * -127.0f;
                float tensor_max_val = ir_tensor->scale * +127.0f;
                t.i()->setChannelDynamicRange(-1, tensor_min_val, tensor_max_val);
            }else if (1 < ir_tensor->quant_param_num){
                for (int ch = 0; ch < ir_tensor->quant_param_num; ++ch)
                {
                    float tensor_min_val = ir_tensor->scale_list[ch] * -127.0f;
                    float tensor_max_val = ir_tensor->scale_list[ch] * +127.0f;
                    t.i()->setChannelDynamicRange(ch, tensor_min_val, tensor_max_val);
                }
            }
            odla_tensor = t.priv();
        }
        else if (spec_type == SPEC_TYPE_DWCONV)
        {
            if (ir_tensor->quant_param_num == 1)
            {
            }
            else if(ir_tensor->quant_param_num > 1)
            {
                std::vector<float> scale_list;
                std::vector<int32_t> zp_list;
                for (int i = 0; i < Dims[0]; i++)
                {
                    scale_list.push_back(ir_tensor->scale_list[i]);
                    zp_list.push_back(ir_tensor->zp_list[i]);
                }

            }
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kIO);
            t.i()->setDataType(datatype);
            t.i()->setName(ir_tensor->name);
            if(ir_tensor->quant_param_num == 1){
                float tensor_min_val = ir_tensor->scale * -127.0f;
                float tensor_max_val = ir_tensor->scale * +127.0f;
                t.i()->setChannelDynamicRange(-1, tensor_min_val, tensor_max_val);
            }
            odla_tensor = t.priv();
        }
        this->odla_tensor_map[ir_tensor_idx] = odla_tensor;
    }

    return 0;
}

int ODLAEngine::Build(struct subgraph* subgraph)
{
#ifdef OPENDLA_DEBUG_DATA
    fprintf(stdout, "%s Entrance.\n", __func__ );
#endif
    struct graph* ir_graph = subgraph->graph;
    std::vector<nvdla::priv::Tensor *> graphInputs;
    std::vector<nvdla::priv::Tensor *> graphOutputs;
    std::vector<nvdla::priv::canonical_ast::Edge *> inputEdges;
    std::vector<nvdla::priv::canonical_ast::Edge *> outputEdges;
    std::map<nvdla::priv::Tensor*,nvdla::priv::Tensor*> originTensor2canTensor;
    inputEdges.resize(subgraph->input_num);
    outputEdges.resize(subgraph->output_num);
    for (int i = 0; i < subgraph->input_num; ++i)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, subgraph->input_tensor_list[i]);
        graphInputs.push_back(nvdla::priv::TensorFactory::priv(this->odla_tensor_map[input_tensor->index]));
    }

    for (int i = 0; i < subgraph->output_num; ++i)
    {
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, subgraph->output_tensor_list[i]);
        graphOutputs.push_back(nvdla::priv::TensorFactory::priv(this->odla_tensor_map[output_tensor->index]));
    }

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;
        nvdla::priv::canonical_ast::Node * Node = nullptr;

        switch (op_type)
        {
            case OP_BATCHNORM:
                Node = this->AddBatchNormalizationNode(ir_node);
                break;
            case OP_CONCAT:
                Node = this->AddConcatNode(ir_node);
                break;
            case OP_CONV:
                Node = this->AddConvolutionNode(ir_node);
                break;
            case OP_CONST:
                continue;
            case OP_DECONV:
                Node = this->AddDeconvlutionNode(ir_node);
                break;
            case OP_ELTWISE:
                Node = this->AddEltwiseNode(ir_node);
                break;
            case OP_FC:
                Node = this->AddFullyConnectionNode(ir_node);
                break;
            case OP_INPUT:
                continue;
            case OP_RELU:
                Node = this->AddReluNode(ir_node);
                break;
            case OP_SCALE:
                Node = this->AddScaleNode(ir_node);
                break;
            case OP_SPLIT:
                Node = this->AddSplitNode(ir_node);
                break;
            case OP_POOL:
                Node = this->AddPoolingNode(ir_node);
                break;
            default:
                fprintf(stderr, "Tengine OpenDLA: Cannot support OP(%d).\n", ir_node->index);
                break;
        }
        if(!Node) {
            fprintf(stderr, "%s: node create failed, op type is : %d .\n", __func__, op_type);
        };
        Node->setGraph(this->graph);
        this->graph->insertNode(Node);
        Node->setId(this->graph->nextNodeId());
        Node->setName(ir_node->name);
        this->odla_node_map[Node] = ir_node;
    }

    for(auto n : this->odla_node_map){
        // iterate each node
        std::vector<nvdla::priv::Tensor*> ioTensors;
        size_t input_tensors = 0, output_tensors = 0, aux_input_tensors = 0;
        auto odla_node = n.first;
        auto ir_node = n.second;

        if(ir_node->op.type == OP_CONV || ir_node->op.type == OP_DECONV || ir_node->op.type == OP_FC){
            // CONV|DECONV|FC Only have one input in OPENDLA but tengine ir regard weights and bias as input.
            struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
            auto tensor = nvdla::priv::TensorFactory::priv(this->odla_tensor_map[input_tensor->index]);
            if(!tensor){
                fprintf(stderr,"%s : Tensor not found .\n", __func__ );
                continue;
            }
            ioTensors.push_back(tensor);
            input_tensors++;
        }
        else for (size_t i = 0; i < ir_node->input_num; ++i)
        {
            struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
            auto tensor = nvdla::priv::TensorFactory::priv(this->odla_tensor_map[input_tensor->index]);
            if(!tensor){
                fprintf(stderr,"%s : Tensor not found .\n", __func__ );
                continue;
            }
            ioTensors.push_back(tensor);
            input_tensors++;
        }
        for (size_t i = 0; i < ir_node->output_num; ++i)
        {
            struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[i]);
            auto tensor = nvdla::priv::TensorFactory::priv(this->odla_tensor_map[output_tensor->index]);
            if(!tensor){
                fprintf(stderr,"%s : Tensor not found .\n", __func__ );
                continue;
            }
            ioTensors.push_back(tensor);
            output_tensors++;
        }


        for (size_t i = 0; i < ioTensors.size(); ++i)
        {
            auto odla_tensor = ioTensors[i];
            bool isInput = i < input_tensors;
            auto edgeSide = isInput ? nvdla::priv::ast::EdgeSideEnum::SECOND:nvdla::priv::ast::EdgeSideEnum::FIRST;
            auto edgeDirection = nvdla::priv::ast::EdgeDirectionEnum::DIRECTED;
            nvdla::priv::canonical_ast::Edge * newEdge = nullptr;
            nvdla::priv::Tensor * newTensor = nullptr;
            auto t2e = this->odla_edge_map.find(odla_tensor);
            if (t2e == this->odla_edge_map.end())
            {
                newEdge = new nvdla::priv::canonical_ast::Edge();
                newEdge->setGraph(this->graph);

                newTensor = odla_tensor->clone();
                newTensor->setNetwork(NULL);
                newTensor->setTensorType(nvdla::TensorType::kIO);
                newEdge->setId(graph->nextEdgeId());
                newEdge->setOriginalTensor(newTensor);
                graph->insertEdge(newEdge);
                this->odla_edge_map[odla_tensor] = newEdge;
                originTensor2canTensor[odla_tensor] = newTensor;
            } else {
                newEdge = t2e->second;
            }
            this->graph->appendNodeToEdge(newEdge, edgeSide, odla_node);

            if(isInput){
                for ( size_t inputIdx = 0; inputIdx < subgraph->input_num; ++inputIdx)
                {
                    if ( odla_tensor == graphInputs[inputIdx])
                    {
                        inputEdges[inputIdx] = newEdge; //把当前edge加入graph的input_edges列表当中
                        newTensor = originTensor2canTensor[odla_tensor];
                        newTensor->setTensorType(nvdla::TensorType::kNW_INPUT);
                        break;
                    }
                }
                odla_node->markInputEdge(newEdge); //告诉当前node，你的这个edge是一个网络inputedge
            }else{
                for ( size_t outputIdx = 0; outputIdx < subgraph->output_num; outputIdx++)
                {
                    if ( odla_tensor == graphOutputs[outputIdx] )
                    {
                        outputEdges[outputIdx] = newEdge;
                        newTensor = originTensor2canTensor[odla_tensor];
                        newTensor->setTensorType(nvdla::TensorType::kNW_OUTPUT);
                        break;
                    }
                }
                odla_node->markOutputEdge(newEdge);
            }
        }
    }

    if ( !inputEdges.empty() )
    {
        graph->setInputEdges(inputEdges);
    }
    if ( !outputEdges.empty() )
    {
        graph->setOutputEdges(outputEdges);
    }
    this->graph->scoredOrdering()->generate();
    this->graph->markClean();
    return 0;
}


int ODLAEngine::ODLAEnginePreRun(struct subgraph* subgraph)
{
#ifdef OPENDLA_DEBUG_DATA
    dump_sub_graph_odla(subgraph);
    fprintf(stdout, "%s Entrance.\n", __func__ );
#endif
    NvDlaError e = NvDlaSuccess;
    struct graph* ir_graph = subgraph->graph;
    /* Add OpenDLA Tensor */
    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        this->ODLATensorMap(ir_graph, ir_tensor_idx, SPEC_TYPE_INPUT);
    }
    for (uint8_t i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        this->ODLATensorMap(ir_graph, ir_tensor_idx, SPEC_TYPE_OUTPUT);
    }
    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        if (ir_node->op.type == OP_CONV)
        {
            auto conv_param = (struct conv_param*)ir_node->op.param_mem;
            if ((conv_param->group == conv_param->output_channel) && (conv_param->output_channel != 1))
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_DWCONV);
            }
            else
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_CONV);
            }
            if (ir_node->input_num > 2)
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[2], SPEC_TYPE_CONV_BIAS);
            }
        }
        else if (ir_node->op.type == OP_PRELU)
        {
            this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_PRELU);
        }
        else if (ir_node->op.type == OP_INTERP)
        {
            if (ir_node->input_num == 3)
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_INTERP);
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[2], SPEC_TYPE_INTERP);
            }
            else if (ir_node->input_num == 2)
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], SPEC_TYPE_INTERP);
            }
        }
        else if (ir_node->op.type == OP_SLICE)
        {
            if (ir_node->input_num > 1)
            {
                for (int FI = 1; FI < ir_node->input_num; FI++)
                {
                    this->ODLATensorMap(ir_graph, ir_node->input_tensors[FI], SPEC_TYPE_SLICE);
                }
            }
        }
    }
    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        for (int j = 0; j < ir_node->input_num; j++)
        {
            int ir_tensor_idx = ir_node->input_tensors[j];
            this->ODLATensorMap(ir_graph, ir_tensor_idx, 0);
        }
        for (int j = 0; j < ir_node->output_num; j++)
        {
            int ir_tensor_idx = ir_node->output_tensors[j];
            this->ODLATensorMap(ir_graph, ir_tensor_idx, 0);
        }
    }


    /* Add OpenDLA Node / Build Canonical AST Graph */
    this->Build(subgraph);

    if (subgraph->node_num > 0)
    {
        std::vector<nvdla::priv::engine_ast::Graph *> engineASTList;
        engineASTList.push_back(nvdla::priv::engine_ast::generateGraph(this->profile, this->targetConfig, this->graph));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: generateGraph. \n");
            engineASTList.pop_back();
            return -1;
        }

        engineASTList.push_back(this->compiler.priv()->registerBuffers(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: registerBuffers. \n");
            engineASTList.pop_back();
            return -1;
        }

        engineASTList.push_back(this->compiler.priv()->preProcessAuxData(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: preProcessAuxData. \n");
            engineASTList.pop_back();
            return -1;
        }
        engineASTList.push_back(this->compiler.priv()->mergeActivationOperations(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: mergeActivationOperations. \n");
            engineASTList.pop_back();
            return -1;
        }
        engineASTList.push_back(this->compiler.priv()->updateScalingFactors(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: updateScalingFactors. \n");
            engineASTList.pop_back();
            return -1;
        }
        engineASTList.push_back(this->compiler.priv()->quantizeAuxData(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: quantizeAuxData. \n");
            engineASTList.pop_back();
            return -1;
        }
        engineASTList.push_back(this->compiler.priv()->fuseOnTheFlyNodes(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: fuseOnTheFlyNodes. \n");
            engineASTList.pop_back();
            return -1;
        }
        engineASTList.push_back(this->compiler.priv()->handleLowPrecisionConversions(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: handleLowPrecisionConversions. \n");
            engineASTList.pop_back();
            return -1;
        }
        engineASTList.push_back(this->compiler.priv()->translateAuxData(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: translateAuxData. \n");
            engineASTList.pop_back();
            return -1;
        }
        engineASTList.push_back(this->compiler.priv()->reserveBuffers(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: reserveBuffers. \n");
            engineASTList.pop_back();
            return -1;
        }
        engineASTList.push_back(this->compiler.priv()->splitNodes(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: splitNodes. \n");
            engineASTList.pop_back();
            return -1;
        }

        engineASTList.push_back(this->compiler.priv()->fuseSubEngineOps(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: fuseSubEngineOps. \n");
            engineASTList.pop_back();
        }

        engineASTList.push_back(this->compiler.priv()->boundGraph(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: boundGraph. \n");
            engineASTList.pop_back();
            return -1;
        }

        engineASTList.push_back(this->compiler.priv()->handleMultiBatch(engineASTList.back()));
        if(!engineASTList.back()){
            fprintf(stderr, "Failed compilation phase: handleMultiBatch. \n");
            engineASTList.pop_back();
            return -1;
        }

        if (this->profile->copyOutDebugSurfaces()){
            engineASTList.push_back(this->compiler.priv()->enableCopyOutDebugSurfaces(engineASTList.back()));
            if(!engineASTList.back()){
                fprintf(stderr, "Failed compilation phase: enableCopyOutDebugSurfaces. \n");
                engineASTList.pop_back();
                return -1;
            }
        }


        // generate Loadable Task info
        bool done = false;
        for ( int pass = 0; !done; ++pass )
        {
            nvdla::priv::engine_ast::NodeSequence topological_order;
            engineASTList.push_back(this->compiler.priv()->generateDependencyParams(engineASTList.back(), topological_order));
            if(!engineASTList.back()){
                fprintf(stderr, "Failed compilation phase: generateDependencyParams. \n");
                engineASTList.pop_back();
                return -1;
            }
            engineASTList.push_back(this->compiler.priv()->resolveMemory(engineASTList.back(),topological_order));
            if(!engineASTList.back()){
                fprintf(stderr, "Failed compilation phase: resolveMemory. \n");
                engineASTList.pop_back();
                return -1;
            }

            done = !engineASTList.back()->dirty();
            ASSERT(done);
        }
        auto finalEngineAST = engineASTList.back();

        const char* env = getenv(OPENDLA_DUMP_LAYER);
        if (env && env[0] == '1'){
            /* debug */
            for(auto & node : finalEngineAST->nodes()){
                if(node->engineOpType() == nvdla::priv::engine_ast::EngineOpTypeEnum::CONVOLUTION_CONV){
                    auto * convNode = (nvdla::priv::engine_ast::ConvCoreNode *) node;
                    auto  rawWeights = convNode->params().rawWeights();
                    struct tensor weightTensor;
                    if(rawWeights.type == nvdla::DataType::INT8){
                        weightTensor.data_type = TENGINE_DT_INT8;
                        weightTensor.tensor_type = TENSOR_TYPE_VAR;
                        weightTensor.layout = TENGINE_LAYOUT_NCHW;
                        weightTensor.name = (char *)(convNode->canonicalNode()->name() + "_weights").c_str() ;
                        weightTensor.dim_num = 4;
                        weightTensor.dims[0] = convNode->auxSurfaces().back()->dimensions().n;
                        weightTensor.dims[1] = convNode->auxSurfaces().back()->dimensions().c;
                        weightTensor.dims[2] = convNode->auxSurfaces().back()->dimensions().h;
                        weightTensor.dims[3] = convNode->auxSurfaces().back()->dimensions().w;
                        weightTensor.elem_size = sizeof(int8_t);
                        weightTensor.elem_num = rawWeights.count;
                        weightTensor.data = (void *)rawWeights.values;
                        extract_feature_from_tensor_odla("weight", convNode->canonicalNode()->name().c_str(), &weightTensor);
                    }
                }else if(node->engineOpType() == nvdla::priv::engine_ast::EngineOpTypeEnum::SDP_BIAS){
                    auto * sdpBiasNode = (nvdla::priv::engine_ast::SDPBiasOpNode *) node;
                    auto  rawBias = sdpBiasNode->params().rawBiasData();
                    struct tensor biasTensor;

                    if(rawBias.type == nvdla::DataType::INT16){
                        biasTensor.data_type = TENGINE_DT_INT16;
                        biasTensor.tensor_type = TENSOR_TYPE_VAR;
                        biasTensor.layout = TENGINE_LAYOUT_NCHW;
                        biasTensor.name = (char *)(sdpBiasNode->canonicalNode()->name() + "bias").c_str() ;
                        biasTensor.dim_num = 4;
                        biasTensor.dims[0] = sdpBiasNode->getAuxDims().n;
                        biasTensor.dims[1] = sdpBiasNode->getAuxDims().c;
                        biasTensor.dims[2] = sdpBiasNode->getAuxDims().h;
                        biasTensor.dims[3] = sdpBiasNode->getAuxDims().w;
                        biasTensor.elem_size = sizeof(int16_t);
                        biasTensor.elem_num = rawBias.count;
                        biasTensor.data = (void *)rawBias.values;
                        extract_feature_from_tensor_odla("bias", sdpBiasNode->canonicalNode()->name().c_str(), &biasTensor);
                    }
                }else if(node->engineOpType() == nvdla::priv::engine_ast::EngineOpTypeEnum::CONVOLUTION_FC){
                    auto * convNode = (nvdla::priv::engine_ast::ConvCoreNode *) node;
                    auto  rawWeights = convNode->params().rawWeights();
                    struct tensor weightTensor;
                    if(rawWeights.type == nvdla::DataType::INT8){
                        weightTensor.data_type = TENGINE_DT_INT8;
                        weightTensor.tensor_type = TENSOR_TYPE_VAR;
                        weightTensor.layout = TENGINE_LAYOUT_NCHW;
                        weightTensor.name = (char *)(convNode->canonicalNode()->name() + "_weights").c_str() ;
                        weightTensor.dim_num = 4;
                        weightTensor.dims[0] = convNode->getAuxDims().n;
                        weightTensor.dims[1] = convNode->getAuxDims().c;
                        weightTensor.dims[2] = convNode->getAuxDims().h;
                        weightTensor.dims[3] = convNode->getAuxDims().w;
                        weightTensor.elem_size = sizeof(int8_t);
                        weightTensor.elem_num = rawWeights.count;
                        weightTensor.data = (void *)rawWeights.values;
                        extract_feature_from_tensor_odla("weight", convNode->canonicalNode()->name().c_str(), &weightTensor);
                    }
                }
            }
        }
        if (this->compiler.priv()->emit(finalEngineAST, this->loadable) != NvDlaSuccess) {
            fprintf(stderr, "Failed to emit Loadable Data. \n");
            return -1;
        }
        (void)this->loadable.priv()->serialize();

        // Get Loadable Image Size
        NvU64 loadableSize = 0;
        this->loadable.priv()->getSerializedDataSize(&loadableSize);
        if(!loadableSize){
            fprintf(stderr, "No Loadable Generated. \n");
            return -1;
        }
        NvU8 * buffer  = (NvU8 *)NvDlaAlloc(loadableSize);
        if (buffer == NULL) {
            fprintf(stderr, "Failed to allocate buffer for loadable. \n");
            return -1;
        }
        this->loadable.priv()->getSerializedData(buffer);

        env = getenv(OPENDLA_DUMP_LAYER);
        if (env && env[0] == '1'){
            NvDlaFileHandle file = 0;
            std::stringstream filename;
            filename << this->profile->getName() << "_subgraph_"<< (int)subgraph->index << ".nvdla" ;

#ifdef OPENDLA_DEBUG_DATA
            fprintf(stdout, "Dump loadable data to : %s . \n", filename.str().c_str());
#endif
            NvDlaFopen(filename.str().c_str(), NVDLA_OPEN_WRITE, &file);
            NvDlaFwrite(file, buffer, loadableSize);
        }
        // deserialize Loadable image
        this->runtime->load(buffer, 0);

        // Allocate Input & Output Buffer
        if (subgraph->input_num > 0)
        {
            this->inputBuffer.reserve(subgraph->input_num);
            for (uint8_t i = 0; i < subgraph->input_num; i++)
            {
                int ir_tensor_idx = subgraph->input_tensor_list[i];
                struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
                nvdla::IRuntime::NvDlaTensor tDesc;
                void *hMem = NULL;

                e = this->runtime->getInputTensorDesc(i, &tDesc);
                if (e != NvDlaSuccess){
                    fprintf(stderr, "getInputTensorDesc failed.\n");
                    return -1;
                }
                e = this->runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, &this->inputBuffer[i]);
                if (e != NvDlaSuccess){
                    fprintf(stderr, "allocateSystemMemory failed.\n");
                    return -1;
                }
                if (!this->runtime->bindInputTensor(i, hMem)){
                    fprintf(stderr, "bindInputTensor failed.\n");
                    return -1;
                }
            }
        }

        if(subgraph->output_num > 0){
            this->outputBuffer.reserve(subgraph->output_num);
            for (uint8_t i = 0; i < subgraph->output_num; i++){
                int ir_tensor_idx = subgraph->output_tensor_list[i];
                struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
                nvdla::IRuntime::NvDlaTensor tDesc;
                void *hMem = nullptr;

                e = this->runtime->getOutputTensorDesc(i, &tDesc);
                if (e != NvDlaSuccess){
                    fprintf(stderr, "getOutputTensorDesc (%d) failed.\n", i);
                    return -1;
                }
                e = this->runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, &this->outputBuffer[i]);
                if (e != NvDlaSuccess){
                    fprintf(stderr, "allocateSystemMemory failed.\n");
                    return -1;
                }
                if (!this->runtime->bindOutputTensor(i, hMem)){
                    fprintf(stderr, "bindOutputTensor failed.\n");
                    return -1;
                }
            }
        }
    }
fail:
    return 0;
};

int ODLAEngine::ODLAEngineRun(struct subgraph* subgraph)
{
#ifdef OPENDLA_DEBUG_DATA
    fprintf(stdout, "%s Entrance.\n", __func__ );
#endif

    NvDlaError e;
    struct graph* ir_graph = subgraph->graph;

    if (subgraph->input_num > 0)
    {
        for (uint8_t i = 0; i < subgraph->input_num; i++)
        {
            int ir_tensor_idx = subgraph->input_tensor_list[i];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            nvdla::IRuntime::NvDlaTensor tDesc;

            e = this->runtime->getInputTensorDesc(i, &tDesc);
            if (e != NvDlaSuccess){
                fprintf(stderr, "getInputTensorDesc failed.\n");
                return -1;
            }
            odla_input_data_convert(this->inputBuffer[i], ir_tensor->data, tDesc);
            const char* env = getenv(OPENDLA_DUMP_LAYER);
            if (env && env[0] == '1'){
                /* debug */
                if (ir_tensor->dim_num <= 5){
                    std::stringstream filename;
                    filename << "subgraph_" << (int)subgraph->index << "_tensor_in_";
                    extract_feature_from_tensor_odla(ir_tensor->name, filename.str().c_str(), ir_tensor);
                }
            }
        }

        struct timeval t1{}, t2{};
        double elapsedTime;

        gettimeofday(&t1, nullptr);
        this->runtime->submit();
        gettimeofday(&t2, nullptr);

#ifdef OPENDLA_DEBUG_DATA
        elapsedTime = t2.tv_sec - t1.tv_sec;
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000000.0;
        fprintf(stdout ,"NVDLA time: %f seconds\n", elapsedTime);
#endif


        /* download data */
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];

            nvdla::IRuntime::NvDlaTensor tDesc;
            e = this->runtime->getOutputTensorDesc(i, &tDesc);
            if (e != NvDlaSuccess){
                fprintf(stderr, "getInputTensorDesc failed.\n");
                return -1;
            }

            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            if (nullptr == ir_tensor->data)
            {
                auto data = (int8_t*)malloc(ir_tensor->elem_size * ir_tensor->elem_num);
                ir_tensor->data = data;

                ir_tensor->free_host_mem = 1;
                ir_tensor->internal_allocated = 0;
            }
            odla_output_data_convert(ir_tensor->data, this->outputBuffer[i], tDesc);
            const char* env = getenv(OPENDLA_DUMP_LAYER);
            if (env && env[0] == '1'){
                /* debug */
                if (ir_tensor->dim_num <= 5){
                    std::stringstream filename;
                    filename << "subgraph_" << (int)subgraph->index << "_tensor_out_";
                    extract_feature_from_tensor_odla(ir_tensor->name, filename.str().c_str(), ir_tensor);
                }
            }
        }
    }


    return 0;
}

void ODLAEngine::ODLAEnginePostRun()
{
    NvDlaError e = NvDlaSuccess;
    for (auto& ptr : this->host_buffer)
        sys_free(ptr);
    if(this->runtime){
        this->runtime->unload();
        nvdla::destroyRuntime(this->runtime);
    }
};
