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

#include "odla_executor.hpp"
#include "odla_define.h"

#ifdef ODLA_MODEL_CACHE
#include "defines.h"
#include "cstdlib"
#endif

#ifdef ODLA_MODEL_CACHE
#include <fstream>
#endif

NvDlaError ODLAEngine::ODLAConfigGenerate(){
    NvDlaError e = NvDlaSuccess;

    nvdla::IProfiler* profiler = nvdla::priv::ProfilerFactory::newProfiler().priv();;
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No profiler available.");
    }

    profile = nvdla::priv::ProfileFactory::priv(profiler->getProfile(this->tp_name.c_str()));
    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Couldn't find profile to compile.");
    }
    // 将 target_config_name 转换为 target_config 对象
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
    NvDlaError e = NvDlaSuccess;

    std::cout << "ODLA Engine Init " << std::endl;

    this->runtime = nvdla::createRuntime();
    this->compiler = nvdla::priv::CompilerFactory::newCompiler();
    this->loadable = nvdla::priv::LoadableFactory::LoadablePrivPair(0, 0);

    this->ODLAConfigGenerate();
    this->graph = new nvdla::priv::canonical_ast::Graph();
    if ( !this->graph )
    {
        fprintf(stderr, "Can't create a new Canonical AST.\n");
    }

//      Should be moved to Pre Run
//    // Init NVDLA GLOBAL_DRAM_POOL、 LOCAL_DRAM_POOL、LOCAL_CVSRAM_POOL
//    e = this->graph->initGraphResources();
//    if (e != NvDlaSuccess)
//    {
//        delete this->graph;
//        this->graph = NULL;
//        fprintf(stderr, "Couldn't initialize all graph resources.\n");
//    }
};


int ODLAEngine::ODLATensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type)
{
    std::cout << "ODLA TensorMap Entrance " << std::endl;
    auto iter = this->odla_tensor_map.find(ir_tensor_idx);

    if (this->odla_tensor_map.end() == iter)
    {
        if (spec_type == NVDLA_LAYER_TYPE_INTERP || spec_type == NVDLA_LAYER_TYPE_SLICE)
        {
            this->odla_tensor_map[ir_tensor_idx] = NULL;
            return 0;
        }
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        auto Dims = (unsigned int*)ir_tensor->dims;

        nvdla::DataType datatype;
        switch(ir_tensor->data_type)
        {
            // Why no Definition of DATATYPE?
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
            case TENGINE_DT_INT32:
                TLOG_ERR("Tensor date type: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
                break;
            default:
                TLOG_ERR("Tensor date type: Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",ir_tensor->name, ir_tensor->index, ir_tensor->data_type);
                break;
        }

        nvdla::Dims4 tensor_shape;

        struct node* ir_node = get_ir_graph_node(ir_graph, ir_tensor->producer);
        if (spec_type == NVDLA_LAYER_TYPE_PRELU)
        {
            tensor_shape.w = 1;
            tensor_shape.h = 1;
            tensor_shape.c = Dims[0];
            tensor_shape.n = 1;
        }
        else
        {
            if(ir_tensor->dim_num == 4){
                tensor_shape.n = Dims[0];
                tensor_shape.c = Dims[1];
                tensor_shape.h = Dims[2];
                tensor_shape.w = Dims[3];
            } else {
                fprintf(stderr, "Dims Number %d Not Supported. \n", ir_tensor->dim_num);
                return -1;
            }
        }

        /* set quant params */
//        tim::vx::Quantization vx_quant(tim::vx::QuantType::ASYMMETRIC, ir_tensor->scale,
//                                       ir_tensor->zero_point);


        /* create the odla tesnor */
        nvdla::priv::TensorFactory::TensorPrivPair t = nvdla::priv::TensorFactory::newTensor();
        nvdla::priv::Tensor* odla_tensor = NULL;
        if (spec_type == NVDLA_LAYER_TYPE_OUTPUT)
        {

            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kNW_OUTPUT);
            t.i()->setDataType(datatype);
            t.i()->setName(ir_tensor->name);
            t.i()->setChannelDynamicRange(-1, -16129, 16129);

            odla_tensor = t.priv();
        }
        else if (ir_tensor->data_type == TENGINE_DT_FP32)
        {
            fprintf(stderr, "DATATYPE TENGINE_DT_FP32 Not Supported. \n");
            return -1;
//            tim::vx::Quantization none_quant(tim::vx::QuantType::NONE, 1, 0);
//            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
//                                        tim::vx::TensorAttribute::CONSTANT, none_quant);
//            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_INPUT || spec_type == NVDLA_LAYER_TYPE_INPUT)
        {
            t.i()->setDimensions(tensor_shape);
            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
            t.i()->setTensorType(nvdla::kNW_INPUT);
            t.i()->setDataType(datatype);
            t.i()->setName(ir_tensor->name);
            t.i()->setChannelDynamicRange(-1, -127, 127);
            odla_tensor = t.priv();
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_VAR)
        {
            const char* env = getenv(TENGINE_DUMP_LAYER);
            if (env && env[0] == '1')
            {
                t.i()->setDimensions(tensor_shape);
                t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
                t.i()->setTensorType(nvdla::kNW_OUTPUT);
                t.i()->setDataType(datatype);
                t.i()->setName(ir_tensor->name);
                t.i()->setChannelDynamicRange(-1, -16129, 16129);
                odla_tensor = t.priv();
            }
            else
            {
                t.i()->setDimensions(tensor_shape);
                t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
                t.i()->setTensorType(nvdla::kUNKNOWN);       // May Not be Right
                t.i()->setDataType(datatype);
                t.i()->setName(ir_tensor->name);
                t.i()->setChannelDynamicRange(-1, -16129, 16129);
                odla_tensor = t.priv();

//                tim::vx::TensorSpec vx_spec(datatype, vx_shape,
//                                            tim::vx::TensorAttribute::TRANSIENT, vx_quant);
            }
        }
        else if (ir_tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            fprintf(stderr, "Tensor_Type Constant Not Supported. \n");
//            return -1;
//            t.i()->setDimensions(tensor_shape);
//            t.i()->setDataFormat(NVDLA_DATA_FORMAT_NCHW);
//            t.i()->setTensorType(nvdla::kBIAS);       // May Not be Right
//            t.i()->setDataType(datatype);
//
//            odla_tensor = std::shared_ptr<nvdla::priv::Tensor>(t.priv());
//            tim::vx::TensorSpec vx_spec(datatype, vx_shape,
//                                        tim::vx::TensorAttribute::CONSTANT, vx_quant);
//            vx_tensor = this->graph->CreateTensor(vx_spec, ir_tensor->data);
        }

        this->odla_tensor_map[ir_tensor_idx] = odla_tensor;
    }

    return 0;
}

int ODLAEngine::Build(struct subgraph* subgraph)
{
    std::cout << "ODLA Build Entrance " << std::endl;
    struct graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;

        switch (op_type)
        {
            case OP_INPUT:
                continue;
            case OP_POOL:
                this->AddPoolingNode(ir_node);
                break;
            default:
                fprintf(stderr, "Tengine OpenDLA: Cannot support OP(%d).\n", ir_node->index);
                break;
        }
    }

    this->graph->scoredOrdering()->generate();
    this->graph->markClean();
    return 0;
}


int ODLAEngine::ODLAEnginePreRun(struct subgraph* subgraph)
{
    std::cout << "ODLA PreRun Entrance " << std::endl;

    NvDlaError e = NvDlaSuccess;
    struct graph* ir_graph = subgraph->graph;
    /* Add OpenDLA Tensor */
    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        this->ODLATensorMap(ir_graph, ir_tensor_idx, NVDLA_LAYER_TYPE_INPUT);
    }
    for (uint8_t i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        this->ODLATensorMap(ir_graph, ir_tensor_idx, NVDLA_LAYER_TYPE_OUTPUT);
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
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], NVDLA_LAYER_TYPE_DECONVOLUTION);
            }
            else
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], NVDLA_LAYER_TYPE_CONVOLUTION);
            }
            if (ir_node->input_num > 2)
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[2], NVDLA_LAYER_TYPE_CONV_BIAS);
            }
        }
        else if (ir_node->op.type == OP_PRELU)
        {
            this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], NVDLA_LAYER_TYPE_PRELU);
        }
        else if (ir_node->op.type == OP_INTERP)
        {
            if (ir_node->input_num == 3)
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], NVDLA_LAYER_TYPE_INTERP);
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[2], NVDLA_LAYER_TYPE_INTERP);
            }
            else if (ir_node->input_num == 2)
            {
                this->ODLATensorMap(ir_graph, ir_node->input_tensors[1], NVDLA_LAYER_TYPE_INTERP);
            }
        }
        else if (ir_node->op.type == OP_SLICE)
        {
            if (ir_node->input_num > 1)
            {
                for (int FI = 1; FI < ir_node->input_num; FI++)
                {
                    this->ODLATensorMap(ir_graph, ir_node->input_tensors[FI], NVDLA_LAYER_TYPE_SLICE);
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
        auto engineASTGraph = nvdla::priv::engine_ast::generateGraph(this->profile, this->targetConfig, this->graph);
        // Optimize pass
        engineASTGraph = this->compiler.priv()->registerBuffers(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->preProcessAuxData(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->updateScalingFactors(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->quantizeAuxData(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->fuseOnTheFlyNodes(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->handleLowPrecisionConversions(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->translateAuxData(engineASTGraph);
        engineASTGraph = this->compiler.priv()->reserveBuffers(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->splitNodes(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->fuseSubEngineOps(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->boundGraph(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->handleMultiBatch(engineASTGraph);
//        engineASTGraph = this->compiler.priv()->enableCopyOutDebugSurfaces(engineASTGraph);

        // generate Loadable Task info
        nvdla::priv::engine_ast::NodeSequence topological_order;
        engineASTGraph = this->compiler.priv()->generateDependencyParams(engineASTGraph, topological_order);
        if (this->compiler.priv()->emit(engineASTGraph, loadable) != NvDlaSuccess) {
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

        // deserialize Loadable image
        this->runtime->load(buffer, 0);

        // Allocate Input & Output Buffer
        if (subgraph->input_num > 0)
        {
            for (uint8_t i = 0; i < subgraph->input_num; i++)
            {
                int ir_tensor_idx = subgraph->input_tensor_list[i];
                struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
                nvdla::IRuntime::NvDlaTensor tDesc;
                void *hMem = NULL;

                this->runtime->getInputTensorDesc(i, &tDesc);
                if (e != NvDlaSuccess){
                    fprintf(stderr, "getInputTensorDesc failed.\n");
                    return -1;
                }
                e = this->runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, &inputBuffer);
                if (e != NvDlaSuccess){
                    fprintf(stderr, "allocateSystemMemory failed.\n");
                    return -1;
                }
                this->inputHandle = (NvU8 *)hMem;
                if (!this->runtime->bindInputTensor(i, hMem)){
                    fprintf(stderr, "bindInputTensor failed.\n");
                    return -1;
                }
            }
        }

        if(subgraph->output_num > 0){
            for (uint8_t i = 0; i < subgraph->output_num; i++){
                int ir_tensor_idx = subgraph->output_tensor_list[i];
                struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
                nvdla::IRuntime::NvDlaTensor tDesc;
                void *hMem = NULL;

                this->runtime->getOutputTensorDesc(i, &tDesc);
                if (e != NvDlaSuccess){
                    fprintf(stderr, "getOutputTensorDesc failed.\n");
                    return -1;
                }
                e = this->runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, &outputBuffer);
                if (e != NvDlaSuccess){
                    fprintf(stderr, "allocateSystemMemory failed.\n");
                    return -1;
                }
                this->outputHandle = (NvU8 *)hMem;
                if (!this->runtime->bindOutputTensor(i, hMem)){
                    fprintf(stderr, "bindOutputTensor failed.\n");
                    return -1;
                }
            }
        }
    }

    return 0;
};

int ODLAEngine::ODLAEngineRun(struct subgraph* subgraph)
{
    std::cout << "ODLA EngineRun Entrance " << std::endl;

    NvDlaError e = NvDlaSuccess;
    struct graph* ir_graph = subgraph->graph;

    if (subgraph->input_num > 0)
    {
        for (uint8_t i = 0; i < subgraph->input_num; i++)
        {
            int ir_tensor_idx = subgraph->input_tensor_list[i];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            memcpy(this->inputBuffer, ir_tensor->data,ir_tensor->elem_num * ir_tensor->elem_size);
        }

        struct timeval t1, t2;
        double elapsedTime;

        this->runtime->initEMU();
        gettimeofday(&t1, NULL);
        this->runtime->submit();
        gettimeofday(&t2, NULL);
        this->runtime->stopEMU();

        elapsedTime = t2.tv_sec - t1.tv_sec;
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000000.0;
        fprintf(stdout ,"NVDLA time: %f seconds\n", elapsedTime);

        /* download data */
        for (uint8_t i = 0; i < subgraph->output_num; i++)
        {
            int ir_tensor_idx = subgraph->output_tensor_list[i];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
            if (nullptr == ir_tensor->data)
            {
                auto data = (int8_t*)malloc(ir_tensor->elem_size * ir_tensor->elem_num);
                ir_tensor->data = data;

                ir_tensor->free_host_mem = 1;
                ir_tensor->internal_allocated = 0;
            }
            memcpy(ir_tensor->data, outputBuffer, ir_tensor->elem_size * ir_tensor->elem_num);
        }
    }

    // Free Buffer
    nvdla::IRuntime::NvDlaTensor tDesc;
    e = runtime->getInputTensorDesc(0, &tDesc);
    if (e != NvDlaSuccess){
        fprintf(stderr, "Failed to getInputTensorDesc.\n");
        return -1;
    }
    if (!this->inputHandle){
        this->runtime->freeSystemMemory(this->inputHandle, tDesc.bufferSize);
    }
    e = runtime->getOutputTensorDesc(0, &tDesc);
    if (e != NvDlaSuccess){
        fprintf(stderr, "Failed to getOutputTensorDesc.\n");
        return -1;
    }
    if (!this->outputHandle){
        this->runtime->freeSystemMemory(this->outputHandle, tDesc.bufferSize);
    }
    return 0;
}

void ODLAEngine::ODLAEnginePostRun()
{
    std::cout << "ODLA PostRun Entrance " << std::endl;

};
