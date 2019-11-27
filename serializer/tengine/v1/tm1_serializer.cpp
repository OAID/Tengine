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
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <string.h>

#include "tengine_c_api.h"
#include "exec_attr.hpp"
#include "data_type.hpp"
#include "operator_manager.hpp"
#include "static_graph.hpp"
#include "graph.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include "compiler.hpp"

#include "tm1_format.h"
#include "tm1_serializer.hpp"
#include "tm1_op_serializer.hpp"

namespace TEngine {

namespace TMSerializer1 {

bool TmSerializer1::IsSaveString(void)
{
    const char* env = std::getenv("TM_WITH_STRING");

    if(env)
        return true;
    else
        return false;
}

bool TmSerializer1::IsSaveData(void)
{
    const char* env = std::getenv("TM_FOR_BENCHMARK");

    if(env)
        return false;
    else
        return true;
}

tm_uoffset_t TmSerializer1::SaveTmTensor(void* const start_ptr, tm_uoffset_t* cur_pos, Tensor* tensor,
                                         unsigned int tensor_id, unsigned int buffer_id)
{
    TM_Tensor tm_tensor;
    tm_tensor.tensor_id = tensor_id;
    tm_tensor.buffer_id = buffer_id;
    tm_tensor.type = tensor->GetType();

    bool tm_with_string = IsSaveString();

    if(tm_with_string)
    {
        std::string name = tensor->GetName();
        TM_String tensor_name;
        tensor_name.size = name.size();
        tensor_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, name.c_str(), tensor_name.size);
        tm_tensor.offset_s_tname = WriteTmObject(start_ptr, cur_pos, &tensor_name, sizeof(TM_String));
    }
    else
        tm_tensor.offset_s_tname = NOT_SET;

    const std::string& data_type = DataType::GetTypeName(tensor->GetDataType());
    if(data_type == "float32")
        tm_tensor.data_type = TM_DT_FLOAT32;
    else if(data_type == "float16")
        tm_tensor.data_type = TM_DT_FLOAT16;
    else if(data_type == "int")
        tm_tensor.data_type = TM_DT_INT32;
    else if(data_type == "int8")
        tm_tensor.data_type = TM_DT_INT8;

    /* Get the dims of the tensor */
    TShape& shape = tensor->GetShape();
    std::vector<int>& dim = shape.GetDim();
    if(dim.size())
    {
        /* Write the vector of dims */
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * dim.size();
        TM_Vector_dims* v_dims = ( TM_Vector_dims* )malloc(vector_size);
        v_dims->v_num = dim.size();
        for(unsigned int i = 0; i < dim.size(); i++)
        {
            v_dims->dims[i] = dim[i];
        }
        tm_tensor.offset_vd_dims = WriteTmObject(start_ptr, cur_pos, v_dims, vector_size);
        free(v_dims);
    }
    else
        tm_tensor.offset_vd_dims = NOT_SET;

    /* Write the tensor */
    return WriteTmObject(start_ptr, cur_pos, &tm_tensor, sizeof(TM_Tensor));
}

tm_uoffset_t TmSerializer1::SaveTmNode(void* const start_ptr, tm_uoffset_t* cur_pos, Node* node,
                                       name_map_t& tensor_name_map)
{
    TM_Node tm_node;
    tm_node.node_id = node->GetNodeIndex();
    tm_node.dynamic_shape = node->IsDynamicShape();

    bool tm_with_string = IsSaveString();

    if(tm_with_string)
    {
        std::string name = node->GetName();
        TM_String node_name;
        node_name.size = name.size();
        node_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, name.c_str(), node_name.size);
        tm_node.offset_s_nname = WriteTmObject(start_ptr, cur_pos, &node_name, sizeof(TM_String));
    }
    else
        tm_node.offset_s_nname = NOT_SET;

    unsigned int input_num = node->GetInputNum();
    unsigned int output_num = node->GetOutputNum();

    if(input_num)
    {
        /* Write the vector of input indices */
        size_t vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * input_num;
        TM_Vector_indices* v_input_indices = ( TM_Vector_indices* )malloc(vector_size);
        v_input_indices->v_num = input_num;
        for(unsigned int i = 0; i < input_num; i++)
        {
            Tensor* p_tensor = node->GetInputTensor(i);
            v_input_indices->indices[i] = tensor_name_map[p_tensor->GetName()];
        }
        tm_node.offset_vi_input_tensors = WriteTmObject(start_ptr, cur_pos, v_input_indices, vector_size);
        free(v_input_indices);
    }
    else
        tm_node.offset_vi_input_tensors = NOT_SET;

    if(output_num)
    {
        /* Write the vector of output indices */
        size_t vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * output_num;
        TM_Vector_indices* v_output_indices = ( TM_Vector_indices* )malloc(vector_size);
        v_output_indices->v_num = output_num;
        for(unsigned int i = 0; i < output_num; i++)
        {
            Tensor* p_tensor = node->GetOutputTensor(i);
            v_output_indices->indices[i] = tensor_name_map[p_tensor->GetName()];
        }
        tm_node.offset_vi_output_tensors = WriteTmObject(start_ptr, cur_pos, v_output_indices, vector_size);
        free(v_output_indices);
    }
    else
        tm_node.offset_vi_output_tensors = NOT_SET;

    tm_node.offset_t_operator = SaveTmOperator(start_ptr, cur_pos, node->GetOp());

    /* Write the node */
    return WriteTmObject(start_ptr, cur_pos, &tm_node, sizeof(TM_Node));
}

tm_uoffset_t TmSerializer1::SaveTmSubgraph(void* const start_ptr, tm_uoffset_t* cur_pos, Graph* graph)
{
    TM_Subgraph tm_subgraph;
    tm_subgraph.subgraph_id = 0; /* subgraph_id starts from 0 */
    tm_subgraph.offset_s_sname = NOT_SET;

    unsigned int tensor_num = 0;
    unsigned int buffer_num = 0;
    std::vector<Tensor*> tensor_ptrs;
    std::vector<void*> buf_ptrs;
    std::vector<unsigned int> buf_sizes;
    name_map_t tensor_name_map; /* map of tensor name and tensor index */
    bool tm_no_data = !IsSaveData();

    /* Write the nodes */
    size_t vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * graph->seq_nodes.size();
    TM_Vector_offsets* v_nodes = ( TM_Vector_offsets* )malloc(vector_size);
    v_nodes->v_num = graph->seq_nodes.size();
    for(unsigned int i = 0; i < graph->seq_nodes.size(); i++)
    {
        Node* p_node = graph->seq_nodes[i];
        for(unsigned int k = 0; k < p_node->GetOutputNum(); k++)
        {
            Tensor* p_tensor = p_node->GetOutputTensor(k);
            tensor_ptrs.push_back(p_tensor);
            tensor_name_map[p_tensor->GetName()] = tensor_num;
            tensor_num++;
        }
        v_nodes->offsets[i] = SaveTmNode(start_ptr, cur_pos, p_node, tensor_name_map);
    }
    /* Write the vector of nodes */
    tm_subgraph.offset_vo_seq_nodes = WriteTmObject(start_ptr, cur_pos, v_nodes, vector_size);

    /* Write the tensors */
    vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * tensor_num;
    TM_Vector_offsets* v_tensors = ( TM_Vector_offsets* )malloc(vector_size);
    v_tensors->v_num = tensor_num;
    for(unsigned int i = 0; i < tensor_num; i++)
    {
        Tensor* p_tensor = tensor_ptrs[i];
        if(p_tensor->GetType() == kConstTensor)
        {
            buf_ptrs.push_back(p_tensor->GetMemAddr());
            buf_sizes.push_back(p_tensor->GetTotalSize());
            buffer_num++;
        }

        v_tensors->offsets[i] = SaveTmTensor(start_ptr, cur_pos, p_tensor, i, buffer_num - 1);
    }
    /* Write the vector of tensors */
    tm_subgraph.offset_vo_tensors = WriteTmObject(start_ptr, cur_pos, v_tensors, vector_size);

    /* Write the buffers */
    vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * buffer_num;
    TM_Vector_offsets* v_buffers = ( TM_Vector_offsets* )malloc(vector_size);
    v_buffers->v_num = buffer_num;
    for(unsigned int i = 0; i < buffer_num; i++)
    {
        TM_Buffer tm_buf;
        tm_buf.size = buf_sizes[i];

        if(tm_no_data)
        {
            /* TM_FOR_BENCHMARK environment variable exists. Not write buf data into the tm file */
            tm_buf.offset_data = NOT_SET;
        }
        else
        {
            /* TM_FOR_BENCHMARK environment variable does not exist */
            tm_buf.offset_data =
                WriteTmFileAlign1(start_ptr, cur_pos, reinterpret_cast<const uint8_t*>(buf_ptrs[i]), tm_buf.size);
        }
        v_buffers->offsets[i] = WriteTmObject(start_ptr, cur_pos, &tm_buf, sizeof(TM_Buffer));
    }
    /* Write the vector of buffers */
    tm_subgraph.offset_vo_buffers = WriteTmObject(start_ptr, cur_pos, v_buffers, vector_size);

    /* Write the vector of input indices */
    vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * graph->input_nodes.size();
    TM_Vector_indices* v_input_indices = ( TM_Vector_indices* )malloc(vector_size);
    v_input_indices->v_num = graph->input_nodes.size();
    for(unsigned int i = 0; i < graph->input_nodes.size(); i++)
    {
        v_input_indices->indices[i] = graph->input_nodes[i]->GetNodeIndex();
    }
    tm_subgraph.offset_vi_input_indices = WriteTmObject(start_ptr, cur_pos, v_input_indices, vector_size);

    /* Write the vector of output indices */
    vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * graph->output_nodes.size();
    TM_Vector_indices* v_output_indices = ( TM_Vector_indices* )malloc(vector_size);
    v_output_indices->v_num = graph->output_nodes.size();
    for(unsigned int i = 0; i < graph->output_nodes.size(); i++)
    {
        v_output_indices->indices[i] = graph->output_nodes[i]->GetNodeIndex();
    }
    tm_subgraph.offset_vi_output_indices = WriteTmObject(start_ptr, cur_pos, v_output_indices, vector_size);

    /* Write the subgraph */
    tm_uoffset_t ret = WriteTmObject(start_ptr, cur_pos, &tm_subgraph, sizeof(TM_Subgraph));

    /* Free the memory of vectors */
    free(v_tensors);
    free(v_buffers);
    free(v_nodes);
    free(v_input_indices);
    free(v_output_indices);

    return ret;
}

bool TmSerializer1::SaveModelIntoMem(void* start_ptr, Graph* graph, uint32_t* tm_model_size)
{
    bool tm_with_string = IsSaveString();

    tm_uoffset_t cur_pos = sizeof(TM_Header);

    /* Define the TM_Header object */
    TM_Header header;
    header.ver_main = TM_FILE_VER_MAIN;
    header.ver_sub = TM_FILE_VER_SUB;
    header.ver_compile = TM_FILE_VER_COMPILE;

    /* Define the TM_Model object */
    TM_Model tm_model;
    if(tm_with_string)
    {
        const std::string& fname = graph->GetName();
        TM_String model_name;
        model_name.size = fname.size();
        model_name.offset_data = WriteTmFileAlign1(start_ptr, &cur_pos, fname.c_str(), model_name.size);
        tm_model.offset_s_mname = WriteTmObject(start_ptr, &cur_pos, &model_name, sizeof(TM_String));
    }
    else
        tm_model.offset_s_mname = NOT_SET;

    /* Write the subgraphs */
    /* Only 1 subgraph is supported currently */
    size_t vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * 1;
    TM_Vector_offsets* v_subgraphs = ( TM_Vector_offsets* )malloc(vector_size);
    v_subgraphs->v_num = 1;
    v_subgraphs->offsets[0] = SaveTmSubgraph(start_ptr, &cur_pos, graph);

    /* Write the vector of subgraphs */
    tm_model.offset_vo_subgraphs = WriteTmObject(start_ptr, &cur_pos, v_subgraphs, vector_size);

    /* Write the model */
    header.offset_root = WriteTmObject(start_ptr, &cur_pos, &tm_model, sizeof(TM_Model));
    *tm_model_size = cur_pos;

    /* Write the header */
    cur_pos = 0;
    WriteTmObject(start_ptr, &cur_pos, &header, sizeof(TM_Header));

    free(v_subgraphs);

    return true;
}

bool TmSerializer1::LoadNode(StaticGraph* graph, StaticNode* node, const TM_Node* tm_node, void* mmap_buf)
{
    if(tm_node->offset_vi_input_tensors != NOT_SET)
    {
        const TM_Vector_indices* v_input_tensors =
            GetTmPtr<TM_Vector_indices>(mmap_buf, tm_node->offset_vi_input_tensors);

        /* Set the input tensors to the node */
        for(unsigned int i = 0; i < v_input_tensors->v_num; i++)
        {
            StaticTensor* tensor = graph->tensor_list[v_input_tensors->indices[i]].get();
            if(!tensor)
            {
                LOG_ERROR() << "The input tensor not exist: " << v_input_tensors->indices[i] << "\n";
                return false;
            }
            AddNodeInputTensor(node, tensor);
        }
    }

    if(tm_node->offset_vi_output_tensors != NOT_SET)
    {
        const TM_Vector_indices* v_output_tensors =
            GetTmPtr<TM_Vector_indices>(mmap_buf, tm_node->offset_vi_output_tensors);

        /* Set the output tensors to the node */
        for(unsigned int i = 0; i < v_output_tensors->v_num; i++)
        {
            StaticTensor* tensor = graph->tensor_list[v_output_tensors->indices[i]].get();
            if(!tensor)
            {
                LOG_ERROR() << "The output tensor not exist: " << v_output_tensors->indices[i] << "\n";
                return false;
            }
            AddNodeOutputTensor(node, tensor);
        }
    }
    return true;
}

bool TmSerializer1::LoadTensor(StaticGraph* graph, const TM_Tensor* tm_tensor, const TM_Buffer* tm_buf, void* mmap_buf)
{
    /* Set the tensor name */
    int idx = tm_tensor->tensor_id;
    std::string tm_tensor_name;
    if(tm_tensor->offset_s_tname == NOT_SET)
        tm_tensor_name = "tensor_" + std::to_string(idx);
    else
    {
        const TM_String* tm_string = GetTmPtr<TM_String>(mmap_buf, tm_tensor->offset_s_tname);
        tm_tensor_name.assign(GetTmPtr<char>(mmap_buf, tm_string->offset_data), tm_string->size);
    }

    /* Create the static tensor */
    StaticTensor* tensor;
    if(tm_tensor->type == kConstTensor)
        tensor = CreateStaticConstTensor(graph, tm_tensor_name);
    else
        tensor = CreateStaticTensor(graph, tm_tensor_name);
    if(!tensor)
    {
        LOG_ERROR() << "Create static const tensor failed: " << tm_tensor_name << "\n";
        return false;
    }

    /* Set the dims */
    if(tm_tensor->offset_vd_dims != NOT_SET)
    {
        const TM_Vector_dims* v_dims = GetTmPtr<TM_Vector_dims>(mmap_buf, tm_tensor->offset_vd_dims);
        if(!v_dims || !(v_dims->v_num))
        {
            LOG_ERROR() << "Get tensor dims failed\n";
            return false;
        }
        std::vector<int> dims;
        for(unsigned int i = 0; i < v_dims->v_num; i++)
            dims.push_back(v_dims->dims[i]);
        SetTensorDim(tensor, dims);
    }

    /* Set the data type */
    if(tm_tensor->data_type == TM_DT_FLOAT32)
        SetTensorDataType(tensor, DataType::GetTypeID("float32"));
    else if(tm_tensor->data_type == TM_DT_FLOAT16)
        SetTensorDataType(tensor, DataType::GetTypeID("float16"));
    else if(tm_tensor->data_type == TM_DT_INT32)
        SetTensorDataType(tensor, DataType::GetTypeID("int"));
    else if(tm_tensor->data_type == TM_DT_INT8)
        SetTensorDataType(tensor, DataType::GetTypeID("int8"));

    /* Set the memory size and pointer */
    if(tm_tensor->type == kConstTensor)
    {
        SetTensorSize(tensor, tm_buf->size);
        void* buf = malloc(tm_buf->size);
        if(tm_buf->offset_data != NOT_SET)
        {
            memcpy(buf, GetTmPtr<void>(mmap_buf, tm_buf->offset_data), tm_buf->size);
        }

        SetConstTensorBuffer(tensor, buf);
        SetConstTensorFileLocation(tensor, -1, 0);
    }

    return true;
}

bool TmSerializer1::LoadGraph(StaticGraph* graph, const TM_Model* tm_model, void* mmap_buf)
{
    const TM_Vector_offsets* v_graphs = GetTmPtr<TM_Vector_offsets>(mmap_buf, tm_model->offset_vo_subgraphs);
    const TM_Subgraph* tm_graph = GetTmPtr<TM_Subgraph>(mmap_buf, v_graphs->offsets[0]);

    const TM_Vector_offsets* v_nodes = GetTmPtr<TM_Vector_offsets>(mmap_buf, tm_graph->offset_vo_seq_nodes);
    const TM_Vector_offsets* v_tensors = GetTmPtr<TM_Vector_offsets>(mmap_buf, tm_graph->offset_vo_tensors);
    const TM_Vector_offsets* v_buffers = GetTmPtr<TM_Vector_offsets>(mmap_buf, tm_graph->offset_vo_buffers);

    /* Load const tensors */
    for(unsigned int i = 0; i < v_tensors->v_num; i++)
    {
        const TM_Tensor* tm_tensor = GetTmPtr<TM_Tensor>(mmap_buf, v_tensors->offsets[i]);
        const TM_Buffer* tm_buf;
        if(tm_tensor->type == kConstTensor)
            tm_buf = GetTmPtr<TM_Buffer>(mmap_buf, v_buffers->offsets[tm_tensor->buffer_id]);
        else
            tm_buf = nullptr;
        LoadTensor(graph, tm_tensor, tm_buf, mmap_buf);
    }

    /* Create static nodes */
    unsigned int i;
    for(i = 0; i < v_nodes->v_num; i++)
    {
        const TM_Node* tm_node = GetTmPtr<TM_Node>(mmap_buf, v_nodes->offsets[i]);
        int idx = tm_node->node_id;
        std::string tm_node_name;
        if(tm_node->offset_s_nname == NOT_SET)
            tm_node_name = "node_" + std::to_string(idx);
        else
        {
            const TM_String* tm_string = GetTmPtr<TM_String>(mmap_buf, tm_node->offset_s_nname);
            tm_node_name.assign(GetTmPtr<char>(mmap_buf, tm_string->offset_data), tm_string->size);
        }

        const TM_Operator* tm_operator = GetTmPtr<TM_Operator>(mmap_buf, tm_node->offset_t_operator);
        const std::string& tm_op_name = GetOpStr(tm_operator->operator_type);

        if(!FindOpLoadMethod(tm_op_name))
        {
            LOG_ERROR() << "cannot find load function for operator: " << tm_op_name << "\n";
            break;
        }

        StaticNode* node = CreateStaticNode(graph, tm_node_name);
        if(!LoadNode(graph, node, tm_node, mmap_buf))
            break;

        op_load_t op_func = any_cast<op_load_t>(GetOpLoadMethod(tm_op_name));

        if(!op_func(graph, node, mmap_buf, tm_operator))
            break;

        /* Set the dynamic shape of the operator */
        node->op->dynamic_shape = tm_node->dynamic_shape;
    }

    if(i < v_nodes->v_num)
        return false;

    const TM_Vector_indices* v_input_nodes = GetTmPtr<TM_Vector_indices>(mmap_buf, tm_graph->offset_vi_input_indices);
    const TM_Vector_indices* v_output_nodes = GetTmPtr<TM_Vector_indices>(mmap_buf, tm_graph->offset_vi_output_indices);

    /* Set the input nodes */
    for(unsigned int i = 0; i < v_input_nodes->v_num; i++)
    {
        StaticNode* node = graph->node_list[v_input_nodes->indices[i]].get();
        if(!node)
        {
            LOG_ERROR() << "Input node #" << v_input_nodes->indices[i] << " not exist\n";
            return false;
        }
        AddGraphInputNode(graph, node);
    }

    /* Set the output nodes */
    for(unsigned int i = 0; i < v_output_nodes->v_num; i++)
    {
        StaticNode* node = graph->node_list[v_output_nodes->indices[i]].get();
        if(!node)
        {
            LOG_ERROR() << "Output node #" << v_output_nodes->indices[i] << " not exist\n";
            return false;
        }
        AddGraphOutputNode(graph, node);
    }

    return true;
}

bool TmSerializer1::LoadModelFromMem(void* mmap_buf, StaticGraph* graph)
{
    const TM_Header* tm_header = reinterpret_cast<const TM_Header*>(mmap_buf);
    /* Check the version of tm file format */
    if(tm_header->ver_main != TM_FILE_VER_MAIN || tm_header->ver_sub != TM_FILE_VER_SUB ||
       tm_header->ver_compile != TM_FILE_VER_COMPILE)
    {
        LOG_ERROR() << "Wrong version of tm file\n";
        return false;
    }

    const TM_Model* tm_model = GetTmPtr<TM_Model>(mmap_buf, tm_header->offset_root);
    if(tm_model->offset_s_mname == NOT_SET)
    {
        SetGraphIdentity(graph, "tengine", "tengine_model", "0");
    }
    else
    {
        std::string tm_model_name;
        const TM_String* tm_string = GetTmPtr<TM_String>(mmap_buf, tm_model->offset_s_mname);
        tm_model_name.assign(GetTmPtr<char>(mmap_buf, tm_string->offset_data), tm_string->size);
        SetGraphIdentity(graph, "tengine", tm_model_name, "0");
    }

    SetModelFormat(graph, MODEL_FORMAT_TENGINE);
    SetGraphLayout(graph, TENGINE_LAYOUT_NCHW);
    SetModelLayout(graph, TENGINE_LAYOUT_NCHW);

    if(LoadGraph(graph, tm_model, mmap_buf))
        return true;
    else
        return false;
}

bool TmSerializerRegisterOpLoader1(void)
{
    TmSerializerPtr serializer;

    if(!TmSerializerManager::SafeGet("tm_v1", serializer))
        return false;

    TmSerializer1* p_tengine = dynamic_cast<TmSerializer1*>(serializer.get());

    for(int i = 0; i < TM_OPTYPE_NUM; i++)
    {
        p_tengine->RegisterOpLoadMethod(GetOpStr(i), op_load_t(LoadTmOpFunc(i)));
    }

    return true;
}

}    // namespace TMSerializer1

}    // namespace TEngine
