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
 * Copyright (c) 2019, Open AI Lab
 * Author: jingyou@openailab.com
 */
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <string.h>
#include <typeinfo>
#include <assert.h>

#include "operator_manager.hpp"
#include "static_graph.hpp"
#include "graph.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include "compiler.hpp"

#include "tm2_format.h"
#include "tm2_serializer.hpp"
#include "tm2_op_serializer.hpp"

#define TYPE_INFO_INT32 1
#define TYPE_INFO_UINT32 2
#define TYPE_INFO_FLOAT 3
#define TYPE_INFO_POINTER 4
#define TYPE_INFO_GENERIC 5

namespace TEngine {

extern int NodeSetParamGeneric(void* node, const char* param_name, const char* type_name, const void* param_val,
                               int size);
extern int NodeAddParamGeneric(void* node, const char* param_name, const char* type_name, int param_size);
}    // namespace TEngine

using namespace TEngine;

namespace TEngine {

namespace TMSerializer2 {

static int typename_to_int(const char* name)
{
    if(name == nullptr)
        return TYPE_INFO_POINTER;

    if(!strcmp(name, typeid(int).name()))
        return TYPE_INFO_INT32;
    if(!strcmp(name, typeid(unsigned int).name()))
        return TYPE_INFO_UINT32;
    if(!strcmp(name, typeid(float).name()))
        return TYPE_INFO_FLOAT;

    return TYPE_INFO_GENERIC;
}

static const char* int_to_typename(int id)
{
    switch(id)
    {
        case TYPE_INFO_INT32:
            return typeid(int).name();
        case TYPE_INFO_UINT32:
            return typeid(unsigned int).name();
        case TYPE_INFO_FLOAT:
            return typeid(float).name();
        case TYPE_INFO_POINTER:
        case TYPE_INFO_GENERIC:
        default:
            return nullptr;
    }
}

bool TmSerializer2::IsSaveString(void)
{
    const char* env = std::getenv("TM_NO_STRING");

    if(env)
        return false;
    else
        return true;
}

bool TmSerializer2::IsSaveData(void)
{
    const char* env = std::getenv("TM_FOR_BENCHMARK");

    if(env)
        return false;
    else
        return true;
}

tm_uoffset_t TmSerializer2::SaveTmTensor(void* const start_ptr, tm_uoffset_t* cur_pos, Tensor* tensor,
                                         unsigned int tensor_id, unsigned int buffer_id)
{
    TM2_Tensor tm_tensor;
    tm_tensor.tensor_id = tensor_id;
    tm_tensor.buffer_id = buffer_id;
    tm_tensor.type = tensor->GetType();
    tm_tensor.data_type = tensor->GetDataType();
    tm_tensor.layout = (tensor->GetShape()).GetDataLayout();

    bool tm_with_string = IsSaveString();

    if(tm_with_string)
    {
        std::string name = tensor->GetName();
        TM2_String tensor_name;
        tensor_name.size = name.size() + 1;    // including trailing \0
        tensor_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, name.c_str(), tensor_name.size);
        tm_tensor.offset_s_tname = WriteTmObject(start_ptr, cur_pos, &tensor_name, sizeof(TM2_String));
    }
    else
        tm_tensor.offset_s_tname = TM2_NOT_SET;

    /* Get the dims of the tensor */
    TShape& shape = tensor->GetShape();
    std::vector<int>& dim = shape.GetDim();
    size_t vector_size;
    if(dim.size())
    {
        /* Write the vector of dims */
        vector_size = sizeof(tm_size_t) + sizeof(int32_t) * dim.size();
        TM2_Vector_dims* v_dims = ( TM2_Vector_dims* )malloc(vector_size);
        v_dims->v_num = dim.size();
        for(unsigned int i = 0; i < dim.size(); i++)
        {
            v_dims->dims[i] = dim[i];
        }
        tm_tensor.offset_vd_dims = WriteTmObject(start_ptr, cur_pos, v_dims, vector_size);
        free(v_dims);
    }
    else
        tm_tensor.offset_vd_dims = TM2_NOT_SET;

    /* Write the quant params */
    std::vector<QuantParam>* params = tensor->GetQuantParam();
    if(params->size() != 0)
    {
        vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * params->size();
        TM2_Vector_offsets* v_qtparams = ( TM2_Vector_offsets* )malloc(vector_size);
        v_qtparams->v_num = params->size();
        for(unsigned int i = 0; i < v_qtparams->v_num; i++)
        {
            QuantParam& p = (*params)[i];
            TM2_QuantParam qtparam;

            qtparam.zero_point = p.zero_point;
            qtparam.scale = p.scale;
            qtparam.width = p.width;

            v_qtparams->offsets[i] = WriteTmObject(start_ptr, cur_pos, &qtparam, sizeof(TM2_QuantParam));
        }

        /* Write the vector of quant params */
        tm_tensor.offect_vo_quantparams = WriteTmObject(start_ptr, cur_pos, v_qtparams, vector_size);
    }
    else
        tm_tensor.offect_vo_quantparams = TM2_NOT_SET;

    /* Write the tensor */
    return WriteTmObject(start_ptr, cur_pos, &tm_tensor, sizeof(TM2_Tensor));
}

tm_uoffset_t TmSerializer2::SaveTmNode(void* const start_ptr, tm_uoffset_t* cur_pos, Node* node,
                                       name_map_t& tensor_name_map)
{
    TM2_Node tm_node;
    tm_node.node_id = node->GetNodeIndex();
    tm_node.dynamic_shape = node->IsDynamicShape();

    bool tm_with_string = IsSaveString();

    if(tm_with_string)
    {
        std::string name = node->GetName();
        TM2_String node_name;
        node_name.size = name.size() + 1;    // including trailing \0
        node_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, name.c_str(), node_name.size);
        tm_node.offset_s_nname = WriteTmObject(start_ptr, cur_pos, &node_name, sizeof(TM2_String));
    }
    else
        tm_node.offset_s_nname = TM2_NOT_SET;

    unsigned int input_num = node->GetInputNum();
    unsigned int output_num = node->GetOutputNum();

    if(input_num)
    {
        /* Write the vector of input indices */
        size_t vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * input_num;
        TM2_Vector_indices* v_input_indices = ( TM2_Vector_indices* )malloc(vector_size);
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
        tm_node.offset_vi_input_tensors = TM2_NOT_SET;

    if(output_num)
    {
        /* Write the vector of output indices */
        size_t vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * output_num;
        TM2_Vector_indices* v_output_indices = ( TM2_Vector_indices* )malloc(vector_size);
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
        tm_node.offset_vi_output_tensors = TM2_NOT_SET;

    /* Write tm operator */
    std::string op_name = node->GetOp()->GetName();
    if(op_name == "Input")
        op_name = TM2_OPSTR_INPUTOP;
    if(!FindOpSaveMethod(op_name))
    {
        LOG_ERROR() << "cannot find save function for operator: " << op_name << "\n";
        return 0;
    }
    op_save_t op_save_func = any_cast<op_save_t>(GetOpSaveMethod(op_name));
    tm_node.offset_t_operator = op_save_func(start_ptr, cur_pos, node->GetOp());

    /* No custom attrs */
    if(!node->ExistAttr(ATTR_CUSTOM_ATTR))
    {
        tm_node.offset_vo_attrs = TM2_NOT_SET;
        /* Write the node */
        return WriteTmObject(start_ptr, cur_pos, &tm_node, sizeof(TM2_Node));
    }

    /* Get custom attrs of node */
    std::vector<TM2_Attr> tm_attrs;
    node_custom_attr_map_t* attr_map = any_cast<node_custom_attr_map_t>(&node->GetAttr(ATTR_CUSTOM_ATTR));
    node_custom_attr_map_t::iterator it = (*attr_map).begin();
    while(it != (*attr_map).end())
    {
        TM2_Attr tm_attr;
        std::string attr_name = it->first;
        CustomNodeAttr attr = it->second;

        TM2_String tm_attr_name, tm_attr_val;
        tm_attr_name.size = attr_name.size() + 1;    // including trailing \0
        tm_attr_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, attr_name.c_str(), attr_name.size());
        tm_attr.offset_s_attrname = WriteTmObject(start_ptr, cur_pos, &tm_attr_name, sizeof(TM2_String));

        tm_attr_val.size = attr.attr_size;    // no trailing \0
        tm_attr_val.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, &(attr.mem), attr.attr_size);
        tm_attr.offset_s_attrval = WriteTmObject(start_ptr, cur_pos, &tm_attr_val, sizeof(TM2_String));

        tm_attr.attr_type = typename_to_int(attr.type_name);

        tm_attrs.push_back(tm_attr);
        ++it;
    }

    /* Write custom attrs */
    size_t vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * tm_attrs.size();
    TM2_Vector_offsets* v_attrs = ( TM2_Vector_offsets* )malloc(vector_size);
    v_attrs->v_num = tm_attrs.size();
    for(unsigned int i = 0; i < tm_attrs.size(); i++)
    {
        v_attrs->offsets[i] = WriteTmObject(start_ptr, cur_pos, &(tm_attrs[i]), sizeof(TM2_Attr));
    }
    tm_node.offset_vo_attrs = WriteTmObject(start_ptr, cur_pos, v_attrs, vector_size);
    free(v_attrs);

    /* Write the node */
    return WriteTmObject(start_ptr, cur_pos, &tm_node, sizeof(TM2_Node));
}

tm_uoffset_t TmSerializer2::SaveTmSubgraph(void* const start_ptr, tm_uoffset_t* cur_pos, Graph* graph)
{
    TM2_Subgraph tm_subgraph;
    tm_subgraph.subgraph_id = 0; /* subgraph_id starts from 0 */
    tm_subgraph.offset_s_sname = TM2_NOT_SET;

    tm_subgraph.graph_layout = graph->GetLayout();
    tm_subgraph.model_layout = graph->GetModelLayout();

    unsigned int tensor_num = 0;
    unsigned int buffer_num = 0;
    std::vector<Tensor*> tensor_ptrs;
    std::vector<void*> buf_ptrs;
    std::vector<unsigned int> buf_sizes;
    name_map_t tensor_name_map; /* map of tensor name and tensor index */
    bool tm_no_data = !IsSaveData();

    /* Write the nodes */
    size_t vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * graph->seq_nodes.size();
    TM2_Vector_offsets* v_nodes = ( TM2_Vector_offsets* )malloc(vector_size);
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
        const auto offset = SaveTmNode(start_ptr, cur_pos, p_node, tensor_name_map);
        if (offset == 0) {
            // save node failed
            return offset;
        }
        v_nodes->offsets[i] = offset;
    }
    /* Write the vector of nodes */
    tm_subgraph.offset_vo_seq_nodes = WriteTmObject(start_ptr, cur_pos, v_nodes, vector_size);

    /* Write the tensors */
    vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * tensor_num;
    TM2_Vector_offsets* v_tensors = ( TM2_Vector_offsets* )malloc(vector_size);
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
    TM2_Vector_offsets* v_buffers = ( TM2_Vector_offsets* )malloc(vector_size);
    v_buffers->v_num = buffer_num;
    for(unsigned int i = 0; i < buffer_num; i++)
    {
        TM2_Buffer tm_buf;
        tm_buf.size = buf_sizes[i];

        if(tm_no_data)
        {
            /* TM2_FOR_BENCHMARK environment variable exists. Not write buf data into the tm file */
            tm_buf.offset_data = TM2_NOT_SET;
        }
        else
        {
            /* TM2_FOR_BENCHMARK environment variable does not exist */
            tm_buf.offset_data =
                WriteTmFileAlign1(start_ptr, cur_pos, reinterpret_cast<const uint8_t*>(buf_ptrs[i]), tm_buf.size);
        }
        v_buffers->offsets[i] = WriteTmObject(start_ptr, cur_pos, &tm_buf, sizeof(TM2_Buffer));
    }
    /* Write the vector of buffers */
    tm_subgraph.offset_vo_buffers = WriteTmObject(start_ptr, cur_pos, v_buffers, vector_size);

    /* Write the vector of input indices */
    vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * graph->input_nodes.size();
    TM2_Vector_indices* v_input_indices = ( TM2_Vector_indices* )malloc(vector_size);
    v_input_indices->v_num = graph->input_nodes.size();
    for(unsigned int i = 0; i < graph->input_nodes.size(); i++)
    {
        v_input_indices->indices[i] = graph->input_nodes[i]->GetNodeIndex();
    }
    tm_subgraph.offset_vi_input_indices = WriteTmObject(start_ptr, cur_pos, v_input_indices, vector_size);

    /* Write the vector of output indices */
    vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * graph->output_nodes.size();
    TM2_Vector_indices* v_output_indices = ( TM2_Vector_indices* )malloc(vector_size);
    v_output_indices->v_num = graph->output_nodes.size();
    for(unsigned int i = 0; i < graph->output_nodes.size(); i++)
    {
        v_output_indices->indices[i] = graph->output_nodes[i]->GetNodeIndex();
    }
    tm_subgraph.offset_vi_output_indices = WriteTmObject(start_ptr, cur_pos, v_output_indices, vector_size);

    /* Write the subgraph */
    tm_uoffset_t ret = WriteTmObject(start_ptr, cur_pos, &tm_subgraph, sizeof(TM2_Subgraph));

    /* Free the memory of vectors */
    free(v_tensors);
    free(v_buffers);
    free(v_nodes);
    free(v_input_indices);
    free(v_output_indices);

    return ret;
}

bool TmSerializer2::SaveModelIntoMem(void* start_ptr, Graph* graph, uint32_t* tm_model_size)
{
    bool tm_with_string = IsSaveString();

    tm_uoffset_t cur_pos = sizeof(TM2_Header);

    /* Define the TM2_Header object */
    TM2_Header header;
    header.ver_main = TM2_FILE_VER_MAIN;
    header.ver_sub = TM2_FILE_VER_SUB;
    header.ver_compile = TM2_FILE_VER_COMPILE;

    /* Define the TM2_Model object */
    TM2_Model tm_model;
    tm_model.orig_format = graph->GetModelFormat();
    tm_model.sub_format = 0;

    if(tm_with_string)
    {
        const std::string& fname = graph->GetName();
        TM2_String model_name;
        model_name.size = fname.size() + 1;    // including trailing \0
        model_name.offset_data = WriteTmFileAlign1(start_ptr, &cur_pos, fname.c_str(), model_name.size);
        tm_model.offset_s_mname = WriteTmObject(start_ptr, &cur_pos, &model_name, sizeof(TM2_String));
    }
    else
        tm_model.offset_s_mname = TM2_NOT_SET;

    /* Write the subgraphs */
    /* Only 1 subgraph is supported currently */
    size_t vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * 1;
    TM2_Vector_offsets* v_subgraphs = ( TM2_Vector_offsets* )malloc(vector_size);
    v_subgraphs->v_num = 1;
    const auto offset = SaveTmSubgraph(start_ptr, &cur_pos, graph);
    if (offset == 0) {
        return false;
    }
    v_subgraphs->offsets[0] = offset;

    /* Write the vector of subgraphs */
    tm_model.offset_vo_subgraphs = WriteTmObject(start_ptr, &cur_pos, v_subgraphs, vector_size);

    /* Write the model */
    header.offset_root = WriteTmObject(start_ptr, &cur_pos, &tm_model, sizeof(TM2_Model));
    *tm_model_size = cur_pos;

    /* Write the header */
    cur_pos = 0;
    WriteTmObject(start_ptr, &cur_pos, &header, sizeof(TM2_Header));

    free(v_subgraphs);

    return true;
}

bool TmSerializer2::LoadNode(StaticGraph* graph, StaticNode* node, const TM2_Node* tm_node, void* mmap_buf)
{
    if(tm_node->offset_vi_input_tensors != TM2_NOT_SET)
    {
        const TM2_Vector_indices* v_input_tensors =
            GetTmPtr<TM2_Vector_indices>(mmap_buf, tm_node->offset_vi_input_tensors);

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

    if(tm_node->offset_vi_output_tensors != TM2_NOT_SET)
    {
        const TM2_Vector_indices* v_output_tensors =
            GetTmPtr<TM2_Vector_indices>(mmap_buf, tm_node->offset_vi_output_tensors);

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

    /* set the custom attributes into static node */
    if(tm_node->offset_vo_attrs == TM2_NOT_SET)
        return true;

    const TM2_Vector_offsets* v_attrs = GetTmPtr<TM2_Vector_offsets>(mmap_buf, tm_node->offset_vo_attrs);
    for(unsigned int i = 0; i < v_attrs->v_num; i++)
    {
        const TM2_Attr* tm_attr = GetTmPtr<TM2_Attr>(mmap_buf, v_attrs->offsets[i]);
        const TM2_String* tm_attr_name = GetTmPtr<TM2_String>(mmap_buf, tm_attr->offset_s_attrname);
        const TM2_String* tm_attr_val = GetTmPtr<TM2_String>(mmap_buf, tm_attr->offset_s_attrval);

        const char* attr_name = GetTmPtr<char>(mmap_buf, tm_attr_name->offset_data);
        const char* attr_val = GetTmPtr<char>(mmap_buf, tm_attr_val->offset_data);
        const char* type_name = int_to_typename(tm_attr->attr_type);

        if(NodeAddParamGeneric(node, attr_name, type_name, tm_attr_val->size) < 0 ||
           NodeSetParamGeneric(node, attr_name, type_name, attr_val, tm_attr_val->size) < 0)
        {
            LOG_ERROR() << "Add and set node param failed\n";
            return false;
        }
    }

    return true;
}

bool TmSerializer2::LoadTensor(StaticGraph* graph, const TM2_Tensor* tm_tensor, const TM2_Buffer* tm_buf,
                               void* mmap_buf)
{
    /* Set the tensor name */
    int idx = tm_tensor->tensor_id;
    std::string tm_tensor_name;
    if(tm_tensor->offset_s_tname == TM2_NOT_SET)
        tm_tensor_name = "tensor_" + std::to_string(idx);
    else
    {
        const TM2_String* tm_str = GetTmPtr<TM2_String>(mmap_buf, tm_tensor->offset_s_tname);
        tm_tensor_name.assign(GetTmPtr<char>(mmap_buf, tm_str->offset_data), tm_str->size - 1);
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
    if(tm_tensor->offset_vd_dims != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_dims = GetTmPtr<TM2_Vector_dims>(mmap_buf, tm_tensor->offset_vd_dims);
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

    /* Set the tensor type and the data type */
    SetTensorType(tensor, tm_tensor->type);
    SetTensorDataType(tensor, tm_tensor->data_type);

    /* Set the memory size and pointer */
    if(tm_tensor->type == kConstTensor)
    {
        SetTensorSize(tensor, tm_buf->size);
        void* buf = malloc(tm_buf->size + 128);
        memset(buf ,0 ,tm_buf->size + 128);
        if(tm_buf->offset_data != TM2_NOT_SET)
        {
            memcpy(buf, GetTmPtr<void>(mmap_buf, tm_buf->offset_data), tm_buf->size);
        }

        SetConstTensorBuffer(tensor, buf);
        SetConstTensorFileLocation(tensor, -1, 0);
    }

    /* Set the quant params */
    if(tm_tensor->offect_vo_quantparams != TM2_NOT_SET)
    {
        const TM2_Vector_offsets* v_quantparams =
            GetTmPtr<TM2_Vector_offsets>(mmap_buf, tm_tensor->offect_vo_quantparams);

        /* currently only support one quant param */
        // assert(v_quantparams->v_num == 1);
        tensor->zero_point.resize(0);
        tensor->scale.resize(0);
        for(unsigned int i = 0; i < v_quantparams->v_num; ++i)
        {
            const TM2_QuantParam* tm_qtparam = GetTmPtr<TM2_QuantParam>(mmap_buf, v_quantparams->offsets[i]);
            tensor->zero_point.push_back(tm_qtparam->zero_point);
            tensor->scale.push_back(tm_qtparam->scale);
            tensor->width = tm_qtparam->width;
        }
    }

    return true;
}

bool TmSerializer2::LoadGraph(StaticGraph* graph, const TM2_Model* tm_model, void* mmap_buf)
{
    const TM2_Vector_offsets* v_graphs = GetTmPtr<TM2_Vector_offsets>(mmap_buf, tm_model->offset_vo_subgraphs);
    const TM2_Subgraph* tm_graph = GetTmPtr<TM2_Subgraph>(mmap_buf, v_graphs->offsets[0]);

    const TM2_Vector_offsets* v_nodes = GetTmPtr<TM2_Vector_offsets>(mmap_buf, tm_graph->offset_vo_seq_nodes);
    const TM2_Vector_offsets* v_tensors = GetTmPtr<TM2_Vector_offsets>(mmap_buf, tm_graph->offset_vo_tensors);
    const TM2_Vector_offsets* v_buffers = GetTmPtr<TM2_Vector_offsets>(mmap_buf, tm_graph->offset_vo_buffers);

    SetGraphLayout(graph, tm_graph->graph_layout);
    SetModelLayout(graph, tm_graph->model_layout);

    /* Load const tensors */
    for(unsigned int i = 0; i < v_tensors->v_num; i++)
    {
        const TM2_Tensor* tm_tensor = GetTmPtr<TM2_Tensor>(mmap_buf, v_tensors->offsets[i]);
        const TM2_Buffer* tm_buf;
        if(tm_tensor->type == kConstTensor)
            tm_buf = GetTmPtr<TM2_Buffer>(mmap_buf, v_buffers->offsets[tm_tensor->buffer_id]);
        else
            tm_buf = nullptr;
        LoadTensor(graph, tm_tensor, tm_buf, mmap_buf);
    }

    /* Create static nodes */
    unsigned int i;
    for(i = 0; i < v_nodes->v_num; i++)
    {
        const TM2_Node* tm_node = GetTmPtr<TM2_Node>(mmap_buf, v_nodes->offsets[i]);
        int idx = tm_node->node_id;
        std::string tm_node_name;
        if(tm_node->offset_s_nname == TM2_NOT_SET)
            tm_node_name = "node_" + std::to_string(idx);
        else
        {
            const TM2_String* tm_str = GetTmPtr<TM2_String>(mmap_buf, tm_node->offset_s_nname);
            tm_node_name.assign(GetTmPtr<char>(mmap_buf, tm_str->offset_data), tm_str->size - 1);
        }

        const TM2_Operator* tm_operator = GetTmPtr<TM2_Operator>(mmap_buf, tm_node->offset_t_operator);
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

    const TM2_Vector_indices* v_input_nodes = GetTmPtr<TM2_Vector_indices>(mmap_buf, tm_graph->offset_vi_input_indices);
    const TM2_Vector_indices* v_output_nodes =
        GetTmPtr<TM2_Vector_indices>(mmap_buf, tm_graph->offset_vi_output_indices);

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

bool TmSerializer2::LoadModelFromMem(void* mmap_buf, StaticGraph* graph)
{
    const TM2_Header* tm_header = reinterpret_cast<const TM2_Header*>(mmap_buf);

    const TM2_Model* tm_model = GetTmPtr<TM2_Model>(mmap_buf, tm_header->offset_root);

    /* Load dla tengine model */
    // if(tm_model->orig_format == MODEL_FORMAT_DLA)
    //    return LoadDlaModel(mmap_buf, graph);

    if(tm_model->offset_s_mname == TM2_NOT_SET)
    {
        SetGraphIdentity(graph, "tengine", "tengine_model", "0");
    }
    else
    {
        std::string tm_model_name;
        const TM2_String* tm_str = GetTmPtr<TM2_String>(mmap_buf, tm_model->offset_s_mname);
        tm_model_name.assign(GetTmPtr<char>(mmap_buf, tm_str->offset_data), tm_str->size - 1);
        SetGraphIdentity(graph, "tengine", tm_model_name, "0");
    }

    SetModelFormat(graph, tm_model->orig_format);

    if(LoadGraph(graph, tm_model, mmap_buf))
        return true;
    else
        return false;
}

bool TmSerializerRegisterOpLoader2(void)
{
    TmSerializerPtr serializer;

    if(!TmSerializerManager::SafeGet("tm_v2", serializer))
        return false;

    TmSerializer2* p_tengine = dynamic_cast<TmSerializer2*>(serializer.get());

    for(int i = 0; i < TM2_OPTYPE_NUM; i++)
    {
        p_tengine->RegisterOpLoadMethod(GetOpStr(i), op_load_t(LoadTmOpFunc(i)));
        p_tengine->RegisterOpSaveMethod(GetOpStr(i), op_save_t(SaveTmOpFunc(i)));
    }

    return true;
}

}    // namespace TMSerializer2

}    // namespace TEngine
