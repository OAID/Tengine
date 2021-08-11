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
 * Author: xlchen@openailab.com
 */

#include "save_graph.hpp"

#define TM_FILE_MAX_SIZE 1 << 30 /* 1G */

using name_map_t = std::unordered_map<std::string, unsigned int>;
using op_save_t = std::function<tm_uoffset_t(void* const, tm_uoffset_t*, ir_node_t*)>;
using op_save_map_t = std::unordered_map<uint16_t, op_save_t>;
op_save_map_t op_save_map_;

bool IsSaveString(void)
{
    const char* env = std::getenv("TM_NO_STRING");

    if (env)
        return false;
    else
        return true;
}

bool IsSaveData(void)
{
    const char* env = std::getenv("TM_FOR_BENCHMARK");

    if (env)
        return false;
    else
        return true;
}

bool RegisterOpSaveMethod(const uint16_t& op_type, const op_save_t& save_func)
{
    if (op_save_map_.count(op_type))
        return false;

    op_save_map_[op_type] = save_func;
    return true;
}

tm_uoffset_t SaveTmTensor(void* const start_ptr, tm_uoffset_t* cur_pos, ir_tensor_t* tensor,
                          unsigned int tensor_id, unsigned int buffer_id)
{
    TM2_Tensor tm_tensor;
    tm_tensor.tensor_id = tensor_id;
    tm_tensor.buffer_id = buffer_id;
    tm_tensor.type = tensor->tensor_type;
    tm_tensor.data_type = tensor->data_type;
    tm_tensor.layout = tensor->layout;

    bool tm_with_string = IsSaveString();

    if (tm_with_string)
    {
        std::string name = tensor->name;
        TM2_String tensor_name;
        tensor_name.size = name.size() + 1; // including trailing \0
        tensor_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, name.c_str(), tensor_name.size);
        tm_tensor.offset_s_tname = WriteTmObject(start_ptr, cur_pos, &tensor_name, sizeof(TM2_String));
    }
    else
        tm_tensor.offset_s_tname = TM2_NOT_SET;

    /* Get the dims of the tensor */
    int* dim = tensor->dims;
    size_t vector_size;
    if (tensor->dim_num)
    {
        /* Write the vector of dims */
        vector_size = sizeof(tm_size_t) + sizeof(int32_t) * tensor->dim_num;
        TM2_Vector_dims* v_dims = (TM2_Vector_dims*)malloc(vector_size);
        v_dims->v_num = tensor->dim_num;
        for (unsigned int i = 0; i < tensor->dim_num; i++)
        {
            v_dims->dims[i] = dim[i];
        }
        tm_tensor.offset_vd_dims = WriteTmObject(start_ptr, cur_pos, v_dims, vector_size);
        free(v_dims);
    }
    else
        tm_tensor.offset_vd_dims = TM2_NOT_SET;

    /* Write the quant params */
    if (tensor->quant_param_num != 0)
    {
        vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * tensor->quant_param_num;
        TM2_Vector_offsets* v_qtparams = (TM2_Vector_offsets*)malloc(vector_size);
        v_qtparams->v_num = tensor->quant_param_num;
        if (v_qtparams->v_num == 1)
        {
            TM2_QuantParam qtparam;
            qtparam.scale = tensor->scale;
            qtparam.zero_point = tensor->zero_point;
            v_qtparams->offsets[0] = WriteTmObject(start_ptr, cur_pos, &qtparam, sizeof(TM2_QuantParam));
        }
        else if (v_qtparams->v_num > 1)
        {
            for (unsigned int i = 0; i < v_qtparams->v_num; i++)
            {
                TM2_QuantParam qtparam;
                qtparam.zero_point = tensor->zp_list[i];
                qtparam.scale = tensor->scale_list[i];

                v_qtparams->offsets[i] = WriteTmObject(start_ptr, cur_pos, &qtparam, sizeof(TM2_QuantParam));
            }
        }

        /* Write the vector of quant params */
        tm_tensor.offect_vo_quantparams = WriteTmObject(start_ptr, cur_pos, v_qtparams, vector_size);
    }
    else
        tm_tensor.offect_vo_quantparams = TM2_NOT_SET;

    /* Write the tensor */
    return WriteTmObject(start_ptr, cur_pos, &tm_tensor, sizeof(TM2_Tensor));
}

tm_uoffset_t SaveTmNode(void* const start_ptr, tm_uoffset_t* cur_pos, ir_graph_t* graph, ir_node_t* node,
                        name_map_t& tensor_name_map)
{
    TM2_Node tm_node;
    memset(&tm_node, 0, sizeof(TM2_Node));
    tm_node.node_id = node->index;
    tm_node.dynamic_shape = node->dynamic_shape;

    bool tm_with_string = IsSaveString();

    if (tm_with_string)
    {
        std::string name = node->name;
        TM2_String node_name;
        node_name.size = name.size() + 1; // including trailing \0
        node_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, name.c_str(), node_name.size);
        tm_node.offset_s_nname = WriteTmObject(start_ptr, cur_pos, &node_name, sizeof(TM2_String));
    }
    else
        tm_node.offset_s_nname = TM2_NOT_SET;

    unsigned int input_num = node->input_num;
    unsigned int output_num = node->output_num;

    if (input_num)
    {
        /* Write the vector of input indices */
        size_t vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * input_num;
        TM2_Vector_indices* v_input_indices = (TM2_Vector_indices*)malloc(vector_size);
        v_input_indices->v_num = input_num;
        for (unsigned int i = 0; i < input_num; i++)
        {
            ir_tensor_t* p_tensor = get_ir_graph_tensor(graph, node->input_tensors[i]);
            v_input_indices->indices[i] = tensor_name_map[p_tensor->name];
        }
        tm_node.offset_vi_input_tensors = WriteTmObject(start_ptr, cur_pos, v_input_indices, vector_size);
        free(v_input_indices);
    }
    else
        tm_node.offset_vi_input_tensors = TM2_NOT_SET;

    if (output_num)
    {
        /* Write the vector of output indices */
        size_t vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * output_num;
        TM2_Vector_indices* v_output_indices = (TM2_Vector_indices*)malloc(vector_size);
        v_output_indices->v_num = output_num;
        for (unsigned int i = 0; i < output_num; i++)
        {
            ir_tensor_t* p_tensor = get_ir_graph_tensor(graph, node->output_tensors[i]);
            v_output_indices->indices[i] = tensor_name_map[p_tensor->name];
        }
        tm_node.offset_vi_output_tensors = WriteTmObject(start_ptr, cur_pos, v_output_indices, vector_size);
        free(v_output_indices);
    }
    else
        tm_node.offset_vi_output_tensors = TM2_NOT_SET;

    /* Write tm operator */
    uint16_t op_type = node->op.type;
    if (!op_save_map_.count(op_type))
    {
        TLOG_ERR("cannot find save function for operator:%d \n", op_type);
        return false;
    }
    op_save_t op_save_func = op_save_map_[op_type];
    tm_node.offset_t_operator = op_save_func(start_ptr, cur_pos, node);

    tm_node.offset_vo_attrs = TM2_NOT_SET;
    /* Write the node */
    return WriteTmObject(start_ptr, cur_pos, &tm_node, sizeof(TM2_Node));
}

tm_uoffset_t SaveTmSubgraph(void* const start_ptr, tm_uoffset_t* cur_pos, ir_graph_t* graph)
{
    TM2_Subgraph tm_subgraph;
    tm_subgraph.subgraph_id = 0; /* subgraph_id starts from 0 */
    tm_subgraph.offset_s_sname = TM2_NOT_SET;

    tm_subgraph.graph_layout = graph->graph_layout;
    tm_subgraph.model_layout = graph->model_layout;

    unsigned int tensor_num = 0;
    unsigned int buffer_num = 0;
    std::vector<ir_tensor_t*> tensor_ptrs;
    std::vector<void*> buf_ptrs;
    std::vector<unsigned int> buf_sizes;
    name_map_t tensor_name_map; /* map of tensor name and tensor index */
    bool tm_no_data = !IsSaveData();
    /* Write the nodes */
    size_t vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * graph->node_num;
    TM2_Vector_offsets* v_nodes = (TM2_Vector_offsets*)malloc(vector_size);
    v_nodes->v_num = graph->node_num;
    for (unsigned int i = 0; i < graph->node_num; i++)
    {
        ir_node_t* p_node = get_ir_graph_node(graph, i);
        for (unsigned int k = 0; k < p_node->output_num; k++)
        {
            ir_tensor_t* p_tensor = get_ir_graph_tensor(graph, p_node->output_tensors[k]);
            tensor_ptrs.push_back(p_tensor);
            tensor_name_map[p_tensor->name] = tensor_num;
            tensor_num++;
        }
        v_nodes->offsets[i] = SaveTmNode(start_ptr, cur_pos, graph, p_node, tensor_name_map);
    }
    /* Write the vector of nodes */
    tm_subgraph.offset_vo_seq_nodes = WriteTmObject(start_ptr, cur_pos, v_nodes, vector_size);

    /* Write the tensors */
    vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * tensor_num;
    TM2_Vector_offsets* v_tensors = (TM2_Vector_offsets*)malloc(vector_size);
    v_tensors->v_num = tensor_num;
    for (unsigned int i = 0; i < tensor_num; i++)
    {
        ir_tensor_t* p_tensor = tensor_ptrs[i];
        if (p_tensor->tensor_type == TENSOR_TYPE_CONST)
        {
            // buf_ptrs.push_back(p_tensor->GetMemAddr());
            buf_ptrs.push_back(p_tensor->data); // may cause bug
            buf_sizes.push_back(p_tensor->elem_num * p_tensor->elem_size);
            buffer_num++;
        }

        v_tensors->offsets[i] = SaveTmTensor(start_ptr, cur_pos, p_tensor, i, buffer_num - 1);
    }
    /* Write the vector of tensors */
    tm_subgraph.offset_vo_tensors = WriteTmObject(start_ptr, cur_pos, v_tensors, vector_size);

    /* Write the buffers */
    vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * buffer_num;
    TM2_Vector_offsets* v_buffers = (TM2_Vector_offsets*)malloc(vector_size);
    v_buffers->v_num = buffer_num;
    for (unsigned int i = 0; i < buffer_num; i++)
    {
        TM2_Buffer tm_buf;
        tm_buf.size = buf_sizes[i];

        if (tm_no_data)
        {
            /* TM2_FOR_BENCHMARK environment variable exists. Not write buf data into the tm file */
            tm_buf.offset_data = TM2_NOT_SET;
        }
        else
        {
            /* TM2_FOR_BENCHMARK environment variable does not exist */
            tm_buf.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, reinterpret_cast<const uint8_t*>(buf_ptrs[i]), tm_buf.size);
        }
        v_buffers->offsets[i] = WriteTmObject(start_ptr, cur_pos, &tm_buf, sizeof(TM2_Buffer));
    }
    /* Write the vector of buffers */
    tm_subgraph.offset_vo_buffers = WriteTmObject(start_ptr, cur_pos, v_buffers, vector_size);

    /* Write the vector of input indices */
    vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * graph->input_num;
    TM2_Vector_indices* v_input_indices = (TM2_Vector_indices*)malloc(vector_size);
    v_input_indices->v_num = graph->input_num;
    for (unsigned int i = 0; i < graph->input_num; i++)
    {
        v_input_indices->indices[i] = graph->input_nodes[i];
    }
    tm_subgraph.offset_vi_input_indices = WriteTmObject(start_ptr, cur_pos, v_input_indices, vector_size);

    /* Write the vector of output indices */
    vector_size = sizeof(tm_size_t) + sizeof(uint32_t) * graph->output_num;
    TM2_Vector_indices* v_output_indices = (TM2_Vector_indices*)malloc(vector_size);
    v_output_indices->v_num = graph->output_num;
    for (unsigned int i = 0; i < graph->output_num; i++)
    {
        v_output_indices->indices[i] = graph->output_nodes[i];
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

bool SaveModelIntoMem(void* start_ptr, ir_graph_t* graph, uint32_t* tm_model_size)
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
    tm_model.orig_format = graph->model_format;
    tm_model.sub_format = 0;

    // if(tm_with_string)
    // {
    //     const std::string& fname = graph->GetName();
    //     TM2_String model_name;
    //     model_name.size = fname.size() + 1;    // including trailing \0
    //     model_name.offset_data = WriteTmFileAlign1(start_ptr, &cur_pos, fname.c_str(), model_name.size);
    //     tm_model.offset_s_mname = WriteTmObject(start_ptr, &cur_pos, &model_name, sizeof(TM2_String));
    // }
    // else
    tm_model.offset_s_mname = TM2_NOT_SET;

    /* Write the subgraphs */
    /* Only 1 subgraph is supported currently */
    size_t vector_size = sizeof(tm_size_t) + sizeof(tm_uoffset_t) * 1;
    TM2_Vector_offsets* v_subgraphs = (TM2_Vector_offsets*)malloc(vector_size);
    v_subgraphs->v_num = 1;
    v_subgraphs->offsets[0] = SaveTmSubgraph(start_ptr, &cur_pos, graph);

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

int save_model(std::vector<void*>& addr_list, std::vector<int>& size_list, ir_graph_t* graph)
{
    uint32_t tm_model_size = 0;

    uint32_t malloc_size = TM_FILE_MAX_SIZE;
    const char* env = std::getenv("TM_FILE_MAX_SIZE");
    if (env)
        malloc_size = std::atoi(env);

    void* start_ptr = (void*)malloc(malloc_size);
    if (start_ptr == nullptr)
    {
        TLOG_ERR("Malloc memory failed: .\n", malloc_size);
        return false;
    }

    bool ret = SaveModelIntoMem(start_ptr, graph, &tm_model_size);

    addr_list.push_back(start_ptr);
    size_list.push_back(tm_model_size);

    return ret;
}

bool save_graph(graph_t graph, const char* fname)
{
    for (uint16_t i = OP_GENERIC + 1; i < OP_BUILTIN_LAST; i++)
    {
        RegisterOpSaveMethod(i, op_save_t(SaveTmOpFunc(i)));
    }

    ir_graph_t* ir_graph = (ir_graph_t*)graph;
    /* Open the tengine model file */
#ifdef _MSC_VER
    FILE* fd = fopen(fname, "wb+");
    if (fd == NULL)
#else
    int fd = open(fname, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1)
#endif
    {
        TLOG_ERR("Could not open %s\n", fname);
        return false;
    }

    std::vector<void*> addr_list;
    std::vector<int> size_list;

    if (!save_model(addr_list, size_list, ir_graph))
    {
#ifdef _MSC_VER
        fclose(fd);
#else
        close(fd);
#endif
        return false;
    }

    void* buf = addr_list[0];
    int size = size_list[0];
#ifdef _MSC_VER
    int ret = fwrite(buf, size, 1, fd) * size;
    fclose(fd);
#else
    int ret = write(fd, buf, size);
    close(fd);
#endif
    free(buf);

    if (ret != size)
        return false;
    else
        return true;
}