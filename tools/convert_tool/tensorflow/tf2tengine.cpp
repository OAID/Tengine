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
 * Author: bzhang@openailab.com
 */

#include "tf2tengine.hpp"

const int OP_VERSION = 1;

bool tensorflow_serializer::find_op_load_method(const std::string& op_name)
{
    if (op_load_map.count(op_name))
        return true;

    return false;
}

ir_tensor_t* tensorflow_serializer::find_tensor(ir_graph_t* graph, const std::string& tensor_name)
{
    for (uint16_t i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        if (tensor->name == tensor_name)
        {
            return tensor;
        }
    }

    return nullptr;
}

static bool GetAttrValue(const tensorflow::NodeDef* node, const char* key, tensorflow::AttrValue& value)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node->attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    if (it != attr.end())
    {
        value = it->second;
        return true;
    }

    return false;
}

int GetTensorContentAndDim(const tensorflow::TensorProto& tf_tensor, int** dims, void** mem_ptr,
                           std::string& layout)
{
    const tensorflow::TensorShapeProto& shape = tf_tensor.tensor_shape();

    int elem_num = 1;
    int dim_num = shape.dim_size();

    int* dim_tmp = (int*)malloc(sizeof(int) * dim_num);
    for (int i = 0; i < dim_num; i++)
    {
        elem_num *= shape.dim(i).size();
        dim_tmp[i] = shape.dim(i).size();
    }

    void* mem_buf = nullptr;

    if (tf_tensor.tensor_content().size())
    {
        int content_size = tf_tensor.tensor_content().size();
        mem_buf = malloc(content_size + 128);
        void* src = (void*)tf_tensor.tensor_content().c_str();
        memcpy(mem_buf, src, content_size);
    }
    else if (tf_tensor.dtype() == tensorflow::DataType::DT_FLOAT)
    {
        // in packed format
        int data_num = tf_tensor.float_val_size();
        mem_buf = malloc(elem_num * sizeof(float));
        float* mem = (float*)mem_buf;
        if (data_num >= elem_num)
        {
            for (int i = 0; i < elem_num; i++)
            {
                mem[i] = tf_tensor.float_val(i);
            }
        }
        else
        {
            // data_num < elem_num
            for (int i = 0; i < data_num; i++)
            {
                mem[i] = tf_tensor.float_val(i);
            }

            for (int i = data_num; i < elem_num; i++)
            {
                mem[i] = mem[data_num - 1];
            }
        }
    }
    else if (tf_tensor.dtype() == tensorflow::DataType::DT_INT32)
    {
        int data_num = tf_tensor.int_val_size();

        mem_buf = malloc(elem_num * sizeof(int));

        int* mem = (int*)mem_buf;

        if (data_num >= elem_num)
        {
            for (int i = 0; i < elem_num; i++)
            {
                mem[i] = tf_tensor.int_val(i);
            }
        }
        else
        {
            // data_num < elem_num
            for (int i = 0; i < data_num; i++)
            {
                mem[i] = tf_tensor.int_val(i);
            }

            for (int i = data_num; i < elem_num; i++)
            {
                mem[i] = mem[data_num - 1];
            }
        }
    }

    *mem_ptr = mem_buf;
    *dims = dim_tmp;
    switch (dim_num)
    {
    case 0:
        layout = "W";
        break;
    case 1:
        layout = "W";
        break;
    case 2:
        layout = "HW";
        break;
    case 4:
        layout = "NCHW";
        break;
    default:
        break;
    }
    return dim_num;
}
int tensorflow_serializer::load_binary_file(std::string model_file)
{
    std::ifstream is(model_file.c_str(), std::ios::in | std::ios::binary);

    if (!is.is_open())
    {
        TLOG_ERR("cannot open file: %s \n", model_file.c_str());
        return false;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);

#if GOOGLE_PROTOBUF_VERSION >= 3011000
    coded_input.SetTotalBytesLimit(INT_MAX);
#else
    coded_input.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif
    bool ret = tf_net.ParseFromCodedStream(&coded_input);

    is.close();

    if (!ret)
    {
        TLOG_ERR("parse file: %s failed\n", model_file.c_str());
        return -1;
    }
    return ret;
}
int load_const_tensor(TFNode* tf_node, ir_graph_t* graph)
{
    ir_node_t* node = create_ir_node(graph, tf_node->name.c_str(), OP_CONST, OP_VERSION);
    ir_tensor_t* tensor = create_ir_tensor(graph, tf_node->name.c_str(), TENGINE_DT_FP32);
    tensorflow::AttrValue value;
    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];
    if (GetAttrValue(node_def, "value", value))
    {
        const tensorflow::TensorProto& tf_tensor = value.tensor();
        void* mem_ptr;
        int* dims;
        std::string layout;
        int dim_num = GetTensorContentAndDim(tf_tensor, &dims, &mem_ptr, layout);
        set_ir_tensor_shape(tensor, dims, dim_num);

        int mem_size = 1;
        for (int i = 0; i < dim_num; i++)
            mem_size *= dims[i];
        tensor->data = (float*)malloc(sizeof(float) * mem_size);
        tensor->tensor_type = TENSOR_TYPE_CONST;

        float* tmp = (float*)mem_ptr;
        float* tr_tmp = (float*)tensor->data;
        for (int i = 0; i < mem_size; i++)
        {
            tr_tmp[i] = tmp[i];
        }
    }
    set_ir_node_output_tensor(node, 0, tensor);
    tf_node->ir_node = node;
    tf_node->ir_tensor = tensor;

    return 0;
}

int tensorflow_serializer::set_graph_input(ir_graph_t* graph)
{
    int node_num = tf_graph.seq_nodes.size();
    std::vector<int16_t> input_nodes;
    for (int i = 0; i < node_num; i++)
    {
        TFNode* tf_node = tf_graph.seq_nodes[i];
        if (tf_node->op == "Placeholder")
        {
            ir_tensor_t* ir_tensor = create_ir_tensor(graph, tf_node->name.c_str(), TENGINE_DT_FP32);
            ir_tensor->tensor_type = TENSOR_TYPE_INPUT;
            tensorflow::AttrValue shape;

            int pb_defs_cnt = tf_node->pb_defs.size();
            int* dims;
            if (pb_defs_cnt == 1)
            {
                if (GetAttrValue(tf_node->pb_defs[0], "shape", shape))
                {
                    int dim_size = shape.shape().dim_size();

                    dims = (int*)sys_malloc(dim_size);
                    memset(dims, 0, sizeof(int) * dim_size);
                    for (int i = 0; i < dim_size; ++i)
                    {
                        dims[i] = shape.shape().dim(i).size();
                    }
                    if (dim_size == 4)
                    {
                        dims[0] = shape.shape().dim(0).size() == -1 ? 1 : shape.shape().dim(0).size();
                        dims[1] = shape.shape().dim(3).size();
                        dims[2] = shape.shape().dim(1).size();
                        dims[3] = shape.shape().dim(2).size();
                    }
                    set_ir_tensor_shape(ir_tensor, dims, dim_size);
                }
            }
            else
            {
                tensorflow::AttrValue value;
                const tensorflow::NodeDef* node_def = tf_node->pb_defs[pb_defs_cnt - 1];
                if (GetAttrValue(node_def, "value", value))
                {
                    const tensorflow::TensorProto& tf_tensor = value.tensor();

                    void* mem_ptr;
                    std::vector<int> tf_dims;
                    std::string layout;
                    int dim_num = GetTensorContentAndDim(tf_tensor, &dims, &mem_ptr, layout);

                    int mem_size = 1;
                    for (int i = 0; i < dim_num; i++)
                        mem_size *= dims[i];
                    ir_tensor->data = (float*)malloc(sizeof(float) * mem_size);
                    ir_tensor->tensor_type = TENSOR_TYPE_CONST;
                    // tensor->data = mem_ptr;
                    float* tmp = (float*)mem_ptr;
                    float* tr_tmp = (float*)ir_tensor->data;
                    for (int i = 0; i < mem_size; i++)
                        tr_tmp[i] = tmp[i];

                    int* reshape_dim = (int*)mem_ptr;
                    for (int i = 0; i < tf_dims[0]; i++)
                    {
                        dims[i] = reshape_dim[i];
                    }

                    for (unsigned int i = 0; i < tf_dims[0]; i++)
                    {
                        if (dims[i] == -1)
                            dims[i] = 1;
                    }
                    free(mem_ptr);
                    set_ir_tensor_shape(ir_tensor, dims, tf_dims[0]);
                }
            }
            ir_node_t* node = create_ir_node(graph, tf_node->name.c_str(), OP_INPUT, OP_VERSION);

            int tensor_id = get_ir_tensor_index_from_name(graph, tf_node->name.c_str());

            set_ir_node_output_tensor(node, 0, ir_tensor);
            input_nodes.push_back(node->index);
        }
    }
    int16_t* node_idx = (int16_t*)sys_malloc(sizeof(int16_t) * input_nodes.size());
    for (int i = 0; i < input_nodes.size(); i++)
    {
        node_idx[i] = input_nodes[i];
    }
    set_ir_graph_input_node(graph, node_idx, input_nodes.size());
    return 0;
}
int tensorflow_serializer::construct_graph()
{
    int node_number = tf_net.node_size();
    std::unordered_map<std::string, TFNode*> node_map;

    /* first scan, setup all nodes */
    for (int i = 0; i < node_number; i++)
    {
        const tensorflow::NodeDef& node_param = tf_net.node(i);

        TFNode* tf_node = new TFNode();

        tf_node->idx = i;
        tf_node->name = node_param.name();
        tf_node->op = node_param.op();
        tf_node->pb_defs.push_back(&tf_net.node(i));

        tf_graph.seq_nodes.push_back(tf_node);

        node_map[tf_node->name] = tf_node;
    }

    /* the second scan, setup connections */
    for (int i = 0; i < node_number; i++)
    {
        const tensorflow::NodeDef& node_param = tf_net.node(i);
        const std::string& name = node_param.name();

        TFNode* cur_node = node_map[name];

        for (int j = 0; j < node_param.input_size(); j++)
        {
            const std::string& input_name = node_param.input(j);
            std::string::size_type pos = input_name.find(":");
            std::string cleanup_name;

            if (pos == std::string::npos)
                pos = input_name.size();

            if (input_name[0] == '^')
                cleanup_name = input_name.substr(1, pos);
            else
                cleanup_name = input_name.substr(0, pos);

            TFNode* input_node = node_map[cleanup_name];

            if (input_node == nullptr)
            {
                TLOG_ERR("cannot find input: %s for node: %s \n", input_name.c_str(), name.c_str());
                return false;
            }
            cur_node->inputs.push_back(input_node);
            input_node->outputs.push_back(cur_node);
        }
    }
    return 0;
}
int DisconnectNode(TFNode* cur_node)
{
    TFNode* input_node;

    for (unsigned int i = 0; i < cur_node->inputs.size(); i++)
    {
        input_node = cur_node->inputs[i];

        auto ir = input_node->outputs.begin();

        while (ir != input_node->outputs.end())
        {
            if (*ir != cur_node)
                ir++;
            else
                break;
        }

        if (ir == input_node->outputs.end())
        {
            TLOG_ERR("ERROR on node connection!!\n");
        }

        input_node->outputs.erase(ir);
    }

    cur_node->inputs.clear();

    TFNode* output_node;

    for (unsigned int i = 0; i < cur_node->outputs.size(); i++)
    {
        output_node = cur_node->outputs[i];

        auto ir = output_node->inputs.begin();

        while (ir != output_node->inputs.end())
        {
            if (*ir != cur_node)
                ir++;
            else
                break;
        }

        if (ir == output_node->inputs.end())
        {
            TLOG_ERR("ERROR on node connection!!\n");
        }

        output_node->inputs.erase(ir);
    }

    cur_node->outputs.clear();

    return 0;
}

int tensorflow_serializer::MergeParentNode(TFNode* base_node, TFNode* parent_node)
{
    /* remove the input for parent node */

    auto input_ir = base_node->inputs.begin();

    while (input_ir != base_node->inputs.end())
    {
        if (*input_ir == parent_node)
            break;

        input_ir++;
    }

    if (parent_node->inputs.size() == 1)
    {
        *input_ir = parent_node->inputs[0];
    }
    else
    {
        base_node->inputs.erase(input_ir);
        /* connect parent's input node and base node */

        base_node->inputs.insert(base_node->inputs.end(), parent_node->inputs.begin(), parent_node->inputs.end());
    }

    /* setup the outputs of parent node's parent */

    for (auto node : parent_node->inputs)
    {
        for (unsigned int i = 0; i < node->outputs.size(); i++)
        {
            if (node->outputs[i] == parent_node)
            {
                node->outputs[i] = base_node;
                break;
            }
        }
    }

    /* bridge parent's output, for those edges do not connect with base node */

    auto output_ir = parent_node->outputs.begin();

    while (output_ir != parent_node->outputs.end())
    {
        TFNode* node = *output_ir;

        if (node != base_node)
        {
            base_node->outputs.push_back(node);

            for (unsigned int i = 0; i < node->inputs.size(); i++)
            {
                if (node->inputs[i] == parent_node)
                {
                    node->inputs[i] = base_node;
                    break;
                }
            }
        }

        output_ir++;
    }

    /* handle TF definitions */

    base_node->pb_defs.insert(base_node->pb_defs.end(), parent_node->pb_defs.begin(), parent_node->pb_defs.end());

    // std::cout<<"base node: "<<base_node->name<<" merge parent: "<<parent_node->name<<"\n";

    parent_node->inputs.clear();
    parent_node->outputs.clear();

    return 0;
}

bool CheckComposedBNAdd(TFNode* cur_node)
{
    if (cur_node->op != "Add")
        return false;

    TFNode* input0 = cur_node->inputs[0];
    TFNode* input1 = cur_node->inputs[1];

    if (input0->op != "Mul" || input1->op != "Sub")
        return false;

    /* further check: /add_1 int name */
    if (cur_node->name.find("/add_1") != std::string::npos)
    {
        if (input0->name.find("/mul_1") != std::string::npos || input1->name.find("/mul_1") != std::string::npos)
            cur_node->BNAddType = 1;
        else
            cur_node->BNAddType = 0;

        return true;
    }

    return false;
}

int tensorflow_serializer::BNRecursiveInputMerge(TFNode* node)
{
    bool mul_1_node = false;
    bool mul_node = false;
    if (node->name.find("/mul") != std::string::npos)
    {
        if (node->BNAddType == 1)
        {
            if (node->name.find("/mul_1") != std::string::npos)
            {
                mul_1_node = true;
            }
            else if (node->name.find("/mul_2") == std::string::npos)
            {
                // disconnect the connection between mul and mul2
                auto ir = node->outputs.begin();

                if ((*ir)->name.find("/mul2") == std::string::npos)
                    ir++;

                TFNode* mul2_node = *ir;

                node->outputs.erase(ir);

                ir = mul2_node->inputs.begin();

                if ((*ir)->name.find("/mul") == std::string::npos)
                    ir++;

                mul2_node->inputs.erase(ir);
            }
        }
        else
        {
            if (node->name.find("/mul_1") != std::string::npos)
            {
                // disconnect the connection between add_1 mul_1
                auto ir = node->inputs.begin();

                if ((*ir)->name.find("/add_1") == std::string::npos)
                    ir++;

                if ((*ir)->name.find("/add_1") != std::string::npos)
                {
                    TFNode* Rsqrt_node = *ir;

                    node->inputs.erase(ir);

                    ir = Rsqrt_node->outputs.begin();

                    if ((*ir)->name.find("/mul_1") == std::string::npos)
                        ir++;

                    Rsqrt_node->outputs.erase(ir);
                }
            }
            else
            {
                mul_node = true;
            }
        }
    }

    int orig_input_size = node->inputs.size();
    std::vector<TFNode*> input_cpy = node->inputs;

    for (int i = 0; i < orig_input_size; i++)
    {
        if (mul_node && i == 0)
            continue;
        if (mul_1_node && i == 0)
            continue;

        TFNode* input_node = input_cpy[i];
        input_node->BNAddType = node->BNAddType;
        if (input_node->op == "Const")
            continue;

        BNRecursiveInputMerge(input_node);
        MergeParentNode(node, input_node);
    }

    return 0;
}

int tensorflow_serializer::FuseComposedBN(TFNode* cur_node)
{
    BNRecursiveInputMerge(cur_node);
    cur_node->op = "ComposedBN";

    /* set new name */
    auto pos = cur_node->name.find("/add_1");
    cur_node->name.replace(pos, strlen("/add_1"), "bn.fused");

    /* skip to create static node for add/y */

    for (unsigned int i = 0; i < cur_node->inputs.size(); i++)
    {
        TFNode* node = cur_node->inputs[i];

        if (node->name.find("/add/y") != std::string::npos)
            node->no_static_node = true;
    }

    return 0;
}

int tensorflow_serializer::MergeChildNode(TFNode* base_node, TFNode* child_node)
{
    auto output_ir = base_node->outputs.begin();

    while (output_ir != base_node->outputs.end())
    {
        if (*output_ir == child_node)
            break;
        output_ir++;
    }

    if (child_node->outputs.size() == 1)
    {
        *output_ir = child_node->outputs[0];
    }
    else
    {
        base_node->outputs.erase(output_ir);
        base_node->outputs.insert(base_node->outputs.end(), child_node->outputs.begin(), child_node->outputs.end());
    }

    for (auto node : child_node->outputs)
    {
        for (unsigned int i = 0; i < node->inputs.size(); i++)
        {
            if (node->inputs[i] == child_node)
            {
                node->inputs[i] = base_node;
                break;
            }
        }
    }

    auto ir = child_node->inputs.begin();

    while (ir != child_node->inputs.end())
    {
        TFNode* node = *ir;

        if (node != base_node)
        {
            base_node->inputs.push_back(node);

            for (unsigned int i = 0; i < node->outputs.size(); i++)
            {
                if (node->outputs[i] == child_node)
                {
                    node->outputs[i] = base_node;
                    break;
                }
            }
        }

        ir++;
    }

    base_node->pb_defs.insert(base_node->pb_defs.end(), child_node->pb_defs.begin(), child_node->pb_defs.end());

    // std::cout<<"base node: "<<base_node->name<<" merge child: "<<child_node->name<<"\n";

    child_node->inputs.clear();
    child_node->outputs.clear();

    return 0;
}

void tensorflow_serializer::CleanupResizeNearestNeighbor()
{
    auto ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->op == "ResizeNearestNeighbor")
        {
            TFNode* data_node = cur_node->inputs[0];
            TFNode* data_shape_node = nullptr;

            for (unsigned int i = 0; i < data_node->outputs.size(); i++)
            {
                if (data_node->outputs[i]->op == "Shape")
                {
                    data_shape_node = data_node->outputs[i];
                }
            }

            assert(data_shape_node != nullptr);
            DisconnectNode(data_shape_node);

            TFNode* mul_node = cur_node->inputs[1];
            TFNode* stride_slice = mul_node->inputs[0];
            DisconnectNode(stride_slice);
            DisconnectNode(mul_node);
        }

        ir++;
    }
}

void tensorflow_serializer::MergeReluMinimum()
{
    for (auto ir = tf_graph.seq_nodes.begin(); ir != tf_graph.seq_nodes.end(); ir++)
    {
        TFNode* cur_node = *ir;

        if (cur_node->inputs.size() == 0)
            continue;

        TFNode* input0 = cur_node->inputs[0];

        if (cur_node->op == "Minimum" && input0->op == "Relu")
        {
            TFNode* const_node = cur_node->inputs[1];

            DisconnectNode(const_node);

            MergeChildNode(input0, cur_node);

            input0->op = "Relu6";
        }
    }
}

int tensorflow_serializer::optimize_graph()
{
    fused_node_count = 0;
    /* first clean up the predictions module of TF */
    auto ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->op == "Reshape")
        {
            /* Reshape should have two inputs */

            TFNode* input_node0 = cur_node->inputs[0];
            TFNode* input_node1 = cur_node->inputs[1];

            if (input_node0->op == "Softmax" || input_node1->op == "Softmax")
            {
                DisconnectNode(cur_node);
                ir = tf_graph.seq_nodes.erase(ir);
                delete cur_node;
                continue;
            }

            TFNode* output_node = cur_node->outputs[0];
            if (NULL == output_node)
                continue;

            if (output_node->op == "Softmax" || output_node->op == "MatMul")
            {
                TFNode* input_node0 = cur_node->inputs[0];
                TFNode* input_node1 = cur_node->inputs[1];
                TFNode* input_node;

                if (input_node0->op == "Const")
                {
                    DisconnectNode(input_node0);
                    input_node = input_node1;
                }
                else
                {
                    DisconnectNode(input_node1);
                    input_node = input_node0;
                }

                MergeChildNode(input_node, cur_node);

                ir = tf_graph.seq_nodes.erase(ir);
                delete cur_node;
                continue;
            }
        }

        ir++;
    }

    /* remove the squeeze node and identity */
    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->op == "Squeeze")
        {
            TFNode* softmax_node = nullptr;
            TFNode* shape_node = nullptr;

            for (unsigned int j = 0; j < cur_node->outputs.size(); j++)
            {
                if (cur_node->outputs[j]->op == "Softmax")
                    softmax_node = cur_node->outputs[j];
                else if (cur_node->outputs[j]->op == "Shape")
                    shape_node = cur_node->outputs[j];
            }

            if (softmax_node)
            {
                if (shape_node)
                    DisconnectNode(shape_node);

                TFNode* input_node = cur_node->inputs[0];
                MergeChildNode(input_node, cur_node);
                ir = tf_graph.seq_nodes.erase(ir);
                delete cur_node;
                continue;
            }

            if (cur_node->outputs.size() == 1 && softmax_node == nullptr)
            {
                TFNode* child_node = cur_node->outputs[0];

                MergeParentNode(child_node, cur_node);
                ir = tf_graph.seq_nodes.erase(ir);
                delete cur_node;
                continue;
            }
        }

        if (cur_node->op == "Identity")
        {
            TFNode* input_node = cur_node->inputs[0];
            MergeChildNode(input_node, cur_node);

            ir = tf_graph.seq_nodes.erase(ir);
            delete cur_node;
            continue;
        }

        if (cur_node->op == "ConcatV2")
        {
            TFNode* axis_node = nullptr;

            for (unsigned int i = 0; i < cur_node->inputs.size(); i++)
            {
                TFNode* check_node = cur_node->inputs[i];

                if (check_node->op == "Const")
                {
                    axis_node = check_node;
                    break;
                }
            }

            if (axis_node)
            {
                cur_node->pb_defs.push_back(axis_node->pb_defs[0]);
                DisconnectNode(axis_node);
            }
        }

        ir++;
    }

    /* merge FIFOQueueV2  DequeueManyV2 */

    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->op == "FIFOQueueV2")
        {
            TFNode* queue_node = cur_node->outputs[0];

            if (queue_node->op == "QueueDequeueManyV2")
            {
                MergeParentNode(queue_node, queue_node->inputs[1]);
            }

            MergeChildNode(cur_node, queue_node);

            break;
        }
        ir++;
    }

    /* remove ExpandDims */
    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->op == "ExpandDims")
        {
            TFNode* input0 = cur_node->inputs[0];
            TFNode* input1 = cur_node->inputs[1];

            if (input0->op == "Constant" && input1->op == "Const")
            {
                TFNode* input1 = cur_node->inputs[1];
                TFNode* child_node = cur_node->outputs[0];

                DisconnectNode(input1);
                DisconnectNode(cur_node);

                child_node->inputs.push_back(input1);
                input1->outputs.push_back(child_node);
            }
            else
            {
                if (input1->op == "Const")
                    DisconnectNode(input1);
                else
                    DisconnectNode(input0);

                TFNode* child_node = cur_node->outputs[0];

                MergeParentNode(child_node, cur_node);
            }

            ir = tf_graph.seq_nodes.erase(ir);
            delete cur_node;

            continue;
        }

        ir++;
    }

    /* merge biasadd and conv */
    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->op == "Conv2D" || cur_node->op == "DepthwiseConv2dNative" || cur_node->op == "MatMul")
        {
            TFNode* output_node = cur_node->outputs[0];

            if (output_node->op == "BiasAdd" || output_node->op == "Add")
            {
                cur_node->biasAdd = 1;
                fused_node_count++;
                MergeChildNode(cur_node, output_node);
            }
            else
            {
                cur_node->biasAdd = 0;
            }
        }
        ir++;
    }

    /* merge composed BatchNormal */

    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (CheckComposedBNAdd(cur_node))
            FuseComposedBN(cur_node);
        ir++;
    }

    /* cleanup ResizeNearestNeighbor */
    CleanupResizeNearestNeighbor();

    /* merge Minimum and Relu */

    MergeReluMinimum();
    /* merge input node and reshape */
    ir = tf_graph.seq_nodes.begin();
    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;
        if (cur_node->op == "Reshape")
        {
            /* Reshape should have two inputs */
            TFNode* input_node0 = cur_node->inputs[0];
            TFNode* input_node1 = cur_node->inputs[1];

            if (input_node0->op == "Placeholder" || input_node1->op == "Placeholder")
            {
                TFNode* input_node;
                TFNode* const_node;

                if (input_node0->op == "Const")
                {
                    const_node = input_node0;
                    input_node = input_node1;
                }
                else
                {
                    const_node = input_node1;
                    input_node = input_node0;
                }

                DisconnectNode(const_node);
                MergeChildNode(input_node, cur_node);
                input_node->pb_defs.insert(input_node->pb_defs.end(), const_node->pb_defs[0]);

                ir = tf_graph.seq_nodes.erase(ir);
                delete cur_node;
                break;
            }
        }
        ir++;
    }

    /* remove the shape and StrideSlice */

    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;
        if (cur_node->op == "StridedSlice")
        {
            /* check if input0 is "shape" */
            TFNode* input_node = cur_node->inputs[0];

            if (input_node->op == "Shape")
            {
                /* here we go */
                DisconnectNode(cur_node);
                DisconnectNode(input_node);
                break;
            }
        }

        ir++;
    }

    /* merge pad and conv */

    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->op == "Conv2D" || cur_node->op == "DepthwiseConv2dNative")
        {
            /* check if input is pad or not */
            TFNode* input_node = cur_node->inputs[0];

            if (input_node->op == "Pad")
            {
                TFNode* padding_args = input_node->inputs[1];

                input_node->pb_defs.push_back(padding_args->pb_defs[0]);

                DisconnectNode(padding_args);
                MergeParentNode(cur_node, input_node);
            }
        }
        ir++;
    }

    /*remove ArgMax node */

    ir = tf_graph.seq_nodes.begin();
    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;
        if (cur_node->op == "ArgMax")
        {
            DisconnectNode(cur_node);
            tf_graph.seq_nodes.erase(ir);

            break;
        }

        ir++;
    }

    /* remove last squeeze */

    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->op == "Squeeze" && cur_node->outputs.empty())
        {
            DisconnectNode(cur_node);
            break;
        }
        ir++;
    }

    /* remove no input and output nodes */

    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->inputs.size() == 0 && cur_node->outputs.size() == 0)
        {
            ir = tf_graph.seq_nodes.erase(ir);
            delete cur_node;
        }
        else
            ir++;
    }

    /* remove no input but not placeholder/const nodes */
    ir = tf_graph.seq_nodes.begin();

    while (ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if (cur_node->inputs.size() == 0 && cur_node->op != "Const" && cur_node->op != "Placeholder" && cur_node->op != "FIFOQueueV2")
        {
            DisconnectNode(cur_node);
            tf_graph.seq_nodes.erase(ir);
            delete cur_node;

            ir = tf_graph.seq_nodes.begin(); // restart
        }
        else
            ir++;
    }

    return 0;
}

int tensorflow_serializer::FindRNNScope(std::string& rnn_scope)
{
    std::string rnn_node;

    std::string::size_type while_pos;

    int rnn_type = -1;

    for (unsigned int i = 0; i < tf_graph.seq_nodes.size(); i++)
    {
        TFNode* node = tf_graph.seq_nodes.at(i);
        std::string& name = node->name;

        while_pos = name.find("while");

        if (while_pos == std::string::npos)
            continue;

        std::string::size_type cell_pos = name.find("lstm_cell", while_pos);

        if (cell_pos != std::string::npos)
        {
            rnn_node = node->name;
            rnn_type = TF_RNN_LSTM;
            break;
        }

        cell_pos = name.find("gru_cell", while_pos);

        if (cell_pos != std::string::npos)
        {
            rnn_node = node->name;
            rnn_type = TF_RNN_GRU;
            break;
        }

        cell_pos = name.find("basic_lstm_cell", while_pos);

        if (cell_pos != std::string::npos)
        {
            rnn_node = node->name;
            rnn_type = TF_RNN_BASIC_LSTM;
            break;
        }

        cell_pos = name.find("basic_rnn_cell", while_pos);

        if (cell_pos != std::string::npos)
        {
            rnn_node = node->name;
            rnn_type = TF_RNN_BASIC_RNN;
            break;
        }
    }

    if (rnn_node.empty())
        return -1;

    std::string rnn_layer = rnn_node.substr(0, while_pos - 1);
    std::string::size_type up_pos = rnn_layer.rfind("/");

    rnn_scope = rnn_layer.substr(0, up_pos + 1);

    return rnn_type;
}

void tensorflow_serializer::ParseLSTMGraph(LSTMNode* lstm_node, std::set<TFNode*>& rnn_graph)
{
    /* parse input node */

    for (unsigned int i = 0; i < lstm_node->inputs.size(); i++)
    {
        TFNode* node = lstm_node->inputs[i];

        if (node->op != "Const")
            continue;

        // node->no_static_node=true; //do not automatically create Static Node

        if (node->name.find("lstm_cell/kernel") != std::string::npos)
        {
            lstm_node->kernel = node;
        }
        else if (node->name.find("lstm_cell/bias") != std::string::npos)
        {
            lstm_node->bias = node;
        }
        else if (node->name.find("lstm_cell/w_f_diag") != std::string::npos)
        {
            lstm_node->w_f_diag = node;
        }
        else if (node->name.find("lstm_cell/w_o_diag") != std::string::npos)
        {
            lstm_node->w_o_diag = node;
        }
        else if (node->name.find("lstm_cell/w_i_diag") != std::string::npos)
        {
            lstm_node->w_i_diag = node;
        }
        else if (node->name.find("lstm_cell/projection/kernel") != std::string::npos)
        {
            lstm_node->projection = node;
        }
    }

    auto rnn_ir = rnn_graph.begin();
    auto rnn_ir_end = rnn_graph.end();

    while (rnn_ir != rnn_ir_end)
    {
        TFNode* node = *rnn_ir;
        int name_len = node->name.size();
        std::string zero_name = "LSTMCellZeroState/zeros";
        std::string zero1_name = "LSTMCellZeroState/zeros_1";
        std::string forget_name = "lstm_cell/add/y";

        if (node->name.find(zero_name, name_len - zero_name.size()) != std::string::npos)
            lstm_node->init_c = node;
        else if (node->name.find(zero1_name, name_len - zero1_name.size()) != std::string::npos)
            lstm_node->init_h = node;
        else if (node->name.find(forget_name, name_len - forget_name.size()) != std::string::npos)
            lstm_node->forget_bias = node;

        rnn_ir++;
    }
}
void ParseGRUGraph(TFGraph& tf_graph, GRUNode* gru_node, std::set<TFNode*>& rnn_graph)
{
    /* parse input node */

    for (unsigned int i = 0; i < gru_node->inputs.size(); i++)
    {
        TFNode* node = gru_node->inputs[i];

        if (node->op != "Const")
            continue;

        // node->no_static_node=true; //do not automatically create Static Node

        if (node->name.find("gru_cell/gates/kernel") != std::string::npos)
        {
            gru_node->gate_kernel = node;
        }
        else if (node->name.find("gru_cell/gates/bias") != std::string::npos)
        {
            gru_node->gate_bias = node;
        }
        else if (node->name.find("gru_cell/candidate/kernel") != std::string::npos)
        {
            gru_node->candidate_kernel = node;
        }
        else if (node->name.find("gru_cell/candidate/bias") != std::string::npos)
        {
            gru_node->candidate_bias = node;
        }
    }

    auto rnn_ir = rnn_graph.begin();
    auto rnn_ir_end = rnn_graph.end();

    while (rnn_ir != rnn_ir_end)
    {
        TFNode* node = *rnn_ir;
        int name_len = node->name.size();
        std::string zero_name = "GRUCellZeroState/zeros";

        if (node->name.find(zero_name, name_len - zero_name.size()) != std::string::npos)
            gru_node->init_h = node;

        rnn_ir++;
    }
}
void ParseRNNGraph(TFGraph& tf_graph, RNNNode* rnn_node, std::set<TFNode*>& rnn_graph)
{
    /* parse input node */

    for (unsigned int i = 0; i < rnn_node->inputs.size(); i++)
    {
        TFNode* node = rnn_node->inputs[i];

        if (node->op != "Const")
            continue;

        // node->no_static_node=true; //do not automatically create Static Node

        if (node->name.find("basic_rnn_cell/kernel") != std::string::npos)
        {
            rnn_node->kernel = node;
        }
        else if (node->name.find("basic_rnn_cell/bias") != std::string::npos)
        {
            rnn_node->bias = node;
        }
    }

    auto rnn_ir = rnn_graph.begin();
    auto rnn_ir_end = rnn_graph.end();

    while (rnn_ir != rnn_ir_end)
    {
        TFNode* node = *rnn_ir;
        int name_len = node->name.size();
        std::string zero_name = "BasicRNNCellZeroState/zeros";

        if (node->name.find(zero_name, name_len - zero_name.size()) != std::string::npos)
            rnn_node->init_h = node;

        rnn_ir++;
    }
}
void tensorflow_serializer::StripRNNScope(std::string& rnn_scope, int rnn_type)
{
    // collect attributes according to rnn_type

    if (rnn_type == TF_RNN_LSTM)
    {
        LSTMNode* lstm_node = new LSTMNode();

        lstm_node->name = rnn_scope + "lstm";
        lstm_node->op = "LSTM";

        std::set<TFNode*>& rnn_graph = lstm_node->rnn_graph;

        std::set<TFNode*> rnn_inputs;
        std::set<TFNode*> rnn_outputs;

        auto ir = tf_graph.seq_nodes.begin();
        std::string::size_type prefix_len = rnn_scope.size();

        while (ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *ir;

            if (node->name.find(rnn_scope.c_str(), 0, prefix_len) == std::string::npos)
            {
                ir++;
                continue;
            }

            /* this is a node, inside rnn scope, remove it from graph first */
            ir = tf_graph.seq_nodes.erase(ir);

            rnn_graph.insert(node);
        }

        auto rnn_ir = rnn_graph.begin();
        auto rnn_end = rnn_graph.end();

        while (rnn_ir != rnn_end)
        {
            TFNode* node = *rnn_ir;

            for (unsigned int i = 0; i < node->inputs.size(); i++)
            {
                TFNode* input = node->inputs[i];

                if (!rnn_graph.count(input))
                    rnn_inputs.insert(input);
            }

            for (unsigned int i = 0; i < node->outputs.size(); i++)
            {
                TFNode* output = node->outputs[i];

                if (!rnn_graph.count(output))
                    rnn_outputs.insert(output);
            }

            rnn_ir++;
        }

        // insert lstm node
        auto seq_ir = tf_graph.seq_nodes.begin();

        while (seq_ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *seq_ir;

            if (rnn_inputs.count(node))
            {
                tf_graph.seq_nodes.insert(seq_ir, lstm_node);
                break;
            }

            seq_ir++;
        }

        // connect inputs and outputs
        auto set_ir = rnn_inputs.begin();
        auto set_ir_end = rnn_inputs.end();

        while (set_ir != set_ir_end)
        {
            TFNode* input_node = *set_ir;

            for (unsigned int j = 0; j < input_node->outputs.size(); j++)
            {
                TFNode* child_node = input_node->outputs[j];

                if (rnn_graph.count(child_node))
                    input_node->outputs[j] = lstm_node;
            }

            lstm_node->inputs.push_back(input_node);

            if (input_node->op == "Identity")
            {
                TFNode* parent_node = input_node->inputs[0];

                MergeChildNode(parent_node, input_node);
            }

            set_ir++;
        }

        set_ir = rnn_outputs.begin();
        set_ir_end = rnn_outputs.end();

        while (set_ir != set_ir_end)
        {
            TFNode* output_node = *set_ir;

            for (unsigned int j = 0; j < output_node->inputs.size(); j++)
            {
                TFNode* parent_node = output_node->inputs[j];

                if (rnn_graph.count(parent_node))
                    output_node->inputs[j] = lstm_node;
            }

            lstm_node->outputs.push_back(output_node);
            set_ir++;
        }

        /* sort input node and output node according to index */
        std::sort(lstm_node->inputs.begin(), lstm_node->inputs.end(),
                  [](const TFNode* a, const TFNode* b) { return a->idx < b->idx; });

        std::sort(lstm_node->outputs.begin(), lstm_node->outputs.end(),
                  [](const TFNode* a, const TFNode* b) { return a->idx < b->idx; });

        ParseLSTMGraph(lstm_node, rnn_graph);
    }

    if (rnn_type == TF_RNN_BASIC_RNN)
    {
        RNNNode* rnn_node = new RNNNode();

        rnn_node->name = rnn_scope + "rnn";
        // std::cout<<rnn_scope<<std::endl;
        rnn_node->op = "RNN";

        std::set<TFNode*>& rnn_graph = rnn_node->rnn_graph;

        std::set<TFNode*> rnn_inputs;
        std::set<TFNode*> rnn_outputs;

        auto ir = tf_graph.seq_nodes.begin();
        std::string::size_type prefix_len = rnn_scope.size();

        while (ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *ir;

            if (node->name.find(rnn_scope.c_str(), 0, prefix_len) == std::string::npos)
            {
                ir++;
                continue;
            }

            /* this is a node, inside rnn scope, remove it from graph first */
            ir = tf_graph.seq_nodes.erase(ir);

            rnn_graph.insert(node);
        }

        auto rnn_ir = rnn_graph.begin();
        auto rnn_end = rnn_graph.end();

        while (rnn_ir != rnn_end)
        {
            TFNode* node = *rnn_ir;

            for (unsigned int i = 0; i < node->inputs.size(); i++)
            {
                TFNode* input = node->inputs[i];

                if (!rnn_graph.count(input))
                    rnn_inputs.insert(input);
            }

            for (unsigned int i = 0; i < node->outputs.size(); i++)
            {
                TFNode* output = node->outputs[i];

                if (!rnn_graph.count(output))
                    rnn_outputs.insert(output);
            }

            rnn_ir++;
        }

        // insert rnn node
        auto seq_ir = tf_graph.seq_nodes.begin();

        while (seq_ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *seq_ir;

            if (rnn_inputs.count(node))
            {
                tf_graph.seq_nodes.insert(seq_ir, rnn_node);
                break;
            }

            seq_ir++;
        }

        // connect inputs and outputs
        auto set_ir = rnn_inputs.begin();
        auto set_ir_end = rnn_inputs.end();

        while (set_ir != set_ir_end)
        {
            TFNode* input_node = *set_ir;

            for (unsigned int j = 0; j < input_node->outputs.size(); j++)
            {
                TFNode* child_node = input_node->outputs[j];

                if (rnn_graph.count(child_node))
                    input_node->outputs[j] = rnn_node;
            }

            rnn_node->inputs.push_back(input_node);

            if (input_node->op == "Identity")
            {
                TFNode* parent_node = input_node->inputs[0];

                MergeChildNode(parent_node, input_node);
            }

            set_ir++;
        }

        set_ir = rnn_outputs.begin();
        set_ir_end = rnn_outputs.end();

        while (set_ir != set_ir_end)
        {
            TFNode* output_node = *set_ir;

            for (unsigned int j = 0; j < output_node->inputs.size(); j++)
            {
                TFNode* parent_node = output_node->inputs[j];

                if (rnn_graph.count(parent_node))
                    output_node->inputs[j] = rnn_node;
            }

            rnn_node->outputs.push_back(output_node);
            set_ir++;
        }

        /* sort input node and output node according to index */
        std::sort(rnn_node->inputs.begin(), rnn_node->inputs.end(),
                  [](const TFNode* a, const TFNode* b) { return a->idx < b->idx; });

        std::sort(rnn_node->outputs.begin(), rnn_node->outputs.end(),
                  [](const TFNode* a, const TFNode* b) { return a->idx < b->idx; });

        ParseRNNGraph(tf_graph, rnn_node, rnn_graph);
    }
    if (rnn_type == TF_RNN_GRU)
    {
        GRUNode* gru_node = new GRUNode();

        gru_node->name = rnn_scope + "gru";
        // std::cout<<rnn_scope<<std::endl;
        gru_node->op = "GRU";

        std::set<TFNode*>& rnn_graph = gru_node->rnn_graph;

        std::set<TFNode*> rnn_inputs;
        std::set<TFNode*> rnn_outputs;

        auto ir = tf_graph.seq_nodes.begin();
        std::string::size_type prefix_len = rnn_scope.size();

        while (ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *ir;

            if (node->name.find(rnn_scope.c_str(), 0, prefix_len) == std::string::npos)
            {
                ir++;
                continue;
            }

            /* this is a node, inside rnn scope, remove it from graph first */
            ir = tf_graph.seq_nodes.erase(ir);

            rnn_graph.insert(node);
        }

        auto rnn_ir = rnn_graph.begin();
        auto rnn_end = rnn_graph.end();

        while (rnn_ir != rnn_end)
        {
            TFNode* node = *rnn_ir;

            for (unsigned int i = 0; i < node->inputs.size(); i++)
            {
                TFNode* input = node->inputs[i];

                if (!rnn_graph.count(input))
                    rnn_inputs.insert(input);
            }

            for (unsigned int i = 0; i < node->outputs.size(); i++)
            {
                TFNode* output = node->outputs[i];

                if (!rnn_graph.count(output))
                    rnn_outputs.insert(output);
            }

            rnn_ir++;
        }

        // insert rnn node
        auto seq_ir = tf_graph.seq_nodes.begin();

        while (seq_ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *seq_ir;

            if (rnn_inputs.count(node))
            {
                tf_graph.seq_nodes.insert(seq_ir, gru_node);
                break;
            }

            seq_ir++;
        }

        // connect inputs and outputs
        auto set_ir = rnn_inputs.begin();
        auto set_ir_end = rnn_inputs.end();

        while (set_ir != set_ir_end)
        {
            TFNode* input_node = *set_ir;

            for (unsigned int j = 0; j < input_node->outputs.size(); j++)
            {
                TFNode* child_node = input_node->outputs[j];

                if (rnn_graph.count(child_node))
                    input_node->outputs[j] = gru_node;
            }

            gru_node->inputs.push_back(input_node);

            if (input_node->op == "Identity")
            {
                TFNode* parent_node = input_node->inputs[0];

                MergeChildNode(parent_node, input_node);
            }

            set_ir++;
        }

        set_ir = rnn_outputs.begin();
        set_ir_end = rnn_outputs.end();

        while (set_ir != set_ir_end)
        {
            TFNode* output_node = *set_ir;

            for (unsigned int j = 0; j < output_node->inputs.size(); j++)
            {
                TFNode* parent_node = output_node->inputs[j];

                if (rnn_graph.count(parent_node))
                    output_node->inputs[j] = gru_node;
            }

            gru_node->outputs.push_back(output_node);
            set_ir++;
        }

        /* sort input node and output node according to index */
        std::sort(gru_node->inputs.begin(), gru_node->inputs.end(),
                  [](const TFNode* a, const TFNode* b) { return a->idx < b->idx; });

        std::sort(gru_node->outputs.begin(), gru_node->outputs.end(),
                  [](const TFNode* a, const TFNode* b) { return a->idx < b->idx; });

        ParseGRUGraph(tf_graph, gru_node, rnn_graph);
    }

    // cleanup zero in/zero out node
    auto seq_ir = tf_graph.seq_nodes.begin();

    while (seq_ir != tf_graph.seq_nodes.end())
    {
        TFNode* node = *seq_ir;

        if (node->inputs.size() == 0 && node->outputs.size() == 0)
        {
            delete node;
            seq_ir = tf_graph.seq_nodes.erase(seq_ir);
        }
        else
        {
            seq_ir++;
        }
    }
}
int tensorflow_serializer::optimize_rnn()
{
    while (1)
    {
        std::string rnn_scope;

        int rnn_type = FindRNNScope(rnn_scope);

        if (rnn_scope.empty())
            break;

        StripRNNScope(rnn_scope, rnn_type);
    }

    return true;
}

int find_tensor_index(ir_graph_t* graph, std::string t_name, TFNode* node)
{
    int tensor_idx = -1;
    for (int i = 0; i < graph->tensor_num; i++)
    {
        const ir_tensor_t* const tensor = graph->tensor_list[i];

        if (tensor->name && 0 == strcmp(tensor->name, node->name.c_str()))
        {
            tensor_idx = i;
        }
    }
    return tensor_idx;
}

int tensorflow_serializer::generate_graph(ir_graph_t* graph)
{
    int node_number = tf_graph.seq_nodes.size();
    int i;

    bool debug_graph = false;
    const char* debug_env = std::getenv("DEBUG_TF");
    if ((debug_env) && (debug_env[0] == '1'))
    {
        debug_graph = true;
    }

    // first: create all tensor node
    for (i = 0; i < node_number; i++)
    {
        TFNode* tf_node = tf_graph.seq_nodes[i];

        if (debug_graph)
        {
            std::cout << i << "\t" << tf_node->op << "\t" << tf_node->name << "\n";
        }

        if (tf_node->no_static_node)
            continue;

        if (tf_node->op == "Placeholder")
            continue;

        if (tf_node->op == "Const")
        {
            load_const_tensor(tf_node, graph);
            continue;
        }
    }

    int count = 0;
    if (fused_node_count > FUSE_NODE)
    {
        count = (int)out_graph.size();
    }
    else
    {
        count = (int)tf_graph.seq_nodes.size();
    }
    for (int i = 0; i < count; i++)
    {
        TFNode* tf_node = nullptr;
        if (fused_node_count > FUSE_NODE)
            tf_node = out_graph[i];
        else
            tf_node = tf_graph.seq_nodes[i];

        if (tf_node->op == "Placeholder" || tf_node->op == "Const")
            continue;

        ir_node_t* ir_node = nullptr;
        int node_idx = get_ir_node_index_from_name(graph, tf_node->name.c_str());
        if (node_idx < 0)
        {
            ir_node = create_ir_node(graph, tf_node->name.c_str(), op_load_map[tf_node->op].first, OP_VERSION);
        }
        else
        {
            ir_node = get_ir_graph_node(graph, node_idx);
        }

        for (int in = 0; in < tf_node->inputs.size(); in++)
        {
            TFNode* node = tf_node->inputs[in];
            std::string t_name = tf_node->name;
            int tensor_idx = get_ir_tensor_index_from_name(graph, node->name.c_str());

            ir_tensor_t* tensor = nullptr;
            if (node->name == "Placeholder")
            {
                continue;
            }

            if (tensor_idx < 0)
            {
                tensor = create_ir_tensor(graph, t_name.c_str(), TENGINE_DT_FP32);
            }
            else
            {
                tensor = get_ir_graph_tensor(graph, tensor_idx);
            }
            set_ir_node_input_tensor(ir_node, in, tensor);
            input_tensors.push_back(node->name.c_str());
        }
        for (int out = 0; out < tf_node->outputs.size(); out++)
        {
            TFNode* node = tf_node->outputs[out];
            ir_tensor_t* tensor = nullptr;
            std::string t_name = tf_node->name;
            int tensor_idx = get_ir_tensor_index_from_name(graph, node->name.c_str());
            if (tensor_idx < 0)
            {
                tensor = create_ir_tensor(graph, t_name.c_str(), TENGINE_DT_FP32);
            }
            else
            {
                tensor = get_ir_graph_tensor(graph, tensor_idx);
            }
            set_ir_node_output_tensor(ir_node, out, tensor);
            output_tensors.push_back(node->name.c_str());
        }

        op_load_t loader = op_load_map[tf_node->op].second;
        if (loader == 0){
           continue;
        }
        if (loader(tf_node, tf_graph, graph, ir_node) < 0)
        {
            fprintf(stderr, "load op %s func failed in node %s .\n", tf_node->op.c_str(), tf_node->name.c_str());
            return -1;
        }
    }

    if (i < node_number)
        return -1;

    return 0;
}

int tensorflow_serializer::set_graph_output(ir_graph_t* graph)
{
    int layer_number = tf_graph.seq_nodes.size();
    std::vector<int16_t> output_nodes;

    std::vector<std::string> graph_outputs;
    for (int i = 0; i < output_tensors.size(); i++)
    {
        int check_flag = true;

        auto it = find(input_tensors.begin(), input_tensors.end(), output_tensors[i]);
        if (it == input_tensors.end())
        {
            graph_outputs.push_back(output_tensors[i]);
        }
    }

    for (int i = 0; i < graph_outputs.size(); i++)
    {
        int tensor_id = get_ir_tensor_index_from_name(graph, graph_outputs[i].c_str());
        ir_tensor_t* tensor = nullptr;
        if (tensor_id < 0)
            tensor = create_ir_tensor(graph, graph_outputs[i].c_str(), TENGINE_DT_FP32);
        else
            tensor = get_ir_graph_tensor(graph, tensor_id);
        int node_idx = get_ir_node_index_from_name(graph, graph_outputs[i].c_str());
        ir_node_t* node = get_ir_graph_node(graph, node_idx);
        set_ir_node_output_tensor(node, 0, tensor);
        output_nodes.push_back(node->index);
    }

    std::vector<int16_t> node_idx;
    for (int i = 0; i < output_nodes.size(); i++)
    {
        node_idx.push_back(output_nodes[i]);
    }
    set_ir_graph_output_node(graph, node_idx.data(), output_nodes.size());
    return 0;
}

bool AllInputCheck(TFNode* node, std::vector<int>& visited)
{
    for (int i = 0; i < node->inputs.size(); i++)
    {
        TFNode* o_node = node->inputs[i];
        if (visited[o_node->idx] != 1)
        {
            if (o_node->op == "Const")
                continue;
            return false;
        }
    }
    return true;
}

int tensorflow_serializer::DFSGraph(ir_graph_t* graph)
{
    std::stack<TFNode*> visit_stack;
    std::vector<int> visited(65535, 0);
    std::vector<TFNode*> starts;
    int node_number = tf_graph.seq_nodes.size();
    /* Find start node */
    for (int i = 0; i < node_number; i++)
    {
        TFNode* node = tf_graph.seq_nodes[i];
        if (node->op == "Placeholder" || node->op == "Const")
            continue;
        if (node->inputs[0]->op == "Placeholder")
        {
            starts.push_back(node);
        }
    }
    for (int i = 0; i < starts.size(); i++)
    {
        visit_stack.push(starts[i]);
    }

    while (!visit_stack.empty())
    {
        TFNode* node = visit_stack.top();
        visited[node->idx] = 1;
        visit_stack.pop();
        for (int out = 0; out < node->outputs.size(); out++)
        {
            TFNode* out_node = node->outputs[out];
            visited[out_node->idx] = 1;
            if (AllInputCheck(out_node, visited))
                visit_stack.push(out_node);
        }
        out_graph.push_back(node);
    }

    return 0;
}

int tensorflow_serializer::load_graph(ir_graph_t* graph)
{
    if (construct_graph() < 0)
        return -1;
    if (optimize_rnn() < 0)
        return false;
    if (optimize_graph() < 0)
        return -1;
    if (fused_node_count > FUSE_NODE)
    {
        if (DFSGraph(graph) < 0)
            return -1;
    }
    if (set_graph_input(graph) < 0)
        return -1;
    fprintf(stderr, "Process 2: Finish set graph input \n");
    if (generate_graph(graph) < 0)
        return -1;
    fprintf(stderr, "Process 3: Finish load graph node \n");

    if (set_graph_output(graph) < 0)
        return -1;
    fprintf(stderr, "Process 4: Finish set graph output \n");

    return 0;
}

int tensorflow_serializer::load_model(ir_graph_t* graph, std::string model_file)
{
    register_op_load();
    if (load_binary_file(model_file) < 0)
        return -1;
    fprintf(stderr, "Process 1: Finish load protobuf file \n");
    load_graph(graph);

    return 0;
}

graph_t tensorflow_serializer::tensorflow2tengine(std::string model_file)
{
    fprintf(stderr, "----------tensorflow2tengine begin----------\n");

    context_t context = create_context(NULL, 1);
    ir_graph_t* ir_graph = create_ir_graph((struct context*)context);
    if (ir_graph == NULL)
    {
        destroy_context(context);
        return NULL;
    }
    ir_graph->attribute->private_context = 1; // new context

    int ret = load_model(ir_graph, model_file);
    if (0 != ret)
    {
        destroy_graph(ir_graph);
        return NULL;
    }
    ir_graph->device = find_default_device();

    fprintf(stderr, "----------tensorflow2tengine done.----------\n");
    return ir_graph;
}

int load_pool(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    TFNode* input = tf_node->inputs[0];
    struct pool_param* param = (struct pool_param*)node->op.param_mem;
    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];
    tensorflow::AttrValue value;

    if (GetAttrValue(node_def, "ksize", value))
    {
        param->kernel_h = value.list().i(1);
        param->kernel_w = value.list().i(2);
    }

    if (GetAttrValue(node_def, "strides", value))
    {
        param->stride_h = value.list().i(1);
        param->stride_w = value.list().i(2);
    }

    if (GetAttrValue(node_def, "padding", value))
    {
        if (value.s() == "VALID")
        {
            param->pad_h0 = 0;
            param->pad_h1 = 0;
            param->pad_w0 = 0;
            param->pad_w1 = 0;
        }
        else if (value.s() == "SAME")
        {
            param->pad_h0 = -1;
            param->pad_h1 = -1;
            param->pad_w0 = -1;
            param->pad_w1 = -1;
        }
    }

    if (tf_node->op == "AvgPool")
    {
        param->pool_method = 1;
    }
    else if (tf_node->op == "MaxPool")
    {
        param->pool_method = 0;
    }
    return 0;
}
int load_conv(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    TFNode* input0 = tf_node->inputs[0]; /* input */
    TFNode* input1 = tf_node->inputs[1]; /* weight */
    TFNode* input2 = nullptr;
    if (tf_node->inputs.size() > 2)
    {
        input2 = tf_node->inputs[2];
    }
    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];
    struct conv_param* param = (struct conv_param*)node->op.param_mem;
    tensorflow::AttrValue value;

    if (GetAttrValue(node_def, "dilations", value))
    {
        param->dilation_h = value.list().i(1);
        param->dilation_w = value.list().i(2);
    }
    if (GetAttrValue(node_def, "padding", value))
    {
        if (value.s() == "VALID")
        {
            param->pad_h0 = 0;
            param->pad_h1 = 0;
            param->pad_w0 = 0;
            param->pad_w1 = 0;
        }
        else if (value.s() == "SAME")
        {
            param->pad_h0 = -1;
            param->pad_h1 = -1;
            param->pad_w0 = -1;
            param->pad_w1 = -1;
        }
    }

    if (GetAttrValue(node_def, "strides", value))
    {
        param->stride_h = value.list().i(1);
        param->stride_w = value.list().i(2);
    }

    int in_channel = 1, out_channel = 1, kernel_h = 0, kernel_w = 0;
    int group = 1;
    // Tensorflow has to get those information from weights

    const tensorflow::NodeDef* weight_def = input1->pb_defs[0];

    if (GetAttrValue(weight_def, "value", value))
    {
        const tensorflow::TensorShapeProto& shape = value.tensor().tensor_shape();

        if (shape.dim_size() == 4)
        {
            kernel_h = shape.dim(0).size();
            kernel_w = shape.dim(1).size();
            in_channel = shape.dim(2).size();
            out_channel = shape.dim(3).size();
        }
        else if (shape.dim_size() == 3)
        {
            kernel_h = 1;
            kernel_w = shape.dim(0).size();
            in_channel = shape.dim(1).size();
            out_channel = shape.dim(2).size();
        }
    }
    ir_tensor_t* weight_tensor = input1->ir_tensor;

    int elem_size = out_channel * in_channel * kernel_h * kernel_w;
    float* new_weight = (float*)malloc(sizeof(float) * elem_size);
    float* src = (float*)weight_tensor->data;
    weight_tensor->data = sys_malloc(elem_size * sizeof(float));
    float* ptr = (float*)weight_tensor->data;

    for (int o = 0; o < out_channel; o++)
        for (int h = 0; h < kernel_h; h++)
            for (int w = 0; w < kernel_w; w++)
                for (int i = 0; i < in_channel; i++)
                {
                    ptr[o * in_channel * kernel_h * kernel_w + i * kernel_h * kernel_w + h * kernel_w + w]
                        = src[h * (kernel_w * in_channel * out_channel) + w * (in_channel * out_channel) + i * out_channel + o];
                }

    free(src);

    weight_tensor->tensor_type = TENSOR_TYPE_CONST;
    if (tf_node->op == "DepthwiseConv2dNative")
    {
        group = in_channel;
        out_channel = in_channel * out_channel;
        in_channel = 1;
    }

    int* dims = (int*)malloc(sizeof(int) * 4);
    dims[0] = out_channel;
    dims[1] = in_channel;
    dims[2] = kernel_h;
    dims[3] = kernel_w;

    // SetTensorDim(weight_tensor, dims);
    set_ir_tensor_shape(weight_tensor, dims, 4);
    param->kernel_h = kernel_h;
    param->kernel_w = kernel_w;
    param->output_channel = out_channel;
    param->group = group;

    auto saved_param = param;

    if (tf_node->op == "DepthwiseConv2dNative")
    {
        in_channel = group;
        out_channel = out_channel / in_channel;
    }

    int pb_def_num = tf_node->pb_defs.size();

    if (pb_def_num > 1)
    {
        // the last one,
        const tensorflow::NodeDef* node_def = tf_node->pb_defs[pb_def_num - 1];

        /* possible pad */
        if (node_def->op() == "Const")
        {
            tensorflow::AttrValue value;

            if (GetAttrValue(node_def, "value", value) && value.has_tensor())
            {
                const tensorflow::TensorProto& tf_tensor = value.tensor();

                int dim_size = tf_tensor.tensor_shape().dim_size();

                if (dim_size == 2 && tf_tensor.tensor_shape().dim(0).size() == 4 && tf_tensor.tensor_shape().dim(1).size() == 2)
                {
                    std::vector<int> shape_data(8);

                    if (tf_tensor.tensor_content().size())
                    {
                        int* ptr = shape_data.data();
                        memcpy(ptr, tf_tensor.tensor_content().c_str(), tf_tensor.tensor_content().size());
                    }
                    else
                    {
                        int data_num = tf_tensor.int_val_size();

                        for (int i = 0; i < data_num; i++)
                        {
                            shape_data[i] = tf_tensor.int_val(i);
                        }
                    }

                    /* h pad */
                    saved_param->pad_h0 = shape_data[2];
                    saved_param->pad_h1 = shape_data[3];
                    /* w pad */
                    saved_param->pad_w0 = shape_data[4];
                    saved_param->pad_w1 = shape_data[5];
                    // printf("%d %d %d %d \n",)
                }
            }
        }
    }
    return 0;
}
int load_batchnorm(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    struct batchnorm_param* param = (struct batchnorm_param*)node->op.param_mem;
    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];
    tensorflow::AttrValue value;

    if (GetAttrValue(node_def, "epsilon", value))
    {
        param->eps = value.f();
    }

    return 0;
}
int load_relu6(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    return 0;
}
int load_softmax(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    struct softmax_param* param = (struct softmax_param*)node->op.param_mem;
    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];
    tensorflow::AttrValue value;
    if (GetAttrValue(node_def, "value", value))
    {
        const tensorflow::TensorProto& tf_tensor = value.tensor();
        int axis = tf_tensor.int_val(0);
        param->axis = axis;
    }
    else
        param->axis = 1;
    return 0;
}
int load_relu(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    TFNode* input = tf_node->inputs[0];
    struct relu_param* param = (struct relu_param*)node->op.param_mem;
    param->negative_slope = 0.f;
    return 0;
}
static EltType MapEltwise(TFNode* tf_node, const std::string& elt_op)
{
    if (elt_op == "Add" || elt_op == "AddN")
        return ELT_SUM;
    else if (elt_op == "Mul")
        return ELT_PROD;
    else if (elt_op == "Sub")
        return ELT_SUB;
    else if (elt_op == "Rsqrt")
        return ELT_RSQRT;
    else if (elt_op == "Minimum")
        return ELT_MIN_SCALAR;
    else if (elt_op == "Exp")
        return ELT_EXP;
    else if (elt_op == "Log")
        return ELT_LOG;
    else if (elt_op == "Pow")
        return ELT_POW;
    else if (elt_op == "RealDiv")
        return ELT_DIV;
    else if (elt_op == "Sqrt")
        return ELT_SQRT;
    else if (elt_op == "Floor")
        return ELT_FLOOR;
    else
        return ELT_LAST;
}
int load_eltwise(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    if (tf_node->op == "Add" || tf_node->op == "Mul" || tf_node->op == "Sub" || tf_node->op == "Minimum" || tf_node->op == "AddN" || tf_node->op == "Pow" || tf_node->op == "RealDiv")
    {
        if (tf_node->inputs.size() != 2)
            return false;
    }
    else if (tf_node->op == "Rsqrt" || tf_node->op == "Exp" || tf_node->op == "Log" || tf_node->op == "Sqrt" || tf_node->op == "Floor")
    {
        if (tf_node->inputs.size() != 1)
            return false;
    }
    else
    {
        TLOG_ERR("Unsupported op: %s \n", tf_node->op.c_str());
        return false;
    }
    struct eltwise_param* param = (struct eltwise_param*)node->op.param_mem;
    param->type = MapEltwise(tf_node, tf_node->op);
    param->caffe_flavor = 1;
    return 0;
}
static void* LoadConstParam(TFNode* tf_node)
{
    tensorflow::AttrValue value;

    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];

    if (GetAttrValue(node_def, "value", value))
    {
        const tensorflow::TensorProto& tf_tensor = value.tensor();
        void* mem_ptr = nullptr;
        int* dims;
        std::string layout;
        int dim_num = GetTensorContentAndDim(tf_tensor, &dims, &mem_ptr, layout);
        return mem_ptr;
    }

    return nullptr;
}

int load_reduction(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    TFNode* input1 = tf_node->inputs[1];
    struct reduction_param* param = (struct reduction_param*)node->op.param_mem;
    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];
    tensorflow::AttrValue value;
    GetAttrValue(node_def, "keep_dims", value);
    param->keepdim = value.b();
    if (tf_node->op == "Sum")
    {
        param->type = 0;
    }
    else if (tf_node->op == "Mean")
    {
        param->type = 1;
    }
    else if (tf_node->op == "Asum")
    {
        param->type = 2;
    }
    else if (tf_node->op == "Sqsum")
    {
        param->type = 3;
    }
    else if (tf_node->op == "Max")
    {
        param->type = 4;
    }
    else if (tf_node->op == "Min")
    {
        param->type = 5;
    }
    else if (tf_node->op == "Prod")
    {
        param->type = 6;
    }
    else if (tf_node->op == "L2")
    {
        param->type = 7;
    }
    else if (tf_node->op == "Logsum")
    {
        param->type = 8;
    }
    else if (tf_node->op == "Logsumexp")
    {
        param->type = 9;
    }
    int* data = (int*)LoadConstParam(input1);
    return 0;
}
int load_pad(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    struct pad_param* param = (struct pad_param*)node->op.param_mem;

    TFNode* input = tf_node->inputs[1];
    int* paddings = (int*)LoadConstParam(input);
    param->mode = 0;
    param->pad_0_h = paddings[0];
    param->pad_0_w = paddings[1];
    param->pad_1_h = paddings[2];
    param->pad_1_w = paddings[3];
    param->pad_2_h = paddings[4];
    param->pad_2_w = paddings[5];
    param->pad_3_h = paddings[6];
    param->pad_3_w = paddings[7];
    return 0;
}

int load_concat(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    struct concat_param* param = (struct concat_param*)node->op.param_mem;
    TFNode* input;

    const tensorflow::NodeDef* node_def = tf_node->pb_defs[1];
    tensorflow::AttrValue value;

    if (GetAttrValue(node_def, "value", value))
    {
        const tensorflow::TensorProto& tf_tensor = value.tensor();

        int axis = tf_tensor.int_val(0);
        param->axis = NCHW_axis_swap[axis];
    }
    else
    {
        param->axis = 3;
    }

    return true;
}
int load_gemm(TFNode* tf_node, TFGraph& tf_graph, ir_graph_t* graph, ir_node_t* node)
{
    TFNode* input1 = tf_node->inputs[1];
    struct gemm_param* param = (struct gemm_param*)node->op.param_mem;

    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];
    tensorflow::AttrValue value;
    if (GetAttrValue(node_def, "transpose_a", value))
    {
        param->transA = value.b();
    }
    if (GetAttrValue(node_def, "transpose_b", value))
    {
        param->transB = value.b();
    }
    param->alpha = 1;
    param->beta = 1;
    ir_tensor_t* weight_tensor = input1->ir_tensor;
    if (!param->transB)
    {
        // swap shape
        int k = weight_tensor->dims[0];
        int n = weight_tensor->dims[1];

        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        float* tmp = (float*)malloc(k * n * sizeof(float));
        float* data = (float*)weight_tensor->data;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }

        memcpy(data, tmp, n * k * sizeof(float));

        free(tmp);
    }

    struct fc_param* fcparam = (struct fc_param*)node->op.param_mem;
    fcparam->num_output = weight_tensor->dims[0];
    return 0;
}

void tensorflow_serializer::register_op_load()
{
    op_load_map["AvgPool"] = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["MaxPool"] = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["Conv2D"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["DepthwiseConv2dNative"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["FusedBatchNorm"] = std::pair<int, op_load_t>(OP_BATCHNORM, load_batchnorm);
    op_load_map["Relu6"] = std::pair<int, op_load_t>(OP_RELU6, load_relu6);
    op_load_map["Relu"] = std::pair<int, op_load_t>(OP_RELU, load_relu);
    op_load_map["Softmax"] = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["Add"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Sub"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Mul"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Minimum"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Rsqrt"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Exp"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Log"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Pow"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["RealDiv"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Sqrt"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["AddN"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Mean"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduction);
    op_load_map["Pad"] = std::pair<int, op_load_t>(OP_PAD, load_pad);
    op_load_map["ConcatV2"] = std::pair<int, op_load_t>(OP_CONCAT, load_concat);
    op_load_map["MatMul"] = std::pair<int, op_load_t>(OP_FC, load_gemm);
}
