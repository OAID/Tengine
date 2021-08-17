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
    if(it != attr.end())
    {
        value = it->second;
        return true;
    }

    return false;
}

static void GetTensorContentAndDim(const tensorflow::TensorProto& tf_tensor, int* dims, void** mem_ptr,
                                   std::string& layout, int dim_num)
{
    const tensorflow::TensorShapeProto& shape = tf_tensor.tensor_shape();

    int elem_num = 1;
    dim_num = shape.dim_size();

    for(int i = 0; i < dim_num; i++)
    {
        elem_num *= shape.dim(i).size();
        dims[i] = shape.dim(i).size();
    }

    void* mem_buf = nullptr;

    if(tf_tensor.tensor_content().size())
    {
        int content_size = tf_tensor.tensor_content().size();

        mem_buf = malloc(content_size + 128);
        void* src = ( void* )tf_tensor.tensor_content().c_str();
        memcpy(mem_buf, src, content_size);
    }
    else if(tf_tensor.dtype() == tensorflow::DataType::DT_FLOAT)
    {
        // in packed format
        int data_num = tf_tensor.float_val_size();
        mem_buf = malloc(elem_num * sizeof(float));
        float* mem = ( float* )mem_buf;

        if(data_num >= elem_num)
        {
            for(int i = 0; i < elem_num; i++)
            {
                mem[i] = tf_tensor.float_val(i);
            }
        }
        else
        {
            // data_num < elem_num
            for(int i = 0; i < data_num; i++)
            {
                mem[i] = tf_tensor.float_val(i);
            }

            for(int i = data_num; i < elem_num; i++)
            {
                mem[i] = mem[data_num - 1];
            }
        }
    }
    else if(tf_tensor.dtype() == tensorflow::DataType::DT_INT32)
    {
        int data_num = tf_tensor.int_val_size();

        mem_buf = malloc(elem_num * sizeof(int));

        int* mem = ( int* )mem_buf;

        if(data_num >= elem_num)
        {
            for(int i = 0; i < elem_num; i++)
            {
                mem[i] = tf_tensor.int_val(i);
            }
        }
        else
        {
            // data_num < elem_num
            for(int i = 0; i < data_num; i++)
            {
                mem[i] = tf_tensor.int_val(i);
            }

            for(int i = data_num; i < elem_num; i++)
            {
                mem[i] = mem[data_num - 1];
            }
        }
    }

    *mem_ptr = mem_buf;

    switch(dim_num)
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
}
int tensorflow_serializer::load_binary_file(std::string model_file)
{
    std::ifstream is(model_file.c_str(), std::ios::in | std::ios::binary);

    if(!is.is_open())
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

    if(!ret)
    {
        TLOG_ERR( "parse file: %s failed\n",model_file.c_str());
        return -1;
    }
    printf("graphd node numebr : %d \n", tf_net.node_size());
    return ret;
}
int load_const_tensor(TFNode* tf_node, ir_graph_t* graph)
{
    ir_node_t* node = create_ir_node(graph, tf_node->name.c_str(), OP_CONST, OP_VERSION);
    ir_tensor_t* tensor = create_ir_tensor(graph, tf_node->name.c_str(),TENGINE_DT_FP32);
    tensorflow::AttrValue value;

    const tensorflow::NodeDef* node_def = tf_node->pb_defs[0];
    if(GetAttrValue(node_def, "value", value))
    {
        const tensorflow::TensorProto& tf_tensor = value.tensor();
        void* mem_ptr;
        int* dims ;
        int dim_num;
        std::string layout;
        GetTensorContentAndDim(tf_tensor, dims, &mem_ptr, layout, dim_num);
        int mem_size = sizeof(float);
        for(unsigned int i = 0; i < dim_num; i++)
        {
            mem_size *= dims[i];
        }
        set_ir_tensor_shape(tensor, dims, dim_num);

        // SetTensorDim(tensor, dims, dims.size());
    }
    set_ir_node_output_tensor(node, 0, tensor);
    tf_node->ir_node = node;
    tf_node->ir_tensor = tensor;
}

int tensorflow_serializer::set_graph_input(ir_graph_t* graph)
{
    int node_num = tf_graph.seq_nodes.size();
    std::vector<int16_t> input_nodes;
    for(int i = 0; i < node_num; i++)
    {
        TFNode* tf_node =tf_graph.seq_nodes[i];
        if(tf_node->op == "Placeholder")
        {
            ir_tensor_t* ir_tensor = create_ir_tensor(graph, tf_node->name.c_str(), TENGINE_DT_FP32);
            tensorflow::AttrValue shape;

            int pb_defs_cnt = tf_node->pb_defs.size();
            int* dims;
            if(pb_defs_cnt == 1)
            {
                if(GetAttrValue(tf_node->pb_defs[0], "shape", shape))
                {
                    int dim_size = shape.shape().dim_size();
  
                    dims = (int*)sys_malloc(dim_size);
                    memset(dims, 0, sizeof(int)*dim_size);
                    for(int i = 0; i < dim_size; ++i)
                    {
                        dims[i] = shape.shape().dim(i).size();
                    }
                    set_ir_tensor_shape(ir_tensor, dims, dim_size);
                }
            }
            else
            {
                tensorflow::AttrValue value;
                const tensorflow::NodeDef* node_def = tf_node->pb_defs[pb_defs_cnt - 1];
                if(GetAttrValue(node_def, "value", value))
                {
                    const tensorflow::TensorProto& tf_tensor = value.tensor();

                    void* mem_ptr;
                    std::vector<int> tf_dims;
                    std::string layout;
                    int dim_num = 0;
                    GetTensorContentAndDim(tf_tensor, dims, &mem_ptr, layout, dim_num);
                    dims = (int*)sys_malloc(tf_dims[0]);
                    memset(dims, 0, sizeof(int)*tf_dims[0]);
                    int* reshape_dim = ( int* )mem_ptr;
                    for(int i = 0; i < tf_dims[0]; i++)
                    {
                        dims[i] = reshape_dim[i];
                    }

                    for(unsigned int i = 0; i < tf_dims[0]; i++)
                    {
                        if(dims[i] == -1)
                            dims[i] = 1;
                    }
                    free(mem_ptr);
                    set_ir_tensor_shape(ir_tensor, dims, tf_dims[0]);
                }
            }
            ir_node_t* node = create_ir_node(graph,  tf_node->name.c_str(), OP_INPUT, OP_VERSION);

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
    printf("tf node_number: %d \n", tf_net.node_size());
    for(int i = 0; i < node_number; i++)
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
    for(int i = 0; i < node_number; i++)
    {
        const tensorflow::NodeDef& node_param = tf_net.node(i);
        const std::string& name = node_param.name();

        TFNode* cur_node = node_map[name];

        for(int j = 0; j < node_param.input_size(); j++)
        {
            const std::string& input_name = node_param.input(j);
            std::string::size_type pos = input_name.find(":");
            std::string cleanup_name;

            if(pos == std::string::npos)
                pos = input_name.size();

            if(input_name[0] == '^')
                cleanup_name = input_name.substr(1, pos);
            else
                cleanup_name = input_name.substr(0, pos);

            TFNode* input_node = node_map[cleanup_name];

            if(input_node == nullptr)
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

    for(unsigned int i = 0; i < cur_node->inputs.size(); i++)
    {
        input_node = cur_node->inputs[i];

        auto ir = input_node->outputs.begin();

        while(ir != input_node->outputs.end())
        {
            if(*ir != cur_node)
                ir++;
            else
                break;
        }

        if(ir == input_node->outputs.end())
        {
            TLOG_ERR("ERROR on node connection!!\n");
        }

        input_node->outputs.erase(ir);
    }

    cur_node->inputs.clear();

    TFNode* output_node;

    for(unsigned int i = 0; i < cur_node->outputs.size(); i++)
    {
        output_node = cur_node->outputs[i];

        auto ir = output_node->inputs.begin();

        while(ir != output_node->inputs.end())
        {
            if(*ir != cur_node)
                ir++;
            else
                break;
        }

        if(ir == output_node->inputs.end())
        {
            TLOG_ERR("ERROR on node connection!!\n");
        }

        output_node->inputs.erase(ir);
    }

    cur_node->outputs.clear();
}

int MergeParentNode(TFNode* base_node, TFNode* parent_node)
{
    /* remove the input for parent node */

    auto input_ir = base_node->inputs.begin();

    while(input_ir != base_node->inputs.end())
    {
        if(*input_ir == parent_node)
            break;

        input_ir++;
    }

    if(parent_node->inputs.size() == 1)
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

    for(auto node : parent_node->inputs)
    {
        for(unsigned int i = 0; i < node->outputs.size(); i++)
        {
            if(node->outputs[i] == parent_node)
            {
                node->outputs[i] = base_node;
                break;
            }
        }
    }

    /* bridge parent's output, for those edges do not connect with base node */

    auto output_ir = parent_node->outputs.begin();

    while(output_ir != parent_node->outputs.end())
    {
        TFNode* node = *output_ir;

        if(node != base_node)
        {
            base_node->outputs.push_back(node);

            for(unsigned int i = 0; i < node->inputs.size(); i++)
            {
                if(node->inputs[i] == parent_node)
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

int CheckComposedBNAdd(TFNode* cur_node)
{
    if(cur_node->op != "Add")
        return 0;

    TFNode* input0 = cur_node->inputs[0];
    TFNode* input1 = cur_node->inputs[1];

    if(input0->op != "Mul" || input1->op != "Sub")
        return 0;

    /* further check: /add_1 int name */
    if(cur_node->name.find("/add_1") != std::string::npos)
    {
        if(input0->name.find("/mul_1") != std::string::npos || input1->name.find("/mul_1") != std::string::npos)
            cur_node->BNAddType = 1;
        else
            cur_node->BNAddType = 0;

        return 0;
    }

    return -1;
}

int BNRecursiveInputMerge(TFNode* node)
{
    bool mul_1_node = false;
    bool mul_node = false;
    if(node->name.find("/mul") != std::string::npos)
    {
        if(node->BNAddType == 1)
        {
            if(node->name.find("/mul_1") != std::string::npos)
            {
                mul_1_node = true;
            }
            else if(node->name.find("/mul_2") == std::string::npos)
            {
                // disconnect the connection between mul and mul2
                auto ir = node->outputs.begin();

                if((*ir)->name.find("/mul2") == std::string::npos)
                    ir++;

                TFNode* mul2_node = *ir;

                node->outputs.erase(ir);

                ir = mul2_node->inputs.begin();

                if((*ir)->name.find("/mul") == std::string::npos)
                    ir++;

                mul2_node->inputs.erase(ir);
            }
        }
        else
        {
            if(node->name.find("/mul_1") != std::string::npos)
            {
                // disconnect the connection between add_1 mul_1
                auto ir = node->inputs.begin();

                if((*ir)->name.find("/add_1") == std::string::npos)
                    ir++;

                if((*ir)->name.find("/add_1") != std::string::npos)
                {
                    TFNode* Rsqrt_node = *ir;

                    node->inputs.erase(ir);

                    ir = Rsqrt_node->outputs.begin();

                    if((*ir)->name.find("/mul_1") == std::string::npos)
                        ir++;

                    Rsqrt_node->outputs.erase(ir);
                }
            }
            else
            {
                mul_node = true;
                // printf("name:%s\n",node->name.c_str());
            }
        }
    }

    int orig_input_size = node->inputs.size();
    std::vector<TFNode*> input_cpy = node->inputs;

    for(int i = 0; i < orig_input_size; i++)
    {
        if(mul_node && i == 0)
            continue;
        if(mul_1_node && i == 0)
            continue;

        TFNode* input_node = input_cpy[i];
        input_node->BNAddType = node->BNAddType;
        if(input_node->op == "Const")
            continue;

        BNRecursiveInputMerge(input_node);
        MergeParentNode(node, input_node);
    }
}
int FuseComposedBN(TFNode* cur_node)
{
    BNRecursiveInputMerge(cur_node);
    cur_node->op = "ComposedBN";

    /* set new name */
    auto pos = cur_node->name.find("/add_1");
    cur_node->name.replace(pos, strlen("/add_1"), "bn.fused");

    /* skip to create static node for add/y */

    for(unsigned int i = 0; i < cur_node->inputs.size(); i++)
    {
        TFNode* node = cur_node->inputs[i];

        if(node->name.find("/add/y") != std::string::npos)
            node->no_static_node = true;
    }
}
int MergeChildNode(TFNode* base_node, TFNode* child_node)
{
    auto output_ir = base_node->outputs.begin();

    while(output_ir != base_node->outputs.end())
    {
        if(*output_ir == child_node)
            break;
        output_ir++;
    }

    if(child_node->outputs.size() == 1)
    {
        *output_ir = child_node->outputs[0];
    }
    else
    {
        base_node->outputs.erase(output_ir);
        base_node->outputs.insert(base_node->outputs.end(), child_node->outputs.begin(), child_node->outputs.end());
    }

    for(auto node : child_node->outputs)
    {
        for(unsigned int i = 0; i < node->inputs.size(); i++)
        {
            if(node->inputs[i] == child_node)
            {
                node->inputs[i] = base_node;
                break;
            }
        }
    }

    auto ir = child_node->inputs.begin();

    while(ir != child_node->inputs.end())
    {
        TFNode* node = *ir;

        if(node != base_node)
        {
            base_node->inputs.push_back(node);

            for(unsigned int i = 0; i < node->outputs.size(); i++)
            {
                if(node->outputs[i] == child_node)
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


void CleanupResizeNearestNeighbor(TFGraph& tf_graph)
{
    auto ir = tf_graph.seq_nodes.begin();

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->op == "ResizeNearestNeighbor")
        {
            TFNode* data_node = cur_node->inputs[0];
            TFNode* data_shape_node = nullptr;

            for(unsigned int i = 0; i < data_node->outputs.size(); i++)
            {
                data_shape_node = data_node->outputs[i];

                if(data_shape_node->op == "Shape")
                    break;
            }

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
    for(auto ir = tf_graph.seq_nodes.begin(); ir != tf_graph.seq_nodes.end(); ir++)
    {
        TFNode* cur_node = *ir;

        if(cur_node->inputs.size() == 0)
            continue;

        TFNode* input0 = cur_node->inputs[0];

        if(cur_node->op == "Minimum" && input0->op == "Relu")
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
    /* first clean up the predictions module of TF */
    auto ir = tf_graph.seq_nodes.begin();
    
    /* remove the squeeze node and identity */
    ir = tf_graph.seq_nodes.begin();

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->op == "Squeeze")
        {
            TFNode* softmax_node = nullptr;
            TFNode* shape_node = nullptr;

            for(unsigned int j = 0; j < cur_node->outputs.size(); j++)
            {
                if(cur_node->outputs[j]->op == "Softmax")
                    softmax_node = cur_node->outputs[j];
                else if(cur_node->outputs[j]->op == "Shape")
                    shape_node = cur_node->outputs[j];
            }

            if(softmax_node)
            {
                if(shape_node)
                    DisconnectNode(shape_node);

                TFNode* input_node = cur_node->inputs[0];
                MergeChildNode(input_node, cur_node);
                ir = tf_graph.seq_nodes.erase(ir);
                delete cur_node;
                continue;
            }

            if(cur_node->outputs.size() == 1 && softmax_node == nullptr)
            {
                TFNode* child_node = cur_node->outputs[0];

                MergeParentNode(child_node, cur_node);
                ir = tf_graph.seq_nodes.erase(ir);
                delete cur_node;
                continue;
            }
        }

        if(cur_node->op == "Identity")
        {
            TFNode* input_node = cur_node->inputs[0];
            MergeChildNode(input_node, cur_node);

            ir = tf_graph.seq_nodes.erase(ir);
            delete cur_node;
            continue;
        }

        if(cur_node->op == "ConcatV2")
        {
            TFNode* axis_node = nullptr;

            for(unsigned int i = 0; i < cur_node->inputs.size(); i++)
            {
                TFNode* check_node = cur_node->inputs[i];

                if(check_node->op == "Const")
                {
                    axis_node = check_node;
                    break;
                }
            }

            if(axis_node)
            {
                cur_node->pb_defs.push_back(axis_node->pb_defs[0]);
                DisconnectNode(axis_node);
            }
        }

        ir++;
    }

    /* merge FIFOQueueV2  DequeueManyV2 */

    ir = tf_graph.seq_nodes.begin();

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->op == "FIFOQueueV2")
        {
            TFNode* queue_node = cur_node->outputs[0];

            if(queue_node->op == "QueueDequeueManyV2")
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

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->op == "ExpandDims")
        {
            TFNode* input0 = cur_node->inputs[0];
            TFNode* input1 = cur_node->inputs[1];

            if(input0->op == "Constant" && input1->op == "Const")
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
                if(input1->op == "Const")
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

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->op == "Conv2D" || cur_node->op == "DepthwiseConv2dNative" || cur_node->op == "MatMul")
        {
            TFNode* output_node = cur_node->outputs[0];

            if(output_node->op == "BiasAdd" || output_node->op == "Add")
            {
                MergeChildNode(cur_node, output_node);
            }
        }

        ir++;
    }

    /* merge composed BatchNormal */

    ir = tf_graph.seq_nodes.begin();

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(CheckComposedBNAdd(cur_node))
            FuseComposedBN(cur_node);
        ir++;
    }

    /* cleanup ResizeNearestNeighbor */
    CleanupResizeNearestNeighbor(tf_graph);

    /* merge Minimum and Relu */

    MergeReluMinimum();
    /* merge input node and reshape */
    ir = tf_graph.seq_nodes.begin();
    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;
        if(cur_node->op == "Reshape")
        {
            /* Reshape should have two inputs */
            TFNode* input_node0 = cur_node->inputs[0];
            TFNode* input_node1 = cur_node->inputs[1];

            if(input_node0->op == "Placeholder" || input_node1->op == "Placeholder")
            {
                TFNode* input_node;
                TFNode* const_node;

                if(input_node0->op == "Const")
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

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;
        if(cur_node->op == "StridedSlice")
        {
            /* check if input0 is "shape" */
            TFNode* input_node = cur_node->inputs[0];

            if(input_node->op == "Shape")
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

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->op == "Conv2D" || cur_node->op == "DepthwiseConv2dNative")
        {
            /* check if input is pad or not */
            TFNode* input_node = cur_node->inputs[0];

            if(input_node->op == "Pad")
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
    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;
        if(cur_node->op == "ArgMax")
        {
            DisconnectNode(cur_node);
            tf_graph.seq_nodes.erase(ir);

            break;
        }

        ir++;
    }

    /* remove last squeeze */

    ir = tf_graph.seq_nodes.begin();

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->op == "Squeeze" && cur_node->outputs.empty())
        {
            DisconnectNode(cur_node);
            break;
        }
        ir++;
    }

    /* remove no input and output nodes */

    ir = tf_graph.seq_nodes.begin();

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->inputs.size() == 0 && cur_node->outputs.size() == 0)
        {
            ir = tf_graph.seq_nodes.erase(ir);
            delete cur_node;
        }
        else
            ir++;
    }

    /* remove no input but not placeholder/const nodes */
    ir = tf_graph.seq_nodes.begin();

    while(ir != tf_graph.seq_nodes.end())
    {
        TFNode* cur_node = *ir;

        if(cur_node->inputs.size() == 0 && cur_node->op != "Const" && cur_node->op != "Placeholder" &&
           cur_node->op != "FIFOQueueV2")
        {
            DisconnectNode(cur_node);
            tf_graph.seq_nodes.erase(ir);
            delete cur_node;

            ir = tf_graph.seq_nodes.begin();    // restart
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

    for(unsigned int i = 0; i < tf_graph.seq_nodes.size(); i++)
    {
        TFNode* node = tf_graph.seq_nodes.at(i);
        std::string& name = node->name;

        while_pos = name.find("while");

        if(while_pos == std::string::npos)
            continue;

        std::string::size_type cell_pos = name.find("lstm_cell", while_pos);

        if(cell_pos != std::string::npos)
        {
            rnn_node = node->name;
            rnn_type = TF_RNN_LSTM;
            break;
        }

        cell_pos = name.find("gru_cell", while_pos);

        if(cell_pos != std::string::npos)
        {
            rnn_node = node->name;
            rnn_type = TF_RNN_GRU;
            break;
        }

        cell_pos = name.find("basic_lstm_cell", while_pos);

        if(cell_pos != std::string::npos)
        {
            rnn_node = node->name;
            rnn_type = TF_RNN_BASIC_LSTM;
            break;
        }

        cell_pos = name.find("basic_rnn_cell", while_pos);

        if(cell_pos != std::string::npos)
        {
            rnn_node = node->name;
            rnn_type = TF_RNN_BASIC_RNN;
            break;
        }
    }

    if(rnn_node.empty())
        return -1;

    std::string rnn_layer = rnn_node.substr(0, while_pos - 1);
    std::string::size_type up_pos = rnn_layer.rfind("/");

    rnn_scope = rnn_layer.substr(0, up_pos + 1);

    return rnn_type;
}

void tensorflow_serializer::ParseLSTMGraph(LSTMNode* lstm_node, std::set<TFNode*>& rnn_graph)
{
    /* parse input node */

    for(unsigned int i = 0; i < lstm_node->inputs.size(); i++)
    {
        TFNode* node = lstm_node->inputs[i];

        if(node->op != "Const")
            continue;

        // node->no_static_node=true; //do not automatically create Static Node

        if(node->name.find("lstm_cell/kernel") != std::string::npos)
        {
            lstm_node->kernel = node;
        }
        else if(node->name.find("lstm_cell/bias") != std::string::npos)
        {
            lstm_node->bias = node;
        }
        else if(node->name.find("lstm_cell/w_f_diag") != std::string::npos)
        {
            lstm_node->w_f_diag = node;
        }
        else if(node->name.find("lstm_cell/w_o_diag") != std::string::npos)
        {
            lstm_node->w_o_diag = node;
        }
        else if(node->name.find("lstm_cell/w_i_diag") != std::string::npos)
        {
            lstm_node->w_i_diag = node;
        }
        else if(node->name.find("lstm_cell/projection/kernel") != std::string::npos)
        {
            lstm_node->projection = node;
        }
    }

    auto rnn_ir = rnn_graph.begin();
    auto rnn_ir_end = rnn_graph.end();

    while(rnn_ir != rnn_ir_end)
    {
        TFNode* node = *rnn_ir;
        int name_len = node->name.size();
        std::string zero_name = "LSTMCellZeroState/zeros";
        std::string zero1_name = "LSTMCellZeroState/zeros_1";
        std::string forget_name = "lstm_cell/add/y";

        if(node->name.find(zero_name, name_len - zero_name.size()) != std::string::npos)
            lstm_node->init_c = node;
        else if(node->name.find(zero1_name, name_len - zero1_name.size()) != std::string::npos)
            lstm_node->init_h = node;
        else if(node->name.find(forget_name, name_len - forget_name.size()) != std::string::npos)
            lstm_node->forget_bias = node;

        rnn_ir++;
    }
}
void ParseGRUGraph(TFGraph& tf_graph, GRUNode* gru_node, std::set<TFNode*>& rnn_graph)
{
    /* parse input node */

    for(unsigned int i = 0; i < gru_node->inputs.size(); i++)
    {
        TFNode* node = gru_node->inputs[i];

        if(node->op != "Const")
            continue;

        // node->no_static_node=true; //do not automatically create Static Node

        if(node->name.find("gru_cell/gates/kernel") != std::string::npos)
        {
            gru_node->gate_kernel = node;
        }
        else if(node->name.find("gru_cell/gates/bias") != std::string::npos)
        {
            gru_node->gate_bias = node;
        }
        else if(node->name.find("gru_cell/candidate/kernel") != std::string::npos)
        {
            gru_node->candidate_kernel = node;
        }
        else if(node->name.find("gru_cell/candidate/bias") != std::string::npos)
        {
            gru_node->candidate_bias = node;
        }
    }

    auto rnn_ir = rnn_graph.begin();
    auto rnn_ir_end = rnn_graph.end();

    while(rnn_ir != rnn_ir_end)
    {
        TFNode* node = *rnn_ir;
        int name_len = node->name.size();
        std::string zero_name = "GRUCellZeroState/zeros";

        if(node->name.find(zero_name, name_len - zero_name.size()) != std::string::npos)
            gru_node->init_h = node;

        rnn_ir++;
    }
}
void ParseRNNGraph(TFGraph& tf_graph, RNNNode* rnn_node, std::set<TFNode*>& rnn_graph)
{
    /* parse input node */

    for(unsigned int i = 0; i < rnn_node->inputs.size(); i++)
    {
        TFNode* node = rnn_node->inputs[i];

        if(node->op != "Const")
            continue;

        // node->no_static_node=true; //do not automatically create Static Node

        if(node->name.find("basic_rnn_cell/kernel") != std::string::npos)
        {
            rnn_node->kernel = node;
        }
        else if(node->name.find("basic_rnn_cell/bias") != std::string::npos)
        {
            rnn_node->bias = node;
        }
    }

    auto rnn_ir = rnn_graph.begin();
    auto rnn_ir_end = rnn_graph.end();

    while(rnn_ir != rnn_ir_end)
    {
        TFNode* node = *rnn_ir;
        int name_len = node->name.size();
        std::string zero_name = "BasicRNNCellZeroState/zeros";

        if(node->name.find(zero_name, name_len - zero_name.size()) != std::string::npos)
            rnn_node->init_h = node;

        rnn_ir++;
    }
}
void tensorflow_serializer::StripRNNScope(std::string& rnn_scope, int rnn_type)
{
    // collect attributes according to rnn_type

    if(rnn_type == TF_RNN_LSTM)
    {
        LSTMNode* lstm_node = new LSTMNode();

        lstm_node->name = rnn_scope + "lstm";
        lstm_node->op = "LSTM";

        std::set<TFNode*>& rnn_graph = lstm_node->rnn_graph;

        std::set<TFNode*> rnn_inputs;
        std::set<TFNode*> rnn_outputs;

        auto ir = tf_graph.seq_nodes.begin();
        std::string::size_type prefix_len = rnn_scope.size();

        while(ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *ir;

            if(node->name.find(rnn_scope.c_str(), 0, prefix_len) == std::string::npos)
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

        while(rnn_ir != rnn_end)
        {
            TFNode* node = *rnn_ir;

            for(unsigned int i = 0; i < node->inputs.size(); i++)
            {
                TFNode* input = node->inputs[i];

                if(!rnn_graph.count(input))
                    rnn_inputs.insert(input);
            }

            for(unsigned int i = 0; i < node->outputs.size(); i++)
            {
                TFNode* output = node->outputs[i];

                if(!rnn_graph.count(output))
                    rnn_outputs.insert(output);
            }

            rnn_ir++;
        }

        // insert lstm node
        auto seq_ir = tf_graph.seq_nodes.begin();

        while(seq_ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *seq_ir;

            if(rnn_inputs.count(node))
            {
                tf_graph.seq_nodes.insert(seq_ir, lstm_node);
                break;
            }

            seq_ir++;
        }

        // connect inputs and outputs
        auto set_ir = rnn_inputs.begin();
        auto set_ir_end = rnn_inputs.end();

        while(set_ir != set_ir_end)
        {
            TFNode* input_node = *set_ir;

            for(unsigned int j = 0; j < input_node->outputs.size(); j++)
            {
                TFNode* child_node = input_node->outputs[j];

                if(rnn_graph.count(child_node))
                    input_node->outputs[j] = lstm_node;
            }

            lstm_node->inputs.push_back(input_node);

            if(input_node->op == "Identity")
            {
                TFNode* parent_node = input_node->inputs[0];

                MergeChildNode(parent_node, input_node);
            }

            set_ir++;
        }

        set_ir = rnn_outputs.begin();
        set_ir_end = rnn_outputs.end();

        while(set_ir != set_ir_end)
        {
            TFNode* output_node = *set_ir;

            for(unsigned int j = 0; j < output_node->inputs.size(); j++)
            {
                TFNode* parent_node = output_node->inputs[j];

                if(rnn_graph.count(parent_node))
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

    if(rnn_type == TF_RNN_BASIC_RNN)
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

        while(ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *ir;

            if(node->name.find(rnn_scope.c_str(), 0, prefix_len) == std::string::npos)
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

        while(rnn_ir != rnn_end)
        {
            TFNode* node = *rnn_ir;

            for(unsigned int i = 0; i < node->inputs.size(); i++)
            {
                TFNode* input = node->inputs[i];

                if(!rnn_graph.count(input))
                    rnn_inputs.insert(input);
            }

            for(unsigned int i = 0; i < node->outputs.size(); i++)
            {
                TFNode* output = node->outputs[i];

                if(!rnn_graph.count(output))
                    rnn_outputs.insert(output);
            }

            rnn_ir++;
        }

        // insert rnn node
        auto seq_ir = tf_graph.seq_nodes.begin();

        while(seq_ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *seq_ir;

            if(rnn_inputs.count(node))
            {
                tf_graph.seq_nodes.insert(seq_ir, rnn_node);
                break;
            }

            seq_ir++;
        }

        // connect inputs and outputs
        auto set_ir = rnn_inputs.begin();
        auto set_ir_end = rnn_inputs.end();

        while(set_ir != set_ir_end)
        {
            TFNode* input_node = *set_ir;

            for(unsigned int j = 0; j < input_node->outputs.size(); j++)
            {
                TFNode* child_node = input_node->outputs[j];

                if(rnn_graph.count(child_node))
                    input_node->outputs[j] = rnn_node;
            }

            rnn_node->inputs.push_back(input_node);

            if(input_node->op == "Identity")
            {
                TFNode* parent_node = input_node->inputs[0];

                MergeChildNode(parent_node, input_node);
            }

            set_ir++;
        }

        set_ir = rnn_outputs.begin();
        set_ir_end = rnn_outputs.end();

        while(set_ir != set_ir_end)
        {
            TFNode* output_node = *set_ir;

            for(unsigned int j = 0; j < output_node->inputs.size(); j++)
            {
                TFNode* parent_node = output_node->inputs[j];

                if(rnn_graph.count(parent_node))
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
    if(rnn_type == TF_RNN_GRU)
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

        while(ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *ir;

            if(node->name.find(rnn_scope.c_str(), 0, prefix_len) == std::string::npos)
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

        while(rnn_ir != rnn_end)
        {
            TFNode* node = *rnn_ir;

            for(unsigned int i = 0; i < node->inputs.size(); i++)
            {
                TFNode* input = node->inputs[i];

                if(!rnn_graph.count(input))
                    rnn_inputs.insert(input);
            }

            for(unsigned int i = 0; i < node->outputs.size(); i++)
            {
                TFNode* output = node->outputs[i];

                if(!rnn_graph.count(output))
                    rnn_outputs.insert(output);
            }

            rnn_ir++;
        }

        // insert rnn node
        auto seq_ir = tf_graph.seq_nodes.begin();

        while(seq_ir != tf_graph.seq_nodes.end())
        {
            TFNode* node = *seq_ir;

            if(rnn_inputs.count(node))
            {
                tf_graph.seq_nodes.insert(seq_ir, gru_node);
                break;
            }

            seq_ir++;
        }

        // connect inputs and outputs
        auto set_ir = rnn_inputs.begin();
        auto set_ir_end = rnn_inputs.end();

        while(set_ir != set_ir_end)
        {
            TFNode* input_node = *set_ir;

            for(unsigned int j = 0; j < input_node->outputs.size(); j++)
            {
                TFNode* child_node = input_node->outputs[j];

                if(rnn_graph.count(child_node))
                    input_node->outputs[j] = gru_node;
            }

            gru_node->inputs.push_back(input_node);

            if(input_node->op == "Identity")
            {
                TFNode* parent_node = input_node->inputs[0];

                MergeChildNode(parent_node, input_node);
            }

            set_ir++;
        }

        set_ir = rnn_outputs.begin();
        set_ir_end = rnn_outputs.end();

        while(set_ir != set_ir_end)
        {
            TFNode* output_node = *set_ir;

            for(unsigned int j = 0; j < output_node->inputs.size(); j++)
            {
                TFNode* parent_node = output_node->inputs[j];

                if(rnn_graph.count(parent_node))
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

    while(seq_ir != tf_graph.seq_nodes.end())
    {
        TFNode* node = *seq_ir;

        if(node->inputs.size() == 0 && node->outputs.size() == 0)
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
    while(1)
    {
        std::string rnn_scope;

        int rnn_type = FindRNNScope(rnn_scope);

        if(rnn_scope.empty())
            break;

        StripRNNScope(rnn_scope, rnn_type);
    }

    return true;
}


int tensorflow_serializer::generate_graph(ir_graph_t* graph)
{
    int node_number = tf_graph.seq_nodes.size();
    int i;

    bool debug_graph = false;
    const char* debug_env = std::getenv("DEBUG_TF");
    if((debug_env) && (debug_env[0] == '1'))
    {
        debug_graph = true;
    }

    printf("node_number: %d \n", node_number);
    // first: create all tensor node
    for(i = 0; i < node_number; i++)
    {
        TFNode* tf_node = tf_graph.seq_nodes[i];

        if(debug_graph)
        {
            std::cout << i << "\t" << tf_node->op << "\t" << tf_node->name << "\n";
        }

        if(tf_node->no_static_node)
            continue;

        if(tf_node->op == "Const")
        {
            load_const_tensor(tf_node, graph);
            continue;
        }


        // StaticNode* node = CreateStaticNode(graph, tf_node->name);
        // std::string& op_name = tf_node->op;
        ir_node_t* ir_node = create_ir_node(graph,tf_node->name.c_str(), op_load_map[tf_node->op].first, OP_VERSION);
        /* create tensor */
        ir_tensor_t* ir_tensor = create_ir_tensor(graph, tf_node->name.c_str(), TENGINE_DT_FP32);

        set_ir_node_output_tensor(ir_node, 0, ir_tensor);
        tf_node->ir_node = ir_node;
        tf_node->ir_tensor = ir_tensor;
    }

    std::vector<std::string> no_supported_op;
    for(i = 0; i < node_number; i++) 
    {    
        TFNode* tf_node = tf_graph.seq_nodes[i];

        if(tf_node->op == "Placeholder" || tf_node->op == "Const")
            continue;

        // if(!FindOpLoadMethod(tf_node->op))
        // {
        //     auto it = find(no_supported_op.begin(),no_supported_op.end(),tf_node->op);
        //     if(it != no_supported_op.end())
        //         no_supported_op.push_back(tf_node->op);
        // }    
    }    
    if(no_supported_op.size())
    {    
        TLOG_ERR("These ops are not supported \n", no_supported_op.size());
        TLOG_ERR("{"); 
        for(int j = 0; j < (int)no_supported_op.size(); j++) 
        {    
            TLOG_ERR("%d ", no_supported_op[j] );
        }    
        TLOG_ERR("}\n");
        return false;
    }
//    for(int i = 0; i < node_number; i++){
//         TFNode* tf_node = tf_graph.seq_nodes[i];
//         if(tf_node->op == "null" || tf_node->op == "Placeholder" || tf_node->op == "Const")
//             continue; 

//         std::vector<std::string>::iterator iter=std::find(support_op.begin(), support_op.end(), tf_node->op);
//         if(iter==support_op.end()){
//             std::vector<std::string>::iterator uniter=std::find(unsupport_op.begin(), unsupport_op.end(), tf_node->op);
//             if(uniter==unsupport_op.end()){
//                 unsupport_op.push_back(tf_node->op);
//             } else {
//                 continue;
//             }
//         } else {
//             continue;
//         }
//     }
//     if(unsupport_op.size() != 0){
//         printf("These ops are not in tensorflow serializer: \n");
//         for(int i = 0; i < (int)unsupport_op.size(); i++){
//             printf("[ %s ]\n", unsupport_op[i].c_str());
//         }
//         printf("\n");
//         return false;
//     }
    for(i = 0; i < node_number; i++)
    {
        TFNode* tf_node = tf_graph.seq_nodes[i];

        if(tf_node->op == "Placeholder" || tf_node->op == "Const")
            continue;


        op_load_t loader = op_load_map[tf_node->op].second;
        if (loader(graph, tf_node->ir_node, tf_node) < 0)
        {
            fprintf(stderr, "load op %s func failed in node %s .\n", tf_node->op.c_str(), tf_node->name.c_str());
            return -1;
        }

        // op_load_t loader = op_load_map[caffe_op_name].second;
        // if (loader(graph, ir_node, layer_param) < 0)
        // {
        //     TLOG_ERR("load op %s func failed in node %s .\n", caffe_op_name.c_str(), ir_node->name);
        //     return -1;
        // }
    }

    if(i < node_number)
        return false;

    return true;
}
int tensorflow_serializer::load_graph(ir_graph_t* graph)
{
    printf("tf_net node num : %d \n", tf_net.node_size());
    if (construct_graph() < 0)
        return -1;
    
    if(optimize_rnn() < 0)
        return false;

    if (optimize_graph() < 0)
        return -1;

    if (set_graph_input(graph) < 0)
        return -1;
    
    if(generate_graph(graph) < 0)
        return -1;

    return 0;
}


int tensorflow_serializer::load_model(ir_graph_t* graph, std::string model_file)
{
    register_op_load();
    if (load_binary_file(model_file) < 0)
        return -1;
    fprintf(stderr, "Process 1: Finish load protobuf file \n");
    // if (load_tensor_data(graph, test_net, train_net) < 0)
    //     return -1;
    // fprintf(stderr, "Process 2: Finish load graph node \n");

    load_graph(graph);
    // fprintf(stderr, "Process 3: Finish set graph input \n");
    // if (load_graph_node(graph, test_net, train_net) < 0)
    //     return -1;
    // fprintf(stderr, "Process 4: Finish load graph node \n");
    // if (set_graph_output(graph, test_net, train_net) < 0)
    //     return -1;
    // fprintf(stderr, "Process 5: Finish set graph output \n");
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

void tensorflow_serializer::register_op_load()
{

}