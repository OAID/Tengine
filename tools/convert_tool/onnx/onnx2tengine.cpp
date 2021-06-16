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

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <vector>

#include "onnx2tengine.hpp"


typedef int (*op_load_t)(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node);
std::unordered_map<std::string, std::pair<int, op_load_t>> op_load_map;
void register_op_load();
const int OP_VERSION = 1;


bool find_op_load_method(const std::string& op_name)
{
    if(op_load_map.count(op_name))
        return true;

    return false;
}

ir_tensor_t* find_tensor(ir_graph_t* graph, const std::string& tensor_name)
{
    for (uint16_t i = 0; i < graph->tensor_num; i++)
    {
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, i);
        if (tensor->name == tensor_name)
            return tensor;
    }
    
    return nullptr;
}

static int load_model_file(std::string model_file, onnx::ModelProto &model)
{
    std::ifstream is(model_file, std::ios::in | std::ios::binary);

    if(!is.is_open())
    {
        TLOG_ERR("cannot open file: %s \n", model_file.c_str());
        return -1;
    }

    google::protobuf::io::IstreamInputStream input_stream(&is);
    google::protobuf::io::CodedInputStream coded_input(&input_stream);

#if GOOGLE_PROTOBUF_VERSION >= 3011000
    coded_input.SetTotalBytesLimit(INT_MAX);
#else
    coded_input.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

    bool ret = model.ParseFromCodedStream(&coded_input);

    is.close();

    if(!ret)
    {
        TLOG_ERR("onnx serializer: parse file: %s \n", model_file.c_str());
        return -1;
    }

    return 0;
}
static onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto& node, const char* key)
{
    for (int i = 0; i < node.attribute_size(); i++)
    {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key)
        {
            return attr.t();
        }
    }

    return onnx::TensorProto();
}


static int load_constant_tensor(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
{
    std::map<std::string, onnx::TensorProto> node_tensor;
    int node_count = onnx_graph.node_size();

    for (int i = 0; i < node_count; i++)
    {
        const onnx::NodeProto& node = onnx_graph.node(i);
        const std::string& op = node.op_type();
        if (op == "Constant")
        {
            onnx::TensorProto node_attr = get_node_attr_tensor(node, "value");
            node_tensor.insert(std::pair<std::string, onnx::TensorProto>(node.output(0), node_attr));
        }
    }
    if (node_tensor.size() == 0)
    {
        return 0;
    }
    for (int i = 0; i < node_count; i++)
    {
        
        const onnx::NodeProto& node = onnx_graph.node(i);

        const std::string& op = node.op_type();

        
        if ((op == "Reshape" || op == "Gather" || op == "Div" || op == "Resize")  )
        {            
            const onnx::TensorProto& onnx_tensor = node_tensor[node.input(1)];
            
            int tensor_date_type = onnx_tensor.data_type() == 1 ? TENGINE_DT_FP32 : TENGINE_DT_INT32;
            const char* name = node.input(1).c_str();
            int dim_num = onnx_tensor.dims_size();
            int *dims = new int[dim_num];
            for (int j = 0; j < dim_num; j++)
            {
                dims[j] = onnx_tensor.dims(j);
            }

            // create ir tensor
            ir_tensor_t* ir_tensor = create_ir_tensor(graph, name, tensor_date_type);
            if (ir_tensor == NULL)
            {
                fprintf(stderr, "create ir tensor failed!\n");
                return -1;
            }
            set_ir_tensor_shape(ir_tensor, dims, dim_num);
            ir_tensor->tensor_type = TENSOR_TYPE_CONST;
            // set tensor data
            if (onnx_tensor.has_raw_data())
            {
                int tensor_size = ir_tensor->elem_size * ir_tensor->elem_num;
                if (onnx_tensor.data_type() == 1) //fp32
                {
                    ir_tensor->data = sys_malloc(tensor_size);
                    uint8_t* mem_buf = (uint8_t*)ir_tensor->data;
                    uint8_t* raw_data = (uint8_t*)onnx_tensor.raw_data().c_str();
                    for (int j = 0; j < tensor_size; j++)
                    {
                        mem_buf[j] = raw_data[2*j];
                    }
                }
                else // int32
                {
                    ir_tensor->data = sys_malloc(tensor_size);
                    int32_t* mem_buf = (int32_t*)ir_tensor->data;
                    int32_t* raw_data = (int32_t*)onnx_tensor.raw_data().data();
                    for (int j = 0; j < ir_tensor->elem_num; j++)
                    {
                        mem_buf[j] = raw_data[2*j];
                    }
                }
            }
            #if 0
            else
            {
                int tensor_size = ir_tensor->elem_size * ir_tensor->elem_num;
                if (onnx_tensor.data_type() == 1) //fp32
                {
                    ir_tensor->data = sys_malloc(tensor_size);
                    float* mem_buf = (float*)ir_tensor->data;
                    float* raw_data = (float*)onnx_tensor.float_data().data();
                    for (int j = 0; j < ir_tensor->elem_num; j++)
                    {
                        mem_buf[j] = raw_data[2*j];
                    }
                }
                else // int32
                {
                    ir_tensor->data = sys_malloc(tensor_size);
                    int32_t* mem_buf = (int32_t*)ir_tensor->data;
                    int32_t* raw_data = (int32_t*)onnx_tensor.int32_data().data();
                    for (int j = 0; j < ir_tensor->elem_num; j++)
                    {
                        mem_buf[j] = raw_data[2*j];
                    }
                }
            }
            #endif
            ir_node_t* ir_node = create_ir_node(graph, name, OP_CONST, OP_VERSION);
            set_ir_node_output_tensor(ir_node, 0, ir_tensor);
        }
        
    }
    
    return 0;
}

static int load_initializer_tensor(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
{
    int const_tensor_num = onnx_graph.initializer_size();
    for (int i = 0; i < const_tensor_num; i++)
    {
        const onnx::TensorProto& onnx_tensor = onnx_graph.initializer(i);
        
        if (onnx_tensor.data_type() != 1 && onnx_tensor.data_type() != 6 && onnx_tensor.data_type() != 7) // fp32 int32 int64
        {
            fprintf(stderr, "const tensor data type is not fp32 or int32 or int64. \n");
            fprintf(stderr, "onnx_tensor.data_type: %d \n", onnx_tensor.data_type());
            return -1;
        }
        int tensor_date_type = onnx_tensor.data_type() == 1 ? TENGINE_DT_FP32 : TENGINE_DT_INT32;
        const char* name = onnx_tensor.name().c_str();
        int dim_num = onnx_tensor.dims_size();
        int *dims = new int[dim_num];
        for (int j = 0; j < dim_num; j++)
        {
            dims[j] = onnx_tensor.dims(j);
        }

        // create ir tensor
        ir_tensor_t* ir_tensor = create_ir_tensor(graph, name, tensor_date_type);
        if (ir_tensor == NULL)
        {
            fprintf(stderr, "create ir tensor failed!\n");
            return -1;
        }
        set_ir_tensor_shape(ir_tensor, dims, dim_num);
        ir_tensor->tensor_type = TENSOR_TYPE_CONST;
        // set tensor data
        if (onnx_tensor.has_raw_data())
        {
            int tensor_size = ir_tensor->elem_size * ir_tensor->elem_num;
            if (onnx_tensor.data_type() == 1) //fp32
            {
                ir_tensor->data = sys_malloc(tensor_size);
                uint8_t* mem_buf = (uint8_t*)ir_tensor->data;
                uint8_t* raw_data = (uint8_t*)onnx_tensor.raw_data().c_str();
                for (int j = 0; j < tensor_size; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
            else // int32
            {
                ir_tensor->data = sys_malloc(tensor_size);
                int32_t* mem_buf = (int32_t*)ir_tensor->data;
                int32_t* raw_data = (int32_t*)onnx_tensor.raw_data().data();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
        }
        else
        {
            int tensor_size = ir_tensor->elem_size * ir_tensor->elem_num;
            if (onnx_tensor.data_type() == 1) //fp32
            {
                ir_tensor->data = sys_malloc(tensor_size);
                float* mem_buf = (float*)ir_tensor->data;
                float* raw_data = (float*)onnx_tensor.float_data().data();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
            else // int32
            {
                ir_tensor->data = sys_malloc(tensor_size);
                int32_t* mem_buf = (int32_t*)ir_tensor->data;
                int32_t* raw_data = (int32_t*)onnx_tensor.int32_data().data();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
        }
        ir_node_t* ir_node = create_ir_node(graph, name, OP_CONST, OP_VERSION);
        set_ir_node_output_tensor(ir_node, 0, ir_tensor);
    }
    return 0;
}


static int set_graph_input(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
{
    std::vector<int16_t> input_nodes;
    for (int i = 0; i < onnx_graph.input_size(); i++)
    {
        const onnx::ValueInfoProto& val = onnx_graph.input(i);
        if(get_ir_tensor_index_from_name(graph, val.name().c_str()) != -1)
            continue;

        // now, catch an input tensor
        const onnx::TypeProto& type = val.type();
        const onnx::TypeProto::Tensor& tensor_type = type.tensor_type();
        const onnx::TensorShapeProto& shape = tensor_type.shape();
        int has_shape = 1;
        int *dims = new int[shape.dim_size()];
        for(int j = 0; j < shape.dim_size(); j++)
        {
            const onnx::TensorShapeProto::Dimension& dim = shape.dim(j);
            if(dim.has_dim_param())
            {
                has_shape = 0;
                break;
            }
            dims[j] = dim.dim_value();
        }

        ir_tensor_t* tensor = create_ir_tensor(graph, val.name().c_str(), TENGINE_DT_FP32);
        if (has_shape)
            set_ir_tensor_shape(tensor, dims, shape.dim_size());
        ir_node_t* node = create_ir_node(graph, val.name().c_str(), OP_INPUT, OP_VERSION);
        set_ir_node_output_tensor(node, 0, tensor);
        input_nodes.push_back(node->index);
    }

    int16_t* node_idx = (int16_t*)sys_malloc(sizeof(int16_t) * input_nodes.size());
    for (int i = 0; i < input_nodes.size(); i++)
    {
        node_idx[i] = input_nodes[i];
    }
    set_ir_graph_input_node(graph, node_idx, input_nodes.size());
    return 0;
}

static int load_graph_node(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
{
    int i;
    std::vector<std::string> no_supported_op;
    for(i = 0; i < onnx_graph.node_size(); i++)
    {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& onnx_op_name = onnx_node.op_type();

        if(!find_op_load_method(onnx_op_name))
        {
            auto it = find(no_supported_op.begin(),no_supported_op.end(),onnx_op_name);
            if(it == no_supported_op.end())
            {
                if(onnx_op_name == "Constant")
                    continue;
                no_supported_op.push_back(onnx_op_name);
            }
        }
    }
    if(no_supported_op.size())
    {
        TLOG_ERR("These %d op are not supported\n{ ", no_supported_op.size());
        for(int j = 0; j < (int) no_supported_op.size(); j++)
        {
            TLOG_ERR("%s ", no_supported_op[j].c_str());
        }
        TLOG_ERR("}\n");
        return -1;
    }

    for(i = 0; i < onnx_graph.node_size(); i++)
    {
        /* create ir node*/
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& op_name = onnx_node.op_type();
        if (op_name == "Constant")
            continue;
        const std::string node_name = onnx_node.name();
        ir_node_t* ir_node = create_ir_node(graph, node_name.c_str(), op_load_map[op_name].first, OP_VERSION);
        if (ir_node == NULL)
            return -1;
        /* set ir node io */
        for (int j = 0; j < onnx_node.input_size(); j++)
        {
            const std::string& input_name = onnx_node.input(j);
            int tensor_id = get_ir_tensor_index_from_name(graph, input_name.c_str());
            if (tensor_id < 0 || tensor_id >= graph->tensor_num)
            {
                TLOG_ERR("can not find input tensor name: %s for node: %s\n", input_name.c_str(), node_name.c_str());
                return -1;
            }
            ir_tensor_t* tensor = get_ir_graph_tensor(graph, tensor_id);
            set_ir_node_input_tensor(ir_node, j, tensor);
        }
        for (int j = 0; j < onnx_node.output_size(); j++)
        {
            if (op_name == "Dropout" && j > 0)
                continue;
            const std::string& output_name = onnx_node.output(j);
            ir_tensor_t* tensor = create_ir_tensor(graph, output_name.c_str(), TENGINE_DT_FP32);
            set_ir_node_output_tensor(ir_node, j, tensor);
        }
        /* exec op load func */
        op_load_t loader = op_load_map[op_name].second;
        if (loader(graph, ir_node, onnx_node) < 0)
        {
            TLOG_ERR("load op %s func failed in node %s .\n", op_name.c_str(), node_name.c_str());
            return -1;
        }
    }
    return 0;
}

static int set_graph_output(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
{
    std::vector<int16_t> output_nodes;
    for (int i = 0; i < onnx_graph.output_size(); i++)
    {
        const onnx::ValueInfoProto& val = onnx_graph.output(i);
        int tensor_id = get_ir_tensor_index_from_name(graph, val.name().c_str());


        const onnx::TypeProto& type = val.type();
        const onnx::TypeProto::Tensor& tensor_type = type.tensor_type();
        const onnx::TensorShapeProto& shape = tensor_type.shape();
        int has_shape = 1;
        int *dims = new int[shape.dim_size()];
        for(int j = 0; j < shape.dim_size(); j++)
        {
            const onnx::TensorShapeProto::Dimension& dim = shape.dim(j);
            if(dim.has_dim_param())
            {
                has_shape = 0;
                break;
            }
            dims[j] = dim.dim_value();
        }
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, tensor_id);
        if (has_shape)
            set_ir_tensor_shape(tensor, dims, shape.dim_size());
        ir_node_t* node = create_ir_node(graph, val.name().c_str(), OP_CONST, OP_VERSION);
        set_ir_node_output_tensor(node, 0, tensor);
        output_nodes.push_back(node->index);
    }
    int16_t* node_idx = (int16_t*)sys_malloc(sizeof(int16_t) * output_nodes.size());
    for (int i = 0; i < output_nodes.size(); i++)
    {
        node_idx[i] = output_nodes[i];
    }
    set_ir_graph_output_node(graph, node_idx, output_nodes.size());
    return 0;
}

static int load_model(ir_graph_t* graph, std::string model_file)
{
    register_op_load();
    onnx::ModelProto model;
    if (load_model_file(model_file, model) < 0)
        return -1;
    const onnx::GraphProto& onnx_graph = model.graph();
    if (load_initializer_tensor(graph, onnx_graph) < 0)
        return -1;
    if (load_constant_tensor(graph, onnx_graph) < 0)
        return -1;
    if (set_graph_input(graph, onnx_graph) < 0)
        return -1;
    if (load_graph_node(graph, onnx_graph) < 0)
        return -1;
    if (set_graph_output(graph, onnx_graph) < 0)
        return -1;
    return 0;
}

graph_t onnx2tengine(std::string model_file)
{
    fprintf(stderr, "----------onnx2tengine begin----------\n");

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

    fprintf(stderr, "----------onnx2tengine done.----------\n");
    return ir_graph;
}

int change_node_op(ir_node_t* node, int new_op_type)
{
    sys_free(node->op.param_mem);
    node->op.type = new_op_type;
    ir_method_t* ir_method = find_op_method(new_op_type, OP_VERSION);
    if ((NULL != ir_method) && (NULL != ir_method->init) && (ir_method->init(&node->op) < 0))
    {
        return -1;
    }

    return 0;
}

int load_conv(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct conv_param* conv_param = ( struct conv_param* )node->op.param_mem;
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "kernel_shape")
        {
            conv_param->kernel_h = attr.ints(0);
            conv_param->kernel_w = attr.ints(1);
        }
        else if (attr.name() == "strides")
        {
            conv_param->stride_h = attr.ints(0);
            conv_param->stride_w = attr.ints(1);
        }
        else if (attr.name() == "pads")
        {
            conv_param->pad_h0 = attr.ints(0);
            conv_param->pad_h1 = attr.ints(2);
            conv_param->pad_w0 = attr.ints(1);
            conv_param->pad_w1 = attr.ints(3);
        }
        else if (attr.name() == "group")
        {
            conv_param->group = attr.i();
        }
        else if (attr.name() == "dilations")
        {
            conv_param->dilation_h = attr.ints(0);
            conv_param->dilation_w = attr.ints(0);
        }
        else if (attr.name() == "auto_pad")
        {
            const std::string& auto_pad = attr.s();

            if (auto_pad == "NOTSET")
            {
                continue;
            }
            else if (auto_pad == "SAME_UPPER")
            {
                // ToDo
                TLOG_ERR("%s attr.name: %s :SAME_UPPER todo implement.\n", node->name, attr.name().c_str());
            }
            else if (auto_pad == "SAME_LOWER" || auto_pad == "VALID")
            {
                // ToDo
                TLOG_ERR("%s attr.name: %s :SAME_LOWER todo implement.\n", node->name, attr.name().c_str());
            }
            else
                TLOG_ERR("%s attr.name: %s : %s not support.\n", node->name, attr.name().c_str(), auto_pad.c_str());
        }
        else
            TLOG_ERR("%s attr.name: %s \n", node->name, attr.name().c_str());
    }

    struct tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    conv_param->output_channel = weight->dims[0]; /* onnx hide the output channel in weight .. */

    return 0;
}

int load_relu(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct relu_param* relu_param = ( struct relu_param* )node->op.param_mem;
    relu_param->negative_slope = 0.f;
    return 0;
}

int load_pool(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct pool_param* pool_param = ( struct pool_param* )node->op.param_mem;
    const std::string& onnx_op = onnx_node.op_type();

    if(onnx_op == "GlobalAveragePool")
    {
        pool_param->global = 1;
        pool_param->pool_method = POOL_AVG;
    }
    else if(onnx_op == "MaxPool" || onnx_op == "AveragePool")
    {
        pool_param->global = 0;

        if(onnx_op == "AveragePool")
            pool_param->pool_method = POOL_AVG;
        else
            pool_param->pool_method = POOL_MAX;

        for(int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);

            if(attr.name() == "kernel_shape")
            {
                pool_param->kernel_h = attr.ints(0);
                pool_param->kernel_w = attr.ints(1);
            }
            else if(attr.name() == "strides")
            {
                pool_param->stride_h = attr.ints(0);
                pool_param->stride_w = attr.ints(1);
            }
            else if(attr.name() == "pads") /* onnx pads: x0_begin, x1_begin, ... , x0_end, x1_end, ... */
            {
                pool_param->pad_h0 = attr.ints(0);
                pool_param->pad_h1 = attr.ints(2);
                pool_param->pad_w0 = attr.ints(1);
                pool_param->pad_w1 = attr.ints(3);
                if (pool_param->pad_h0 == 0 && pool_param->pad_h1 == 1 && pool_param->pad_w0 == 0 && pool_param->pad_w1 == 1)
                    pool_param->caffe_flavor = 1;
            }
        }
    }
    else
    {
        TLOG_ERR("UKNOWN POOLING: %s \n", onnx_op.c_str());
        return -1;
    }
    return 0;
}

int load_flatten(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct flatten_param* flatten_param = ( struct flatten_param* )node->op.param_mem;
    flatten_param->axis = 1;

    if (1 == onnx_node.attribute_size())
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(0);
        flatten_param->axis = attr.i();
    }
    return 0;
}

int load_gemm(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct gemm_param* gemm_param = ( struct gemm_param* )node->op.param_mem;
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "alpha")
            gemm_param->alpha = attr.f();
        else if (attr.name() == "beta")
            gemm_param->beta = attr.f();
        else if (attr.name() == "transA")
            gemm_param->transA = attr.i();
        else if (attr.name() == "transB")
            gemm_param->transB = attr.i();
    }

    ir_tensor_t* weight_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    ir_tensor_t* bias_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);

    if (gemm_param->transA)
    {
        return 0;
    }

    // create fc instead
    if (!gemm_param->transB)
    {
        // swap shape
        int k = weight_tensor->dims[0];
        int n = weight_tensor->dims[1];
        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        float* tmp = ( float* )sys_malloc(k * n * sizeof(float));
        float* data = ( float* )weight_tensor->data;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }

        memcpy(data, tmp, n * k * sizeof(float));
        sys_free(tmp);
    }

    if (gemm_param->alpha != 1)
    {
        float* data = ( float* )weight_tensor->data;
        int tensor_size = weight_tensor->dims[0] * weight_tensor->dims[1];

        for (int i = 0; i < tensor_size; i++)
            data[i] *= gemm_param->alpha;
    }

    if (gemm_param->beta != 1)
    {
        float* data = ( float* )bias_tensor->data;
        int tensor_size = weight_tensor->dims[0];

        for (int i = 0; i < tensor_size; i++)
            data[i] *= gemm_param->beta;
    }

    if (change_node_op(node, OP_FC) < 0)
        return -1;
    struct fc_param* fc_param = (struct fc_param*)node->op.param_mem;
    fc_param->num_output = weight_tensor->dims[0];
    
    return 0;
}

int load_concat(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct concat_param* concat_param = ( struct concat_param* )node->op.param_mem;

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axis")
        {
            concat_param->axis = attr.i();
        }
    }

    return 0;
}

int load_bn(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct batchnorm_param* batchnorm_param = ( struct batchnorm_param* )node->op.param_mem;

    // get espilon
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "epsilon")
            batchnorm_param->eps = attr.f();
    }

    return 0;
}

int load_eltwise(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct eltwise_param* eltwise_param = ( struct eltwise_param* )node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();
    if (op_name == "Add")
        eltwise_param->type = ELT_SUM;
    else if (op_name == "Mul")
    {
        eltwise_param->type = ELT_PROD;
        for(int i = 0; i < onnx_node.input().size(); ++i)
        {
            ir_tensor_t* tensor = find_tensor(graph, onnx_node.input(i));
            if(tensor->dim_num == 0)
            {
                tensor->dim_num = 1;
                tensor->dims[0] = 1;
            }
        }
    }
    else if (op_name == "Div")
    {
        eltwise_param->type = ELT_DIV;
        for(int i = 0; i < onnx_node.input().size(); ++i)
        {
            ir_tensor_t* tensor = find_tensor(graph, onnx_node.input(i));
            if(tensor->dim_num == 0)
            {
                tensor->dim_num = 1;
                tensor->dims[0] = 1;
            }
        }
    }
    else if (op_name == "Floor")
        eltwise_param->type = ELT_FLOOR;
    else if (op_name == "Exp")
        eltwise_param->type = ELT_EXP;
    else if (op_name == "Sub")
        eltwise_param->type = ELT_SUB;
    else if (op_name == "Pow")
        eltwise_param->type = ELT_POW;
    else if (op_name == "Sqrt")
        eltwise_param->type = ELT_SQRT;

    return 0;
}

int load_transpose(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct transpose_param* transpose_param = ( struct transpose_param* )node->op.param_mem;
    
    const onnx::AttributeProto& attr = onnx_node.attribute(0);
    int size = attr.ints_size();
    transpose_param->tr_shape = (int*)sys_malloc(sizeof(int) * size);
    transpose_param->tr_shape_size = size;
    for (int i = 0; i < size; i++)
    {
        transpose_param->tr_shape[i] = attr.ints(i);
    }

    return 0;
}

int load_clip(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct clip_param* clip_param = ( struct clip_param* )node->op.param_mem;

    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "max")
        {
            clip_param->max = attr.f();
        }
        else if (attr.name() == "min")
        {
            clip_param->min = attr.f();
        }
    }

    return 0;
}

int load_reshape(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct reshape_param* reshape_param = ( struct reshape_param* )node->op.param_mem;

    ir_tensor_t* shape_tensor = find_tensor(graph, onnx_node.input(1));
    if (shape_tensor == nullptr)
    {
        fprintf(stderr, "find shape tensor of reshape node failed.\n");
        return -1;
    }
    reshape_param->is_onnx = 1;
    int size = shape_tensor->elem_num;
    reshape_param->re_shape = (int*)sys_malloc(sizeof(int) * size);
    reshape_param->dim_size = size;
    int64_t* data = (int64_t*)shape_tensor->data;
    for (int i = 0; i < size; i++)
    {
        reshape_param->re_shape[i] = data[i];
    }

    return 0;
}

int load_no_param(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    // no param
    return 0;
}

int load_softmax(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct softmax_param* softmax_param = ( struct softmax_param* )node->op.param_mem;

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axis")
        {
            softmax_param->axis = attr.i();
        }
    }

    return 0;
}

int load_elu(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct elu_param* elu_param = ( struct elu_param* )node->op.param_mem;

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "alpha")
            elu_param->alpha = attr.f();
    }

    return 0;
}

int load_interp(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    std::string mode = "nearest";
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);

        if (attr.name() == "mode")
        {
            mode = attr.s();
        }
    }
    if(mode != "nearest")
    {
        struct interp_param* interp_param = ( struct interp_param* )node->op.param_mem;

        if (onnx_node.input_size() == 1)
        {
            for (int k = 0; k < onnx_node.attribute_size(); k++)
            {
                const onnx::AttributeProto& attr = onnx_node.attribute(k);
                if (attr.name() == "scales")
                {
                    if (attr.floats_size() == 4)
                    {
                        float num0 = attr.floats(0);
                        float num1 = attr.floats(1);
                        float num2 = attr.floats(2);
                        float num3 = attr.floats(3);
                        interp_param->height_scale = num2 / num0;
                        interp_param->width_scale = num3 / num1;
                    }
                    else
                    {
                        interp_param->height_scale = attr.f();
                        interp_param->width_scale = attr.f();
                    }
                }
            }
        }
        else
        {
            const std::string& input_name = onnx_node.input(1);
            ir_tensor_t* tensor = find_tensor(graph, input_name);
            float* data = ( float* )tensor->data;

            interp_param->height_scale = data[2];
            interp_param->width_scale = data[3];
        }
        if (mode == "nearest")
        {
            interp_param->resize_type = 1;
        }
        else if (mode == "bilinear" || mode == "linear")
        {
            interp_param->resize_type = 2;
        }
    } 
    else
    {
        /* change op */
        sys_free(node->op.param_mem);
        int new_op_type = OP_RESIZE;
        node->op.type = new_op_type;
        ir_method_t* ir_method = find_op_method(new_op_type, OP_VERSION);
        if ((NULL != ir_method) && (NULL != ir_method->init) && (ir_method->init(&node->op) < 0))
        {
            return -1;
        }
        struct resize_param* resize_param = (struct resize_param*)node->op.param_mem;

        if (onnx_node.input_size() == 2)
        {
            const std::string& input_name = onnx_node.input(1);
            ir_tensor_t* tensor = find_tensor(graph, input_name);
            float* data = ( float* )tensor->data;
            resize_param->scale_h = data[2];
            resize_param->scale_w = data[3];
        }
        else
        {
            resize_param->scale_w = 1.f;
            resize_param->scale_h = 1.f;
        }
    }

    return 0;
}

int load_leaky_relu(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct relu_param* relu_param = ( struct relu_param* )node->op.param_mem;
    const onnx::AttributeProto& attr = onnx_node.attribute(0);
    relu_param->negative_slope = attr.f();

    return 0;
}

int load_slice(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct slice_param* slice_param = ( struct slice_param* )node->op.param_mem;

    slice_param->step = 1;
    slice_param->axis = 0;
    slice_param->begin = 0;
    slice_param->end = -1;
    if (onnx_node.input_size() == 1)
    {
        for (int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);
            if (attr.name() == "axes")
            {
                slice_param->axis = attr.ints(0);
            }
            else if (attr.name() == "ends")
            {
                long long end = attr.ints(0);
                if (end > INT_MAX)
                    end = INT_MAX;
                slice_param->end = ( int )end;
            }
            else if (attr.name() == "starts")
            {
                slice_param->begin = attr.ints(0);
            }
        }
    }
    else
    {
        ir_tensor_t* node_tensor = nullptr;
        node_tensor = find_tensor(graph, onnx_node.input(1));
        slice_param->begin = (int)(*(int64_t*)(node_tensor->data));

        node_tensor = find_tensor(graph, onnx_node.input(2));
        slice_param->end = (int)(*(int64_t*)(node_tensor->data));

        if (onnx_node.input_size() >= 4)
        {
            node_tensor = find_tensor(graph, onnx_node.input(3));
            slice_param->axis = (int)(*(int64_t*)(node_tensor->data));
        }

        if (onnx_node.input_size() >= 5)
        {
            node_tensor = find_tensor(graph, onnx_node.input(4));
            slice_param->step = (int)(*(int64_t*)(node_tensor->data));
        }
    }

    slice_param->iscaffe = 0;
    slice_param->ismxnet = 0;
    slice_param->isonnx = 1;
    return 0;
}

int load_split(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct split_param* split_param = ( struct split_param* )node->op.param_mem;
    split_param->is_onnx = true;
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axis")
        {
            split_param->axis = attr.i();
        }
        else if (attr.name() == "split")
        {
            int size = attr.ints_size();
            struct vector* new_shape = create_vector(sizeof(int), NULL);
            split_param->split_dim = size;
            for (int i = 0; i < size; i++)
            {
                int tmp = attr.ints(i);
                push_vector_data(new_shape, &tmp);
            }
            split_param->split_sizes_ = new_shape;
        }
    }
    split_param->is_caffe = false;

    return 0;
}

int load_unsqueeze(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct unsqueeze_param* unsqueeze_param = ( struct unsqueeze_param* )node->op.param_mem;

    std::vector<int> axises;
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axes")
        {
            for (int i = 0; i < attr.ints_size(); i++)
            {
                axises.push_back(attr.ints(i));
            }
        }
    }
    sort(axises.begin(), axises.end());

    unsqueeze_param->axises_size = axises.size();
    unsqueeze_param->axises = (int*)sys_malloc(sizeof(int) * unsqueeze_param->axises_size);
    for (size_t i = 0; i < axises.size(); i++)
    {
        unsqueeze_param->axises[i] = axises[i];
    }
    
    return 0;
}

int load_squeeze(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct squeeze_param* squeeze_param = ( struct squeeze_param* )node->op.param_mem;

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axes")
        {
            for (int i = 0; i < attr.ints_size(); i++)
            {
                if (0 == attr.ints(i))
                {
                    squeeze_param->dim_0 = 1;
                }
                else if (1 == attr.ints(i))
                {
                    squeeze_param->dim_1 = 1;
                }
                else if (2 == attr.ints(i))
                {
                    squeeze_param->dim_2 = 1;
                }
                else if (3 == attr.ints(i))
                {
                    squeeze_param->dim_3 = 1;
                }
            }
        }
    }
    
    return 0;
}

int load_matmul(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    ir_tensor_t* input_tensor = find_tensor(graph, onnx_node.input(0));
    ir_tensor_t* weight_tensor = find_tensor(graph, onnx_node.input(1));

    if(2 == weight_tensor->dim_num && weight_tensor->tensor_type == TENSOR_TYPE_CONST)
    {
        // swap shape
        int k = weight_tensor->dims[0];
        int n = weight_tensor->dims[1];

        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        float* tmp = ( float* )sys_malloc(k * n * sizeof(float));
        float* data = ( float* )weight_tensor->data;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }
        }
        memcpy(data, tmp, n * k * sizeof(float));
        free(tmp);

        if (change_node_op(node, OP_FC) < 0)
            return -1;
        struct fc_param* fc_param = ( struct fc_param* )node->op.param_mem;
        fc_param->num_output = weight_tensor->dims[0];
    }
    
    return 0;
}

int load_reducel2(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct reducel2_param* reducel2_param = ( struct reducel2_param* )node->op.param_mem;

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axes")
        {
            reducel2_param->axis = attr.ints(0);    // TODO:Support muti axis
        }
        if (attr.name() == "keepdims")
        {
            reducel2_param->keepdim = attr.i();
        }
    }
    
    return 0;
}

int load_gather(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct gather_param* gather_param = ( struct gather_param* )node->op.param_mem;

    ir_tensor_t* indices_tensor = find_tensor(graph, onnx_node.input(1));
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "axis")
        {
            gather_param->axis = attr.i();
        }
    }
    int64_t* data = ( int64_t* )indices_tensor->data;
    gather_param->indices_num = *data;
    gather_param->is_onnx = 1;
    
    return 0;
}

int load_comparison(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct comparison_param* comparison_param = ( struct comparison_param* )node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();

    if (op_name == "Greater")
        comparison_param->type = COMP_GREATER;
    else if (op_name == "Equal")
        comparison_param->type = COMP_EQUAL;
    else if (op_name == "Less")
        comparison_param->type = COMP_LESS;

    return 0;
}

int load_LRN(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct lrn_param* lrn_param = ( struct lrn_param* )node->op.param_mem;
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "alpha")
        {
            lrn_param->alpha = attr.f();    // TODO:Support muti axis
        }
        if (attr.name() == "beta")
        {
            lrn_param->beta = attr.f();
        }
        if (attr.name() == "bias")
        {
            lrn_param->k = attr.f();
        }
        if (attr.name() == "size")
        {
            lrn_param->local_size = attr.i();
        }
    }
    
    return 0;
}

int load_unary(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct unary_param* unary_param = ( struct unary_param* )node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();

    if (op_name == "Abs")
        unary_param->type = 0;
    else if (op_name == "Neg")
        unary_param->type = 1;
    else if (op_name == "Ceil")
        unary_param->type = 3;
    else if (op_name == "Log")
        unary_param->type = 8;
    else if (op_name == "Cos")
        unary_param->type = 10;
    else if (op_name == "Asin")
        unary_param->type = 12;
    else if (op_name == "Acos")
        unary_param->type = 13;
    else if (op_name == "Atan")
        unary_param->type = 14;
    
    return 0;
}

int load_logical(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct logical_param* logical_param = ( struct logical_param* )node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();

    if (op_name == "And")
        logical_param->type = 0;
    else if (op_name == "Or")
        logical_param->type = 1;
    
    return 0;
}

int load_pad(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct pad_param* pad_param = ( struct pad_param* )node->op.param_mem;
    
    if (onnx_node.attribute_size() == 1){  // since opset 11, 'pads' and 'value' have been moved from attributes to inputs
        const std::string& input_name_pad = onnx_node.input(1);
        ir_tensor_t* tensor_pad = find_tensor(graph, input_name_pad);
        int64_t* data_pad = ( int64_t * )tensor_pad->data;
        pad_param->pad_0_h = data_pad[0];
        pad_param->pad_0_w = data_pad[4];
        pad_param->pad_1_h = data_pad[1];
        pad_param->pad_1_w = data_pad[5];
        pad_param->pad_2_h = data_pad[2];
        pad_param->pad_2_w = data_pad[6];
        pad_param->pad_3_h = data_pad[3];
        pad_param->pad_3_w = data_pad[7];

        const std::string& input_name_value = onnx_node.input(2);
        ir_tensor_t* tensor_value = find_tensor(graph, input_name_value);
        float* data_value = ( float * )tensor_value->data;
        pad_param->value = data_value[0];
    }
    
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "mode")
        {
            if (attr.s() == "constant")
            {
                pad_param->mode = 0;
            }
            else if (attr.s() == "reflect")
            {
                pad_param->mode = 1;
            }
            else
            {
                pad_param->mode = 2;
            }
        }
        if (attr.name() == "pads")
        {
            pad_param->pad_0_h = attr.ints(0);
            pad_param->pad_0_w = attr.ints(4);
            pad_param->pad_1_h = attr.ints(1);
            pad_param->pad_1_w = attr.ints(5);
            pad_param->pad_2_h = attr.ints(2);
            pad_param->pad_2_w = attr.ints(6);
            pad_param->pad_3_h = attr.ints(3);
            pad_param->pad_3_w = attr.ints(7);
        }
        if (attr.name() == "value")
        {
            pad_param->value = attr.f();
        }
    }
    if(onnx_node.input_size() > 1){
        ir_tensor_t* shape_tensor = find_tensor(graph, onnx_node.input(1));
        int size = shape_tensor->dims[0];
        int64_t* data = ( int64_t* )shape_tensor->data;
        for (int i = 0; i < size; i++)
        {
                pad_param->pad_0_h = data[0];
                pad_param->pad_0_w = data[4];
                pad_param->pad_1_h = data[1];
                pad_param->pad_1_w = data[5];
                pad_param->pad_2_h = data[2];
                pad_param->pad_2_w = data[6];
                pad_param->pad_3_h = data[3];
                pad_param->pad_3_w = data[7];
        }
    }
    
    return 0;
}

int load_reduce(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct reduction_param* reduction_param = ( struct reduction_param* )node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();

    if (op_name == "ReduceSum")
        reduction_param->type = 0;
    else if (op_name == "ReduceMean")
        reduction_param->type = 1;
    else if (op_name == "ReduceSumSquare")
        reduction_param->type = 3;
    else if (op_name == "ReduceMax")
        reduction_param->type = 4;
    else if (op_name == "ReduceMin")
        reduction_param->type = 5;
    else if (op_name == "ReduceProd")
        reduction_param->type = 6;
    else if (op_name == "ReduceLogSum")
        reduction_param->type = 9;
    else if (op_name == "ReduceLogSumExp")
        reduction_param->type = 10;

    reduction_param->dim_0 = -2;
    reduction_param->dim_1 = -2;
    reduction_param->dim_2 = -2;
    reduction_param->dim_3 = -2;
    reduction_param->keepdim = 1;
    
    int size = onnx_node.attribute_size();
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "keepdims")
        {
            reduction_param->keepdim = attr.i();
        }
        else if (attr.name() == "axes")
        {
            int axis_size = attr.ints_size();
            if (axis_size == 1)
            {
                int attr_0 = attr.ints(0);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                reduction_param->dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                reduction_param->dim_0 = attr_0;
                reduction_param->dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                reduction_param->dim_0 = attr_0;
                reduction_param->dim_1 = attr_1;
                reduction_param->dim_2 = attr_2;
            }
            else if (axis_size == 4)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                int attr_3 = attr.ints(3);
                if (attr.ints(0) < 0)
                    attr_0 = 4 + attr.ints(0);
                if (attr.ints(1) < 0)
                    attr_0 = 4 + attr.ints(1);
                if (attr.ints(2) < 0)
                    attr_0 = 4 + attr.ints(2);
                if (attr.ints(3) < 0)
                    attr_0 = 4 + attr.ints(3);
                reduction_param->dim_0 = attr_0;
                reduction_param->dim_1 = attr_1;
                reduction_param->dim_2 = attr_2;
                reduction_param->dim_3 = attr_3;
            }
        }
    }
    
    return 0;
}

int load_argmax(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct argmax_param* argmax_param = ( struct argmax_param* )node->op.param_mem;
    
    int size = onnx_node.attribute_size();
    argmax_param->axis = 0;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
            argmax_param->axis = attr.i();
        if (attr.name() == "keepdims")
            argmax_param->keepdims = attr.i();
    }
    
    return 0;
}

int load_argmin(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct argmin_param* argmin_param = ( struct argmin_param* )node->op.param_mem;
    
    int size = onnx_node.attribute_size();
    argmin_param->axis = 0;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
            argmin_param->axis = attr.i();
        if (attr.name() == "keepdims")
            argmin_param->keepdims = attr.i();
    }
    
    return 0;
}

int load_log_softmax(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct logsoftmax_param* logsoftmax_param = ( struct logsoftmax_param* )node->op.param_mem;
    
    int size = onnx_node.attribute_size();
    logsoftmax_param->axis = 1;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
            logsoftmax_param->axis = attr.i();
    }
    
    return 0;
}

int load_deconv(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct deconv_param* deconv_param = ( struct deconv_param* )node->op.param_mem;
    
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "kernel_shape")
        {
            deconv_param->kernel_h = attr.ints(0);
            deconv_param->kernel_w = attr.ints(1);
        }
        else if (attr.name() == "strides")
        {
            deconv_param->stride_h = attr.ints(0);
            deconv_param->stride_w = attr.ints(1);
        }
        else if (attr.name() == "output_padding")
        {
            deconv_param->output_pad_h0 = attr.ints(0);
            deconv_param->output_pad_w0 = attr.ints(1);
        }
        else if (attr.name() == "pads")
        {
            deconv_param->pad_h0 = attr.ints(0);
            deconv_param->pad_h1 = attr.ints(2);
            deconv_param->pad_w0 = attr.ints(1);
            deconv_param->pad_w1 = attr.ints(3);
        }
        else if (attr.name() == "group")
        {
            deconv_param->group = attr.i();
        }
        else if (attr.name() == "dilations")
        {
            deconv_param->dilation_h = attr.ints(0);
            deconv_param->dilation_w = attr.ints(0);
        }
        else
            TLOG_ERR("attr.name: %s \n", attr.name().c_str());
    }

    /* update the input tensor data layout */
    for (int k = 0; k < onnx_node.input_size(); k++)
    {
        const std::string& input_name = onnx_node.input(k);
        ir_tensor_t* tensor = find_tensor(graph, input_name);
        if (k == 1)    // weight
        {
            int* dim = tensor->dims;
            /* onnx hide the output channel in weight ..*/
            deconv_param->num_output = dim[1];
            deconv_param->kernel_h = dim[2];
            deconv_param->kernel_w = dim[3];
        }
    }
    
    return 0;
}

int load_scatter(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct scatter_param* scatter_param = ( struct scatter_param* )node->op.param_mem;
    
    int size = onnx_node.attribute_size();
    scatter_param->axis = 0;
    scatter_param->is_onnx = 1;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
            scatter_param->axis = attr.i();
    }
    
    return 0;
}

int load_selu(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct selu_param* selu_param = ( struct selu_param* )node->op.param_mem;
    
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "alpha")
            selu_param->alpha = attr.f();
        else if (attr.name() == "gamma")
            selu_param->lambda = attr.f();
    }
    
    return 0;
}

int load_hard_sigmoid(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct hard_sigmoid_param* hard_sigmoid_param = ( struct hard_sigmoid_param* )node->op.param_mem;
    
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "alpha")
            hard_sigmoid_param->alpha = attr.f();
        else if (attr.name() == "beta")
            hard_sigmoid_param->beta = attr.f();
    }
    
    return 0;
}

int load_tile(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct tile_param* tile_param = ( struct tile_param* )node->op.param_mem;
    tile_param->frame_flag = 1;
    
    return 0;
}

int load_cast(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct cast_param* cast_param = ( struct cast_param* )node->op.param_mem;

    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "to")
            cast_param->type_to = attr.i();
    }
    cast_param->type_from = 1;

    return 0;
}

int load_depth_to_space(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct depthtospace_param* depthtospace_param = ( struct depthtospace_param* )node->op.param_mem;

    for(int k = 0; k < onnx_node.attribute_size(); k++){
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "block_size"){
            depthtospace_param->block_size = attr.i();
        }
    }

    return 0;
}

int load_instance_norm(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct instancenorm_Param* instancenorm_param = ( struct instancenorm_Param* )node->op.param_mem;

    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "epsilon")
            instancenorm_param->eps = attr.f();
    }

    return 0;
}

int load_resize(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct interp_param* interp_param = ( struct interp_param* )node->op.param_mem;

    if(onnx_node.input_size() == 1)
    {
        for(int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);
            if(attr.name() == "scales")
            {
                interp_param->height_scale = attr.f();
                interp_param->width_scale = attr.f();
            }
        }
    }
    else if(onnx_node.input_size() == 2) // opset 10
    {
        const std::string& input_name = onnx_node.input(1);
        ir_tensor_t* tensor = find_tensor(graph, input_name);
        float* data = ( float* )tensor->data;

        interp_param->height_scale = data[2];
        interp_param->width_scale = data[3];
    }
    else if(onnx_node.input_size() == 3) // opset 11
    {
        const std::string& input_name = onnx_node.input(2);
        ir_tensor_t* tensor = find_tensor(graph, input_name);
        float* data = ( float* )tensor->data;

        interp_param->height_scale = data[2];
        interp_param->width_scale = data[3];
    }
    else
    {
        fprintf(stderr, "Not support the num of inputs > 3, please check the onnx model or update the codes of convert tool\n");
        return -1;
    }

    std::string mode = "nearest";
    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "mode")
            mode = attr.s();
    }

    if (mode == "nearest")
        interp_param->resize_type = 1;
    else if (mode == "bilinear" || mode == "linear")
        interp_param->resize_type = 2;

    return 0;
}

int load_LSTM(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct lstm_param* lstm_param = ( struct lstm_param* )node->op.param_mem;

    int s_size;
    std::string lstm_type;
    for(int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if(attr.name() == "hidden_size")
            s_size = attr.i();
        if(attr.name() == "direction")
            lstm_type = attr.s();
    }

    // if(lstm_type == "bidirectional")
    //     lstm_param->algorithm = 0;
    // else
    //     lstm_param->algorithm = 0;

    lstm_param->mxnet_flag = 0;
    lstm_param->hidden_size = s_size;
    lstm_param->cell_size = s_size;

    return 0;
}

void register_op_load()
{
    op_load_map["Abs"]                   = std::pair<int, op_load_t>(OP_UNARY,        load_unary);
    op_load_map["Acos"]                  = std::pair<int, op_load_t>(OP_UNARY,        load_unary);
    op_load_map["And"]                   = std::pair<int, op_load_t>(OP_LOGICAL,      load_logical);
    op_load_map["ArgMax"]                = std::pair<int, op_load_t>(OP_ARGMAX,       load_argmax);
    op_load_map["ArgMin"]                = std::pair<int, op_load_t>(OP_ARGMIN,       load_argmin);
    op_load_map["Asin"]                  = std::pair<int, op_load_t>(OP_UNARY,        load_unary);
    op_load_map["Atan"]                  = std::pair<int, op_load_t>(OP_UNARY,        load_unary);
    op_load_map["AveragePool"]           = std::pair<int, op_load_t>(OP_POOL,         load_pool);
    op_load_map["Add"]                   = std::pair<int, op_load_t>(OP_ELTWISE,      load_eltwise);
    op_load_map["BatchNormalization"]    = std::pair<int, op_load_t>(OP_BATCHNORM,    load_bn);
    op_load_map["Conv"]                  = std::pair<int, op_load_t>(OP_CONV,         load_conv);
    op_load_map["ConvTranspose"]         = std::pair<int, op_load_t>(OP_DECONV,       load_deconv);
    op_load_map["Concat"]                = std::pair<int, op_load_t>(OP_CONCAT,       load_concat);
    op_load_map["Clip"]                  = std::pair<int, op_load_t>(OP_CLIP,         load_clip);
    op_load_map["Ceil"]                  = std::pair<int, op_load_t>(OP_UNARY,        load_unary);
    op_load_map["Cos"]                   = std::pair<int, op_load_t>(OP_UNARY,        load_unary);
    op_load_map["Cast"]                  = std::pair<int, op_load_t>(OP_CAST,         load_cast);
    op_load_map["Dropout"]               = std::pair<int, op_load_t>(OP_DROPOUT,      load_no_param);
    op_load_map["DepthToSpace"]          = std::pair<int, op_load_t>(OP_DEPTHTOSPACE, load_depth_to_space);
    op_load_map["Div"]                   = std::pair<int, op_load_t>(OP_ELTWISE,      load_eltwise);
    op_load_map["Elu"]                   = std::pair<int, op_load_t>(OP_ELU,          load_elu);
    op_load_map["Exp"]                   = std::pair<int, op_load_t>(OP_ELTWISE,      load_eltwise);
    op_load_map["Equal"]                 = std::pair<int, op_load_t>(OP_COMPARISON,   load_comparison);
    op_load_map["Flatten"]               = std::pair<int, op_load_t>(OP_FLATTEN,      load_flatten);
    op_load_map["Floor"]                 = std::pair<int, op_load_t>(OP_ELTWISE,      load_eltwise);
    op_load_map["Gemm"]                  = std::pair<int, op_load_t>(OP_GEMM,         load_gemm);
    op_load_map["Gather"]                = std::pair<int, op_load_t>(OP_GATHER,       load_gather);
    op_load_map["Greater"]               = std::pair<int, op_load_t>(OP_COMPARISON,   load_comparison);
    op_load_map["GlobalAveragePool"]     = std::pair<int, op_load_t>(OP_POOL,         load_pool);
    op_load_map["HardSwish"]             = std::pair<int, op_load_t>(OP_HARDSWISH,    load_no_param);
    op_load_map["HardSigmoid"]           = std::pair<int, op_load_t>(OP_HARDSIGMOID,  load_hard_sigmoid);
    op_load_map["InstanceNormalization"] = std::pair<int, op_load_t>(OP_INSTANCENORM, load_instance_norm);
    op_load_map["Log"]                   = std::pair<int, op_load_t>(OP_UNARY,        load_unary);
    op_load_map["LRN"]                   = std::pair<int, op_load_t>(OP_LRN,          load_LRN);
    op_load_map["Less"]                  = std::pair<int, op_load_t>(OP_COMPARISON,   load_comparison);
    op_load_map["LSTM"]                  = std::pair<int, op_load_t>(OP_LSTM,         load_LSTM);
    op_load_map["LeakyRelu"]             = std::pair<int, op_load_t>(OP_RELU,         load_leaky_relu);
    op_load_map["LogSoftmax"]            = std::pair<int, op_load_t>(OP_LOGSOFTMAX,   load_log_softmax);
    op_load_map["Mul"]                   = std::pair<int, op_load_t>(OP_ELTWISE,      load_eltwise);
    op_load_map["Max"]                   = std::pair<int, op_load_t>(OP_MAXIMUM,      load_no_param);
    op_load_map["Min"]                   = std::pair<int, op_load_t>(OP_MINIMUM,      load_no_param);
    op_load_map["Mean"]                  = std::pair<int, op_load_t>(OP_MEAN,         load_no_param);
    op_load_map["Matmul"]                = std::pair<int, op_load_t>(OP_MATMUL,       load_matmul);
    op_load_map["MaxPool"]               = std::pair<int, op_load_t>(OP_POOL,         load_pool);
    op_load_map["Neg"]                   = std::pair<int, op_load_t>(OP_UNARY,        load_unary);
    op_load_map["Or"]                    = std::pair<int, op_load_t>(OP_LOGICAL,      load_logical);
    op_load_map["Pad"]                   = std::pair<int, op_load_t>(OP_PAD,          load_pad);
    op_load_map["Pow"]                   = std::pair<int, op_load_t>(OP_ELTWISE,      load_eltwise);
    op_load_map["PRelu"]                 = std::pair<int, op_load_t>(OP_PRELU,        load_no_param);
    op_load_map["Relu"]                  = std::pair<int, op_load_t>(OP_RELU,         load_relu);
    op_load_map["Resize"]                = std::pair<int, op_load_t>(OP_INTERP,       load_resize);
    op_load_map["Reshape"]               = std::pair<int, op_load_t>(OP_RESHAPE,      load_reshape);
    op_load_map["ReduceL2"]              = std::pair<int, op_load_t>(OP_REDUCEL2,     load_reducel2);
    op_load_map["ReduceMean"]            = std::pair<int, op_load_t>(OP_REDUCTION,    load_reduce);
    op_load_map["ReduceLogSumExp"]       = std::pair<int, op_load_t>(OP_REDUCTION,    load_reduce);
    op_load_map["ReduceLogSum"]          = std::pair<int, op_load_t>(OP_REDUCTION,    load_reduce);
    op_load_map["ReduceMax"]             = std::pair<int, op_load_t>(OP_REDUCTION,    load_reduce);
    op_load_map["ReduceMin"]             = std::pair<int, op_load_t>(OP_REDUCTION,    load_reduce);
    op_load_map["ReduceProd"]            = std::pair<int, op_load_t>(OP_REDUCTION,    load_reduce);
    op_load_map["ReduceSumSquare"]       = std::pair<int, op_load_t>(OP_REDUCTION,    load_reduce);
    op_load_map["Reciprocal"]            = std::pair<int, op_load_t>(OP_RECIPROCAL,   load_no_param);
    op_load_map["Sub"]                   = std::pair<int, op_load_t>(OP_ELTWISE,      load_eltwise);
    op_load_map["Selu"]                  = std::pair<int, op_load_t>(OP_SELU,         load_selu);
    op_load_map["Sqrt"]                  = std::pair<int, op_load_t>(OP_ELTWISE,      load_eltwise);
    op_load_map["Slice"]                 = std::pair<int, op_load_t>(OP_SLICE,        load_slice);
    op_load_map["Split"]                 = std::pair<int, op_load_t>(OP_SPLIT,        load_split);
    op_load_map["Shape"]                 = std::pair<int, op_load_t>(OP_SHAPE,        load_no_param);
    op_load_map["Squeeze"]               = std::pair<int, op_load_t>(OP_SQUEEZE,      load_squeeze);
    op_load_map["Scatter"]               = std::pair<int, op_load_t>(OP_SCATTER,      load_scatter);
    op_load_map["Sigmoid"]               = std::pair<int, op_load_t>(OP_SIGMOID,      load_no_param);
    op_load_map["Softmax"]               = std::pair<int, op_load_t>(OP_SOFTMAX,      load_softmax);
    op_load_map["Softplus"]              = std::pair<int, op_load_t>(OP_SOFTPLUS,     load_no_param);
    op_load_map["Tanh"]                  = std::pair<int, op_load_t>(OP_TANH,         load_no_param);
    op_load_map["Tile"]                  = std::pair<int, op_load_t>(OP_TILE,         load_tile);
    op_load_map["Transpose"]             = std::pair<int, op_load_t>(OP_TRANSPOSE,    load_transpose);
    op_load_map["Upsample"]              = std::pair<int, op_load_t>(OP_INTERP,       load_interp);
    op_load_map["Unsqueeze"]             = std::pair<int, op_load_t>(OP_UNSQUEEZE,    load_unsqueeze);
    op_load_map["Where"]                 = std::pair<int, op_load_t>(OP_WHERE,        load_no_param);
}