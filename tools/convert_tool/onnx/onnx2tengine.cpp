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

extern "C"
{
    #include "convolution_param.h"
    #include "relu_param.h"
    #include "pooling_param.h"
    #include "flatten_param.h"
    #include "fc_param.h"
    #include "gemm_param.h"
}

typedef int (*op_load_t)(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node);
std::unordered_map<std::string, std::pair<int, op_load_t>> op_load_map;
void register_op_load();
const int OP_VERSION = 1;

bool FindOpLoadMethod(const std::string& op_name)
{
    if(op_load_map.count(op_name))
        return true;

    return false;
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

static int load_const_tensor(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
{
    int const_tensor_num = onnx_graph.initializer_size();
    for (int i = 0; i < const_tensor_num; i++)
    {
        const onnx::TensorProto& onnx_tensor = onnx_graph.initializer(i);
        if (onnx_tensor.data_type() != 1 && onnx_tensor.data_type() != 6) // fp32 int32
        {
            fprintf(stderr, "const tensor data type is not fp32 or int32. \n");
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

        if(!FindOpLoadMethod(onnx_op_name))
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
        TLOG_ERR("These %d op are not supported\n {", no_supported_op.size());
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
        int node_id = get_ir_tensor_index_from_name(graph, val.name().c_str());
        if (node_id == -1 || node_id > graph->node_num)
        {
            TLOG_ERR("find graph output node failed!\n");
            return -1;
        }
        output_nodes.push_back(node_id);
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
    if (load_const_tensor(graph, onnx_graph) < 0)
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
    fprintf(stderr, "onnx2tengine begin\n");

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

    fprintf(stderr, "onnx2tengine done.\n");
    return ir_graph;
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

    /* change op */
    sys_free(node->op.param_mem);
    int new_op_type = OP_FC;
    node->op.type = new_op_type;
    ir_method_t* ir_method = find_op_method(new_op_type, OP_VERSION);
    if ((NULL != ir_method) && (NULL != ir_method->init) && (ir_method->init(&node->op) < 0))
    {
        return -1;
    }
    struct fc_param* fc_param = (struct fc_param*)node->op.param_mem;
    fc_param->num_output = weight_tensor->dims[0];
    
    return 0;
}

void register_op_load()
{
    op_load_map["Conv"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["Relu"] = std::pair<int, op_load_t>(OP_RELU, load_relu);
    op_load_map["GlobalAveragePool"] = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["Flatten"] = std::pair<int, op_load_t>(OP_FLATTEN, load_flatten);
    op_load_map["Gemm"] = std::pair<int, op_load_t>(OP_GEMM, load_gemm);
}