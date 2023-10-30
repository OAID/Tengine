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
           bzhang@openailab.com
 */

#include "onnx2tengine.hpp"

/*
*   SELF DEFINE VARIABLE
*   FOR ONNX SERIALIZER
*/
const int OP_VERSION = 1;
static int op_set;

/*
*   ASSIST FUNCTIONS FOR ONNX SERIALIZER START
*/
bool onnx_serializer::find_op_load_method(const std::string& op_name)
{
    if (op_load_map.count(op_name))
        return true;

    return false;
}

ir_tensor_t* find_tensor(ir_graph_t* graph, const std::string& tensor_name)
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

static int change_node_op(ir_node_t* node, int new_op_type)
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

const int get_onnx_tensor_data_type(const onnx::TensorProto& onnx_tensor)
{
    int tensor_data_type = -1;
    switch (onnx_tensor.data_type())
    {
    case 1:
        tensor_data_type = TENGINE_DT_FP32;
        break;
    case 2:
        tensor_data_type = TENGINE_DT_UINT8;
        break;
    case 3:
        tensor_data_type = TENGINE_DT_INT8;
        break;
    case 5:
        tensor_data_type = TENGINE_DT_INT16;
        break;
    case 6: // int 32
    case 7: // int 64
        tensor_data_type = TENGINE_DT_INT32;
        break;
    case 10:
        tensor_data_type = TENGINE_DT_FP16;
        break;

    default:
        fprintf(stderr, "tensor: %s. data type unsupported in get data type: %d.\n", onnx_tensor.name().c_str(), onnx_tensor.data_type());
        return -1;
    }

    return tensor_data_type;
}

onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto& node, const char* key)
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

/*
*   ASSIST FUNCTIONS FOR ONNX SERIALIZER END
*/

int onnx_serializer::load_model_file(std::string model_file, onnx::ModelProto& model)
{
    std::ifstream is(model_file, std::ios::in | std::ios::binary);

    if (!is.is_open())
    {
        fprintf(stderr, "cannot open file: %s \n", model_file.c_str());
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

    if (!ret)
    {
        fprintf(stderr, "onnx serializer: parse file: %s \n", model_file.c_str());
        return -1;
    }

    /* get model op set */
    op_set = -1;
    if (model.opset_import_size())
    {
        const onnx::OperatorSetIdProto opset_import = model.opset_import(0);
        if (opset_import.has_version())
        {
            op_set = opset_import.version();
        }
    }
    if (op_set != -1)
    {
        fprintf(stderr, "Model op set is: %d\n", op_set);
    }

    return 0;
}

#define TASSERT(x)                                \
    if (!(x))                                     \
    {                                             \
        throw std::runtime_error("check failed"); \
    }

// this class is used to force the user to explicitly specify the template argument type
// of GetAttributeOrDefault
// Ref: https://stackoverflow.com/a/28171644
template<typename T>
struct Identity
{
    using type = T;
};

class NoAttrWithGivenNameError : public std::runtime_error
{
public:
    explicit NoAttrWithGivenNameError(const std::string& msg)
        : std::runtime_error(msg)
    {
    }
};

template<typename T>
T GetAttributeOrThrow(const onnx::NodeProto& node, const std::string& name);

template<typename T>
T GetAttributeOrDefault(const onnx::NodeProto& node, const std::string& name, typename Identity<T>::type default_val);

#define DEFINE_GetAttribute_FOR_SCALAR(cpp_type, onnx_proto_type, onnx_attr_getter) \
    template<>                                                                      \
    cpp_type GetAttributeOrThrow<cpp_type>(const onnx::NodeProto& node,             \
                                           const std::string& name)                 \
    {                                                                               \
        for (int k = 0; k < node.attribute_size(); k++)                             \
        {                                                                           \
            const onnx::AttributeProto& attr = node.attribute(k);                   \
                                                                                    \
            if (attr.name() == name)                                                \
            {                                                                       \
                if (attr.type() != onnx::AttributeProto::onnx_proto_type)           \
                {                                                                   \
                    throw std::invalid_argument(                                    \
                        "the type of attr " + name + " is "                         \
                        + onnx::AttributeProto::AttributeType_Name(attr.type())     \
                        + ", expected "                                             \
                        + onnx::AttributeProto::AttributeType_Name(                 \
                            onnx::AttributeProto::onnx_proto_type));                \
                }                                                                   \
                return attr.onnx_attr_getter();                                     \
            }                                                                       \
        }                                                                           \
        throw NoAttrWithGivenNameError("cannot find attr " + name);                 \
    }                                                                               \
    template<>                                                                      \
    cpp_type GetAttributeOrDefault<cpp_type>(                                       \
        const onnx::NodeProto& node, const std::string& name, cpp_type default_val) \
    {                                                                               \
        try                                                                         \
        {                                                                           \
            return GetAttributeOrThrow<cpp_type>(node, name);                       \
        }                                                                           \
        catch (const NoAttrWithGivenNameError&)                                     \
        {                                                                           \
            return default_val;                                                     \
        }                                                                           \
    }

DEFINE_GetAttribute_FOR_SCALAR(float, FLOAT, f);
DEFINE_GetAttribute_FOR_SCALAR(int, INT, i);
DEFINE_GetAttribute_FOR_SCALAR(std::string, STRING, s);

#undef DEFINE_GetAttribute_FOR_SCALAR

#define DEFINE_GetAttribute_FOR_VECTOR(cpp_type, onnx_proto_type, onnx_attr_getter) \
    template<>                                                                      \
    cpp_type GetAttributeOrThrow<cpp_type>(const onnx::NodeProto& node,             \
                                           const std::string& name)                 \
    {                                                                               \
        for (int k = 0; k < node.attribute_size(); k++)                             \
        {                                                                           \
            const onnx::AttributeProto& attr = node.attribute(k);                   \
                                                                                    \
            if (attr.name() == name)                                                \
            {                                                                       \
                if (attr.type() != onnx::AttributeProto::onnx_proto_type)           \
                {                                                                   \
                    throw std::invalid_argument(                                    \
                        "the type of attr " + name + " is "                         \
                        + onnx::AttributeProto::AttributeType_Name(attr.type())     \
                        + ", expected "                                             \
                        + onnx::AttributeProto::AttributeType_Name(                 \
                            onnx::AttributeProto::onnx_proto_type));                \
                }                                                                   \
                cpp_type res;                                                       \
                auto size = attr.onnx_attr_getter##_size();                         \
                for (int i = 0; i < size; i++)                                      \
                {                                                                   \
                    res.push_back(attr.onnx_attr_getter(i));                        \
                }                                                                   \
                return res;                                                         \
            }                                                                       \
        }                                                                           \
        throw NoAttrWithGivenNameError("cannot find attr " + name);                 \
    }                                                                               \
    template<>                                                                      \
    cpp_type GetAttributeOrDefault<cpp_type>(                                       \
        const onnx::NodeProto& node, const std::string& name, cpp_type default_val) \
    {                                                                               \
        try                                                                         \
        {                                                                           \
            return GetAttributeOrThrow<cpp_type>(node, name);                       \
        }                                                                           \
        catch (const NoAttrWithGivenNameError&)                                     \
        {                                                                           \
            return default_val;                                                     \
        }                                                                           \
    }

DEFINE_GetAttribute_FOR_VECTOR(std::vector<float>, FLOATS, floats);
DEFINE_GetAttribute_FOR_VECTOR(std::vector<int>, INTS, ints);

#undef DEFINE_GetAttribute_FOR_VECTOR

int onnx_serializer::load_constant_tensor(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
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
        bool logged = false;

        if (node.input_size() > 1)
        {
            // iter over constant inputs and create ir_tensor for constant tensor
            for (int inp_idx = 0; inp_idx < node.input_size(); ++inp_idx)
            {
                if (node_tensor.count(node.input(inp_idx)) == 0)
                    continue;
                if (!logged)
                {
                    logged = true;
                    if (!(op == "Reshape" || op == "Gather" || op == "Div" || op == "Resize" || op == "Upsample"
                          || op == "Clip" || op == "Slice" || op == "Expand"))
                    {
                        auto msg = "Load a Constant node \"%s\" as input[%d] of node \"%s\".\n";
                        printf(msg, node.input(inp_idx).c_str(), inp_idx, node.name().c_str());
                    }
                }
                const onnx::TensorProto& onnx_tensor = node_tensor[node.input(inp_idx)];
                std::pair<std::string, bool> t(node.input(inp_idx), 0);
                tensor_check.insert(t);
                int tensor_data_type = get_onnx_tensor_data_type(onnx_tensor);
                if (tensor_data_type < 0)
                {
                    return -1;
                }

                const char* name = node.input(inp_idx).c_str();
                int dim_num = onnx_tensor.dims_size();
                std::vector<int> dims(dim_num);
                for (int j = 0; j < dim_num; j++)
                {
                    dims[j] = onnx_tensor.dims(j);
                }

                // create ir tensor
                ir_tensor_t* ir_tensor = create_ir_tensor(graph, name, tensor_data_type);
                if (ir_tensor == NULL)
                {
                    fprintf(stderr, "create ir tensor failed!\n");
                    return -1;
                }
                set_ir_tensor_shape(ir_tensor, dims.data(), dim_num);
                ir_tensor->tensor_type = TENSOR_TYPE_CONST;
                // set tensor data
                if (7 == onnx_tensor.data_type())
                {
                    int tensor_size = ir_tensor->elem_num * sizeof(int64_t);
                    ir_tensor->data = sys_malloc(tensor_size);
                    int64_t* mem_buf = (int64_t*)ir_tensor->data;
                    if (onnx_tensor.has_raw_data())
                    {
                        int64_t* raw_data = (int64_t*)onnx_tensor.raw_data().data();
                        for (int j = 0; j < ir_tensor->elem_num; j++)
                        {
                            mem_buf[j] = raw_data[j];
                        }
                    }
                    else
                    {
                        int64_t* raw_data = (int64_t*)onnx_tensor.int64_data().data();
                        for (int j = 0; j < ir_tensor->elem_num; j++)
                        {
                            mem_buf[j] = raw_data[j];
                        }
                    }
                }
                else if (tensor_data_type == TENGINE_DT_FP32)
                {
                    // to support float type constant data loading
                    int tensor_size = ir_tensor->elem_num * sizeof(float_t);
                    ir_tensor->data = sys_malloc(tensor_size);
                    float_t* mem_buf = (float_t*)ir_tensor->data;
                    if (onnx_tensor.has_raw_data())
                    {
                        float_t* raw_data = (float_t*)onnx_tensor.raw_data().data();
                        for (int j = 0; j < ir_tensor->elem_num; j++)
                        {
                            mem_buf[j] = raw_data[j];
                        }
                    }
                    else
                    {
                        int32_t* raw_data = (int32_t*)onnx_tensor.int32_data().data();
                        for (int j = 0; j < ir_tensor->elem_num; j++)
                        {
                            mem_buf[j] = raw_data[j];
                        }
                    }
                }
                else
                {
                    int tensor_size = ir_tensor->elem_num * sizeof(uint8_t);
                    ir_tensor->data = sys_malloc(tensor_size);
                    uint8_t* mem_buf = (uint8_t*)ir_tensor->data;
                    if (onnx_tensor.has_raw_data())
                    {
                        uint8_t* raw_data = (uint8_t*)onnx_tensor.raw_data().data();
                        for (int j = 0; j < ir_tensor->elem_num; j++)
                        {
                            mem_buf[j] = raw_data[j];
                        }
                    }
                    else
                    {
                        uint8_t* raw_data = (uint8_t*)onnx_tensor.int32_data().data();
                        for (int j = 0; j < ir_tensor->elem_num; j++)
                        {
                            mem_buf[j] = raw_data[j];
                        }
                    }
                }
                ir_node_t* ir_node = create_ir_node(graph, name, OP_CONST, OP_VERSION);
                set_ir_node_output_tensor(ir_node, 0, ir_tensor);
            }
        }
    }

    return 0;
}

int onnx_serializer::load_initializer_tensor(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
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
        std::pair<std::string, int> t(onnx_tensor.name(), 0);
        tensor_check.insert(t);
        int tensor_data_type = get_onnx_tensor_data_type(onnx_tensor);
        if (tensor_data_type < 0)
        {
            return -1;
        }
        const char* name = onnx_tensor.name().c_str();
        int dim_num = onnx_tensor.dims_size();
        std::vector<int> dims(dim_num);
        for (int j = 0; j < dim_num; j++)
        {
            dims[j] = onnx_tensor.dims(j);
        }

        // create ir tensor
        ir_tensor_t* ir_tensor = create_ir_tensor(graph, name, tensor_data_type);
        if (ir_tensor == NULL)
        {
            fprintf(stderr, "create ir tensor failed!\n");
            return -1;
        }
        set_ir_tensor_shape(ir_tensor, dims.data(), dim_num);
        ir_tensor->tensor_type = TENSOR_TYPE_CONST;
        if (ir_tensor->dim_num == 0)
        {
            ir_tensor->dim_num = 1;
            ir_tensor->dims[0] = 1;
        }

        if (onnx_tensor.has_raw_data())
        {
            if (onnx_tensor.data_type() == 1) //fp32
            {
                ir_tensor->data_type = TENGINE_DT_FP32;
                int tensor_size = ir_tensor->elem_num * sizeof(float);
                ir_tensor->data = sys_malloc(tensor_size);
                float* mem_buf = (float*)ir_tensor->data;
                float* raw_data = (float*)onnx_tensor.raw_data().c_str();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
            else if (onnx_tensor.data_type() == 6) // int32
            {
                ir_tensor->data_type = TENGINE_DT_INT32;
                int tensor_size = ir_tensor->elem_num * sizeof(int32_t);
                ir_tensor->data = sys_malloc(tensor_size);
                int32_t* mem_buf = (int32_t*)ir_tensor->data;
                int32_t* raw_data = (int32_t*)onnx_tensor.raw_data().data();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
            else if (onnx_tensor.data_type() == 7) // int64
            {
                // ir_tensor->data_type = TENGINE_DT_INT32;
                int tensor_size = ir_tensor->elem_num * sizeof(int64_t);
                ir_tensor->data = sys_malloc(tensor_size);
                int64_t* mem_buf = (int64_t*)ir_tensor->data;
                int64_t* raw_data = (int64_t*)onnx_tensor.raw_data().data();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
            else
            {
                fprintf(stderr, "tensor: %s data type unsupported in set raw data.\n", onnx_tensor.name().c_str());
                return -1;
            }
        }
        else
        {
            if (onnx_tensor.data_type() == 1) //fp32
            {
                ir_tensor->data_type = TENGINE_DT_FP32;
                int tensor_size = ir_tensor->elem_num * sizeof(float);
                ir_tensor->data = sys_malloc(tensor_size);
                float* mem_buf = (float*)ir_tensor->data;
                float* raw_data = (float*)onnx_tensor.float_data().data();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
            else if (onnx_tensor.data_type() == 6) // int32
            {
                ir_tensor->data_type = TENGINE_DT_INT32;
                int tensor_size = ir_tensor->elem_num * sizeof(int32_t);
                ir_tensor->data = sys_malloc(tensor_size);
                int32_t* mem_buf = (int32_t*)ir_tensor->data;
                int32_t* raw_data = (int32_t*)onnx_tensor.int32_data().data();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
            else if (onnx_tensor.data_type() == 7) // int64
            {
                // ir_tensor->data_type = TENGINE_DT_INT32;
                int tensor_size = ir_tensor->elem_num * sizeof(int64_t);
                ir_tensor->data = sys_malloc(tensor_size);
                int64_t* mem_buf = (int64_t*)ir_tensor->data;
                int64_t* raw_data = (int64_t*)onnx_tensor.int64_data().data();
                for (int j = 0; j < ir_tensor->elem_num; j++)
                {
                    mem_buf[j] = raw_data[j];
                }
            }
            else
            {
                fprintf(stderr, "tensor: %s data type unsupported in set data.\n", onnx_tensor.name().c_str());
                return -1;
            }
        }

        ir_node_t* ir_node = create_ir_node(graph, name, OP_CONST, OP_VERSION);
        set_ir_node_output_tensor(ir_node, 0, ir_tensor);
    }
    return 0;
}

int onnx_serializer::set_graph_input(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
{
    std::vector<int16_t> input_nodes;
    for (int i = 0; i < onnx_graph.input_size(); i++)
    {
        const onnx::ValueInfoProto& val = onnx_graph.input(i);
        if (get_ir_tensor_index_from_name(graph, val.name().c_str()) != -1)
            continue;

        // now, catch an input tensor
        const onnx::TypeProto& type = val.type();
        const onnx::TypeProto::Tensor& tensor_type = type.tensor_type();
        const onnx::TensorShapeProto& shape = tensor_type.shape();
        int has_shape = 1;
        std::vector<int> dims(shape.dim_size());
        for (int j = 0; j < shape.dim_size(); j++)
        {
            const onnx::TensorShapeProto::Dimension& dim = shape.dim(j);
            if (dim.has_dim_param())
            {
                has_shape = 0;
                break;
            }
            dims[j] = dim.dim_value();
        }

        ir_tensor_t* tensor = create_ir_tensor(graph, val.name().c_str(), TENGINE_DT_FP32);
        if (has_shape)
        {
            set_ir_tensor_shape(tensor, dims.data(), shape.dim_size());
        }
        tensor->tensor_type = TENSOR_TYPE_INPUT;
        ir_node_t* node = create_ir_node(graph, val.name().c_str(), OP_INPUT, OP_VERSION);
        set_ir_node_output_tensor(node, 0, tensor);
        input_nodes.push_back(node->index);
    }

    set_ir_graph_input_node(graph, input_nodes.data(), input_nodes.size());
    return 0;
}

int onnx_serializer::load_graph_node(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
{
    int i;
    std::vector<std::string> no_supported_op;
    for (i = 0; i < onnx_graph.node_size(); i++)
    {
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& onnx_op_name = onnx_node.op_type();

        if (!find_op_load_method(onnx_op_name))
        {
            auto it = find(no_supported_op.begin(), no_supported_op.end(), onnx_op_name);
            if (it == no_supported_op.end())
            {
                if (onnx_op_name == "Constant")
                    continue;
                no_supported_op.push_back(onnx_op_name);
            }
        }
    }
    if (no_supported_op.size())
    {
        fprintf(stderr, "These %zu op are not supported\n{ ", no_supported_op.size());
        for (int j = 0; j < (int)no_supported_op.size(); j++)
        {
            fprintf(stderr, "%s ", no_supported_op[j].c_str());
        }
        fprintf(stderr, "}\n");
        return -1;
    }

    for (i = 0; i < onnx_graph.node_size(); i++)
    {
        /* create ir node*/
        const onnx::NodeProto& onnx_node = onnx_graph.node(i);
        const std::string& op_name = onnx_node.op_type();
        if (op_name == "Constant")
        {
            continue;
        }
        std::string node_name = onnx_node.name();
        if (node_name.empty())
        {
            node_name = std::to_string(i);
        }
        ir_node_t* ir_node = create_ir_node(graph, node_name.c_str(), op_load_map[op_name].first, OP_VERSION);
        if (ir_node == NULL)
        {
            return -1;
        }
        /* set ir node io */
        for (int j = 0; j < onnx_node.input_size(); j++)
        {
            const std::string& input_name = onnx_node.input(j);
            if (input_name == "")
            {
                continue;
            }
            int tensor_id = get_ir_tensor_index_from_name(graph, input_name.c_str());
            ir_tensor_t* tensor = get_ir_graph_tensor(graph, tensor_id);
            tensor_check[tensor->name] = tensor_check[tensor->name] + 1;
            set_ir_node_input_tensor(ir_node, ir_node->input_num, tensor);
        }

        for (int j = 0; j < onnx_node.output_size(); j++)
        {
            if (op_name == "Dropout" && j > 0)
            {
                continue;
            }
            const std::string& output_name = onnx_node.output(j);
            ir_tensor_t* tensor = create_ir_tensor(graph, output_name.c_str(), TENGINE_DT_FP32);
            set_ir_node_output_tensor(ir_node, j, tensor);
        }
        /* exec op load func */
        op_load_t loader = op_load_map[op_name].second;
        if (loader(graph, ir_node, onnx_node) < 0)
        {
            fprintf(stderr, "load op %s func failed in node %s .\n", op_name.c_str(), node_name.c_str());
            return -1;
        }
    }
    return 0;
}

int onnx_serializer::set_graph_output(ir_graph_t* graph, const onnx::GraphProto& onnx_graph)
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
        std::vector<int> dims(shape.dim_size());
        for (int j = 0; j < shape.dim_size(); j++)
        {
            const onnx::TensorShapeProto::Dimension& dim = shape.dim(j);
            if (dim.has_dim_param())
            {
                has_shape = 0;
                break;
            }
            dims[j] = dim.dim_value();
        }
        ir_tensor_t* tensor = get_ir_graph_tensor(graph, tensor_id);
        if (has_shape)
        {
            set_ir_tensor_shape(tensor, dims.data(), shape.dim_size());
        }
        ir_node_t* node = get_ir_graph_node(graph, tensor->producer);
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

static int deal_old_softmax(ir_graph_t* graph)
{
    std::vector<ir_node_t*> old_spec_softmax_nodes;

    // get all softmax case.
    if (op_set < 13)
    {
        int node_num = graph->node_num;
        for (int i = 0; i < node_num; ++i)
        {
            ir_node_t* node = get_ir_graph_node(graph, i);
            if (node->op.type != OP_SOFTMAX)
            {
                continue;
            }
            struct softmax_param* param = (struct softmax_param*)node->op.param_mem;
            // check if transpose + softmax + transpose
            ir_tensor_t* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
            ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
            ir_node_t* pre_node = get_ir_graph_node(graph, input->producer);
            ir_node_t* next_node = nullptr;
            if (output->consumer_num > 0)
            {
                next_node = get_ir_graph_node(graph, output->consumer[0]);
            }
            if (next_node != nullptr && pre_node->op.type == OP_TRANSPOSE && next_node->op.type == OP_TRANSPOSE)
            {
                if (param->axis == input->dim_num - 1)
                {
                    // TODO: optimize to softmax(new spec).
                    continue;
                }
                else
                {
                    continue;
                }
            }
            if (param->axis == input->dim_num - 1)
            {
                // old spec == new spec
                continue;
            }

            old_spec_softmax_nodes.push_back(node);
        }
    }

    // support old spec onnx softmax.
    for (auto& node : old_spec_softmax_nodes)
    {
        ir_tensor_t* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
        struct softmax_param* param = (struct softmax_param*)node->op.param_mem;

        // support old spec by adding reshape node.
        // reshape in
        std::string node_name = node->name;
        std::string name = node_name + "_reshape_in";
        int reshape_in_id = add_node_above(graph, node->index, OP_RESHAPE, name.c_str());
        ir_node_t* reshape_in = get_ir_graph_node(graph, reshape_in_id);
        struct reshape_param* reshape_in_param = (struct reshape_param*)reshape_in->op.param_mem;

        std::vector<int> re_shape;
        for (int j = 0; j < input->dim_num; ++j)
        {
            if (j == param->axis)
            {
                re_shape.push_back(-1);
                break;
            }
            re_shape.push_back(input->dims[j]);
        }
        reshape_in_param->is_onnx = 1;
        reshape_in_param->dim_size = re_shape.size();
        reshape_in_param->re_shape = (int*)sys_malloc(reshape_in_param->dim_size * sizeof(int));
        for (int j = 0; j < reshape_in_param->dim_size; ++j)
        {
            reshape_in_param->re_shape[j] = re_shape[j];
        }

        // reshape out
        name = node_name + "_reshape_out";
        int reshape_out_id = add_node_below(graph, node->index, OP_RESHAPE, name.c_str());
        ir_node_t* reshape_out = get_ir_graph_node(graph, reshape_out_id);
        struct reshape_param* reshape_out_param = (struct reshape_param*)reshape_out->op.param_mem;
        reshape_out_param->is_onnx = 1;
        reshape_out_param->dim_size = input->dim_num;
        reshape_out_param->re_shape = (int*)sys_malloc(reshape_out_param->dim_size * sizeof(int));
        for (int j = 0; j < reshape_out_param->dim_size; ++j)
        {
            reshape_out_param->re_shape[j] = input->dims[j];
        }
    }

    return 0;
}

static int reduce2avgpool(ir_graph_t* graph)
{
    std::vector<ir_node_t*> reduce_mean_nodes;

    // get all softmax case.
    if (op_set < 13)
    {
        int node_num = graph->node_num;
        for (int i = 0; i < node_num; ++i)
        {
            ir_node_t* node = get_ir_graph_node(graph, i);
            if (node->op.type != OP_REDUCTION)
            {
                continue;
            }
            struct reduction_param* param = (struct reduction_param*)node->op.param_mem;

            if (param->type != 1 || param->dim_0 != 2 || param->dim_1 != 3 || param->dim_2 != -2 || param->dim_3 != -2)
            {
                continue;
            }

            reduce_mean_nodes.push_back(node);
        }
    }

    // support old spec onnx softmax.
    for (auto& node : reduce_mean_nodes)
    {
        if (change_node_op(node, OP_POOL) < 0)
        {
            return -1;
        }
        struct pool_param* pool_param = (struct pool_param*)node->op.param_mem;
        pool_param->global = 1;
        pool_param->pool_method = POOL_AVG;
    }

    return 0;
}

int onnx_serializer::optimize_graph(ir_graph_t* graph)
{
    set_log_level(LOG_EMERG);
    if (infer_ir_graph_shape(graph) < 0)
    {
        fprintf(stderr, "Skip internal optimize in onnx serializer.\n");
        return 0;
    }
    if (deal_old_softmax(graph) < 0)
        return -1;
    if (reduce2avgpool(graph) < 0)
        return -1;
    fprintf(stderr, "Internal optimize in onnx serializer done.\n");

    return 0;
}

int onnx_serializer::load_model(ir_graph_t* graph, std::string model_file)
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
    if (optimize_graph(graph) < 0)
        return -1;

    graph->model_format = MODEL_FORMAT_ONNX;
    graph->graph_layout = TENGINE_LAYOUT_NCHW;
    graph->model_layout = TENGINE_LAYOUT_NCHW;
    return 0;
}

graph_t onnx_serializer::onnx2tengine(std::string model_file)
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

static int load_conv(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct conv_param* conv_param = (struct conv_param*)node->op.param_mem;
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
            conv_param->dilation_w = attr.ints(1);
        }
        else if (attr.name() == "auto_pad")
        {
            /*
             * real pad will be calculated in infer shape.
             * flag:
             *      -1 : SAME_UPPER
             *      -2 : SAME_LOWER
             */
            const std::string& auto_pad = attr.s();

            if (auto_pad == "NOTSET")
            {
                continue;
            }
            else if (auto_pad == "SAME_UPPER")
            {
                conv_param->pad_w0 = -1;
                conv_param->pad_w1 = -1;
                conv_param->pad_h0 = -1;
                conv_param->pad_h1 = -1;
            }
            else if (auto_pad == "SAME_LOWER")
            {
                conv_param->pad_w0 = -2;
                conv_param->pad_w1 = -2;
                conv_param->pad_h0 = -2;
                conv_param->pad_h1 = -2;
            }
            else if (auto_pad == "VALID")
            {
                conv_param->pad_w0 = 0;
                conv_param->pad_w1 = 0;
                conv_param->pad_h0 = 0;
                conv_param->pad_h1 = 0;
            }
            else
                fprintf(stderr, "%s attr.name: %s : %s not support.\n", node->name, attr.name().c_str(), auto_pad.c_str());
        }
        else
            fprintf(stderr, "%s attr.name: %s \n", node->name, attr.name().c_str());
    }

    struct tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    conv_param->output_channel = weight->dims[0]; /* onnx hide the output channel in weight .. */

    return 0;
}

static int load_relu(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct relu_param* relu_param = (struct relu_param*)node->op.param_mem;
    relu_param->negative_slope = 0.f;
    return 0;
}

static int load_pool(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct pool_param* pool_param = (struct pool_param*)node->op.param_mem;
    const std::string& onnx_op = onnx_node.op_type();

    // set default param
    pool_param->pad_h0 = 0;
    pool_param->pad_h1 = 0;
    pool_param->pad_w0 = 0;
    pool_param->pad_w1 = 0;
    pool_param->stride_h = 1;
    pool_param->stride_w = 1;
    pool_param->global = 0;
    pool_param->caffe_flavor = 0;

    if (onnx_op == "GlobalAveragePool")
    {
        pool_param->global = 1;
        pool_param->pool_method = POOL_AVG;
    }
    else if (onnx_op == "MaxPool" || onnx_op == "AveragePool")
    {
        pool_param->global = 0;

        if (onnx_op == "AveragePool")
            pool_param->pool_method = POOL_AVG;
        else
            pool_param->pool_method = POOL_MAX;

        for (int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);

            if (attr.name() == "kernel_shape")
            {
                pool_param->kernel_h = attr.ints(0);
                pool_param->kernel_w = attr.ints(1);
            }
            else if (attr.name() == "strides")
            {
                pool_param->stride_h = attr.ints(0);
                pool_param->stride_w = attr.ints(1);
            }
            else if (attr.name() == "pads") /* onnx pads: x0_begin, x1_begin, ... , x0_end, x1_end, ... */
            {
                pool_param->pad_h0 = attr.ints(0);
                pool_param->pad_h1 = attr.ints(2);
                pool_param->pad_w0 = attr.ints(1);
                pool_param->pad_w1 = attr.ints(3);
                if (pool_param->pad_h0 != pool_param->pad_h1 && pool_param->pad_w0 != pool_param->pad_w1)
                {
                    pool_param->caffe_flavor = 1;
                }
            }
        }
    }
    else
    {
        fprintf(stderr, "UNKNOWN POOLING: %s \n", onnx_op.c_str());
        return -1;
    }

    pool_param->pad_h0_org = pool_param->pad_h0;
    pool_param->pad_h1_org = pool_param->pad_h1;
    pool_param->pad_w0_org = pool_param->pad_w0;
    pool_param->pad_w1_org = pool_param->pad_w1;

    return 0;
}

static int load_flatten(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct flatten_param* flatten_param = (struct flatten_param*)node->op.param_mem;
    flatten_param->axis = 1;

    if (1 == onnx_node.attribute_size())
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(0);
        flatten_param->axis = attr.i();
    }
    return 0;
}

static int load_gemm(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct gemm_param* gemm_param = (struct gemm_param*)node->op.param_mem;
    // set default
    gemm_param->alpha = GetAttributeOrDefault<float>(onnx_node, "alpha", 1.0f);
    gemm_param->beta = GetAttributeOrDefault<float>(onnx_node, "beta", 1.0f);
    gemm_param->transA = GetAttributeOrDefault<int>(onnx_node, "transA", 0);
    gemm_param->transB = GetAttributeOrDefault<int>(onnx_node, "transB", 0);

    ir_tensor_t* weight_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    ir_tensor_t* bias_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);

    if (gemm_param->transA)
    {
        return -1;
    }

    // create fc instead
    if (!gemm_param->transB)
    {
        // swap shape
        int k = weight_tensor->dims[0];
        int n = weight_tensor->dims[1];
        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        // float* tmp = ( float* )sys_malloc(k * n * sizeof(float));
        std::vector<float> tmp(k * n);
        float* data = (float*)weight_tensor->data;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }

        memcpy(data, tmp.data(), n * k * sizeof(float));
        // sys_free(tmp);
    }

    if (gemm_param->alpha != 1)
    {
        float* data = (float*)weight_tensor->data;
        int tensor_size = weight_tensor->dims[0] * weight_tensor->dims[1];

        for (int i = 0; i < tensor_size; i++)
            data[i] *= gemm_param->alpha;
    }

    if (gemm_param->beta != 1)
    {
        float* data = (float*)bias_tensor->data;
        int tensor_size = weight_tensor->dims[0];

        for (int i = 0; i < tensor_size; i++)
            data[i] *= gemm_param->beta;
    }

    if (change_node_op(node, OP_FC) < 0)
    {
        return -1;
    }
    struct fc_param* fc_param = (struct fc_param*)node->op.param_mem;
    fc_param->num_output = weight_tensor->dims[0];

    return 0;
}

static int load_concat(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct concat_param* concat_param = (struct concat_param*)node->op.param_mem;

    concat_param->axis = GetAttributeOrThrow<int>(onnx_node, "axis");

    return 0;
}

static int load_bn(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct batchnorm_param* batchnorm_param = (struct batchnorm_param*)node->op.param_mem;

    batchnorm_param->eps = GetAttributeOrThrow<float>(onnx_node, "epsilon");

    return 0;
}

static int load_eltwise(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct eltwise_param* eltwise_param = (struct eltwise_param*)node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();
    if (op_name == "Add")
    {
        eltwise_param->type = ELT_SUM;
    }
    else if (op_name == "Mul")
    {
        eltwise_param->type = ELT_PROD;
    }
    else if (op_name == "Div")
    {
        eltwise_param->type = ELT_DIV;
    }
    else if (op_name == "Floor")
    {
        eltwise_param->type = ELT_FLOOR;
    }
    else if (op_name == "Exp")
    {
        eltwise_param->type = ELT_EXP;
    }
    else if (op_name == "Sub")
    {
        eltwise_param->type = ELT_SUB;
    }
    else if (op_name == "Pow")
    {
        eltwise_param->type = ELT_POW;
    }
    else if (op_name == "Sqrt")
    {
        eltwise_param->type = ELT_SQRT;
    }

    return 0;
}

static int load_transpose(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct transpose_param* transpose_param = (struct transpose_param*)node->op.param_mem;

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

static int load_clip(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct clip_param* clip_param = (struct clip_param*)node->op.param_mem;

    if (node->input_num == 1)
    {
        clip_param->max = GetAttributeOrThrow<float>(onnx_node, "max");
        clip_param->min = GetAttributeOrThrow<float>(onnx_node, "min");
    }
    else if (node->input_num == 3)
    {
        ir_tensor_t* min = find_tensor(graph, onnx_node.input(1));
        ir_tensor_t* max = find_tensor(graph, onnx_node.input(2));
        if (min->tensor_type == TENSOR_TYPE_CONST && max->tensor_type == TENSOR_TYPE_CONST)
        {
            float* min_data = (float*)min->data;
            float* max_data = (float*)max->data;
            clip_param->min = min_data[0];
            clip_param->max = max_data[0];
            node->input_num = 1;
        }
    }
    else
    {
        throw std::invalid_argument("unsupported clip op input num: " + std::to_string(node->input_num));
    }

    return 0;
}

static int load_reshape(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct reshape_param* reshape_param = (struct reshape_param*)node->op.param_mem;

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

static int load_no_param(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    // no param
    return 0;
}

static int load_softmax(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct softmax_param* softmax_param = (struct softmax_param*)node->op.param_mem;
    softmax_param->axis = 1; // default

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

static int load_elu(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct elu_param* elu_param = (struct elu_param*)node->op.param_mem;
    elu_param->alpha = GetAttributeOrDefault<float>(onnx_node, "alpha", 1.f);

    return 0;
}

static int load_interp(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    std::string mode = GetAttributeOrDefault<std::string>(onnx_node, "mode", "nearest");
    if (mode != "nearest")
    {
        struct interp_param* interp_param = (struct interp_param*)node->op.param_mem;

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
            float* data = (float*)tensor->data;

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
        if (change_node_op(node, OP_RESIZE) < 0)
        {
            return -1;
        }
        struct resize_param* resize_param = (struct resize_param*)node->op.param_mem;

        if (onnx_node.input_size() == 2)
        {
            const std::string& input_name = onnx_node.input(1);
            ir_tensor_t* tensor = find_tensor(graph, input_name);
            float* data = (float*)tensor->data;
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

static int load_leaky_relu(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct relu_param* relu_param = (struct relu_param*)node->op.param_mem;
    const onnx::AttributeProto& attr = onnx_node.attribute(0);
    relu_param->negative_slope = attr.f();

    return 0;
}

static int load_slice(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct slice_param* slice_param = (struct slice_param*)node->op.param_mem;

    slice_param->step = 1;
    slice_param->axis = 0;
    slice_param->begin = 0;
    slice_param->end = -1;
    slice_param->slice_point_ = nullptr;
    slice_param->begin_ = nullptr;
    slice_param->size_ = nullptr;

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
                {
                    end = INT_MAX;
                }
                slice_param->end = (int)end;
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

static int load_split(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct split_param* split_param = (struct split_param*)node->op.param_mem;
    split_param->is_onnx = true;
    split_param->axis = GetAttributeOrDefault<int>(onnx_node, "axis", 0);
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "split")
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

static int load_unsqueeze(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct unsqueeze_param* unsqueeze_param = (struct unsqueeze_param*)node->op.param_mem;

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

    /* opset 13 */
    if (axises.empty() && node->input_num == 2)
    {
        ir_tensor_t* axes_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
        int* data = (int*)axes_tensor->data;
        for (int i = 0; i < axes_tensor->elem_num; i++)
        {
            axises.push_back(data[i]);
        }

        // remove axes tensor
        node->input_num = 1;
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

static int load_squeeze(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct squeeze_param* squeeze_param = (struct squeeze_param*)node->op.param_mem;

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

static int load_matmul(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    ir_tensor_t* input_tensor = find_tensor(graph, onnx_node.input(0));
    ir_tensor_t* weight_tensor = find_tensor(graph, onnx_node.input(1));

    if (2 == input_tensor->dim_num && weight_tensor->tensor_type == TENSOR_TYPE_CONST)
    {
        // swap shape
        int k = weight_tensor->dims[0];
        int n = weight_tensor->dims[1];

        weight_tensor->dims[0] = n;
        weight_tensor->dims[1] = k;

        // float* tmp = ( float* )sys_malloc(k * n * sizeof(float));
        std::vector<float> tmp(k * n);
        float* data = (float*)weight_tensor->data;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                tmp[i * k + j] = data[j * n + i];
            }
        }
        memcpy(data, tmp.data(), n * k * sizeof(float));
        // free(tmp);

        if (change_node_op(node, OP_FC) < 0)
        {
            return -1;
        }
        struct fc_param* fc_param = (struct fc_param*)node->op.param_mem;
        fc_param->num_output = weight_tensor->dims[0];
    }

    return 0;
}

static int load_reducel2(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct reducel2_param* reducel2_param = (struct reducel2_param*)node->op.param_mem;
    const auto axes = GetAttributeOrThrow<std::vector<int> >(onnx_node, "axes");
    TASSERT(axes.size() == 1);
    reducel2_param->axis = axes[0];
    reducel2_param->keepdim = GetAttributeOrDefault<int>(onnx_node, "keepdims", 1);

    return 0;
}

static int load_gather(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct gather_param* gather_param = (struct gather_param*)node->op.param_mem;

    gather_param->axis = GetAttributeOrDefault<int>(onnx_node, "axis", 0);
    ir_tensor_t* indices_tensor = find_tensor(graph, onnx_node.input(1));
    int64_t* data = (int64_t*)indices_tensor->data;
    gather_param->indices_num = *data;
    gather_param->is_onnx = 1;

    return 0;
}

static int load_comparison(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct comparison_param* comparison_param = (struct comparison_param*)node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();

    if (op_name == "Greater")
    {
        comparison_param->type = COMP_GREATER;
    }
    else if (op_name == "Equal")
    {
        comparison_param->type = COMP_EQUAL;
    }
    else if (op_name == "Less")
    {
        comparison_param->type = COMP_LESS;
    }

    return 0;
}

static int load_LRN(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct lrn_param* lrn_param = (struct lrn_param*)node->op.param_mem;
    lrn_param->alpha = GetAttributeOrDefault<float>(onnx_node, "alpha", 0.0001);
    lrn_param->beta = GetAttributeOrDefault<float>(onnx_node, "alpha", 0.0001);
    lrn_param->k = GetAttributeOrDefault<float>(onnx_node, "bias", 1.);
    lrn_param->local_size = GetAttributeOrThrow<int>(onnx_node, "size");

    return 0;
}

static int load_unary(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct unary_param* unary_param = (struct unary_param*)node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();

    if (op_name == "Abs")
    {
        unary_param->type = 0;
    }
    else if (op_name == "Neg")
    {
        unary_param->type = 1;
    }
    else if (op_name == "Ceil")
    {
        unary_param->type = 3;
    }
    else if (op_name == "Log")
    {
        unary_param->type = 8;
    }
    else if (op_name == "Cos")
    {
        unary_param->type = 10;
    }
    else if (op_name == "Asin")
    {
        unary_param->type = 12;
    }
    else if (op_name == "Acos")
    {
        unary_param->type = 13;
    }
    else if (op_name == "Atan")
    {
        unary_param->type = 14;
    }

    return 0;
}

static int load_logical(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct logical_param* logical_param = (struct logical_param*)node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();

    if (op_name == "And")
    {
        logical_param->type = 0;
    }
    else if (op_name == "Or")
    {
        logical_param->type = 1;
    }

    return 0;
}

static int load_pad(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct pad_param* pad_param = (struct pad_param*)node->op.param_mem;

    if (onnx_node.attribute_size() == 1) // since opset 11, 'pads' and 'value' have been moved from attributes to inputs
    {
        const std::string& input_name_pad = onnx_node.input(1);
        ir_tensor_t* tensor_pad = find_tensor(graph, input_name_pad);
        int64_t* data_pad = (int64_t*)tensor_pad->data;
        pad_param->pad_0_h = data_pad[0];
        pad_param->pad_0_w = data_pad[4];
        pad_param->pad_1_h = data_pad[1];
        pad_param->pad_1_w = data_pad[5];
        pad_param->pad_2_h = data_pad[2];
        pad_param->pad_2_w = data_pad[6];
        pad_param->pad_3_h = data_pad[3];
        pad_param->pad_3_w = data_pad[7];

        if (onnx_node.input_size() > 2)
        {
            const std::string& input_name_value = onnx_node.input(2);
            ir_tensor_t* tensor_value = find_tensor(graph, input_name_value);
            float* data_value = (float*)tensor_value->data;
            pad_param->value = data_value[0];
        }
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
            else if (attr.s() == "edge")
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

    return 0;
}

static int load_reduce(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct reduction_param* reduction_param = (struct reduction_param*)node->op.param_mem;
    const std::string& op_name = onnx_node.op_type();

    if (op_name == "ReduceSum")
    {
        reduction_param->type = 0;
    }
    else if (op_name == "ReduceMean")
    {
        reduction_param->type = 1;
    }
    else if (op_name == "ReduceSumSquare")
    {
        reduction_param->type = 3;
    }
    else if (op_name == "ReduceMax")
    {
        reduction_param->type = 4;
    }
    else if (op_name == "ReduceMin")
    {
        reduction_param->type = 5;
    }
    else if (op_name == "ReduceProd")
    {
        reduction_param->type = 6;
    }
    else if (op_name == "ReduceLogSum")
    {
        reduction_param->type = 9;
    }
    else if (op_name == "ReduceLogSumExp")
    {
        reduction_param->type = 10;
    }

    reduction_param->dim_0 = -2;
    reduction_param->dim_1 = -2;
    reduction_param->dim_2 = -2;
    reduction_param->dim_3 = -2;
    reduction_param->keepdim = 1;

    ir_tensor_t* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    int input_dim_num = input_tensor->dim_num;
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
                {
                    attr_0 = input_dim_num + attr.ints(0);
                }
                reduction_param->dim_0 = attr_0;
            }
            else if (axis_size == 2)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                if (attr.ints(0) < 0)
                {
                    attr_0 = input_dim_num + attr.ints(0);
                }
                if (attr.ints(1) < 0)
                {
                    attr_0 = input_dim_num + attr.ints(1);
                }
                reduction_param->dim_0 = attr_0;
                reduction_param->dim_1 = attr_1;
            }
            else if (axis_size == 3)
            {
                int attr_0 = attr.ints(0);
                int attr_1 = attr.ints(1);
                int attr_2 = attr.ints(2);
                if (attr.ints(0) < 0)
                {
                    attr_0 = input_dim_num + attr.ints(0);
                }
                if (attr.ints(1) < 0)
                {
                    attr_0 = input_dim_num + attr.ints(1);
                }
                if (attr.ints(2) < 0)
                {
                    attr_0 = input_dim_num + attr.ints(2);
                }
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
                {
                    attr_0 = input_dim_num + attr.ints(0);
                }
                if (attr.ints(1) < 0)
                {
                    attr_0 = input_dim_num + attr.ints(1);
                }
                if (attr.ints(2) < 0)
                {
                    attr_0 = input_dim_num + attr.ints(2);
                }
                if (attr.ints(3) < 0)
                {
                    attr_0 = input_dim_num + attr.ints(3);
                }
                reduction_param->dim_0 = attr_0;
                reduction_param->dim_1 = attr_1;
                reduction_param->dim_2 = attr_2;
                reduction_param->dim_3 = attr_3;
            }
        }
    }
    return 0;
}

static int load_argmax(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct argmax_param* argmax_param = (struct argmax_param*)node->op.param_mem;

    argmax_param->axis = GetAttributeOrDefault<int>(onnx_node, "axis", 0);
    argmax_param->keepdims = GetAttributeOrDefault<int>(onnx_node, "keepdims", 1);

    return 0;
}

static int load_argmin(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct argmin_param* argmin_param = (struct argmin_param*)node->op.param_mem;

    argmin_param->axis = GetAttributeOrDefault<int>(onnx_node, "axis", 0);
    argmin_param->keepdims = GetAttributeOrDefault<int>(onnx_node, "keepdims", 1);

    return 0;
}

static int load_log_softmax(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct logsoftmax_param* logsoftmax_param = (struct logsoftmax_param*)node->op.param_mem;

    logsoftmax_param->axis = GetAttributeOrDefault<int>(onnx_node, "axis", -1);

    return 0;
}

static int load_deconv(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct deconv_param* deconv_param = (struct deconv_param*)node->op.param_mem;

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
            deconv_param->dilation_w = attr.ints(1);
        }
        else
            fprintf(stderr, "attr.name: %s \n", attr.name().c_str());
    }

    /* update the input tensor data layout */
    for (int k = 0; k < onnx_node.input_size(); k++)
    {
        const std::string& input_name = onnx_node.input(k);
        ir_tensor_t* tensor = find_tensor(graph, input_name);
        if (k == 1) // weight
        {
            int* dim = tensor->dims;
            /* onnx hide the output channel in weight ..*/
            /* The number of channels in the output should be equal to W.shape[1] * group */
            deconv_param->num_output = dim[1] * deconv_param->group;
            deconv_param->kernel_h = dim[2];
            deconv_param->kernel_w = dim[3];
        }
    }

    return 0;
}

static int load_scatter(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct scatter_param* scatter_param = (struct scatter_param*)node->op.param_mem;

    int size = onnx_node.attribute_size();
    scatter_param->axis = 0;
    scatter_param->is_onnx = 1;
    for (int i = 0; i < size; i++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(i);
        if (attr.name() == "axis")
        {
            scatter_param->axis = attr.i();
        }
    }

    return 0;
}

static int load_selu(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct selu_param* selu_param = (struct selu_param*)node->op.param_mem;

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "alpha")
        {
            selu_param->alpha = attr.f();
        }
        else if (attr.name() == "gamma")
        {
            selu_param->lambda = attr.f();
        }
    }

    return 0;
}

static int load_hard_sigmoid(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct hard_sigmoid_param* hard_sigmoid_param = (struct hard_sigmoid_param*)node->op.param_mem;
    hard_sigmoid_param->alpha = 1 / 6.f;
    hard_sigmoid_param->beta = 0.5f;
    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "alpha")
        {
            hard_sigmoid_param->alpha = attr.f();
        }
        else if (attr.name() == "beta")
        {
            hard_sigmoid_param->beta = attr.f();
        }
    }

    return 0;
}

static int load_tile(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct tile_param* tile_param = (struct tile_param*)node->op.param_mem;
    tile_param->frame_flag = 1;

    return 0;
}

static int load_cast(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct cast_param* cast_param = (struct cast_param*)node->op.param_mem;

    for (int k = 0; k < onnx_node.attribute_size(); k++)
    {
        const onnx::AttributeProto& attr = onnx_node.attribute(k);
        if (attr.name() == "to")
            cast_param->type_to = attr.i();
    }
    cast_param->type_from = 1;

    return 0;
}

static int load_depth_to_space(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct depthtospace_param* depthtospace_param = (struct depthtospace_param*)node->op.param_mem;
    depthtospace_param->block_size = GetAttributeOrThrow<int>(onnx_node, "blocksize");

    return 0;
}

static int load_instance_norm(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct instancenorm_Param* instancenorm_param = (struct instancenorm_Param*)node->op.param_mem;
    instancenorm_param->eps = GetAttributeOrDefault<float>(onnx_node, "epsilon", 1e-5);

    return 0;
}

static int load_resize(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct interp_param* interp_param = (struct interp_param*)node->op.param_mem;
    interp_param->height_scale = 0;
    interp_param->width_scale = 0;

    std::string coordinate_transformation_mode = GetAttributeOrDefault<std::string>(onnx_node, "coordinate_transformation_mode", "half_pixel");
    TASSERT(coordinate_transformation_mode == "half_pixel" || coordinate_transformation_mode == "align_corners" || coordinate_transformation_mode == "asymmetric");
    int align_corner = (coordinate_transformation_mode == "align_corners");

    if (onnx_node.input_size() == 1)
    {
        for (int k = 0; k < onnx_node.attribute_size(); k++)
        {
            const onnx::AttributeProto& attr = onnx_node.attribute(k);
            if (attr.name() == "scales")
            {
                interp_param->height_scale = attr.f();
                interp_param->width_scale = attr.f();
            }
        }
    }
    else if (onnx_node.input_size() == 2) // opset 10
    {
        const std::string& input_name = onnx_node.input(1);
        ir_tensor_t* tensor = find_tensor(graph, input_name);
        float* data = (float*)tensor->data;

        interp_param->height_scale = data[2];
        interp_param->width_scale = data[3];
    }
    else if (onnx_node.input_size() == 3) // opset 11
    {
        const std::string& input_name = onnx_node.input(2);
        ir_tensor_t* tensor = find_tensor(graph, input_name);
        float* data = (float*)tensor->data;

        interp_param->height_scale = data[2];
        interp_param->width_scale = data[3];
    }
    else if (onnx_node.input_size() == 4)
    {
        const std::string& size_name = onnx_node.input(3); // sizes
        ir_tensor_t* size_tensor = find_tensor(graph, size_name);

        int64_t* output_dims = (int64_t*)size_tensor->data;
        if (size_tensor->elem_num == 4)
        {
            interp_param->output_height = output_dims[2];
            interp_param->output_width = output_dims[3];
        }
    }
    else
    {
        fprintf(stderr, "Not support the num of inputs > 3, please check the onnx model or update the codes of convert tool\n");
        return -1;
    }

    std::string mode = GetAttributeOrDefault<std::string>(onnx_node, "mode", "nearest");
    if (mode == "nearest")
    {
        interp_param->resize_type = 1;
    }
    else if (mode == "bilinear" || mode == "linear")
    {
        interp_param->resize_type = align_corner == 0 ? 2 : 4;
    }

    return 0;
}

static int load_LSTM(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct lstm_param* lstm_param = (struct lstm_param*)node->op.param_mem;

    lstm_param->mxnet_flag = 0;
    lstm_param->hidden_size = GetAttributeOrThrow<int>(onnx_node, "hidden_size");
    lstm_param->cell_size = lstm_param->hidden_size;

    return 0;
}

static int load_expand(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct expand_param* expand_param = (struct expand_param*)node->op.param_mem;

    ir_tensor_t* shape_tensor = find_tensor(graph, onnx_node.input(1));
    if (shape_tensor == nullptr)
    {
        fprintf(stderr, "find shape tensor of expand node failed.\n");
        return -1;
    }
    int size = shape_tensor->elem_num;
    expand_param->ex_shape = (int*)sys_malloc(sizeof(int) * size);
    expand_param->dim_num = size;
    int64_t* data = (int64_t*)shape_tensor->data;
    for (int i = 0; i < size; i++)
    {
        expand_param->ex_shape[i] = data[i];
    }
    return 0;
}

static int load_gru(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    gru_param* param = (gru_param*)node->op.param_mem;
    param->hidden_size = GetAttributeOrThrow<int>(onnx_node, "hidden_size");
    param->clip = 0;
    param->output_len = 1;
    param->sequence_len = 1;
    param->input_size = 1;
    param->has_clip = 0;
    param->has_gate_bias = 0;
    param->has_candidate_bias = 0;
    param->has_init_state = 0;

    return 0;
}

static int load_layer_norm(ir_graph_t* graph, ir_node_t* node, const onnx::NodeProto& onnx_node)
{
    struct layernorm_Param* layernorm_param = (struct layernorm_Param*)node->op.param_mem;
    layernorm_param->eps = GetAttributeOrDefault<float>(onnx_node, "epsilon", 1e-5);

    return 0;
}

/*
*   OPERAOTR REGISTER FUNCTION DEFINE FOR ONNX SERIALIZER START
*/
void onnx_serializer::register_op_load()
{
    op_load_map["Abs"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["Acos"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["And"] = std::pair<int, op_load_t>(OP_LOGICAL, load_logical);
    op_load_map["ArgMax"] = std::pair<int, op_load_t>(OP_ARGMAX, load_argmax);
    op_load_map["ArgMin"] = std::pair<int, op_load_t>(OP_ARGMIN, load_argmin);
    op_load_map["Asin"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["Atan"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["AveragePool"] = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["Add"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["BatchNormalization"] = std::pair<int, op_load_t>(OP_BATCHNORM, load_bn);
    op_load_map["Conv"] = std::pair<int, op_load_t>(OP_CONV, load_conv);
    op_load_map["ConvTranspose"] = std::pair<int, op_load_t>(OP_DECONV, load_deconv);
    op_load_map["Concat"] = std::pair<int, op_load_t>(OP_CONCAT, load_concat);
    op_load_map["Clip"] = std::pair<int, op_load_t>(OP_CLIP, load_clip);
    op_load_map["Ceil"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["Cos"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["Cast"] = std::pair<int, op_load_t>(OP_CAST, load_cast);
    op_load_map["Dropout"] = std::pair<int, op_load_t>(OP_DROPOUT, load_no_param);
    op_load_map["DepthToSpace"] = std::pair<int, op_load_t>(OP_DEPTHTOSPACE, load_depth_to_space);
    op_load_map["Div"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Elu"] = std::pair<int, op_load_t>(OP_ELU, load_elu);
    op_load_map["Exp"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Expand"] = std::pair<int, op_load_t>(OP_EXPAND, load_expand);
    op_load_map["Equal"] = std::pair<int, op_load_t>(OP_COMPARISON, load_comparison);
    op_load_map["Flatten"] = std::pair<int, op_load_t>(OP_FLATTEN, load_flatten);
    op_load_map["Floor"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Gemm"] = std::pair<int, op_load_t>(OP_GEMM, load_gemm);
    op_load_map["Gather"] = std::pair<int, op_load_t>(OP_GATHER, load_gather);
    op_load_map["Greater"] = std::pair<int, op_load_t>(OP_COMPARISON, load_comparison);
    op_load_map["GRU"] = std::pair<int, op_load_t>(OP_GRU, load_gru);
    op_load_map["GlobalAveragePool"] = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["HardSwish"] = std::pair<int, op_load_t>(OP_HARDSWISH, load_no_param);
    op_load_map["HardSigmoid"] = std::pair<int, op_load_t>(OP_HARDSIGMOID, load_hard_sigmoid);
    op_load_map["InstanceNormalization"] = std::pair<int, op_load_t>(OP_INSTANCENORM, load_instance_norm);
    op_load_map["Log"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["LRN"] = std::pair<int, op_load_t>(OP_LRN, load_LRN);
    op_load_map["Less"] = std::pair<int, op_load_t>(OP_COMPARISON, load_comparison);
    op_load_map["LSTM"] = std::pair<int, op_load_t>(OP_LSTM, load_LSTM);
    op_load_map["LeakyRelu"] = std::pair<int, op_load_t>(OP_RELU, load_leaky_relu);
    op_load_map["LogSoftmax"] = std::pair<int, op_load_t>(OP_LOGSOFTMAX, load_log_softmax);
    op_load_map["Mul"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Max"] = std::pair<int, op_load_t>(OP_MAXIMUM, load_no_param);
    op_load_map["Min"] = std::pair<int, op_load_t>(OP_MINIMUM, load_no_param);
    op_load_map["Mean"] = std::pair<int, op_load_t>(OP_MEAN, load_no_param);
    op_load_map["MatMul"] = std::pair<int, op_load_t>(OP_MATMUL, load_matmul);
    op_load_map["MaxPool"] = std::pair<int, op_load_t>(OP_POOL, load_pool);
    op_load_map["Neg"] = std::pair<int, op_load_t>(OP_UNARY, load_unary);
    op_load_map["Or"] = std::pair<int, op_load_t>(OP_LOGICAL, load_logical);
    op_load_map["Pad"] = std::pair<int, op_load_t>(OP_PAD, load_pad);
    op_load_map["Pow"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["PRelu"] = std::pair<int, op_load_t>(OP_PRELU, load_no_param);
    op_load_map["Relu"] = std::pair<int, op_load_t>(OP_RELU, load_relu);
    op_load_map["Resize"] = std::pair<int, op_load_t>(OP_INTERP, load_resize);
    op_load_map["Reshape"] = std::pair<int, op_load_t>(OP_RESHAPE, load_reshape);
    op_load_map["ReduceL2"] = std::pair<int, op_load_t>(OP_REDUCEL2, load_reducel2);
    op_load_map["ReduceMean"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduce);
    op_load_map["ReduceLogSumExp"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduce);
    op_load_map["ReduceLogSum"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduce);
    op_load_map["ReduceMax"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduce);
    op_load_map["ReduceMin"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduce);
    op_load_map["ReduceProd"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduce);
    op_load_map["ReduceSumSquare"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduce);
    op_load_map["ReduceSum"] = std::pair<int, op_load_t>(OP_REDUCTION, load_reduce);
    op_load_map["Reciprocal"] = std::pair<int, op_load_t>(OP_RECIPROCAL, load_no_param);
    op_load_map["Sub"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Selu"] = std::pair<int, op_load_t>(OP_SELU, load_selu);
    op_load_map["Sqrt"] = std::pair<int, op_load_t>(OP_ELTWISE, load_eltwise);
    op_load_map["Slice"] = std::pair<int, op_load_t>(OP_SLICE, load_slice);
    op_load_map["Split"] = std::pair<int, op_load_t>(OP_SPLIT, load_split);
    op_load_map["Shape"] = std::pair<int, op_load_t>(OP_SHAPE, load_no_param);
    op_load_map["Squeeze"] = std::pair<int, op_load_t>(OP_SQUEEZE, load_squeeze);
    op_load_map["Scatter"] = std::pair<int, op_load_t>(OP_SCATTER, load_scatter);
    op_load_map["Sigmoid"] = std::pair<int, op_load_t>(OP_SIGMOID, load_no_param);
    op_load_map["Softmax"] = std::pair<int, op_load_t>(OP_SOFTMAX, load_softmax);
    op_load_map["Softplus"] = std::pair<int, op_load_t>(OP_SOFTPLUS, load_no_param);
    op_load_map["Tanh"] = std::pair<int, op_load_t>(OP_TANH, load_no_param);
    op_load_map["Tile"] = std::pair<int, op_load_t>(OP_TILE, load_tile);
    op_load_map["Transpose"] = std::pair<int, op_load_t>(OP_TRANSPOSE, load_transpose);
    op_load_map["Upsample"] = std::pair<int, op_load_t>(OP_INTERP, load_interp);
    op_load_map["Unsqueeze"] = std::pair<int, op_load_t>(OP_UNSQUEEZE, load_unsqueeze);
    op_load_map["Where"] = std::pair<int, op_load_t>(OP_WHERE, load_no_param);
    op_load_map["Gelu"] = std::pair<int, op_load_t>(OP_GELU, load_no_param);
    op_load_map["LayerNorm"] = std::pair<int, op_load_t>(OP_LAYERNORM, load_layer_norm);
}
/*
*   OPERATOR REGISTER FUNCTION DEFINE FOR ONNX SERIALIZER END
*/
